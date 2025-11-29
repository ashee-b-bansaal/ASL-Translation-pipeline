"""
Script to analyze expression offsets from alignment file.
Creates a CSV file marking sentences with cumulative offset thresholds.
A sentence with all expressions within n offset is marked for all columns >= n.
"""

import re
import csv
from collections import defaultdict
from pathlib import Path


def extract_expressions_and_positions(text):
    """
    Extract expressions and their word positions from text.
    Returns: dict mapping expression -> list of word indices
    """
    # First, split by spaces to get words
    words = text.split()
    expression_positions = defaultdict(list)
    
    # Track word index (ignoring expressions in count)
    word_idx = 0
    
    for word in words:
        # Check if word has expressions
        # Pattern: WORD(expr1,expr2) or WORD(expr)
        match = re.match(r'^(\w+)(\([^)]+\))?$', word)
        if match:
            base_word = match.group(1)
            expr_str = match.group(2)
            
            if expr_str:
                # Remove parentheses and split by comma
                expressions = expr_str[1:-1].split(',')
                for expr in expressions:
                    expr = expr.strip()
                    # Only track non-emotion expressions
                    if expr in ['raise', 'furrow', 'shake', 'mm', 'th', 'cha', 'cs']:
                        expression_positions[expr].append(word_idx)
            
            # Increment word index for the base word
            word_idx += 1
    
    return expression_positions


def get_word_list_without_expressions(text):
    """Get list of words without expression annotations."""
    words = text.split()
    word_list = []
    
    for word in words:
        # Extract base word (remove expression annotations)
        match = re.match(r'^(\w+)', word)
        if match:
            word_list.append(match.group(1))
    
    return word_list


def calculate_offset(gt_pos, pred_pos):
    """Calculate absolute offset between ground truth and prediction positions."""
    return abs(pred_pos - gt_pos)


def get_max_offset_for_sentence(gt_text, pred_text):
    """
    Calculate the maximum offset for all expressions in a sentence.
    
    Args:
        gt_text: Ground truth text (emotions already removed)
        pred_text: Predicted text
    
    Returns:
        Maximum offset value, or None if no expressions found
    """
    # Extract expressions and positions from GT
    gt_expressions = extract_expressions_and_positions(gt_text)
    pred_expressions = extract_expressions_and_positions(pred_text)
    
    # Expressions to track
    expressions_to_track = ['raise', 'furrow', 'shake', 'mm', 'th', 'cha', 'cs']
    
    max_offset = -1
    has_expressions = False
    
    # For each expression in GT, find corresponding in prediction
    for expr in expressions_to_track:
        if expr in gt_expressions:
            has_expressions = True
            # Create a copy of prediction positions for this expression
            # so we can remove matched ones
            available_pred_positions = list(pred_expressions.get(expr, []))
            
            for gt_pos in gt_expressions[expr]:
                # Find closest available position in prediction
                if available_pred_positions:
                    # Find closest position
                    best_offset = None
                    best_pred_pos = None
                    best_idx = None
                    
                    for idx, pred_pos in enumerate(available_pred_positions):
                        offset = calculate_offset(gt_pos, pred_pos)
                        if best_offset is None or offset < best_offset:
                            best_offset = offset
                            best_pred_pos = pred_pos
                            best_idx = idx
                    
                    if best_offset is not None:
                        max_offset = max(max_offset, best_offset)
                        # Remove matched position to avoid double-counting
                        available_pred_positions.pop(best_idx)
                else:
                    # Expression not found in prediction - use a high offset
                    # Get sentence length to use as offset
                    gt_words = get_word_list_without_expressions(gt_text)
                    pred_words = get_word_list_without_expressions(pred_text)
                    max_sentence_length = max(len(gt_words), len(pred_words))
                    missing_offset = max_sentence_length + 1
                    max_offset = max(max_offset, missing_offset)
    
    if not has_expressions:
        return None
    
    return max_offset if max_offset >= 0 else None


def analyze_alignment_file(input_file: str, output_file: str, max_offset_column: int = 10):
    """
    Analyze alignment file and create CSV with offset thresholds.
    
    Args:
        input_file: Path to alignment_without_LM_emo_sep.txt
        output_file: Path to save CSV file
        max_offset_column: Maximum offset column to include (default: 10)
    """
    print(f"Reading alignment file: {input_file}")
    
    sentences_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Skip metrics section
            if line.startswith('=') or line.startswith('Exact') or line.startswith('SENTENCES') or line.startswith('Total') or line.startswith('Line'):
                continue
            
            # Format: "clean_ground_truth - clean_aligned_sentence --- gt_emotion - rec_emotion"
            if ' --- ' not in line:
                continue
            
            parts = line.split(' --- ')
            if len(parts) != 2:
                continue
            
            sentence_part = parts[0].strip()
            
            # Split sentence part: "gt_text - pred_text"
            if ' - ' not in sentence_part:
                continue
            
            gt_text, pred_text = sentence_part.split(' - ', 1)
            gt_text = gt_text.strip()
            pred_text = pred_text.strip()
            
            # Get max offset for this sentence
            max_offset = get_max_offset_for_sentence(gt_text, pred_text)
            
            # Create row data
            row_data = {
                'sentence_id': line_num,
                'ground_truth': gt_text,
                'predicted': pred_text
            }
            
            # Mark all columns from 0 to max_offset_column as 1 if max_offset <= offset
            # If max_offset is None (no expressions), mark all as 0
            if max_offset is None:
                # No expressions found - mark all as 0
                for offset in range(max_offset_column + 1):
                    row_data[f'offset_{offset}'] = 0
            else:
                # If sentence has all expressions within max_offset, then it qualifies
                # for all thresholds >= max_offset. So mark all columns from 0 to max_offset_column as 1
                # (since having expressions within n offset means they're also within n+1, n+2, etc.)
                for offset in range(max_offset_column + 1):
                    if max_offset <= offset:
                        row_data[f'offset_{offset}'] = 1
                    else:
                        row_data[f'offset_{offset}'] = 0
            
            # Store max_offset for reference
            row_data['max_offset'] = max_offset if max_offset is not None else 'N/A'
            
            sentences_data.append(row_data)
    
    print(f"Processed {len(sentences_data)} sentences")
    
    # Write CSV file
    print(f"Writing CSV file: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create column headers
    fieldnames = ['sentence_id', 'ground_truth', 'predicted', 'max_offset']
    for offset in range(max_offset_column + 1):
        fieldnames.append(f'offset_{offset}')
    
    # Count sentences for each offset threshold and collect examples
    offset_counts = defaultdict(int)
    offset_examples = defaultdict(list)  # Store example sentences for each offset
    sentences_with_expressions = 0
    
    for row in sentences_data:
        if row['max_offset'] != 'N/A':
            sentences_with_expressions += 1
            max_off = row['max_offset']
            # Count how many sentences have all expressions within each offset
            for offset in range(max_offset_column + 1):
                if row[f'offset_{offset}'] == 1:
                    offset_counts[offset] += 1
                    # Store example - prefer examples that have max_offset equal to the threshold
                    # or close to it (for better representation)
                    if len(offset_examples[offset]) < 3:
                        # Prefer examples where max_offset equals the threshold
                        if max_off == offset:
                            # Insert at beginning if it's an exact match
                            offset_examples[offset].insert(0, {
                                'gt': row['ground_truth'],
                                'pred': row['predicted'],
                                'max_offset': max_off
                            })
                        elif len(offset_examples[offset]) < 3:
                            # Otherwise append
                            offset_examples[offset].append({
                                'gt': row['ground_truth'],
                                'pred': row['predicted'],
                                'max_offset': max_off
                            })
                    # Keep only first 3 examples
                    if len(offset_examples[offset]) > 3:
                        offset_examples[offset] = offset_examples[offset][:3]
    
    # Write CSV file with data rows
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sentences_data)
        
        # Add empty row for separation
        writer.writerow({})
        
        # Add summary header row
        summary_header = {
            'sentence_id': '',
            'ground_truth': 'OFFSET STATISTICS',
            'predicted': '',
            'max_offset': ''
        }
        for offset in range(max_offset_column + 1):
            summary_header[f'offset_{offset}'] = ''
        writer.writerow(summary_header)
        
        # Add empty row
        writer.writerow({})
        
        # Add total sentences row
        total_row = {
            'sentence_id': '',
            'ground_truth': 'Total Sentences',
            'predicted': '',
            'max_offset': len(sentences_data)
        }
        for offset in range(max_offset_column + 1):
            total_row[f'offset_{offset}'] = ''
        writer.writerow(total_row)
        
        # Add sentences with expressions row
        expr_row = {
            'sentence_id': '',
            'ground_truth': 'Sentences with Expressions',
            'predicted': '',
            'max_offset': sentences_with_expressions
        }
        for offset in range(max_offset_column + 1):
            expr_row[f'offset_{offset}'] = ''
        writer.writerow(expr_row)
        
        # Add empty row
        writer.writerow({})
        
        # Add offset count rows with examples
        for offset in range(max_offset_column + 1):
            count = offset_counts[offset]
            percentage = (count / sentences_with_expressions * 100) if sentences_with_expressions > 0 else 0.0
            
            # First row: offset header with count and percentage
            offset_row = {
                'sentence_id': '',
                'ground_truth': f'Offset ≤ {offset}',
                'predicted': '',
                'max_offset': f'{count} ({percentage:.2f}%)'
            }
            for off in range(max_offset_column + 1):
                offset_row[f'offset_{off}'] = ''
            writer.writerow(offset_row)
            
            # Add example sentences for this offset (if any)
            if offset in offset_examples and offset_examples[offset]:
                for idx, example in enumerate(offset_examples[offset], 1):
                    example_row = {
                        'sentence_id': '',
                        'ground_truth': f'  Example {idx} GT: {example["gt"]}',
                        'predicted': f'Pred: {example["pred"]}',
                        'max_offset': f'max_offset={example["max_offset"]}'
                    }
                    for off in range(max_offset_column + 1):
                        example_row[f'offset_{off}'] = ''
                    writer.writerow(example_row)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("OFFSET ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"Total sentences: {len(sentences_data)}")
    print(f"Sentences with expressions: {sentences_with_expressions}")
    print(f"Sentences without expressions: {len(sentences_data) - sentences_with_expressions}")
    print("\nCumulative offset statistics:")
    print(f"{'Offset':<10} {'Sentences':<15} {'Percentage':<15}")
    print("-" * 40)
    
    for offset in range(max_offset_column + 1):
        count = offset_counts[offset]
        percentage = (count / sentences_with_expressions * 100) if sentences_with_expressions > 0 else 0.0
        print(f"≤ {offset:<8} {count:<15} {percentage:>6.2f}%")
    
    print(f"\nCSV file saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze expression offsets and create CSV with cumulative offset thresholds'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to alignment_without_LM_emo_sep.txt file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to save CSV file'
    )
    parser.add_argument(
        '--max_offset',
        type=int,
        default=10,
        help='Maximum offset column to include (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    
    analyze_alignment_file(args.input_file, args.output_file, args.max_offset)

