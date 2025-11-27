#!/usr/bin/env python3
"""
Alignment Analysis Script
Calculates offset statistics for facial expressions in alignment results.
"""

import re
import os
from collections import defaultdict
from pathlib import Path


def remove_emotions(text):
    """Remove emotion annotations (happy, sad, angry) but keep expressions."""
    # Remove emotion annotations: (happy), (sad), (angry)
    text = re.sub(r'\(happy\)', '', text)
    text = re.sub(r'\(sad\)', '', text)
    text = re.sub(r'\(angry\)', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


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


def analyze_alignment_file(input_file, output_file):
    """
    Analyze alignment file and calculate expression offset statistics.
    """
    # Track statistics: expression -> offset -> count
    expression_stats = defaultdict(lambda: defaultdict(int))
    expression_totals = defaultdict(int)
    
    # Expressions to track
    expressions_to_track = ['raise', 'furrow', 'shake', 'mm', 'th', 'cha', 'cs']
    
    print(f"Reading alignment file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines and separator lines
        if not line or line.startswith('='):
            continue
        
        # Split by ' - ' to get GT and prediction
        if ' - ' not in line:
            continue
        
        gt_text, pred_text = line.split(' - ', 1)
        
        # Remove emotions from GT (but keep expressions)
        gt_cleaned = remove_emotions(gt_text)
        pred_cleaned = pred_text  # Keep prediction as is
        
        # Extract expressions and positions from GT
        gt_expressions = extract_expressions_and_positions(gt_cleaned)
        
        # Extract expressions and positions from prediction
        pred_expressions = extract_expressions_and_positions(pred_cleaned)
        
        # Get word lists to calculate max possible offset
        gt_words = get_word_list_without_expressions(gt_cleaned)
        pred_words = get_word_list_without_expressions(pred_cleaned)
        max_sentence_length = max(len(gt_words), len(pred_words))
        
        # For each expression in GT, find corresponding in prediction
        # Use one-to-one matching to avoid double-counting
        for expr in expressions_to_track:
            if expr in gt_expressions:
                # Create a copy of prediction positions for this expression
                # so we can remove matched ones
                available_pred_positions = list(pred_expressions.get(expr, []))
                
                for gt_pos in gt_expressions[expr]:
                    expression_totals[expr] += 1
                    
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
                            expression_stats[expr][best_offset] += 1
                            # Remove matched position to avoid double-counting
                            available_pred_positions.pop(best_idx)
                    else:
                        # Expression not found in prediction - use max sentence length as offset
                        # This represents expressions that are completely missing
                        missing_offset = max_sentence_length + 1
                        expression_stats[expr][missing_offset] += 1
    
    # Generate report
    print(f"\nGenerating analysis report: {output_file}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALIGNMENT ANALYSIS - EXPRESSION OFFSET STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total lines processed: {len([l for l in lines if l.strip() and not l.strip().startswith('=')])}\n\n")
        
        # For each expression, calculate statistics
        for expr in expressions_to_track:
            if expression_totals[expr] == 0:
                continue
            
            f.write("-" * 80 + "\n")
            f.write(f"EXPRESSION: {expr.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total occurrences: {expression_totals[expr]}\n\n")
            
            # Get all offsets and sort (treat missing as last)
            offsets = sorted(expression_stats[expr].keys())
            
            for offset in offsets:
                count = expression_stats[expr][offset]
                percentage = (count / expression_totals[expr]) * 100
                if offset > 10:  # Likely a missing expression
                    f.write(f"  Missing/High offset (>{offset}): {count} occurrences ({percentage:.2f}%)\n")
                else:
                    f.write(f"  {offset} offset: {count} occurrences ({percentage:.2f}%)\n")
            
            f.write("\n")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Expression':<15} {'Total':<10} {'0 offset':<15} {'1 offset':<15} {'2+ offset':<15}\n")
        f.write("-" * 80 + "\n")
        
        for expr in expressions_to_track:
            if expression_totals[expr] == 0:
                continue
            
            total = expression_totals[expr]
            offset_0 = expression_stats[expr].get(0, 0)
            offset_1 = expression_stats[expr].get(1, 0)
            offset_2plus = sum(count for off, count in expression_stats[expr].items() if off >= 2)
            
            pct_0 = (offset_0 / total) * 100
            pct_1 = (offset_1 / total) * 100
            pct_2plus = (offset_2plus / total) * 100
            
            f.write(f"{expr:<15} {total:<10} {offset_0} ({pct_0:.1f}%){'':<5} {offset_1} ({pct_1:.1f}%){'':<5} {offset_2plus} ({pct_2plus:.1f}%)\n")
    
    print(f"Analysis complete! Results saved to: {output_file}")


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    alignment_file = base_dir / "final_pipeline_txt_220110" / "alignment" / "alignment_without_LM.txt"
    output_dir = base_dir / "final_pipeline_txt_220110" / "analysis"
    output_file = output_dir / "alignment_analysis.txt"
    
    # Check if input file exists
    if not alignment_file.exists():
        print(f"Error: Alignment file not found: {alignment_file}")
        return
    
    # Run analysis
    analyze_alignment_file(str(alignment_file), str(output_file))


if __name__ == "__main__":
    main()

