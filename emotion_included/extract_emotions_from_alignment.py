"""
Script to extract emotions from alignment_without_LM.txt file.
Reads alignment file and extracts emotions from both ground truth and aligned sentences.
Removes emotions (happy, sad, angry) from both sentences in the output.
Outputs: clean_ground_truth - clean_aligned_sentence --- gt_emotion - rec_emotion
"""

import re
from typing import Optional, List, Tuple
from collections import Counter
from pathlib import Path


# Emotional expressions to extract
EMOTION_EXPRESSIONS = {"happy", "sad", "angry"}


def parse_gloss(gloss: str) -> List[Tuple[str, List[str]]]:
    """
    Parse ASL gloss format: "MY FATHER WANT(sad) SELL(sad) HIS CAR(mm)"
    
    Returns:
        List of tuples: (sign, [expressions])
    """
    pattern = r'(\w+(?:-\w+)*)(?:\(([^)]+)\))?'
    matches = re.findall(pattern, gloss)
    
    result = []
    for sign, exp_str in matches:
        expressions = [e.strip() for e in exp_str.split(',')] if exp_str else []
        result.append((sign, expressions))
    
    return result


def remove_emotions_from_gloss(gloss: str) -> str:
    """
    Remove emotion-only expressions (happy, sad, angry) from a gloss string.
    Preserves other expressions and drops empty parentheses.
    
    Args:
        gloss: ASL gloss string (e.g., "MY FATHER(sad) WANT(sad) SELL(sad) HIS CAR(mm)")
    
    Returns:
        Cleaned gloss string with emotions removed (e.g., "MY FATHER WANT SELL HIS CAR(mm)")
    """
    pattern = r'(\S+)\(([^)]+)\)'

    def clean(match):
        sign = match.group(1)
        exprs = [expr.strip() for expr in match.group(2).split(',')]
        filtered = [expr for expr in exprs if expr.lower() not in EMOTION_EXPRESSIONS]
        if not filtered:
            return sign
        return f"{sign}({','.join(filtered)})"

    return re.sub(pattern, clean, gloss)


def extract_emotion_from_gloss(gloss: str) -> Optional[str]:
    """
    Extract emotion (happy, sad, or angry) from an ASL gloss sentence.
    If multiple emotions exist, choose the most frequent one.
    If tie, choose the first one found.
    
    Args:
        gloss: ASL gloss string (e.g., "MY FATHER(sad) WANT(sad) SELL(sad) HIS CAR")
    
    Returns:
        Emotion string ("happy", "sad", or "angry") or None if no emotion found
    """
    if not gloss or not gloss.strip():
        return None
    
    # Parse the gloss to get all expressions
    parsed = parse_gloss(gloss)
    
    # Collect all emotions from all expressions
    all_emotions = []
    for sign, expressions in parsed:
        for expr in expressions:
            if expr.lower() in EMOTION_EXPRESSIONS:
                all_emotions.append(expr.lower())
    
    if not all_emotions:
        return None
    
    # Count frequency of each emotion
    emotion_counts = Counter(all_emotions)
    
    # Find emotion with max frequency
    max_count = max(emotion_counts.values())
    candidates = [emotion for emotion, count in emotion_counts.items() if count == max_count]
    
    # If single candidate with max frequency, return it
    if len(candidates) == 1:
        return candidates[0]
    
    # If tie, choose the first one that appears in the sentence
    # (preserve order from original sentence)
    for emotion in all_emotions:
        if emotion in candidates:
            return emotion
    
    # Fallback (shouldn't reach here)
    return candidates[0] if candidates else None


def process_alignment_file(input_file: str, output_file: str):
    """
    Process alignment file and extract emotions for each line.
    
    Args:
        input_file: Path to alignment_without_LM.txt
        output_file: Path to output file (alignment_without_LM_emo_sep.txt)
    """
    print(f"Reading alignment file: {input_file}")
    
    results = []
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Skip metrics section
            if line.startswith('=') or line.startswith('ACCURACY') or line.startswith('Simple') or line.startswith('Exact') or line.startswith('Total'):
                continue
            
            # Format: "ground_truth - aligned_sentence"
            if ' - ' not in line:
                continue
            
            parts = line.split(' - ', 1)
            if len(parts) != 2:
                continue
            
            ground_truth = parts[0].strip()
            aligned_sentence = parts[1].strip()
            
            # Extract emotions first (before removing them)
            gt_emotion = extract_emotion_from_gloss(ground_truth)
            rec_emotion = extract_emotion_from_gloss(aligned_sentence)
            
            # Remove emotions from both sentences
            clean_ground_truth = remove_emotions_from_gloss(ground_truth)
            clean_aligned_sentence = remove_emotions_from_gloss(aligned_sentence)
            
            # Format: "clean_ground_truth - clean_aligned_sentence --- gt_emotion - rec_emotion"
            gt_emotion_str = gt_emotion if gt_emotion else 'none'
            rec_emotion_str = rec_emotion if rec_emotion else 'none'
            
            output_line = f"{clean_ground_truth} - {clean_aligned_sentence} --- {gt_emotion_str} - {rec_emotion_str}"
            results.append(output_line)
            total_lines += 1
    
    # Write output file
    print(f"Writing output file: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')
    
    # Calculate and write metrics
    exact_matches = 0
    sentence_match_emotion_mismatch = []
    
    for idx, result in enumerate(results, 1):
        # Parse the result line
        if ' --- ' in result:
            parts = result.split(' --- ')
            if len(parts) == 2:
                sentence_part = parts[0]
                emotion_part = parts[1]
                
                if ' - ' in sentence_part:
                    gt_sent, rec_sent = sentence_part.split(' - ', 1)
                    gt_sent = gt_sent.strip()
                    rec_sent = rec_sent.strip()
                    
                    if ' - ' in emotion_part:
                        gt_emo, rec_emo = emotion_part.split(' - ', 1)
                        gt_emo = gt_emo.strip()
                        rec_emo = rec_emo.strip()
                        
                        # Check if exact match: sentence matches AND emotions match
                        if gt_sent == rec_sent and gt_emo == rec_emo:
                            exact_matches += 1
                        # Track sentences where sentence matches but emotions don't
                        elif gt_sent == rec_sent and gt_emo != rec_emo:
                            sentence_match_emotion_mismatch.append({
                                'line': idx,
                                'sentence': gt_sent,
                                'gt_emotion': gt_emo,
                                'rec_emotion': rec_emo
                            })
    
    # Write metrics at the end
    with open(output_file, 'a', encoding='utf-8') as f:
        exact_match_rate = exact_matches / total_lines if total_lines > 0 else 0.0
        f.write(f"\nExact Match Accuracy: {exact_matches}/{total_lines} = {100*exact_match_rate:.2f}%\n")
        
        if sentence_match_emotion_mismatch:
            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"SENTENCES WHERE SENTENCE MATCHES BUT EMOTIONS DON'T MATCH\n")
            f.write(f"Total: {len(sentence_match_emotion_mismatch)}\n")
            f.write("=" * 80 + "\n\n")
            for item in sentence_match_emotion_mismatch:
                f.write(f"Line {item['line']}: {item['sentence']} --- GT Emotion: {item['gt_emotion']} | Recognized Emotion: {item['rec_emotion']}\n")
        else:
            f.write(f"\nNo sentences found where sentence matches but emotions don't match.\n")
    
    print(f"\nDone! Processed {total_lines} samples")
    print(f"Exact Match Accuracy: {exact_matches}/{total_lines} ({100*exact_match_rate:.2f}%)")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract emotions from alignment_without_LM.txt file'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Path to alignment_without_LM.txt file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Path to output file (alignment_without_LM_emo_sep.txt)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    
    process_alignment_file(args.input_file, args.output_file)

