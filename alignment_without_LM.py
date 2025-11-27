"""
Script to align CTC outputs from signs and expressions files (Version 2).
Implements Multi-Stream CTC Fusion with emotion filtering and improved assignment rules.
Outputs: ground_truth_ASL_gloss - aligned_ASL_gloss
Calculates two accuracy metrics: exact match and proportional match.
"""

import re
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import sys
import os
from pathlib import Path


# Emotional expressions to ignore
EMOTION_EXPRESSIONS = {"happy", "sad", "angry"}


def remove_emotions_from_gloss(gloss: str) -> str:
    """
    Remove emotion-only expressions (happy, sad, angry) from a gloss string.
    Preserves other expressions and drops empty parentheses.
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


def parse_file_line(line: str) -> Tuple[str, str, str, str, str, str]:
    """
    Parse a line from the data files.
    For signs: ground_truth - recognized <<<<< CTC_sequence >>>> ASL_gloss ----- session, sample
    For expressions: expressions - expressions ----- emotions - emotions <<<<< CTC_sequence >>>> ASL_gloss ----- session, sample
    """
    parts = line.split('-----')
    session_sample = parts[-1].strip() if len(parts) > 1 else ""
    session, sample = session_sample.split(',') if ',' in session_sample else ("", "")
    
    ctc_match = re.search(r'<<<<<(.*?)>>>>', line)
    ctc_sequence = ctc_match.group(1).strip() if ctc_match else ""
    
    gloss_match = re.search(r'>>>>(.*?)-----', line)
    asl_gloss = gloss_match.group(1).strip() if gloss_match else ""
    
    before_ctc = line.split('<<<<<')[0].strip()
    
    # Check if it's expressions format (has ----- in the middle)
    if ' ----- ' in before_ctc:
        # Expressions format: expressions - expressions ----- emotions - emotions
        parts_before = before_ctc.split(' ----- ')
        exp_part = parts_before[0].strip()
        if ' - ' in exp_part:
            ground_truth, recognized = exp_part.split(' - ', 1)
        else:
            ground_truth = exp_part
            recognized = exp_part
    else:
        # Signs format: ground_truth - recognized
        if ' - ' in before_ctc:
            ground_truth, recognized = before_ctc.split(' - ', 1)
            ground_truth = ground_truth.strip()
            recognized = recognized.strip()
        else:
            ground_truth = before_ctc
            recognized = before_ctc
    
    return ground_truth, recognized, ctc_sequence, asl_gloss, session.strip(), sample.strip()


def parse_gloss(gloss: str) -> List[Tuple[str, List[str]]]:
    """Parse ASL gloss format: "MY FATHER WANT(sad) SELL(sad) HIS CAR(mm)" """
    pattern = r'(\w+(?:-\w+)*)(?:\(([^)]+)\))?'
    matches = re.findall(pattern, gloss)
    
    result = []
    for sign, exp_str in matches:
        expressions = [e.strip() for e in exp_str.split(',')] if exp_str else []
        result.append((sign, expressions))
    
    return result


class Segment:
    """Represents a segment with token and time span."""
    def __init__(self, token: str, start: int, end: int):
        self.token = token
        self.start = start
        self.end = end
        self.expressions: List[str] = []
    
    def __repr__(self):
        return f"Segment(token={self.token}, start={self.start}, end={self.end}, exprs={self.expressions})"
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "expressions": self.expressions.copy()
        }


def collapse_ctc(frames: List[str]) -> List[Segment]:
    """
    STEP 1: Collapse CTC frame sequence into segments with time spans.
    
    Rules:
    - Remove blanks (#)
    - Collapse repeated tokens into a single segment
    - Merge segments if same token AND separated by <= 1 blank frame
    
    Args:
        frames: List of tokens, one per frame (e.g., ["#", "#", "FATHER", "FATHER", "WANT", ...])
    
    Returns:
        List of Segment objects with token, start, end
    """
    if not frames:
        return []
    
    segments = []
    current_token = frames[0]
    start_idx = 0
    
    for i in range(1, len(frames)):
        if frames[i] != current_token:
            # End of current segment
            if current_token != '#':  # Only add non-blank segments
                segments.append(Segment(current_token, start_idx, i - 1))
            current_token = frames[i]
            start_idx = i
    
    # Add last segment
    if current_token != '#':
        segments.append(Segment(current_token, start_idx, len(frames) - 1))
    
    # Merge identical tokens that are contiguous or separated by <= 1 blank frame
    merged_segments = []
    i = 0
    while i < len(segments):
        current = segments[i]
        j = i + 1
        
        # Look ahead to merge identical tokens
        while j < len(segments):
            next_seg = segments[j]
            # Check if same token and contiguous or separated by <= 1 frame
            gap = next_seg.start - current.end - 1
            if current.token == next_seg.token and gap <= 1:
                # Merge: extend current segment's end
                current = Segment(current.token, current.start, next_seg.end)
                j += 1
            else:
                break
        
        merged_segments.append(current)
        i = j
    
    return merged_segments


def filter_expressions(segments: List[Segment]) -> List[Segment]:
    """
    STEP 2: Filter expression segments to remove emotional expressions.
    
    Removes segments whose token is in EMOTION_EXPRESSIONS.
    Also handles comma-separated tokens (e.g., "shake,cha").
    
    Args:
        segments: List of expression segments
    
    Returns:
        Filtered list of segments (emotions removed, comma-separated tokens split)
    """
    filtered = []
    
    for seg in segments:
        # Handle comma-separated tokens (e.g., "shake,cha")
        tokens = [t.strip() for t in seg.token.split(',')]
        
        for token in tokens:
            # Skip emotional expressions
            if token in EMOTION_EXPRESSIONS:
                continue
            
            # Create a new segment for each token (sharing the same time span)
            filtered.append(Segment(token, seg.start, seg.end))
    
    return filtered


def compute_overlap(segA: Segment, segB: Segment) -> int:
    """
    Compute temporal overlap between two segments.
    
    Returns:
        Number of overlapping frames
    """
    overlap_start = max(segA.start, segB.start)
    overlap_end = min(segA.end, segB.end)
    return max(0, overlap_end - overlap_start + 1)


def choose_sign_for_expression(expr: Segment, sign_segments: List[Segment]) -> Optional[int]:
    """
    STEP 3: Choose the best sign for an expression using the specified rules.
    
    Rules:
    A. Maximum temporal overlap
    B. If overlaps tie: expression boundary rule
    C. If still tied: choose nearest-center
    D. Expression outside sign range: attach to first/last/nearest
    E. Expression spans many words: max overlap, then middle sign
    
    Returns:
        Index of chosen sign segment, or None if no suitable sign found
    """
    if not sign_segments:
        return None
    
    first_sign_start = sign_segments[0].start
    last_sign_end = sign_segments[-1].end
    
    # Rule D: Expression outside sign range
    if expr.end < first_sign_start:
        # Before first sign: attach to first sign
        return 0
    elif expr.start > last_sign_end:
        # After last sign: attach to last sign
        return len(sign_segments) - 1
    
    # Rule A: Find sign with maximum overlap
    max_overlap = -1
    candidates = []
    
    for idx, sign in enumerate(sign_segments):
        overlap = compute_overlap(expr, sign)
        if overlap > max_overlap:
            max_overlap = overlap
            candidates = [(idx, sign, overlap)]
        elif overlap == max_overlap and overlap > 0:
            candidates.append((idx, sign, overlap))
    
    # If no overlap found, use Rule D: nearest sign by center distance
    if max_overlap == 0:
        expr_center = (expr.start + expr.end) / 2
        best_idx = None
        min_distance = float('inf')
        
        for idx, sign in enumerate(sign_segments):
            sign_center = (sign.start + sign.end) / 2
            distance = abs(expr_center - sign_center)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx
        
        return best_idx
    
    # If single candidate with max overlap, return it
    if len(candidates) == 1:
        return candidates[0][0]
    
    # Rule B: If overlaps tie, use expression boundary rule
    # Choose sign whose boundary falls inside expression range
    for idx, sign, _ in candidates:
        if (expr.start <= sign.start <= expr.end) or (expr.start <= sign.end <= expr.end):
            return idx
    
    # Rule C: If still tied, choose nearest-center
    expr_center = (expr.start + expr.end) / 2
    best_idx = None
    min_distance = float('inf')
    
    for idx, sign, _ in candidates:
        sign_center = (sign.start + sign.end) / 2
        distance = abs(expr_center - sign_center)
        if distance < min_distance:
            min_distance = distance
            best_idx = idx
    
    return best_idx


def assign_expressions(sign_segments: List[Segment], expr_segments: List[Segment]):
    """
    STEP 3: Assign each expression to the best matching sign segment.
    
    Handles:
    - Rule F: Multiple expressions can attach to same sign (in time order)
    - All assignment rules from choose_sign_for_expression
    """
    for expr in expr_segments:
        chosen_idx = choose_sign_for_expression(expr, sign_segments)
        
        if chosen_idx is not None:
            # Rule F: Append expressions in time order
            # Since expr_segments are already in time order, just append
            sign_segments[chosen_idx].expressions.append(expr.token)


def merge_output(sign_segments_with_exprs: List[Segment]) -> str:
    """
    STEP 4: Convert sign segments with expressions to final ASL gloss string.
    
    Example:
    - FATHER WANT SELL(shake) HIS(shake) CAR
    
    Multiple expressions:
    - CAR(cs,raise)
    """
    words = []
    for seg in sign_segments_with_exprs:
        if seg.expressions:
            # Remove duplicates while preserving order
            unique_exprs = []
            seen = set()
            for exp in seg.expressions:
                if exp not in seen:
                    unique_exprs.append(exp)
                    seen.add(exp)
            
            expr_str = ','.join(unique_exprs)
            words.append(f"{seg.token}({expr_str})")
        else:
            words.append(seg.token)
    
    return ' '.join(words)


def align_ctc_streams(sign_ctc: str, exp_ctc: str) -> str:
    """
    Main function: Combine sign and expression CTC streams into ASL gloss.
    
    Args:
        sign_ctc: Comma-separated CTC sequence for signs (e.g., "#,MY,MY,FATHER,...")
        exp_ctc: Comma-separated CTC sequence for expressions (e.g., "#,#,#,raise,raise,...")
    
    Returns:
        ASL gloss string (e.g., "MY FATHER WANT SELL HIS CAR(raise)")
    """
    # Parse CTC sequences into frame lists
    sign_frames = [t.strip() for t in sign_ctc.split(',')]
    expr_frames = [t.strip() for t in exp_ctc.split(',')]
    
    # Ensure same length (pad if necessary)
    max_len = max(len(sign_frames), len(expr_frames))
    sign_frames.extend(['#'] * (max_len - len(sign_frames)))
    expr_frames.extend(['#'] * (max_len - len(expr_frames)))
    
    # STEP 1: Collapse each CTC stream but keep time spans
    sign_segments = collapse_ctc(sign_frames)
    expr_segments = collapse_ctc(expr_frames)
    
    if not sign_segments:
        return ""
    
    # STEP 2: Filter expression segments (remove emotions, split comma-separated)
    expr_segments = filter_expressions(expr_segments)
    
    # STEP 3: Assign expressions to signs
    assign_expressions(sign_segments, expr_segments)
    
    # STEP 4: Construct output
    return merge_output(sign_segments)


class TemporalAligner:
    """
    Multi-Stream CTC Fusion Aligner (Version 2).
    Uses temporal overlap with emotion filtering and improved assignment rules.
    """
    
    def __init__(self, sign_step: int = 40, exp_step: int = 40):
        # Parameters kept for compatibility but not used in new approach
        self.sign_step = sign_step
        self.exp_step = exp_step
        
    def align(self, sign_ctc: str, exp_ctc: str) -> str:
        """
        Align expressions to signs using multi-stream CTC fusion (v2).
        
        Returns: Aligned ASL gloss string like "MY FATHER WANT(sad) SELL(sad) HIS CAR(mm)"
        """
        return align_ctc_streams(sign_ctc, exp_ctc)


def extract_emotion_from_expression(expression_str: str) -> Optional[str]:
    """
    Extract emotion (happy, sad, or angry) from an expression string.
    Only returns one emotion per sentence - the first one found.
    
    Args:
        expression_str: Comma-separated expression string (e.g., "angry,cs,raise" or "happy")
    
    Returns:
        Emotion string ("happy", "sad", or "angry") or None if no emotion found
    """
    if not expression_str or not expression_str.strip():
        return None
    
    expressions = [e.strip() for e in expression_str.split(',')]
    
    for expr in expressions:
        if expr.lower() in EMOTION_EXPRESSIONS:
            return expr.lower()
    
    return None


def extract_recognized_emotion_with_frequency(recognized_str: str, ctc_sequence: str) -> Optional[str]:
    """
    Extract recognized emotion from recognized expression string.
    If multiple emotions exist, choose the one with max frequency in CTC sequence.
    If tie, choose the one that occurs first.
    
    Args:
        recognized_str: Recognized expression string (e.g., "angry" or "angry,cs,happy")
        ctc_sequence: CTC sequence string (e.g., "#,#,angry,angry,angry,angry,#,#,...")
    
    Returns:
        Emotion string ("happy", "sad", or "angry") or None if no emotion found
    """
    if not recognized_str or not recognized_str.strip():
        return None
    
    # Extract all emotions from recognized string
    expressions = [e.strip() for e in recognized_str.split(',')]
    emotions_in_recognized = [e for e in expressions if e.lower() in EMOTION_EXPRESSIONS]
    
    if not emotions_in_recognized:
        return None
    
    # If only one emotion, return it
    if len(emotions_in_recognized) == 1:
        return emotions_in_recognized[0].lower()
    
    # Multiple emotions: count frequency in CTC sequence
    ctc_tokens = [t.strip() for t in ctc_sequence.split(',')]
    emotion_frequencies = {}
    emotion_first_occurrence = {}
    
    for emotion in emotions_in_recognized:
        emotion_lower = emotion.lower()
        count = ctc_tokens.count(emotion_lower)
        emotion_frequencies[emotion_lower] = count
        
        # Find first occurrence
        try:
            first_idx = ctc_tokens.index(emotion_lower)
            emotion_first_occurrence[emotion_lower] = first_idx
        except ValueError:
            emotion_first_occurrence[emotion_lower] = float('inf')
    
    # Find emotion with max frequency
    max_freq = max(emotion_frequencies.values())
    candidates = [emotion for emotion, freq in emotion_frequencies.items() if freq == max_freq]
    
    # If single candidate with max frequency, return it
    if len(candidates) == 1:
        return candidates[0]
    
    # If tie, choose the one that occurs first
    best_emotion = None
    earliest_occurrence = float('inf')
    
    for emotion in candidates:
        first_occ = emotion_first_occurrence[emotion]
        if first_occ < earliest_occurrence:
            earliest_occurrence = first_occ
            best_emotion = emotion
    
    # If still tied (all have same first occurrence), return the first one in the recognized string
    if best_emotion is None:
        return emotions_in_recognized[0].lower()
    
    return best_emotion


def calculate_proportional_match(gt_gloss: str, aligned_gloss: str) -> float:
    """
    Calculate proportional match score that:
    - Rewards correct signs and expressions
    - Penalizes misalignment
    
    Returns: score between 0 and 1
    """
    gt_signs_exps = parse_gloss(gt_gloss)
    aligned_signs_exps = parse_gloss(aligned_gloss)
    
    # Extract all signs and expressions
    gt_signs = [s[0] for s in gt_signs_exps]
    aligned_signs = [s[0] for s in aligned_signs_exps]
    
    # Calculate sign accuracy
    min_len = min(len(gt_signs), len(aligned_signs))
    max_len = max(len(gt_signs), len(aligned_signs))
    
    if max_len == 0:
        return 1.0
    
    # Count matching signs at each position
    sign_matches = 0
    for i in range(min_len):
        if gt_signs[i] == aligned_signs[i]:
            sign_matches += 1
    
    sign_score = sign_matches / max_len if max_len > 0 else 0.0
    
    # Calculate expression accuracy
    # Collect all expressions with their sign positions
    gt_all_exps = []
    aligned_all_exps = []
    
    for i, (sign, exps) in enumerate(gt_signs_exps):
        for exp in exps:
            gt_all_exps.append((i, sign, exp))
    
    for i, (sign, exps) in enumerate(aligned_signs_exps):
        for exp in exps:
            aligned_all_exps.append((i, sign, exp))
    
    # Match expressions (considering sign position)
    gt_exp_set = set(gt_all_exps)
    aligned_exp_set = set(aligned_all_exps)
    
    # Exact matches (same position, same sign, same expression)
    exact_exp_matches = len(gt_exp_set & aligned_exp_set)
    
    # Partial matches (same expression, but different position/sign)
    gt_exp_types = {exp for _, _, exp in gt_all_exps}
    aligned_exp_types = {exp for _, _, exp in aligned_all_exps}
    exp_type_matches = len(gt_exp_types & aligned_exp_types)
    
    # Expression score: weighted combination
    total_gt_exps = len(gt_all_exps)
    total_aligned_exps = len(aligned_all_exps)
    max_exps = max(total_gt_exps, total_aligned_exps, 1)
    
    if max_exps > 0:
        # Exact match score (position matters)
        exact_score = exact_exp_matches / max_exps
        # Type match score (position doesn't matter)
        type_score = exp_type_matches / max(len(gt_exp_types), len(aligned_exp_types), 1) if (gt_exp_types or aligned_exp_types) else 1.0
        # Combined: 70% exact, 30% type
        exp_score = 0.7 * exact_score + 0.3 * type_score
    else:
        exp_score = 1.0
    
    # Combined score: 60% signs, 40% expressions
    proportional_score = 0.6 * sign_score + 0.4 * exp_score
    
    return proportional_score


def load_and_align_files(signs_file: str, exp_file: str, output_dir: str):
    """
    Load both files, align expressions to signs, and calculate accuracy metrics.
    """
    print("Loading data...")
    
    signs_data = {}
    exp_data = {}
    
    # Load signs data
    with open(signs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gt, rec, ctc, gloss, session, sample = parse_file_line(line)
            key = f"{session},{sample}"
            signs_data[key] = {
                'ground_truth': gt,
                'recognized': rec,
                'ctc_sequence': ctc,
                'ground_truth_gloss': gloss,  # This is the ground truth ASL gloss
                'session': session,
                'sample': sample
            }
    
    # Load expressions data
    with open(exp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gt, rec, ctc, gloss, session, sample = parse_file_line(line)
            key = f"{session},{sample}"
            exp_data[key] = {
                'ground_truth': gt,  # Ground truth expression string (e.g., "angry,cs,raise")
                'recognized': rec,  # Recognized expression string (e.g., "angry")
                'ctc_sequence': ctc,  # CTC sequence (emotions may still be present)
                'ground_truth_gloss': gloss,  # This is the ground truth ASL gloss (emotions removed)
                'session': session,
                'sample': sample
            }
    
    print(f"Loaded {len(signs_data)} signs entries and {len(exp_data)} expressions entries")
    
    # Find common keys - preserve order from signs file
    common_keys = []
    seen_keys = set()
    # First, preserve order from signs file
    with open(signs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gt, rec, ctc, gloss, session, sample = parse_file_line(line)
            key = f"{session},{sample}"
            if key in exp_data and key not in seen_keys:
                common_keys.append(key)
                seen_keys.add(key)
    
    print(f"Found {len(common_keys)} common samples (order preserved from signs file)")
    
    # Create aligner
    print("Initializing aligner...")
    aligner = TemporalAligner()
    
    # Align each sample
    print("Aligning samples...")
    results = []
    exact_matches = 0
    # Keep exact match count only (ground-truth cleansed of emotions)
    
    for key in common_keys:
        sign_info = signs_data[key]
        exp_info = exp_data[key]
        
        # Get ground truth ASL gloss (from signs file - it has the complete ground truth)
        gt_gloss = sign_info['ground_truth_gloss']
        clean_gt_gloss = remove_emotions_from_gloss(gt_gloss)
        
        # Align expressions to signs
        aligned_gloss = aligner.align(
            sign_info['ctc_sequence'],
            exp_info['ctc_sequence']
        )
        
        # Calculate metrics
        exact_match = (clean_gt_gloss == aligned_gloss)
        if exact_match:
            exact_matches += 1
        
        # Extract emotions from expressions data
        gt_emotion = extract_emotion_from_expression(exp_info['ground_truth'])
        recognized_emotion = extract_recognized_emotion_with_frequency(
            exp_info['recognized'],
            exp_info['ctc_sequence']
        )
        
        results.append({
            'key': key,
            'ground_truth_gloss': gt_gloss,
            'aligned_gloss': aligned_gloss,
            'exact_match': exact_match,
            'clean_ground_truth_gloss': clean_gt_gloss,
            'ground_truth_emotion': gt_emotion if gt_emotion else 'none',
            'recognized_emotion': recognized_emotion if recognized_emotion else 'none'
        })
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_file = output_dir_path / "alignment_without_LM.txt"
    
    # Write output file
    print(f"Writing alignment results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['ground_truth_gloss']} - {result['aligned_gloss']}\n")
        
        # Write metrics
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        exact_match_rate = exact_matches / len(results) if results else 0.0
        
        f.write(f"Simple Exact Match Accuracy (emotions removed from GT): {exact_matches}/{len(results)} ({100*exact_match_rate:.2f}%)\n")
        f.write(f"   - Exact match: {exact_matches}\n")
        f.write(f"   - Total samples: {len(results)}\n")
        f.write(f"   - Accuracy: {100*exact_match_rate:.2f}%\n")
    
    # Write additional output file with emotions
    output_file_with_emotions = output_dir_path / "alignment_with_emotion_without_LM.txt"
    print(f"Writing alignment results with emotions to {output_file_with_emotions}...")
    
    # Count exact matches (both aligned sentence and emotions match)
    exact_matches_with_emotions = 0
    # Track sentences where sentence matches but emotions don't
    sentence_match_emotion_mismatch = []
    
    with open(output_file_with_emotions, 'w', encoding='utf-8') as f:
        for idx, result in enumerate(results, 1):
            # Format: ground truth (without emotion) - aligned sentence --- ground truth emotion - recognized emotion
            f.write(f"{result['clean_ground_truth_gloss']} - {result['aligned_gloss']} --- {result['ground_truth_emotion']} - {result['recognized_emotion']}\n")
            
            # Check if exact match: aligned sentence matches clean ground truth AND emotions match
            if (result['clean_ground_truth_gloss'] == result['aligned_gloss'] and 
                result['ground_truth_emotion'] == result['recognized_emotion']):
                exact_matches_with_emotions += 1
            
            # Track sentences where sentence matches but emotions don't
            elif (result['clean_ground_truth_gloss'] == result['aligned_gloss'] and 
                  result['ground_truth_emotion'] != result['recognized_emotion']):
                sentence_match_emotion_mismatch.append({
                    'line': idx,
                    'sentence': result['clean_ground_truth_gloss'],
                    'gt_emotion': result['ground_truth_emotion'],
                    'rec_emotion': result['recognized_emotion']
                })
        
        # Write exact match accuracy at the end
        exact_match_rate_with_emotions = exact_matches_with_emotions / len(results) if results else 0.0
        f.write(f"\nExact Match Accuracy: {exact_matches_with_emotions}/{len(results)} = {100*exact_match_rate_with_emotions:.2f}%\n")
        
        # Write sentences where sentence matches but emotions don't
        if sentence_match_emotion_mismatch:
            f.write(f"\n" + "=" * 80 + "\n")
            f.write(f"SENTENCES WHERE SENTENCE MATCHES BUT EMOTIONS DON'T MATCH\n")
            f.write(f"Total: {len(sentence_match_emotion_mismatch)}\n")
            f.write("=" * 80 + "\n\n")
            for item in sentence_match_emotion_mismatch:
                f.write(f"Line {item['line']}: {item['sentence']} --- GT Emotion: {item['gt_emotion']} | Recognized Emotion: {item['rec_emotion']}\n")
        else:
            f.write(f"\nNo sentences found where sentence matches but emotions don't match.\n")
    
    # Calculate rate for printing (same calculation as above)
    exact_match_rate_with_emotions = exact_matches_with_emotions / len(results) if results else 0.0
    
    print(f"\nDone! Processed {len(results)} samples")
    print(f"Simple Exact Match Accuracy: {exact_matches}/{len(results)} ({100*exact_match_rate:.2f}%)")
    print(f"Exact Match Accuracy (with emotions): {exact_matches_with_emotions}/{len(results)} ({100*exact_match_rate_with_emotions:.2f}%)")
    print(f"Output file: {output_file}")
    print(f"Output file with emotions: {output_file_with_emotions}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Align CTC outputs from signs and expressions files. Preserves input order.'
    )
    parser.add_argument(
        '--signs_file',
        type=str,
        required=True,
        help='Path to the signs input file'
    )
    parser.add_argument(
        '--exp_file',
        type=str,
        required=True,
        help='Path to the expressions input file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where alignment_without_LM.txt will be saved'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.signs_file):
        print(f"Error: Signs file '{args.signs_file}' not found.")
        sys.exit(1)
    if not os.path.exists(args.exp_file):
        print(f"Error: Expressions file '{args.exp_file}' not found.")
        sys.exit(1)
    
    load_and_align_files(args.signs_file, args.exp_file, args.output_dir)

