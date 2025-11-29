#!/usr/bin/env python3
"""
Translation Analysis Script
Categorizes translation errors into: misrecognition (sign), misrecognition (exp), 
misalignment, mistranslation, and misaligned but accurate translation.
"""

import re
import os
from pathlib import Path
from collections import defaultdict, Counter


def remove_emotions(text):
    """Remove emotion annotations (happy, sad, angry) but keep expressions."""
    text = re.sub(r'\(happy\)', '', text)
    text = re.sub(r'\(sad\)', '', text)
    text = re.sub(r'\(angry\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_signs(sign_text):
    """Normalize sign text for comparison."""
    # Remove expressions and emotions, keep only base signs
    text = remove_emotions(sign_text)
    # Remove expression annotations
    text = re.sub(r'\([^)]+\)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to uppercase and split
    signs = [s.strip() for s in text.split() if s.strip()]
    return signs


def extract_expressions(text):
    """Extract expressions (excluding emotions) from text."""
    expressions = []
    # Find all expression annotations
    matches = re.findall(r'\(([^)]+)\)', text)
    for match in matches:
        # Split by comma and filter out emotions
        exprs = [e.strip() for e in match.split(',')]
        for expr in exprs:
            if expr not in ['happy', 'sad', 'angry']:
                expressions.append(expr)
    return sorted(expressions)


def parse_translation_file(filepath):
    """Parse translation file and return list of (gt, pred, bleu) tuples."""
    translations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('='):
                continue
            # Format: GT - PRED - BLEU
            parts = line.split(' - ')
            if len(parts) >= 3:
                gt = parts[0].strip()
                pred = parts[1].strip()
                bleu_str = parts[2].strip()
                try:
                    bleu = float(bleu_str)
                except ValueError:
                    bleu = 0.0
                translations.append((gt, pred, bleu))
    return translations


def parse_alignment_file(filepath):
    """Parse alignment file and return list of (gt, pred) tuples."""
    alignments = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('='):
                continue
            # Format: GT - PRED
            if ' - ' in line:
                parts = line.split(' - ', 1)
                gt = parts[0].strip()
                pred = parts[1].strip()
                alignments.append((gt, pred))
    return alignments


def parse_exp_outputs(filepath):
    """Parse expression outputs file."""
    exp_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: GT_EXP - PRED_EXP <<<<< ... >>>> GT_SIGNS ----- ID, index
            if ' - ' in line and '<<<<<' in line and '>>>>' in line:
                parts = line.split(' - ', 1)
                gt_exp_str = parts[0].strip()
                rest = parts[1]
                
                # Extract predicted expressions
                if '<<<<<' in rest:
                    pred_exp_str = rest.split('<<<<<')[0].strip()
                    # Extract GT signs from after >>>>
                    if '>>>>' in rest:
                        after_arrow = rest.split('>>>>')[1]
                        if '-----' in after_arrow:
                            gt_signs = after_arrow.split('-----')[0].strip()
                            exp_data.append((gt_exp_str, pred_exp_str, gt_signs))
    return exp_data


def parse_sign_outputs(filepath):
    """Parse sign outputs file."""
    sign_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: GT_SIGNS - PRED_SIGNS <<<<< ... >>>> GT_SIGNS_WITH_EXP ----- ID, index
            if ' - ' in line and '<<<<<' in line and '>>>>' in line:
                parts = line.split(' - ', 1)
                gt_signs_str = parts[0].strip()
                rest = parts[1]
                
                # Extract predicted signs
                if '<<<<<' in rest:
                    pred_signs_str = rest.split('<<<<<')[0].strip()
                    sign_data.append((gt_signs_str, pred_signs_str))
    return sign_data


def check_sign_misrecognition(gt_signs_str, pred_signs_str):
    """Check if signs are misrecognized."""
    gt_signs = [s.strip() for s in gt_signs_str.split(',') if s.strip()]
    pred_signs = [s.strip() for s in pred_signs_str.split(',') if s.strip()]
    return gt_signs != pred_signs


def check_exp_misrecognition(gt_exp_str, pred_exp_str):
    """Check if expressions are misrecognized (ignoring emotions)."""
    gt_exps = [e.strip() for e in gt_exp_str.split(',') if e.strip() and e.strip() not in ['happy', 'sad', 'angry']]
    pred_exps = [e.strip() for e in pred_exp_str.split(',') if e.strip() and e.strip() not in ['happy', 'sad', 'angry']]
    return sorted(gt_exps) != sorted(pred_exps)


def check_misalignment(gt_alignment, pred_alignment):
    """Check if alignment is wrong (ignoring emotions)."""
    # Remove emotions from both
    gt_clean = remove_emotions(gt_alignment)
    pred_clean = remove_emotions(pred_alignment)
    
    # Extract normalized signs (base words only)
    gt_signs = normalize_signs(gt_clean)
    pred_signs = normalize_signs(pred_clean)
    
    # If base signs don't match, it's misalignment (or sign misrecognition, but we check that separately)
    if gt_signs != pred_signs:
        return True
    
    # Extract expressions with their positions
    gt_exps_with_pos = extract_expressions_with_positions(gt_clean)
    pred_exps_with_pos = extract_expressions_with_positions(pred_clean)
    
    # Check if expressions match (ignoring positions for now - if same expressions exist)
    gt_exps_set = set(gt_exps_with_pos.keys())
    pred_exps_set = set(pred_exps_with_pos.keys())
    
    if gt_exps_set != pred_exps_set:
        return True
    
    # Check if expression positions are significantly different
    for exp in gt_exps_set:
        gt_positions = sorted(gt_exps_with_pos[exp])
        pred_positions = sorted(pred_exps_with_pos[exp])
        if gt_positions != pred_positions:
            return True
    
    return False


def extract_expressions_with_positions(text):
    """Extract expressions and their word positions."""
    words = text.split()
    exp_positions = defaultdict(list)
    word_idx = 0
    
    for word in words:
        match = re.match(r'^(\w+)(\([^)]+\))?$', word)
        if match:
            base_word = match.group(1)
            expr_str = match.group(2)
            
            if expr_str:
                expressions = expr_str[1:-1].split(',')
                for expr in expressions:
                    expr = expr.strip()
                    if expr not in ['happy', 'sad', 'angry']:
                        exp_positions[expr].append(word_idx)
            
            word_idx += 1
    
    return exp_positions


def analyze_translations(translation_file, alignment_file, exp_file, sign_file, output_file):
    """Main analysis function."""
    
    print("Loading files...")
    translations = parse_translation_file(translation_file)
    alignments = parse_alignment_file(alignment_file)
    exp_data = parse_exp_outputs(exp_file)
    sign_data = parse_sign_outputs(sign_file)
    
    print(f"Loaded {len(translations)} translations, {len(alignments)} alignments, "
          f"{len(exp_data)} expression outputs, {len(sign_data)} sign outputs")
    
    # Categories
    categories = {
        'misrecognition_sign': [],
        'misrecognition_exp': [],
        'misalignment': [],
        'mistranslation': [],
        'misaligned_but_accurate': [],
        'correct': []
    }
    
    # Statistics
    stats = defaultdict(int)
    bleu_by_category = defaultdict(list)
    
    # Expression-wise statistics
    # Track: expression -> {'misalignment_mistranslation': count, 'misalignment_correct': count, 'both_correct': count, 'total': count}
    expr_stats = defaultdict(lambda: {'misalignment_mistranslation': 0, 'misalignment_correct': 0, 'both_correct': 0, 'total': 0})
    # Track translation accuracy: expression -> {'total_sentences': count, 'correct_translations': count}
    expr_translation_stats = defaultdict(lambda: {'total_sentences': 0, 'correct_translations': 0})
    expressions_to_track = ['cha', 'th', 'cs', 'mm', 'raise', 'furrow', 'shake']
    
    # Process each translation
    for idx, (gt_trans, pred_trans, bleu) in enumerate(translations):
        if idx >= len(alignments) or idx >= len(exp_data) or idx >= len(sign_data):
            continue
        
        gt_alignment, pred_alignment = alignments[idx]
        gt_exp_str, pred_exp_str, _ = exp_data[idx]
        gt_signs_str, pred_signs_str = sign_data[idx]
        
        # Determine error category (priority order matters)
        error_category = None
        
        # 1. Check sign misrecognition
        sign_misrec = check_sign_misrecognition(gt_signs_str, pred_signs_str)
        
        # 2. Check expression misrecognition
        exp_misrec = check_exp_misrecognition(gt_exp_str, pred_exp_str)
        
        # 3. Check misalignment (only if signs match - otherwise it's sign misrecognition)
        misaligned = False
        if not sign_misrec:
            misaligned = check_misalignment(gt_alignment, pred_alignment)
        
        # 4. Check if translation is correct (BLEU = 1.0)
        translation_correct = (bleu == 1.0)
        
        # Extract expressions from ground truth alignment for expression-wise analysis
        gt_expressions_in_alignment = extract_expressions(gt_alignment)
        
        # Track translation accuracy for all sentences containing each expression (regardless of recognition status)
        for expr in expressions_to_track:
            if expr in gt_expressions_in_alignment:
                expr_translation_stats[expr]['total_sentences'] += 1
                if translation_correct:
                    expr_translation_stats[expr]['correct_translations'] += 1
        
        # Track expression-wise statistics (only for expressions that appear in GT)
        # Only track if signs are correct (no sign misrecognition) and expressions are correct (no exp misrecognition)
        if not sign_misrec and not exp_misrec:
            for expr in expressions_to_track:
                if expr in gt_expressions_in_alignment:
                    expr_stats[expr]['total'] += 1
                    
                    if misaligned:
                        if translation_correct:
                            expr_stats[expr]['misalignment_correct'] += 1
                        else:
                            expr_stats[expr]['misalignment_mistranslation'] += 1
                    elif translation_correct:
                        # Both alignment and translation are correct
                        expr_stats[expr]['both_correct'] += 1
        
        # Categorize based on priority (most impactful first)
        # Priority: sign misrecognition > exp misrecognition > misalignment > mistranslation
        if sign_misrec:
            error_category = 'misrecognition_sign'
        elif exp_misrec:
            error_category = 'misrecognition_exp'
        elif misaligned:
            if translation_correct:
                error_category = 'misaligned_but_accurate'
            else:
                error_category = 'misalignment'
        elif not translation_correct:
            # Signs, expressions, and alignment are correct, but translation is wrong
            error_category = 'mistranslation'
        else:
            # Everything is correct: signs, expressions, alignment, and translation
            error_category = 'correct'
        
        # Store result
        categories[error_category].append({
            'idx': idx + 1,
            'gt_trans': gt_trans,
            'pred_trans': pred_trans,
            'bleu': bleu,
            'gt_alignment': gt_alignment,
            'pred_alignment': pred_alignment
        })
        
        stats[error_category] += 1
        bleu_by_category[error_category].append(bleu)
    
    # Generate report
    print(f"\nGenerating analysis report: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSLATION ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Translation file: {translation_file}\n")
        f.write(f"Alignment file: {alignment_file}\n")
        f.write(f"Expression outputs file: {exp_file}\n")
        f.write(f"Sign outputs file: {sign_file}\n\n")
        
        # Summary statistics
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        total_sentences = len(translations)
        total_errors = sum(stats[cat] for cat in ['misrecognition_sign', 'misrecognition_exp', 
                                                   'misalignment', 'mistranslation', 'misaligned_but_accurate'])
        total_correct = stats['correct']
        
        f.write(f"{'Category':<35} {'Count':<10} {'Percentage':<15} {'Avg BLEU':<15}\n")
        f.write("-" * 80 + "\n")
        
        # First show error categories
        for category in ['misrecognition_sign', 'misrecognition_exp', 'misalignment', 
                         'mistranslation', 'misaligned_but_accurate']:
            count = stats[category]
            if total_sentences > 0:
                pct = (count / total_sentences) * 100
            else:
                pct = 0.0
            avg_bleu = sum(bleu_by_category[category]) / len(bleu_by_category[category]) if bleu_by_category[category] else 0.0
            f.write(f"{category.replace('_', ' ').title():<35} {count:<10} {pct:.2f}%{'':<10} {avg_bleu:.4f}\n")
        
        # Then show correct category
        if total_sentences > 0:
            correct_pct = (total_correct / total_sentences) * 100
        else:
            correct_pct = 0.0
        avg_bleu_correct = sum(bleu_by_category['correct']) / len(bleu_by_category['correct']) if bleu_by_category['correct'] else 1.0
        f.write(f"{'Correct':<35} {total_correct:<10} {correct_pct:.2f}%{'':<10} {avg_bleu_correct:.4f}\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{'Total Sentences':<35} {total_sentences:<10}\n")
        f.write(f"{'Total Errors':<35} {total_errors:<10}\n\n")
        
        # Detailed examples for each category
        for category in ['misrecognition_sign', 'misrecognition_exp', 'misalignment', 
                         'mistranslation', 'misaligned_but_accurate', 'correct']:
            if not categories[category]:
                continue
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{category.replace('_', ' ').upper()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total: {len(categories[category])} examples\n")
            f.write(f"Average BLEU: {sum(bleu_by_category[category]) / len(bleu_by_category[category]):.4f}\n\n")
            
            # Show first 10 examples
            examples = categories[category][:10]
            for ex in examples:
                f.write(f"Example {ex['idx']}:\n")
                f.write(f"  BLEU: {ex['bleu']:.4f}\n")
                f.write(f"  GT Translation: {ex['gt_trans']}\n")
                f.write(f"  Pred Translation: {ex['pred_trans']}\n")
                f.write(f"  GT Alignment: {ex['gt_alignment']}\n")
                f.write(f"  Pred Alignment: {ex['pred_alignment']}\n")
                f.write("\n")
            
            if len(categories[category]) > 10:
                f.write(f"... and {len(categories[category]) - 10} more examples\n\n")
        
        # Expression-wise analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXPRESSION-WISE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Analysis for expressions when they are correctly recognized (signs and expressions correct).\n")
        f.write("Shows impact of misalignment on translation quality.\n\n")
        
        f.write(f"{'Expression':<15} {'Total':<10} {'Misalign+Wrong':<20} {'Misalign+Correct':<20} {'Both Correct':<20}\n")
        f.write("-" * 80 + "\n")
        
        for expr in expressions_to_track:
            stats_expr = expr_stats[expr]
            total = stats_expr['total']
            
            if total == 0:
                continue
            
            misalign_wrong = stats_expr['misalignment_mistranslation']
            misalign_correct = stats_expr['misalignment_correct']
            both_correct = stats_expr['both_correct']
            
            misalign_wrong_pct = (misalign_wrong / total * 100) if total > 0 else 0.0
            misalign_correct_pct = (misalign_correct / total * 100) if total > 0 else 0.0
            both_correct_pct = (both_correct / total * 100) if total > 0 else 0.0
            
            f.write(f"{expr:<15} {total:<10} {misalign_wrong} ({misalign_wrong_pct:.2f}%){'':<10} "
                   f"{misalign_correct} ({misalign_correct_pct:.2f}%){'':<10} "
                   f"{both_correct} ({both_correct_pct:.2f}%)\n")
        
        f.write("\n")
        f.write("Legend:\n")
        f.write("  Total: Total occurrences of expression when correctly recognized\n")
        f.write("  Misalign+Wrong: Misalignment causing mistranslation\n")
        f.write("  Misalign+Correct: Misalignment but correct translation\n")
        f.write("  Both Correct: Both alignment and translation are correct\n\n")
        
        # Translation accuracy for expressions
        f.write("\n" + "=" * 80 + "\n")
        f.write("TRANSLATION ACCURACY BY EXPRESSION\n")
        f.write("=" * 80 + "\n\n")
        f.write("Translation accuracy: Total correct translations / Total sentences where expression is present\n\n")
        
        f.write(f"{'Expression':<15} {'Total Sentences':<20} {'Correct Translations':<25} {'Accuracy':<15}\n")
        f.write("-" * 80 + "\n")
        
        for expr in expressions_to_track:
            stats_expr = expr_translation_stats[expr]
            total_sentences = stats_expr['total_sentences']
            correct_translations = stats_expr['correct_translations']
            
            if total_sentences == 0:
                continue
            
            accuracy_pct = (correct_translations / total_sentences * 100) if total_sentences > 0 else 0.0
            f.write(f"{expr:<15} {total_sentences:<20} {correct_translations:<25} {accuracy_pct:.2f}%\n")
        
        f.write("\n")
    
    print(f"Analysis complete! Results saved to: {output_file}")


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    translation_file = base_dir / "final_pipeline_txt_220110" / "Translations" / "translation_without_LM.txt"
    alignment_file = base_dir / "final_pipeline_txt_220110" / "alignment" / "alignment_without_LM.txt"
    exp_file = base_dir / "final_pipeline_txt_220110" / "best_outputs" / "exp_best_outputs.txt"
    sign_file = base_dir / "final_pipeline_txt_220110" / "best_outputs" / "sign_best_outputs.txt"
    output_dir = base_dir / "final_pipeline_txt_220110" / "analysis"
    output_file = output_dir / "translation_analysis_v2.txt"
    
    # Check if input files exist
    for filepath, name in [(translation_file, "Translation"), 
                          (alignment_file, "Alignment"),
                          (exp_file, "Expression outputs"),
                          (sign_file, "Sign outputs")]:
        if not filepath.exists():
            print(f"Error: {name} file not found: {filepath}")
            return
    
    # Run analysis
    analyze_translations(
        str(translation_file),
        str(alignment_file),
        str(exp_file),
        str(sign_file),
        str(output_file)
    )


if __name__ == "__main__":
    main()

