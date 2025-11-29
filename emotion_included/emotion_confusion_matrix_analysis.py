"""
Script to analyze emotion recognition performance from alignment file.
Creates a confusion matrix for emotion recognition (happy, sad, angry, none).
Generates both text and visual confusion matrix with percentages.
"""

import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# All possible emotion labels
EMOTION_LABELS = ["happy", "sad", "angry", "none"]


def parse_alignment_file(input_file: str):
    """
    Parse alignment file and extract emotion pairs.
    
    Args:
        input_file: Path to alignment_without_LM_emo_sep.txt
    
    Returns:
        List of tuples: [(gt_emotion, rec_emotion), ...]
    """
    emotion_pairs = []
    
    print(f"Reading alignment file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
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
            
            emotion_part = parts[1].strip()
            
            # Parse emotion pair: "gt_emotion - rec_emotion"
            if ' - ' in emotion_part:
                gt_emotion, rec_emotion = emotion_part.split(' - ', 1)
                gt_emotion = gt_emotion.strip().lower()
                rec_emotion = rec_emotion.strip().lower()
                
                # Normalize emotions (ensure they're in the valid set)
                if gt_emotion not in EMOTION_LABELS:
                    gt_emotion = 'none'
                if rec_emotion not in EMOTION_LABELS:
                    rec_emotion = 'none'
                
                emotion_pairs.append((gt_emotion, rec_emotion))
    
    return emotion_pairs


def build_confusion_matrix(emotion_pairs):
    """
    Build confusion matrix from emotion pairs.
    
    Args:
        emotion_pairs: List of (gt_emotion, rec_emotion) tuples
    
    Returns:
        confusion_matrix: 2D dictionary [gt_emotion][rec_emotion] = count
        total_pairs: Total number of pairs
    """
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for gt_emotion, rec_emotion in emotion_pairs:
        confusion_matrix[gt_emotion][rec_emotion] += 1
    
    return confusion_matrix, len(emotion_pairs)


def print_confusion_matrix(confusion_matrix, total_pairs, output_file=None):
    """
    Print confusion matrix in a readable format.
    
    Args:
        confusion_matrix: 2D dictionary [gt_emotion][rec_emotion] = count
        total_pairs: Total number of pairs
        output_file: Optional file path to write output
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("EMOTION RECOGNITION CONFUSION MATRIX")
    lines.append("=" * 80)
    lines.append(f"Total samples: {total_pairs}")
    lines.append("")
    
    # Header row
    header = "GT \\ Predicted".ljust(15)
    for label in EMOTION_LABELS:
        header += f"{label:>12}"
    header += "   Total"
    lines.append(header)
    lines.append("-" * 80)
    
    # Data rows
    row_totals = defaultdict(int)
    col_totals = defaultdict(int)
    
    for gt_label in EMOTION_LABELS:
        row = f"{gt_label:>14}"
        row_total = 0
        
        for pred_label in EMOTION_LABELS:
            count = confusion_matrix[gt_label][pred_label]
            row += f"{count:>12}"
            row_total += count
            col_totals[pred_label] += count
        
        row += f"   {row_total:>6}"
        lines.append(row)
        row_totals[gt_label] = row_total
    
    # Column totals row
    lines.append("-" * 80)
    total_row = "Total".ljust(15)
    grand_total = 0
    for label in EMOTION_LABELS:
        total_row += f"{col_totals[label]:>12}"
        grand_total += col_totals[label]
    total_row += f"   {grand_total:>6}"
    lines.append(total_row)
    lines.append("")
    
    # Calculate metrics
    lines.append("=" * 80)
    lines.append("PERFORMANCE METRICS")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall accuracy
    correct = sum(confusion_matrix[label][label] for label in EMOTION_LABELS)
    overall_accuracy = (correct / total_pairs * 100) if total_pairs > 0 else 0.0
    lines.append(f"Overall Accuracy: {correct}/{total_pairs} = {overall_accuracy:.2f}%")
    lines.append("")
    
    # Per-class metrics
    lines.append("Per-Class Metrics:")
    lines.append("-" * 80)
    lines.append(f"{'Emotion':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    lines.append("-" * 80)
    
    for label in EMOTION_LABELS:
        # True positives: correctly predicted as this label
        tp = confusion_matrix[label][label]
        
        # False positives: predicted as this label but actually something else
        fp = sum(confusion_matrix[other][label] for other in EMOTION_LABELS if other != label)
        
        # False negatives: actually this label but predicted as something else
        fn = sum(confusion_matrix[label][other] for other in EMOTION_LABELS if other != label)
        
        # Support: total ground truth instances of this label
        support = row_totals[label]
        
        # Precision
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        
        # F1-score
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        lines.append(f"{label:<12} {precision:>11.2f}% {recall:>11.2f}% {f1:>11.2f}% {support:>12}")
    
    lines.append("")
    
    # Confusion details
    lines.append("=" * 80)
    lines.append("CONFUSION DETAILS")
    lines.append("=" * 80)
    lines.append("")
    
    for gt_label in EMOTION_LABELS:
        for pred_label in EMOTION_LABELS:
            if gt_label != pred_label:
                count = confusion_matrix[gt_label][pred_label]
                if count > 0:
                    percentage = (count / row_totals[gt_label] * 100) if row_totals[gt_label] > 0 else 0.0
                    lines.append(f"GT: {gt_label:>6} → Predicted: {pred_label:>6} : {count:>4} ({percentage:>5.2f}%)")
    
    lines.append("")
    
    # Print to console
    output_text = "\n".join(lines)
    print(output_text)
    
    # Write to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nAnalysis saved to: {output_file}")
    
    return output_text


def create_confusion_matrix_image(confusion_matrix, total_pairs, output_file: str):
    """
    Create a visual confusion matrix image with percentages.
    
    Args:
        confusion_matrix: 2D dictionary [gt_emotion][rec_emotion] = count
        total_pairs: Total number of pairs
        output_file: Path to save the image
    """
    # Create numpy array for confusion matrix
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=int)
    
    for i, gt_label in enumerate(EMOTION_LABELS):
        for j, pred_label in enumerate(EMOTION_LABELS):
            matrix[i, j] = confusion_matrix[gt_label][pred_label]
    
    # Calculate percentages (row-wise, i.e., percentage of each GT emotion)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    percentage_matrix = (matrix / row_sums * 100).round(1)
    
    # Create figure with better size
    plt.figure(figsize=(11, 9))
    
    # Create annotation strings with count and percentage
    annot_matrix = []
    for i in range(len(EMOTION_LABELS)):
        row = []
        for j in range(len(EMOTION_LABELS)):
            count = matrix[i, j]
            percentage = percentage_matrix[i, j]
            if count > 0:
                row.append(f'{count}\n({percentage:.1f}%)')
            else:
                row.append('0\n(0.0%)')
        annot_matrix.append(row)
    
    # Create heatmap with custom annotations
    ax = sns.heatmap(
        percentage_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={'label': 'Percentage (%)'},
        vmin=0,
        vmax=100,
        linewidths=1.5,
        linecolor='white',
        square=True,
        annot_kws={'size': 11, 'weight': 'bold', 'color': 'black'}
    )
    
    # Customize labels
    plt.xlabel('Predicted Emotion', fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel('Ground Truth Emotion', fontsize=13, fontweight='bold', labelpad=15)
    
    # Add title
    correct = sum(confusion_matrix[label][label] for label in EMOTION_LABELS)
    overall_accuracy = (correct / total_pairs * 100) if total_pairs > 0 else 0.0
    plt.title(f'Emotion Recognition Confusion Matrix\nOverall Accuracy: {overall_accuracy:.2f}% ({correct}/{total_pairs})', 
              fontsize=15, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix image saved to: {output_file}")
    plt.close()


def main(input_file: str, output_file: str = None, image_file: str = None):
    """
    Main function to run emotion recognition analysis.
    
    Args:
        input_file: Path to alignment_without_LM_emo_sep.txt
        output_file: Optional path to save analysis results (text)
        image_file: Optional path to save confusion matrix image
    """
    # Parse alignment file
    emotion_pairs = parse_alignment_file(input_file)
    
    if not emotion_pairs:
        print("Error: No emotion pairs found in the file.")
        return
    
    print(f"Found {len(emotion_pairs)} emotion pairs")
    print("")
    
    # Build confusion matrix
    confusion_matrix, total_pairs = build_confusion_matrix(emotion_pairs)
    
    # Print confusion matrix and metrics
    print_confusion_matrix(confusion_matrix, total_pairs, output_file)
    
    # Create visual confusion matrix if image_file is specified
    if image_file:
        create_confusion_matrix_image(confusion_matrix, total_pairs, image_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze emotion recognition performance and create confusion matrix'
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
        default=None,
        help='Optional path to save analysis results (text file)'
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default=None,
        help='Optional path to save confusion matrix image (PNG file)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    
    main(args.input_file, args.output_file, args.image_file)

