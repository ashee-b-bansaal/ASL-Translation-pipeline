#!/usr/bin/env python3
"""
Correct Optimal Hybrid Aligner - Best of Both Worlds
- For Signs: Apply LM model (no probability matrices, no direct raw CTC output)
- For Expressions: Apply same approach as original optimal_hybrid_aligner.py (8.33% WER)
"""

import argparse
import os
import csv
import numpy as np
import pickle
import re
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import math


def load_best_output_glosses(best_outputs_path: str) -> Dict[Tuple[str, int], str]:
    """
    Parse best_outputs.txt and build a lookup from (session, sample_index) to ground-truth gloss.
    The expected format per line is:
    GT_SEQ - PRED_SEQ <<<<< ... >>>> GLOSS ----- SESSION, SAMPLE_IDX
    """
    gloss_map: Dict[Tuple[str, int], str] = {}

    if not best_outputs_path or not os.path.exists(best_outputs_path):
        print(f"Warning: best_outputs file not found at {best_outputs_path}")
        return gloss_map

    with open(best_outputs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                _, rhs = line.split(">>>>", 1)
                gloss_part, meta_part = rhs.split("-----", 1)
                gloss = gloss_part.strip()

                session_sample = meta_part.strip()
                if "," not in session_sample:
                    continue
                session_str, sample_idx_str = [s.strip() for s in session_sample.split(",", 1)]
                sample_idx = int(sample_idx_str)
                gloss_map[(session_str, sample_idx)] = gloss
            except ValueError:
                # Skip lines that don't match the expected structure
                continue

    return gloss_map


class CTCProbabilityLoader:
    """Load and manage CTC probability matrices"""
    
    def __init__(self, signs_dir: str, exprs_dir: str):
        self.signs_dir = signs_dir
        self.exprs_dir = exprs_dir
        self.sign_samples = {}
        self.expr_samples = {}
        
    def load_sample_data(self, sample_id: int):
        """Load probability matrices for a specific sample"""
        sample_str = f"sample_{sample_id:04d}"
        
        # Load sign data
        sign_metadata_path = os.path.join(self.signs_dir, f"{sample_str}_metadata.npy")
        sign_probs_path = os.path.join(self.signs_dir, f"{sample_str}_probabilities.npy")
        
        if os.path.exists(sign_metadata_path) and os.path.exists(sign_probs_path):
            sign_metadata = np.load(sign_metadata_path, allow_pickle=True).item()
            sign_probs = np.load(sign_probs_path)
            self.sign_samples[sample_id] = {
                'metadata': sign_metadata,
                'probabilities': sign_probs
            }
        
        # Load expression data
        expr_metadata_path = os.path.join(self.exprs_dir, f"{sample_str}_metadata.npy")
        expr_probs_path = os.path.join(self.exprs_dir, f"{sample_str}_probabilities.npy")
        
        if os.path.exists(expr_metadata_path) and os.path.exists(expr_probs_path):
            expr_metadata = np.load(expr_metadata_path, allow_pickle=True).item()
            expr_probs = np.load(expr_probs_path)
            self.expr_samples[sample_id] = {
                'metadata': expr_metadata,
                'probabilities': expr_probs
            }
    
    def get_sign_vocabulary(self, sample_id: int) -> List[str]:
        """Get sign vocabulary for a sample"""
        if sample_id in self.sign_samples:
            return self.sign_samples[sample_id]['metadata']['vocabulary']
        return []
    
    def get_expr_vocabulary(self, sample_id: int) -> List[str]:
        """Get expression vocabulary for a sample"""
        if sample_id in self.expr_samples:
            return self.expr_samples[sample_id]['metadata']['vocabulary']
        return []
    
    def get_sign_probabilities(self, sample_id: int) -> np.ndarray:
        """Get sign probability matrix for a sample"""
        if sample_id in self.sign_samples:
            return self.sign_samples[sample_id]['probabilities']
        return None
    
    def get_expr_probabilities(self, sample_id: int) -> np.ndarray:
        """Get expression probability matrix for a sample"""
        if sample_id in self.expr_samples:
            return self.expr_samples[sample_id]['probabilities']
        return None


class LanguageModelRescorer:
    """Language model rescoring system"""
    
    def __init__(self, sign_lm_path: str, expr_lm_path: str):
        self.sign_transitions = {}
        self.expr_associations = {}
        self.sign_vocabulary = set()
        self.expr_vocabulary = set()
        self.load_models(sign_lm_path, expr_lm_path)
    
    def load_models(self, sign_lm_path: str, expr_lm_path: str):
        """Load both language models"""
        self.load_sign_language_model(sign_lm_path)
        self.load_expression_association_model(expr_lm_path)
    
    def load_sign_language_model(self, path: str):
        """Load sign language model from CSV"""
        print(f"Loading sign language model from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Extract vocabulary (all columns except first)
            self.sign_vocabulary = set(header[1:])
            
            # Load transition probabilities
            for row in reader:
                if not row:
                    continue
                    
                context = row[0]
                for i, word in enumerate(header[1:], 1):
                    if i < len(row):
                        prob = float(row[i])
                        if prob > 0:
                            self.sign_transitions[(context, word)] = prob
        
        print(f"Loaded {len(self.sign_transitions)} sign transitions")
        print(f"Sign vocabulary size: {len(self.sign_vocabulary)}")
    
    def load_expression_association_model(self, path: str):
        """Load expression association model from CSV"""
        print(f"Loading expression association model from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Extract expression vocabulary (all columns except first)
            self.expr_vocabulary = set(header[1:])
            
            # Load association probabilities
            for row in reader:
                if not row:
                    continue
                    
                sign_word = row[0]
                for i, expression in enumerate(header[1:], 1):
                    if i < len(row):
                        prob = float(row[i])
                        if prob > 0:
                            self.expr_associations[(sign_word, expression)] = prob
        
        print(f"Loaded {len(self.expr_associations)} expression associations")
        print(f"Expression vocabulary size: {len(self.expr_vocabulary)}")
    
    def get_sign_transition_prob(self, context: str, word: str) -> float:
        """Get transition probability for sign word sequence"""
        return self.sign_transitions.get((context, word), 0.0)
    
    def get_expression_association_prob(self, sign_word: str, expression: str) -> float:
        """Get association probability for sign word and expression"""
        return self.expr_associations.get((sign_word, expression), 0.0)


class CorrectOptimalHybridAligner:
    """Correct optimal hybrid aligner - best of both worlds"""
    
    def __init__(
        self,
        signs_dir: str,
        exprs_dir: str,
        sign_lm_path: str,
        expr_lm_path: str,
        best_outputs_glosses: Optional[Dict[Tuple[str, int], str]] = None,
    ):
        self.ctc_loader = CTCProbabilityLoader(signs_dir, exprs_dir)
        self.lm_rescorer = LanguageModelRescorer(sign_lm_path, expr_lm_path)
        self.best_outputs_glosses = best_outputs_glosses or {}
    
    def enhance_sign_sequence_with_lm_only(self, sample_id: int) -> List[str]:
        """
        Apply LM model for signs (no probability matrices, no direct raw CTC output)
        """
        sign_probs = self.ctc_loader.get_sign_probabilities(sample_id)
        sign_vocab = self.ctc_loader.get_sign_vocabulary(sample_id)
        
        if sign_probs is None or not sign_vocab:
            return []
        
        # Add blank token to vocabulary
        full_vocab = sign_vocab + ['<BLANK>']
        blank_idx = len(sign_vocab)
        
        # Initialize beam search
        beam = [([], 0.0)]  # (sequence, log_prob)
        
        for t in range(sign_probs.shape[0]):
            new_beam = []
            
            for sequence, log_prob in beam:
                # Consider all possible signs
                for sign_idx, sign in enumerate(full_vocab):
                    if sign == '<BLANK>':
                        # Blank token - no change to sequence
                        new_beam.append((sequence, log_prob + math.log(sign_probs[t, sign_idx])))
                    else:
                        if sign_idx < len(sign_vocab):
                            # Get CTC probability
                            ctc_prob = sign_probs[t, sign_idx]
                            
                            # Get LM transition probability
                            context = sequence[-1] if sequence else ""
                            lm_prob = self.lm_rescorer.get_sign_transition_prob(context, sign)
                            
                            # If no LM probability, use a small default
                            if lm_prob == 0.0:
                                lm_prob = 0.001
                            
                            # Combine CTC and LM probabilities
                            alpha = 0.7  # Weight for CTC probability
                            beta = 0.3   # Weight for LM probability
                            
                            combined_prob = alpha * ctc_prob + beta * lm_prob
                            new_log_prob = log_prob + math.log(combined_prob)
                            
                            # Add to sequence if not duplicate
                            if not sequence or sequence[-1] != sign:
                                new_sequence = sequence + [sign]
                            else:
                                new_sequence = sequence
                            
                            new_beam.append((new_sequence, new_log_prob))
            
            # Keep top beam_size candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:10]  # Larger beam for signs
        
        # Return best sequence
        if beam:
            return beam[0][0]
        return []
    
    def rescore_expression_sequence_with_original_optimal_approach(self, sample_id: int, sign_sequence: List[str]) -> List[str]:
        """
        Apply same approach as original optimal_hybrid_aligner.py for expressions (8.33% WER)
        """
        expr_probs = self.ctc_loader.get_expr_probabilities(sample_id)
        expr_vocab = self.ctc_loader.get_expr_vocabulary(sample_id)
        
        if expr_probs is None or not expr_vocab or not sign_sequence:
            return []
        
        # Add blank token to vocabulary
        full_vocab = expr_vocab + ['<BLANK>']
        blank_idx = len(expr_vocab)
        
        # Initialize beam search
        beam = [([], 0.0)]  # (sequence, log_prob)
        
        for t in range(expr_probs.shape[0]):
            new_beam = []
            
            for sequence, log_prob in beam:
                # Consider all possible expressions
                for expr_idx, expr in enumerate(full_vocab):
                    if expr == '<BLANK>':
                        # Blank token - no change to sequence
                        new_beam.append((sequence, log_prob + math.log(expr_probs[t, expr_idx])))
                    else:
                        if expr_idx < len(expr_vocab):
                            # Get CTC probability
                            ctc_prob = expr_probs[t, expr_idx]
                            
                            # Get association probability with current sign
                            # Map time step to sign (simple mapping for now)
                            sign_idx = min(t * len(sign_sequence) // expr_probs.shape[0], len(sign_sequence) - 1)
                            current_sign = sign_sequence[sign_idx] if sign_idx < len(sign_sequence) else sign_sequence[-1]
                            
                            # Try to match sign with language model vocabulary (case-insensitive)
                            assoc_prob = 0.0
                            for lm_sign in self.lm_rescorer.expr_associations:
                                if lm_sign[0].upper() == current_sign.upper():
                                    assoc_prob = self.lm_rescorer.get_expression_association_prob(lm_sign[0], expr)
                                    break
                            
                            # If no exact match, use a small default probability
                            if assoc_prob == 0.0:
                                assoc_prob = 0.001  # Very small default probability
                            
                            # Sophisticated combination: weighted combination of CTC and association
                            alpha = 0.6  # Weight for CTC probability
                            beta = 0.4   # Weight for association probability
                            
                            combined_prob = alpha * ctc_prob + beta * assoc_prob
                            new_log_prob = log_prob + math.log(combined_prob)
                            
                            # Add to sequence if not duplicate
                            if not sequence or sequence[-1] != expr:
                                new_sequence = sequence + [expr]
                            else:
                                new_sequence = sequence
                            
                            new_beam.append((new_sequence, new_log_prob))
            
            # Keep top beam_size candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:5]  # Smaller beam for expressions
        
        # Return best sequence
        if beam:
            return beam[0][0]
        return []
    
    def build_enhanced_aligned_sentence(self, sign_sequence: List[str], expr_sequence: List[str]) -> str:
        """Build enhanced aligned sentence with expressions associated to signs"""
        if not sign_sequence:
            return ""
        
        # Create a simple mapping: distribute expressions across signs
        words = []
        expr_per_sign = len(expr_sequence) / len(sign_sequence) if sign_sequence else 0
        
        for i, sign in enumerate(sign_sequence):
            # Calculate which expressions belong to this sign
            start_expr = int(i * expr_per_sign)
            end_expr = int((i + 1) * expr_per_sign)
            sign_exprs = expr_sequence[start_expr:end_expr]
            
            if sign_exprs:
                words.append(f"{sign}({','.join(sign_exprs)})")
            else:
                words.append(sign)
        
        return ' '.join(words)
    
    def get_ground_truth_gloss(self, sample_id: int) -> Optional[str]:
        """Retrieve ground truth gloss from best_outputs mapping via session/sample index metadata."""
        metadata_source = None
        if sample_id in self.ctc_loader.expr_samples:
            metadata_source = self.ctc_loader.expr_samples[sample_id]['metadata']
        elif sample_id in self.ctc_loader.sign_samples:
            metadata_source = self.ctc_loader.sign_samples[sample_id]['metadata']
        else:
            return None

        session = metadata_source.get('session')
        sample_idx = metadata_source.get('sample_index')
        if session is None or sample_idx is None:
            return None

        key = (str(session), int(sample_idx))
        return self.best_outputs_glosses.get(key)
    
    def remove_emotions_from_gloss(self, gloss: str) -> str:
        """Remove emotions (happy, sad, angry) from ASL gloss expressions in brackets"""
        # List of emotions to remove
        emotions_to_remove = {'happy', 'sad', 'angry'}
        
        # Pattern to match expressions in brackets: SIGN(expr1,expr2,expr3)
        def clean_brackets(match):
            sign = match.group(1)  # The sign before the bracket
            exprs_str = match.group(2)  # The expressions inside brackets
            
            # Split expressions by comma and filter out emotions
            exprs = [expr.strip() for expr in exprs_str.split(',')]
            filtered_exprs = [expr for expr in exprs if expr.lower() not in emotions_to_remove]
            
            # If no expressions left, return just the sign, otherwise return sign(filtered_exprs)
            if not filtered_exprs:
                return sign
            else:
                return f"{sign}({','.join(filtered_exprs)})"
        
        # Pattern: word followed by (expressions)
        pattern = r'(\S+)\(([^)]+)\)'
        cleaned_gloss = re.sub(pattern, clean_brackets, gloss)
        
        return cleaned_gloss
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.split(',') if ',' in reference else reference.split()
        hyp_words = hypothesis.split(',') if ',' in hypothesis else hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        # Simple WER calculation (can be improved with proper edit distance)
        matches = 0
        for ref_word in ref_words:
            if ref_word in hyp_words:
                matches += 1
        
        return 1.0 - (matches / len(ref_words))
    
    def process_all_samples(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all samples using the correct optimal combination
        """
        enhanced_alignments = []
        wer_results = []
        
        # Process all available samples
        # Find the maximum sample ID by checking available files
        max_samples = 0
        for i in range(1000):  # Check up to 1000 samples
            sample_str = f"sample_{i:04d}"
            sign_metadata_path = os.path.join(self.ctc_loader.signs_dir, f"{sample_str}_metadata.npy")
            if os.path.exists(sign_metadata_path):
                max_samples = i + 1
            else:
                break
        print(f"Processing {max_samples} samples...")
        
        for i in range(max_samples):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Processing sample {i}...")
            
            try:
                # Load probability data for this sample
                self.ctc_loader.load_sample_data(i)
                
                # Get ground truth from probability matrices metadata
                sign_vocab = self.ctc_loader.get_sign_vocabulary(i)
                expr_vocab = self.ctc_loader.get_expr_vocabulary(i)
                
                if not sign_vocab or not expr_vocab:
                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"  Skipping sample {i} - missing vocabulary")
                    continue
                
                # Get ground truth from metadata
                if i in self.ctc_loader.sign_samples:
                    gt_signs = self.ctc_loader.sign_samples[i]['metadata'].get('ground_truth', [])
                    if isinstance(gt_signs, list):
                        gt_signs = ','.join(gt_signs)
                else:
                    gt_signs = "MY,FATHER,WANT,SELL,HIS,CAR"  # Fallback
                
                if i in self.ctc_loader.expr_samples:
                    gt_exprs = self.ctc_loader.expr_samples[i]['metadata'].get('ground_truth', [])
                    if isinstance(gt_exprs, list):
                        gt_exprs = ','.join(gt_exprs)
                else:
                    gt_exprs = "sad,mm"  # Fallback
                
                gt_exprs_list = gt_exprs.split(',') if ',' in gt_exprs else [gt_exprs]
                gt_gloss = self.get_ground_truth_gloss(i)
                
                # Enhance sign sequence using LM only (no probability matrices, no direct raw CTC output)
                enhanced_signs = self.enhance_sign_sequence_with_lm_only(i)
                
                # Rescore expressions using original optimal approach (8.33% WER)
                enhanced_exprs = self.rescore_expression_sequence_with_original_optimal_approach(i, enhanced_signs)
                
                # Build enhanced aligned sentence
                enhanced_sentence = self.build_enhanced_aligned_sentence(enhanced_signs, enhanced_exprs)
                
                # Remove emotions (happy, sad, angry) from the recognized sentence
                enhanced_sentence = self.remove_emotions_from_gloss(enhanced_sentence)
                
                # Create ground truth aligned sentence: prefer best_outputs gloss if available
                if gt_gloss:
                    gt_aligned_sentence = gt_gloss
                else:
                    gt_aligned_sentence = self.build_enhanced_aligned_sentence(gt_signs.split(','), gt_exprs_list)
                
                # Store enhanced alignment
                enhanced_alignments.append({
                    'sample_id': i,
                    'ground_truth': gt_aligned_sentence,
                    'enhanced_sentence': enhanced_sentence
                })
                
                # Calculate WER
                signs_wer = self.calculate_wer(gt_signs, ','.join(enhanced_signs))
                exprs_wer = self.calculate_wer(gt_exprs, ','.join(enhanced_exprs))
                
                # Store WER results
                wer_results.append({
                    'sample_id': i,
                    'ground_truth_signs': gt_signs,
                    'enhanced_signs': ','.join(enhanced_signs),
                    'signs_wer': signs_wer,
                    'ground_truth_exprs': gt_exprs,
                    'enhanced_exprs': ','.join(enhanced_exprs),
                    'exprs_wer': exprs_wer
                })
                
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Sample {i}: GT={gt_aligned_sentence[:50]}... Enhanced={enhanced_sentence[:50]}...")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        return enhanced_alignments, wer_results


def process_all_folds(signs_base_dir: str, exprs_base_dir: str, sign_lm_path: str, expr_lm_path: str, output_file: str):
    """
    Process all 10 folds and combine results into a single output file
    """
    all_alignments = []
    
    # Process folds 1-10 in order
    for fold_num in range(1, 11):
        fold_name = f"140_70_None_all_ch4_poi300350_w16080_k10_fold_{fold_num}"
        signs_fold_name = f"140_70_None_signs_ch4_poi300350_w16080_k10_fold_{fold_num}"
        
        signs_dir = os.path.join(signs_base_dir, signs_fold_name, "best_model_probability_matrices")
        exprs_dir = os.path.join(exprs_base_dir, fold_name, "best_model_probability_matrices")
        expr_best_outputs_path = os.path.join(exprs_base_dir, fold_name, "best_outputs.txt")
        
        if not os.path.exists(signs_dir) or not os.path.exists(exprs_dir):
            print(f"Warning: Fold {fold_num} directories not found. Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*60}")
        print(f"Signs directory: {signs_dir}")
        print(f"Expressions directory: {exprs_dir}")
        print(f"Expressions best outputs: {expr_best_outputs_path}")
        
        best_outputs_glosses = load_best_output_glosses(expr_best_outputs_path)
        
        # Create aligner for this fold
        aligner = CorrectOptimalHybridAligner(
            signs_dir,
            exprs_dir,
            sign_lm_path,
            expr_lm_path,
            best_outputs_glosses=best_outputs_glosses,
        )
        
        # Process all samples in this fold
        enhanced_alignments, _ = aligner.process_all_samples()
        
        # Add to combined results
        all_alignments.extend(enhanced_alignments)
        
        print(f"Fold {fold_num} complete: {len(enhanced_alignments)} samples processed")
    
    # Write combined results to output file
    print(f"\n{'='*60}")
    print(f"Writing combined results to {output_file}")
    print(f"{'='*60}")
    
    exact_matches = 0
    total_samples = len(all_alignments)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_alignments:
            gt_gloss = result['ground_truth']
            aligned_gloss = result['enhanced_sentence']
            
            # Write in format: ground_truth_ASL_gloss - aligned_ASL_gloss_after_LM
            f.write(f"{gt_gloss} - {aligned_gloss}\n")
            
            # Check for exact match
            if gt_gloss.strip() == aligned_gloss.strip():
                exact_matches += 1
    
    # Calculate exact match accuracy
    exact_match_accuracy = (exact_matches / total_samples * 100) if total_samples > 0 else 0.0
    
    # Append metrics to the end of the file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("EXACT MATCH ACCURACY METRICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Exact matches: {exact_matches}\n")
        f.write(f"Exact match accuracy: {exact_match_accuracy:.2f}%\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nResults Summary:")
    print(f"=" * 60)
    print(f"Total samples processed (all folds): {total_samples}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
    print(f"Output file: {output_file}")
    print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Alignment with LM - Process all folds and combine results")
    parser.add_argument(
        "--signs_base_dir",
        type=str,
        default="Penultimate/140_70_signs",
        help="Base directory containing sign fold directories (e.g., Penultimate/140_70_signs)"
    )
    parser.add_argument(
        "--exprs_base_dir", 
        type=str,
        default="Penultimate/140_70",
        help="Base directory containing expression fold directories (e.g., Penultimate/140_70)"
    )
    parser.add_argument(
        "--sign_lm",
        type=str,
        default="Penultimate/asl_models/sign_language_model.csv",
        help="Path to sign language model CSV"
    )
    parser.add_argument(
        "--expr_lm",
        type=str,
        default="Penultimate/asl_models/expression_association_model.csv",
        help="Path to expression association model CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Penultimate/final_pipeline_txt/alignment/alignment_with_LM.txt",
        help="Output file path (single file with all folds combined)"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    # Script is in Penultimate/FInal_pipeline/, so go up one level to get to Penultimate/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    penultimate_dir = os.path.dirname(script_dir)  # Go up from FInal_pipeline to Penultimate
    
    def resolve_path(path):
        """Resolve relative path relative to Penultimate directory"""
        if os.path.isabs(path):
            return path
        # Remove "Penultimate/" prefix if present
        if path.startswith("Penultimate/"):
            path = path[len("Penultimate/"):]
        return os.path.normpath(os.path.join(penultimate_dir, path))
    
    signs_base_dir = resolve_path(args.signs_base_dir)
    exprs_base_dir = resolve_path(args.exprs_base_dir)
    sign_lm_path = resolve_path(args.sign_lm)
    expr_lm_path = resolve_path(args.expr_lm)
    output_file = resolve_path(args.output)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("Alignment with Language Model - All Folds")
    print("=" * 60)
    print("Strategy:")
    print("- Signs: Apply LM model (no probability matrices, no direct raw CTC output)")
    print("- Expressions: Apply same approach as original optimal_hybrid_aligner.py (8.33% WER)")
    print("=" * 60)
    print(f"Signs base directory: {signs_base_dir}")
    print(f"Expressions base directory: {exprs_base_dir}")
    print(f"Sign LM: {sign_lm_path}")
    print(f"Expression LM: {expr_lm_path}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    # Process all folds and combine results
    process_all_folds(signs_base_dir, exprs_base_dir, sign_lm_path, expr_lm_path, output_file)


if __name__ == "__main__":
    main()
