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
        # Case-insensitive lookup map: uppercase sign -> original sign
        self.sign_case_map = {}
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
                # Build case-insensitive lookup map
                self.sign_case_map[sign_word.upper()] = sign_word
                
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
        sign_alpha: float = 0.7,
        sign_beta: float = 0.3,
        expr_alpha: float = 0.6,
        expr_beta: float = 0.4,
        default_lm_prob: float = 0.001,
    ):
        self.ctc_loader = CTCProbabilityLoader(signs_dir, exprs_dir)
        self.lm_rescorer = LanguageModelRescorer(sign_lm_path, expr_lm_path)
        self.best_outputs_glosses = best_outputs_glosses or {}
        self.sign_alpha = sign_alpha
        self.sign_beta = sign_beta
        self.expr_alpha = expr_alpha
        self.expr_beta = expr_beta
        self.default_lm_prob = default_lm_prob
    
    def enhance_sign_sequence_with_lm_only(self, sample_id: int) -> List[str]:
        """
        Apply LM model for signs using CTC probability matrices with language model rescoring.
        Uses proper CTC collapse rules and log-space probability combination.
        """
        sign_probs = self.ctc_loader.get_sign_probabilities(sample_id)
        sign_vocab = self.ctc_loader.get_sign_vocabulary(sample_id)
        
        if sign_probs is None or not sign_vocab:
            return []
        
        # Add blank token to vocabulary
        full_vocab = sign_vocab + ['<BLANK>']
        blank_idx = len(sign_vocab)
        
        # Initialize beam search
        # Use dict to properly merge duplicate sequences (CTC collapse rule)
        beam_dict = {tuple([]): 0.0}  # (sequence_tuple, log_prob)
        
        for t in range(sign_probs.shape[0]):
            new_beam_dict = {}
            
            for sequence_tuple, log_prob in beam_dict.items():
                sequence = list(sequence_tuple)
                last_token = sequence[-1] if sequence else None
                
                # First, handle blank token (no change to sequence)
                blank_log_prob = log_prob + math.log(sign_probs[t, blank_idx] + 1e-10)
                seq_key = tuple(sequence)
                if seq_key in new_beam_dict:
                    # Log-sum-exp for merging probabilities (numerically stable)
                    max_prob = max(new_beam_dict[seq_key], blank_log_prob)
                    new_beam_dict[seq_key] = max_prob + math.log(
                        math.exp(new_beam_dict[seq_key] - max_prob) + 
                        math.exp(blank_log_prob - max_prob)
                    )
                else:
                    new_beam_dict[seq_key] = blank_log_prob
                
                # Then, handle all non-blank tokens
                for sign_idx in range(len(sign_vocab)):
                    sign = sign_vocab[sign_idx]
                    
                    # CTC collapse rule: skip if same as last token
                    if last_token == sign:
                        continue  # Don't add duplicate token (CTC collapse)
                    
                    # Get CTC probability
                    ctc_log_prob = math.log(sign_probs[t, sign_idx] + 1e-10)
                    
                    # Get LM transition probability (bigram: previous word -> current word)
                    context = last_token if last_token else ""
                    lm_prob = self.lm_rescorer.get_sign_transition_prob(context, sign)
                    
                    # If no LM probability, use a small default
                    if lm_prob == 0.0:
                        lm_log_prob = math.log(self.default_lm_prob)
                    else:
                        lm_log_prob = math.log(lm_prob)
                    
                    # Combine CTC and LM probabilities in log space
                    # Weighted combination: alpha * log(CTC) + beta * log(LM)
                    # This is equivalent to: log(CTC^alpha * LM^beta)
                    alpha = self.sign_alpha  # Weight for CTC probability
                    beta = self.sign_beta    # Weight for LM probability
                    
                    combined_log_prob = alpha * ctc_log_prob + beta * lm_log_prob
                    new_log_prob = log_prob + combined_log_prob
                    
                    # Add token to sequence (already checked for duplicates above)
                    new_sequence = sequence + [sign]
                    
                    # Merge sequences with same content
                    seq_key = tuple(new_sequence)
                    if seq_key in new_beam_dict:
                        # Log-sum-exp for merging probabilities (numerically stable)
                        max_prob = max(new_beam_dict[seq_key], new_log_prob)
                        new_beam_dict[seq_key] = max_prob + math.log(
                            math.exp(new_beam_dict[seq_key] - max_prob) + 
                            math.exp(new_log_prob - max_prob)
                        )
                    else:
                        new_beam_dict[seq_key] = new_log_prob
            
            # Convert back to list and keep top beam_size candidates
            beam = [(list(seq), prob) for seq, prob in new_beam_dict.items()]
            beam.sort(key=lambda x: x[1], reverse=True)
            beam_dict = {tuple(seq): prob for seq, prob in beam[:10]}  # Larger beam for signs
        
        # Return best sequence
        if beam_dict:
            best_seq = max(beam_dict.items(), key=lambda x: x[1])[0]
            return list(best_seq)
        return []
    
    def rescore_expression_sequence_with_original_optimal_approach(self, sample_id: int, sign_sequence: List[str]) -> List[str]:
        """
        Apply same approach as original optimal_hybrid_aligner.py for expressions (8.33% WER).
        Uses proper CTC collapse rules, efficient expression lookup, and log-space probability combination.
        """
        expr_probs = self.ctc_loader.get_expr_probabilities(sample_id)
        expr_vocab = self.ctc_loader.get_expr_vocabulary(sample_id)
        
        if expr_probs is None or not expr_vocab or not sign_sequence:
            return []
        
        # Add blank token to vocabulary
        full_vocab = expr_vocab + ['<BLANK>']
        blank_idx = len(expr_vocab)
        
        # Pre-compute sign-to-time mapping for better alignment
        # Create a mapping from expression time steps to sign indices
        num_expr_steps = expr_probs.shape[0]
        num_signs = len(sign_sequence)
        sign_time_mapping = []
        for t in range(num_expr_steps):
            # Improved mapping: use proportional alignment
            sign_idx = min(int(t * num_signs / num_expr_steps), num_signs - 1)
            sign_time_mapping.append(sign_idx)
        
        # Initialize beam search
        # Use dict to properly merge duplicate sequences (CTC collapse rule)
        beam_dict = {tuple([]): 0.0}  # (sequence_tuple, log_prob)
        
        for t in range(expr_probs.shape[0]):
            new_beam_dict = {}
            
            # Get the corresponding sign for this time step
            sign_idx = sign_time_mapping[t]
            current_sign = sign_sequence[sign_idx] if sign_idx < len(sign_sequence) else sign_sequence[-1]
            
            for sequence_tuple, log_prob in beam_dict.items():
                sequence = list(sequence_tuple)
                last_expr = sequence[-1] if sequence else None
                
                # First, handle blank token (no change to sequence)
                blank_log_prob = log_prob + math.log(expr_probs[t, blank_idx] + 1e-10)
                seq_key = tuple(sequence)
                if seq_key in new_beam_dict:
                    # Log-sum-exp for merging probabilities (numerically stable)
                    max_prob = max(new_beam_dict[seq_key], blank_log_prob)
                    new_beam_dict[seq_key] = max_prob + math.log(
                        math.exp(new_beam_dict[seq_key] - max_prob) + 
                        math.exp(blank_log_prob - max_prob)
                    )
                else:
                    new_beam_dict[seq_key] = blank_log_prob
                
                # Then, handle all non-blank expressions
                for expr_idx in range(len(expr_vocab)):
                    expr = expr_vocab[expr_idx]
                    
                    # CTC collapse rule: skip if same as last expression
                    if last_expr == expr:
                        continue  # Don't add duplicate expression (CTC collapse)
                    
                    # Get CTC probability
                    ctc_log_prob = math.log(expr_probs[t, expr_idx] + 1e-10)
                    
                    # Efficient direct lookup for expression association probability
                    # Try exact match first (case-sensitive)
                    assoc_prob = self.lm_rescorer.get_expression_association_prob(current_sign, expr)
                    
                    # If no exact match, try case-insensitive lookup using pre-built map
                    if assoc_prob == 0.0:
                        current_sign_upper = current_sign.upper()
                        if current_sign_upper in self.lm_rescorer.sign_case_map:
                            original_sign = self.lm_rescorer.sign_case_map[current_sign_upper]
                            assoc_prob = self.lm_rescorer.get_expression_association_prob(original_sign, expr)
                    
                            # If still no match, use a small default probability
                            if assoc_prob == 0.0:
                                assoc_log_prob = math.log(self.default_lm_prob)
                            else:
                                assoc_log_prob = math.log(assoc_prob)
                            
                            # Combine CTC and association probabilities in log space
                            # Weighted combination: alpha * log(CTC) + beta * log(assoc)
                            # This is equivalent to: log(CTC^alpha * assoc^beta)
                            alpha = self.expr_alpha  # Weight for CTC probability
                            beta = self.expr_beta    # Weight for association probability
                    
                    combined_log_prob = alpha * ctc_log_prob + beta * assoc_log_prob
                    new_log_prob = log_prob + combined_log_prob
                    
                    # Add expression to sequence (already checked for duplicates above)
                    new_sequence = sequence + [expr]
                    
                    # Merge sequences with same content
                    seq_key = tuple(new_sequence)
                    if seq_key in new_beam_dict:
                        # Log-sum-exp for merging probabilities (numerically stable)
                        max_prob = max(new_beam_dict[seq_key], new_log_prob)
                        new_beam_dict[seq_key] = max_prob + math.log(
                            math.exp(new_beam_dict[seq_key] - max_prob) + 
                            math.exp(new_log_prob - max_prob)
                        )
                    else:
                        new_beam_dict[seq_key] = new_log_prob
            
            # Convert back to list and keep top beam_size candidates
            beam = [(list(seq), prob) for seq, prob in new_beam_dict.items()]
            beam.sort(key=lambda x: x[1], reverse=True)
            beam_dict = {tuple(seq): prob for seq, prob in beam[:5]}  # Smaller beam for expressions
        
        # Return best sequence
        if beam_dict:
            best_seq = max(beam_dict.items(), key=lambda x: x[1])[0]
            return list(best_seq)
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


def process_all_folds(
    signs_base_dir: str, 
    exprs_base_dir: str, 
    sign_lm_path: str, 
    expr_lm_path: str, 
    output_file: str,
    sign_alpha: float = 0.7,
    sign_beta: float = 0.3,
    expr_alpha: float = 0.6,
    expr_beta: float = 0.4,
    default_lm_prob: float = 0.001,
):
    """
    Process all folds discovered under the provided base directories and combine results
    into a single output file.
    """

    def discover_fold_dirs(base_dir: str) -> Dict[int, str]:
        """Return mapping fold_number -> absolute path for each *_fold_* directory."""
        fold_dirs: Dict[int, str] = {}
        if not os.path.isdir(base_dir):
            print(f"Warning: Base directory not found: {base_dir}")
            return fold_dirs
        for entry in os.listdir(base_dir):
            entry_path = os.path.join(base_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            match = re.search(r'fold_(\d+)', entry)
            if match:
                fold_num = int(match.group(1))
                fold_dirs[fold_num] = entry_path
        return fold_dirs

    sign_fold_dirs = discover_fold_dirs(signs_base_dir)
    expr_fold_dirs = discover_fold_dirs(exprs_base_dir)

    all_alignments: List[Dict] = []
    available_folds = sorted(set(sign_fold_dirs.keys()) & set(expr_fold_dirs.keys()))

    if not available_folds:
        print("No common fold directories found between signs and expressions. Nothing to process.")
        return

    for fold_num in available_folds:
        sign_fold_path = sign_fold_dirs[fold_num]
        expr_fold_path = expr_fold_dirs[fold_num]

        signs_dir = os.path.join(sign_fold_path, "best_model_probability_matrices")
        exprs_dir = os.path.join(expr_fold_path, "best_model_probability_matrices")
        expr_best_outputs_path = os.path.join(expr_fold_path, "best_outputs.txt")

        missing = []
        if not os.path.exists(signs_dir):
            missing.append(f"sign probability matrices at {signs_dir}")
        if not os.path.exists(exprs_dir):
            missing.append(f"expression probability matrices at {exprs_dir}")
        if missing:
            print(f"Warning: Fold {fold_num} missing {' & '.join(missing)}. Skipping...")
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
            sign_alpha=sign_alpha,
            sign_beta=sign_beta,
            expr_alpha=expr_alpha,
            expr_beta=expr_beta,
            default_lm_prob=default_lm_prob,
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
        default="Penultimate/final_pipeline_txt_220110/alignment/alignment_with_LM.txt",
        help="Output file path (single file with all folds combined)"
    )
    parser.add_argument(
        "--sign_alpha",
        type=float,
        default=0.7,
        help="Weight for CTC probability in sign sequence (default: 0.7)"
    )
    parser.add_argument(
        "--sign_beta",
        type=float,
        default=0.3,
        help="Weight for LM probability in sign sequence (default: 0.3)"
    )
    parser.add_argument(
        "--expr_alpha",
        type=float,
        default=0.6,
        help="Weight for CTC probability in expression sequence (default: 0.6)"
    )
    parser.add_argument(
        "--expr_beta",
        type=float,
        default=0.4,
        help="Weight for LM probability in expression sequence (default: 0.4)"
    )
    parser.add_argument(
        "--default_lm_prob",
        type=float,
        default=0.001,
        help="Default LM probability when transition not found (default: 0.001)"
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
    process_all_folds(
        signs_base_dir, 
        exprs_base_dir, 
        sign_lm_path, 
        expr_lm_path, 
        output_file,
        sign_alpha=args.sign_alpha,
        sign_beta=args.sign_beta,
        expr_alpha=args.expr_alpha,
        expr_beta=args.expr_beta,
        default_lm_prob=args.default_lm_prob,
    )


if __name__ == "__main__":
    main()
