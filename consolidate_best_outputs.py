#!/usr/bin/env python3
"""
Script to consolidate best_outputs.txt files from all folds in order.
Reads best_outputs.txt from each fold directory (fold_1, fold_2, ..., fold_10)
and consolidates them into a single file in the order of the folds.

Terminal command: 
python consolidate_best_outputs.py root_directory_path --output-dir "output_directory_path"
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple


def find_fold_folders(base_path: str) -> List[Tuple[int, str]]:
    """
    Find all fold folders in the base path.
    Returns: List of (fold_number, folder_path) tuples sorted by fold number.
    
    Args:
        base_path: Path to search in
    """
    base = Path(base_path)
    
    if not base.exists():
        raise ValueError(f"Path does not exist: {base_path}")
    
    fold_folders = []
    
    # Pattern to match folders containing 'fold_X' where X is 1-10
    for item in base.iterdir():
        if item.is_dir():
            # Try to match pattern like '...fold_1' or '...fold_10'
            match = re.search(r'fold[_\s]*(\d+)', item.name, re.IGNORECASE)
            if match:
                fold_num = int(match.group(1))
                if 1 <= fold_num <= 10:
                    fold_folders.append((fold_num, str(item)))
    
    # Sort by fold number
    fold_folders.sort(key=lambda x: x[0])
    return fold_folders


def read_best_outputs(folder_path: str) -> List[str]:
    """Read best_outputs.txt from a fold folder."""
    file_path = Path(folder_path) / "best_outputs.txt"
    if not file_path.exists():
        print(f"Warning: {file_path} not found, skipping...")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate best_outputs.txt files from all folds in order'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to folder containing fold directories (e.g., Penultimate/140_70)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory where consolidated file will be saved'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='exp_best_outputs.txt',
        help='Output filename (default: exp_best_outputs.txt)'
    )
    
    args = parser.parse_args()
    
    # Find all fold folders
    fold_folders = find_fold_folders(args.folder_path)
    if not fold_folders:
        raise ValueError(f"No fold folders found in {args.folder_path}")
    
    print(f"Found {len(fold_folders)} fold folders: {[f[0] for f in fold_folders]}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Collect all outputs in fold order
    all_outputs = []
    
    for fold_num, fold_path in fold_folders:
        print(f"Processing Fold {fold_num}...")
        
        # Read best_outputs.txt
        outputs = read_best_outputs(fold_path)
        all_outputs.extend(outputs)
        print(f"  Added {len(outputs)} lines from best_outputs.txt")
    
    # Write consolidated file
    output_file = output_dir / args.output_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_outputs:
            f.write(line + '\n')
    
    print(f"\n✓ Wrote {len(all_outputs)} total lines to {output_file}")
    print("✅ Consolidation complete!")


if __name__ == "__main__":
    main()

