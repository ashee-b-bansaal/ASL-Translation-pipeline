#!/usr/bin/env python3
"""
Script to extract recognized signs from sign_best_outputs.txt
and save them without commas to a new text file.

Terminal command: 
python alignment_without_facial_exp.py input_file --output_dir output_directory --output_filename output_file.txt
"""

import os
import argparse
from pathlib import Path


def extract_recognized_signs(input_file, output_dir, output_filename):
    """
    Extract recognized signs and ground truth ASL gloss from the input file.
    Output format: ground_truth_ASL_gloss - recognized_signs
    
    Args:
        input_file: Path to the input sign_best_outputs.txt file
        output_dir: Directory where the output file will be saved
        output_filename: Name of the output file
    """
    output_lines = []
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                output_lines.append('')
                continue
            
            # Parse the line: ground_truth - recognized_signs <<<<< CTC_output >>>> ASL_gloss ----- session, sample
            # Extract:
            # 1. Ground truth ASL gloss (between '>>>>' and '-----')
            # 2. Recognized signs (between '-' and '<<<<<')
            
            ground_truth_gloss = ""
            recognized_signs = ""
            
            # Extract ground truth ASL gloss (between '>>>>' and '-----')
            if ' >>>> ' in line and ' ----- ' in line:
                gloss_match = line.split(' >>>> ', 1)
                if len(gloss_match) == 2:
                    gloss_part = gloss_match[1].split(' ----- ', 1)[0]
                    ground_truth_gloss = gloss_part.strip()
            
            # Extract recognized signs (between '-' and '<<<<<')
            if ' - ' in line and ' <<<<< ' in line:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    recognized_part = parts[1].split(' <<<<< ', 1)[0]
                    # Remove commas and replace with spaces
                    recognized_signs = recognized_part.replace(',', ' ').strip()
            
            # Format output: ground_truth_ASL_gloss - recognized_signs
            if ground_truth_gloss and recognized_signs:
                output_lines.append(f"{ground_truth_gloss} - {recognized_signs}")
            elif recognized_signs:
                # If no ground truth gloss found, just output recognized signs
                output_lines.append(f" - {recognized_signs}")
            elif ground_truth_gloss:
                # If no recognized signs found, just output ground truth gloss
                output_lines.append(f"{ground_truth_gloss} - ")
            else:
                # If format doesn't match, append empty line
                output_lines.append('')
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Combine output directory and filename
    output_file = output_dir_path / output_filename
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for output_line in output_lines:
            f.write(output_line + '\n')
    
    print(f"Successfully extracted {len([s for s in output_lines if s])} lines")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ground truth ASL gloss and recognized signs from sign_best_outputs.txt. Output format: ground_truth_ASL_gloss - recognized_signs'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input sign_best_outputs.txt file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where the output file will be saved'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        required=False,
        default='alignment_without_facial_exp.txt',
        help='Name of the output file'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return
    
    # Extract and save recognized signs
    extract_recognized_signs(args.input_file, args.output_dir, args.output_filename)


if __name__ == '__main__':
    main()

