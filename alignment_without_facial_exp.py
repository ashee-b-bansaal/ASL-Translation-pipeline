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
    Extract recognized signs from the input file and save without commas.
    
    Args:
        input_file: Path to the input sign_best_outputs.txt file
        output_dir: Directory where the output file will be saved
        output_filename: Name of the output file
    """
    recognized_signs_list = []
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                recognized_signs_list.append('')
                continue
            
            # Parse the line: ground_truth - recognized_signs <<<<< CTC_output >>>> ASL_gloss ----- session, sample
            # Extract the recognized signs part (between '-' and '<<<<<')
            if ' - ' in line and ' <<<<< ' in line:
                parts = line.split(' - ', 1)
                if len(parts) == 2:
                    recognized_part = parts[1].split(' <<<<< ', 1)[0]
                    # Remove commas and replace with spaces
                    recognized_signs = recognized_part.replace(',', ' ').strip()
                    recognized_signs_list.append(recognized_signs)
                else:
                    recognized_signs_list.append('')
            else:
                # If format doesn't match, append empty line
                recognized_signs_list.append('')
    
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Combine output directory and filename
    output_file = output_dir_path / output_filename
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for signs in recognized_signs_list:
            f.write(signs + '\n')
    
    print(f"Successfully extracted {len([s for s in recognized_signs_list if s])} lines with recognized signs")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract recognized signs from sign_best_outputs.txt without commas'
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
        required=True,
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

