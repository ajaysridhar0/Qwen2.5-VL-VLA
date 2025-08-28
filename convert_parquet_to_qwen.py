#!/usr/bin/env python3
"""
Convert parquet format with Q&A and media to Qwen training format.

Input parquet structure:
- question: Text question from user
- answer: Array containing the response text
- qid: Question ID  
- image: Array of image/video frame paths
- is_video: Boolean indicating if media is video frames

Output Qwen format:
{
    "conversations": [
        {"from": "human", "value": "question with <image> tokens"},
        {"from": "gpt", "value": "response text"}
    ],

    "video": ["path1.jpg", "path2.jpg", ...] or "path.jpg" for single frame,
    "image": ["path1.jpg", "path2.jpg", ...] or "path.jpg" for single image
}
"""

import pandas as pd
import json
import argparse
import os
from typing import List, Dict, Any, Union
import numpy as np


def convert_parquet_row_to_qwen(row: pd.Series, base_path: str = None, system_prompt: str = None) -> Dict[str, Any]:
    """
    Convert a single parquet row to Qwen training format.
    
    Args:
        row: Pandas Series representing one row from the parquet file
        base_path: Optional base directory path to prepend to relative media paths
        system_prompt: Optional system prompt to include (uses default if None)
    """
    # Extract fields
    question = row['question']
    answer = row['answer']
    qid = row['qid']
    image_paths = row['image']
    is_video = row['is_video']
    
    # Handle answer field (convert from array to string if needed)
    if isinstance(answer, np.ndarray):
        answer_text = answer[0] if len(answer) > 0 else ""
    elif isinstance(answer, list):
        answer_text = answer[0] if len(answer) > 0 else ""
    else:
        answer_text = str(answer)
    
    # Handle image paths (convert from numpy array to list if needed)
    if isinstance(image_paths, np.ndarray):
        image_list = image_paths.tolist()
    elif isinstance(image_paths, list):
        image_list = image_paths
    else:
        # Single image path
        image_list = [str(image_paths)]
    
    # Prepend base_path if provided and paths are relative
    if base_path:
        processed_paths = []
        for path in image_list:
            if not os.path.isabs(path):
                full_path = os.path.join(base_path, path)
            else:
                full_path = path
            processed_paths.append(full_path)
        image_list = processed_paths
    
    # Create the question with appropriate tokens based on media type
    # Add one <image> or <video> token for each frame/image after the question
    if is_video:
        media_tokens = "<video>"
    else:
        media_tokens = "\n".join(["<image>"] * len(image_list))
    question_with_tokens = f"{question}\n{media_tokens}".strip()
    
    # Build conversations with system prompt (use default if none provided)
    conversations = []
    
    # Use default system prompt if none provided
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    
    # Add system message
    conversations.append({
        "from": "system",
        "value": system_prompt
    })
    
    # Add user and assistant messages
    conversations.extend([
        {"from": "human", "value": question_with_tokens},
        {"from": "gpt", "value": answer_text}
    ])
    
    # Build result
    result = {
        "conversations": conversations
    }
    
    # Add media files if present
    if image_list:
        if is_video:
            # Video frames
            if len(image_list) == 1:
                result["video"] = image_list[0]  # Single video frame
            else:
                result["video"] = image_list  # Multiple video frames
        else:
            # Static images
            if len(image_list) == 1:
                result["image"] = image_list[0]  # Single image
            else:
                result["image"] = image_list  # Multiple images
    
    # Preserve original metadata
    result["qid"] = qid
    
    return result


def convert_parquet_to_qwen(input_file: str, output_file: str, base_path: str = None, sample_size: int = None, system_prompt: str = None):
    """
    Convert a parquet file to Qwen training format.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output Qwen format JSON file  
        base_path: Optional base directory path to prepend to relative media paths
        sample_size: Optional number of samples to convert for testing
        system_prompt: Optional system prompt to include in all conversations
    """
    print(f"Loading parquet dataset from: {input_file}")
    
    # Load the parquet file
    df = pd.read_parquet(input_file)
    
    print(f"Found {len(df)} rows in parquet file")
    print(f"Columns: {list(df.columns)}")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling first {sample_size} rows for testing")
        df = df.head(sample_size)
    
    # Convert each row
    converted_data = []
    errors = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        try:
            converted_item = convert_parquet_row_to_qwen(row, base_path, system_prompt)
            converted_data.append(converted_item)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)} rows...")
                
        except Exception as e:
            error_msg = f"Row {i} (index {idx}): {str(e)}"
            errors.append(error_msg)
            print(f"Error processing row {i}: {e}")
    
    print(f"Successfully converted {len(converted_data)} items")
    if errors:
        print(f"Failed to convert {len(errors)} items")
        print("Errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    # Save the converted data
    print(f"Saving converted dataset to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Conversion complete! Saved {len(converted_data)} items to {output_file}")
    
    # Show a sample for verification
    if converted_data:
        print("\nSample converted item:")
        print(json.dumps(converted_data[0], indent=2))
        
        # Show statistics
        video_count = sum(1 for item in converted_data if "video" in item)
        image_count = sum(1 for item in converted_data if "image" in item)
        print(f"\nStatistics:")
        print(f"- Video samples: {video_count}")
        print(f"- Image samples: {image_count}")


def main():
    parser = argparse.ArgumentParser(description="Convert parquet format to Qwen training format")
    parser.add_argument("input_file", help="Input parquet file")
    parser.add_argument("output_file", help="Output Qwen training JSON file")
    parser.add_argument("--sample-size", type=int, help="Convert only first N items for testing")
    parser.add_argument("--base-path", type=str, help="Base directory path to prepend to relative media file paths")
    parser.add_argument("--system-prompt", type=str, help="System prompt to include in all conversations (optional)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return 1
    
    try:
        convert_parquet_to_qwen(args.input_file, args.output_file, args.base_path, args.sample_size, args.system_prompt)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
