#!/usr/bin/env python3
"""
Convert VLA JSON format to Qwen training format.

Your current format:
{
    "qid": "0",
    "mode": "keyframe_selection", 
    "message_list": [
        {"role": "system", "content": [{"text": "..."}]},
        {"role": "user", "content": [{"text": "..."}, {"video": ["path.png"]}]},
        {"role": "assistant", "content": [{"text": "JSON response"}]}
    ],
    "metadata": {...}
}

Expected format:
{
    "conversations": [
        {"from": "human", "value": "combined text content with <image> tokens"},
        {"from": "gpt", "value": "response text"}
    ],
    "video": ["path1.png", "path2.png", ...] or "path.png" for single frame
}
"""

import json
import argparse
import os
from typing import List, Dict, Any, Union


def convert_content_list_to_text(content_list: List[Dict[str, Any]], base_path: str = None) -> tuple[str, List[str]]:
    """
    Convert content list to text string and extract media files.
    
    Args:
        content_list: List of content items with text/image/video
        base_path: Optional base directory path to prepend to relative media paths
    
    Returns:
        tuple: (combined_text, media_files)
    """
    text_parts = []
    media_files = []
    
    for content_item in content_list:
        if "text" in content_item:
            text_parts.append(content_item["text"])
        elif "video" in content_item:
            # Add image token for each video frame
            frames = content_item["video"]
            if isinstance(frames, list):
                for frame in frames:
                    text_parts.append("<image>")
                    # Prepend base_path if provided and frame is relative
                    full_path = os.path.join(base_path, frame) if base_path and not os.path.isabs(frame) else frame
                    media_files.append(full_path)
            else:
                text_parts.append("<image>")
                # Prepend base_path if provided and frame is relative
                full_path = os.path.join(base_path, frames) if base_path and not os.path.isabs(frames) else frames
                media_files.append(full_path)
        elif "image" in content_item:
            # Handle image content if present
            images = content_item["image"]
            if isinstance(images, list):
                for image in images:
                    text_parts.append("<image>")
                    # Prepend base_path if provided and image is relative
                    full_path = os.path.join(base_path, image) if base_path and not os.path.isabs(image) else image
                    media_files.append(full_path)
            else:
                text_parts.append("<image>")
                # Prepend base_path if provided and image is relative
                full_path = os.path.join(base_path, images) if base_path and not os.path.isabs(images) else images
                media_files.append(full_path)
    
    combined_text = "\n".join(text_parts).strip()
    return combined_text, media_files


def convert_vla_item(vla_item: Dict[str, Any], base_path: str = None) -> Dict[str, Any]:
    """
    Convert a single VLA item to Qwen training format.
    
    Args:
        vla_item: VLA format item to convert
        base_path: Optional base directory path to prepend to relative media paths
    """
    if "message_list" not in vla_item:
        raise ValueError(f"Missing 'message_list' in item {vla_item.get('qid', 'unknown')}")
    
    message_list = vla_item["message_list"]
    conversations = []
    all_media_files = []
    
    # Role mapping
    role_mapping = {
        "system": "system",  # Keep system as system
        "user": "human", 
        "assistant": "gpt"
    }
    
    for message in message_list:
        role = message.get("role", "")
        content = message.get("content", [])
        
        if not isinstance(content, list):
            # Handle case where content is already a string
            content_text = str(content)
            media_files = []
        else:
            content_text, media_files = convert_content_list_to_text(content, base_path)
        
        # Map role
        mapped_role = role_mapping.get(role, role)
        
        # Keep system messages - they contain important task-specific instructions
        
        # Add to conversations
        if content_text:  # Only add non-empty content
            conversations.append({
                "from": mapped_role,
                "value": content_text
            })
        
        # Collect media files
        all_media_files.extend(media_files)
    
    # Build result
    result = {
        "conversations": conversations
    }
    
    # Add media files if present
    if all_media_files:
        if len(all_media_files) == 1:
            result["video"] = all_media_files[0]  # Single frame
        else:
            result["video"] = all_media_files  # Multiple frames
    
    # Preserve original metadata if needed
    if "qid" in vla_item:
        result["qid"] = vla_item["qid"]
    if "metadata" in vla_item:
        result["metadata"] = vla_item["metadata"]
    
    return result


def convert_vla_dataset(input_file: str, output_file: str, base_path: str = None):
    """
    Convert a full VLA dataset file to Qwen training format.
    
    Args:
        input_file: Path to input VLA JSON file
        output_file: Path to output Qwen format JSON file  
        base_path: Optional base directory path to prepend to relative media paths
    """
    print(f"Loading VLA dataset from: {input_file}")
    
    # Load the input JSON
    with open(input_file, 'r') as f:
        vla_data = json.load(f)
    
    if not isinstance(vla_data, list):
        raise ValueError("Expected input to be a list of VLA items")
    
    print(f"Found {len(vla_data)} VLA items to convert")
    
    # Convert each item
    converted_data = []
    errors = []
    
    for i, vla_item in enumerate(vla_data):
        try:
            converted_item = convert_vla_item(vla_item, base_path)
            converted_data.append(converted_item)
        except Exception as e:
            error_msg = f"Error converting item {i} (qid: {vla_item.get('qid', 'unknown')}): {str(e)}"
            errors.append(error_msg)
            print(f"WARNING: {error_msg}")
    
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


def main():
    parser = argparse.ArgumentParser(description="Convert VLA JSON format to Qwen training format")
    parser.add_argument("input_file", help="Input VLA JSON file")
    parser.add_argument("output_file", help="Output Qwen training JSON file")
    parser.add_argument("--sample-size", type=int, help="Convert only first N items for testing")
    parser.add_argument("--base-path", type=str, help="Base directory path to prepend to relative media file paths")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return 1
    
    # Handle sample size
    if args.sample_size:
        print(f"Converting only first {args.sample_size} items for testing")
        with open(args.input_file, 'r') as f:
            full_data = json.load(f)
        
        sample_data = full_data[:args.sample_size]
        
        # Create temporary file for sample
        sample_file = args.input_file.replace('.json', f'_sample_{args.sample_size}.json')
        with open(sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        try:
            convert_vla_dataset(sample_file, args.output_file, args.base_path)
        finally:
            # Clean up sample file
            if os.path.exists(sample_file):
                os.remove(sample_file)
    else:
        convert_vla_dataset(args.input_file, args.output_file, args.base_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
