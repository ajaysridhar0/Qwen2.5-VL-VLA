"""
Converter for transforming Qwen API format conversations to training data format.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path


def convert_qwen_api_to_training_format(api_messages: List[Dict[str, Any]], 
                                       base_data_path: str = "",
                                       include_system_messages: bool = True) -> Dict[str, Any]:
    """
    Convert Qwen API format messages to training data format.
    
    Args:
        api_messages: List of messages in Qwen API format
        base_data_path: Base path for media files (optional)
        include_system_messages: Whether to include system messages (default: False)
        
    Returns:
        Dictionary in training data format
    """
    conversations = []
    all_images = []
    all_videos = []
    video_as_frames = []  # For videos provided as image sequences
    
    for msg in api_messages:
        role = msg["role"]
        content_array = msg["content"]
        
        # Skip system messages if requested
        if role == "system" and not include_system_messages:
            continue
            
        # Process content array to extract text and media
        text_parts = []
        msg_images = []
        msg_videos = []
        
        for item in content_array:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                elif "image" in item:
                    msg_images.append(item["image"])
                    # Add image token where the image appears
                    text_parts.append("<image>")
                elif "video" in item:
                    # Video provided as list of image frames
                    video_frames = item["video"]
                    if isinstance(video_frames, list) and all(".jpg" in f or ".png" in f for f in video_frames):
                        # This is a video represented as frames
                        video_as_frames.append(video_frames)
                        text_parts.append("<video>")
                    else:
                        # Regular video file
                        msg_videos.append(item["video"])
                        text_parts.append("<video>")
        
        # Add images to the global list (no duplicates)
        all_images.extend(msg_images)
        
        # Combine text parts
        content_text = " ".join(text_parts).strip()
        
        # Create conversation entry
        conv_entry = {
            "role": role,
            "content": content_text
        }
        
        # Only add non-empty conversations
        if content_text:
            conversations.append(conv_entry)
    
    # Build the training data format
    result = {
        "conversations": conversations
    }
    
    # Add media files if present
    if all_images:
        result["image"] = all_images if len(all_images) > 1 else all_images[0]
        
    if video_as_frames:
        # For now, we'll treat video frames as a special case
        # You might need to handle this differently based on your needs
        result["video_frames"] = video_as_frames
        
    if base_data_path:
        result["data_path"] = base_data_path
        
    return result


def convert_video_frames_to_video_format(frames_data: Dict[str, Any], 
                                        output_dir: str = "/tmp/converted_videos") -> Dict[str, Any]:
    """
    Convert video frames format to standard video format.
    The data loader now supports frame sequences directly.
    
    Args:
        frames_data: Data dictionary with video_frames
        output_dir: Directory to save converted data
        
    Returns:
        Updated data dictionary
    """
    if "video_frames" in frames_data:
        video_frames = frames_data["video_frames"]
        if video_frames:
            # For single video with frames, store the frames directly
            # The data loader will detect it's a list of frames
            if len(video_frames) == 1:
                frames_data["video"] = video_frames[0]
            else:
                # Multiple videos - store as list
                frames_data["video"] = video_frames
        
        # Remove the temporary video_frames key
        del frames_data["video_frames"]
    
    return frames_data


def process_jsonl_file(input_file: str, output_file: str, base_data_path: str = ""):
    """
    Process a JSONL file containing Qwen API format conversations.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        base_data_path: Base path for media files
    """
    converted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            api_messages = json.loads(line.strip())
            converted = convert_qwen_api_to_training_format(api_messages, base_data_path)
            converted = convert_video_frames_to_video_format(converted)
            converted_data.append(converted)
    
    # Write converted data
    with open(output_file, 'w') as f:
        for item in converted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(converted_data)} samples from {input_file} to {output_file}")


def test_conversion():
    """Test the conversion with the provided example."""
    # Example from the user
    example_messages = [
        {
            "role": "system",
            "content": [
                {
                    "text": "You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions..."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "Which object was put down by the person?\nA: The food.\nB: The laptop.\nC: The cup/glass/bottle.\nD: The shoe. Please think step by step. Put your final answer in \\boxed{}"
                },
                {
                    "video": [
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/1.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/2.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/3.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/4.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/5.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/6.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/7.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/8.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/9.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/10.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/11.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/12.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/13.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/14.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/15.jpg",
                        "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/16.jpg"
                    ]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "The video shows a woman entering a room holding a cup or glass. She walks toward the door and places the cup or glass on a surface near the door. Afterward, she appears to interact with her phone while standing near the door.  \n\nNow I will select some frames to look clearer at which object was put down by the person.<tool_call>{\"name\": \"select_frames\", \"arguments\": {\"target_frames\": [1, 2, 3, 4]}}</tool_call>"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "Here are the selected frames: "
                },
                {
                    "image": "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/17.jpg"
                },
                {
                    "image": "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/18.jpg"
                },
                {
                    "image": "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/19.jpg"
                },
                {
                    "image": "/home/ma-user/work/haozhe/muze/modelartsdata/starqa/612/20.jpg"
                }
            ]
        }
    ]
    
    # Convert the example
    converted = convert_qwen_api_to_training_format(example_messages)
    converted = convert_video_frames_to_video_format(converted)
    
    print("Converted format:")
    print(json.dumps(converted, indent=2))
    
    return converted


if __name__ == "__main__":
    # Run test
    test_conversion()
