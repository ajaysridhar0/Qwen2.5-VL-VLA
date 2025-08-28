#!/usr/bin/env python3
"""
Simple test to verify JSON evaluation is working correctly.
"""

import torch
from transformers import AutoTokenizer, AutoProcessor
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_mixed_vla import MixedVLADataset
from types import SimpleNamespace
import os

# Mock data args
data_args = SimpleNamespace(
    droid_data_dir="/iliad2/u/ajaysri/episodic_memory/droid_rlds",
    droid_dataset_name="droid_100",
    action_chunk_size=15,
    use_joint_velocity=True,
    shuffle_buffer_size=1000,
    num_droid_samples=100,
    image_height=180,
    image_width=320,
    enable_cotrain=True,
    cotrain_json_paths=["/iris/u/ajaysri/datasets/scratch_pad_combined_with_action_framesub_5_context_16/compiled_new_qwen.json"],
    cotrain_json_ratio=1.0,
)

# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
data_args.image_processor = processor.image_processor

# Mock action tokenizer
from transformers import AutoProcessor as ActionProcessor
action_tokenizer = ActionProcessor.from_pretrained("KarlP/fast-droid", trust_remote_code=True)

# Token mappings (from your training setup)
token_mappings = {
    "state_token_ids": list(range(148000, 148256)),
    "action_token_ids": list(range(148256, 149280)),
    "control_mappings": {
        "<|action_start|>": 149280,
        "<|action_end|>": 149281,
        "<|state_start|>": 149282,
        "<|state_end|>": 149283,
    },
    "action_start_id": 149280,
    "action_end_id": 149281,
    "state_start_id": 149282,
    "state_end_id": 149283,
}

print("Creating MixedVLADataset with cotrain_json_ratio=1.0...")
dataset = MixedVLADataset(
    tokenizer=tokenizer,
    data_args=data_args,
    action_tokenizer=action_tokenizer,
    model_max_length=5000,
    token_mappings=token_mappings,
    image_size=(180, 320),
    cotrain_json_ratio=1.0,
)

print(f"\nDataset created! Length: {len(dataset)}")
print(f"Dataset type: {type(dataset)}")

# Test getting a few examples
print("\nTesting first 5 examples:")
for i in range(min(5, len(dataset))):
    try:
        example = dataset[i]
        print(f"\nExample {i}:")
        print(f"  Keys: {list(example.keys())}")
        
        # Check if it has action tokens (VLA) or not (JSON)
        input_ids = example['input_ids']
        if hasattr(input_ids, 'flatten'):
            input_ids_list = input_ids.flatten().tolist()
        else:
            input_ids_list = input_ids.tolist()
            
        has_action_tokens = any(token_id in token_mappings['action_token_ids'] 
                              for token_id in input_ids_list)
        has_action_start = token_mappings['action_start_id'] in input_ids_list
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Has action tokens: {has_action_tokens}")
        print(f"  Has action_start token: {has_action_start}")
        print(f"  Type: {'VLA' if has_action_tokens else 'JSON'}")
        
        # Print a snippet of the text
        text_snippet = tokenizer.decode(input_ids_list[:100], skip_special_tokens=False)
        print(f"  Text snippet: {text_snippet[:200]}...")
        
    except Exception as e:
        print(f"\nError processing example {i}: {e}")
        import traceback
        traceback.print_exc()

print("\nTest complete!")
