#!/usr/bin/env python3
"""
Standalone script to run evaluation on JSON examples and log generations.
"""

import os
import torch
from transformers import Trainer, TrainingArguments
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.train.train_qwen_vla import (
    VLAModelArguments, VLADataArguments, VLATrainingArguments,
    get_action_state_token_mappings, set_model
)
from qwenvl.data.data_mixed_vla import make_mixed_vla_data_module
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser
)

def main():
    # Parse arguments
    parser = HfArgumentParser(
        (VLAModelArguments, VLADataArguments, VLATrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load model from checkpoint
    checkpoint_dir = training_args.output_dir
    print(f"Loading model from {checkpoint_dir}")
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Load action tokenizer and mappings
    action_tokenizer = AutoProcessor.from_pretrained(
        model_args.action_tokenizer_path, 
        trust_remote_code=True
    )
    token_mappings = get_action_state_token_mappings(
        tokenizer, 
        action_tokenizer.vocab_size,
        output_dir=training_args.output_dir
    )
    
    # Load image processor
    data_args.image_processor = AutoProcessor.from_pretrained(
        checkpoint_dir
    ).image_processor
    data_args.model_type = "qwen2.5vl"
    
    # Create eval dataset
    print(f"Creating evaluation dataset with cotrain_json_ratio={data_args.cotrain_json_ratio}")
    data_module = make_mixed_vla_data_module(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args,
        model_max_length=training_args.model_max_length,
        token_mappings=token_mappings,
        image_size=(data_args.image_height, data_args.image_width),
        cotrain_json_ratio=data_args.cotrain_json_ratio,
    )
    
    eval_dataset = data_module['eval_dataset']
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Test a few examples
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    num_examples = min(5, len(eval_dataset))
    print(f"\nGenerating outputs for {num_examples} examples...")
    
    for i in range(num_examples):
        print(f"\n{'='*80}")
        print(f"Example {i+1}/{num_examples}")
        
        example = eval_dataset[i]
        
        # Check if it's a JSON or VLA example
        input_ids = example['input_ids']
        if hasattr(input_ids, 'flatten'):
            input_ids_list = input_ids.flatten().tolist()
        else:
            input_ids_list = input_ids.tolist()
            
        has_action_tokens = any(
            token_id in token_mappings['action_token_ids'] 
            for token_id in input_ids_list
        )
        
        print(f"Type: {'VLA' if has_action_tokens else 'JSON'}")
        
        # Find where assistant response starts
        labels = example['labels']
        assistant_start_pos = None
        for idx, label in enumerate(labels.tolist()):
            if label != -100:
                assistant_start_pos = idx
                break
        
        if assistant_start_pos is None:
            print("Could not find assistant response start")
            continue
        
        # Get input text
        input_text = tokenizer.decode(input_ids[:assistant_start_pos].tolist(), skip_special_tokens=False)
        print(f"Input (last 300 chars): ...{input_text[-300:]}")
        
        # Get ground truth
        gt_token_ids = [t for t in labels.tolist() if t != -100]
        gt_text = tokenizer.decode(gt_token_ids, skip_special_tokens=True)
        print(f"Ground Truth: {gt_text[:200]}{'...' if len(gt_text) > 200 else ''}")
        
        # Generate
        with torch.no_grad():
            gen_input_ids = input_ids[:assistant_start_pos].unsqueeze(0).to(device)
            
            # Handle vision inputs if present
            pixel_values = None
            image_grid_thw = None
            if 'pixel_values' in example and example['pixel_values'] is not None:
                pixel_values = example['pixel_values'].unsqueeze(0).to(device)
            if 'image_grid_thw' in example and example['image_grid_thw'] is not None:
                image_grid_thw = example['image_grid_thw'].unsqueeze(0).to(device)
            
            outputs = model.generate(
                input_ids=gen_input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_tokens = outputs[0][len(gen_input_ids[0]):].tolist()
            pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Generated: {pred_text[:200]}{'...' if len(pred_text) > 200 else ''}")
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
