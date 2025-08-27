"""
Generation logging callback for Qwen VLA training.
Handles logging of generation outputs and calculating accuracy during evaluation.
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import wandb

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


def rank0_print(*args):
    """Print only on rank 0 to avoid duplicate output in distributed training."""
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


class GenerationLoggingCallback(TrainerCallback):
    """Callback for logging generation outputs and calculating accuracy during evaluation."""
    
    def __init__(
        self, 
        tokenizer, 
        action_tokenizer,
        token_mappings,
        num_examples=10,
        log_file="generations.txt",
        log_to_wandb=True,
    ):
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.token_mappings = token_mappings
        self.num_examples = num_examples
        self.log_file = log_file
        self.log_to_wandb = log_to_wandb
        self.action_token_ids = set(token_mappings['action_token_ids'])
        self.action_start_id = token_mappings['action_start_id']
        self.action_end_id = token_mappings['action_end_id']
        self.model = None
        self.eval_dataloader = None
        
    def setup_trainer_components(self, model, eval_dataloader):
        """Set up the callback with required trainer components."""
        self.model = model
        self.eval_dataloader = eval_dataloader
        rank0_print(f"[GenerationLogger] Callback setup complete - eval_dataloader: {eval_dataloader is not None}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        rank0_print(f"[GenerationLogger] on_train_begin called - callback is active!")
        return control
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called during evaluation to log generation outputs."""
        rank0_print(f"\n[GenerationLogger] ========== on_evaluate called at step {state.global_step} ==========")
        rank0_print(f"[GenerationLogger] model is None: {self.model is None}")
        rank0_print(f"[GenerationLogger] eval_dataloader is None: {self.eval_dataloader is None}")
        rank0_print(f"[GenerationLogger] kwargs keys: {list(kwargs.keys())}")
        rank0_print(f"[GenerationLogger] num_examples configured: {self.num_examples}")
        
        # Check if we have the required components
        if self.model is None or self.eval_dataloader is None:
            rank0_print("[GenerationLogger] Skipping generation logging - model or eval_dataloader not available")
            return control
        if state.global_step == 0:
            rank0_print("[GenerationLogger] Skipping generation logging - step 0")
            return control
            
        try:
            model = self.model
            eval_dataloader = self.eval_dataloader
            
            model.eval()
            rank0_print(f"\n[GenerationLogger] Generating {self.num_examples} examples at step {state.global_step}")
            
            vla_generation_logs = []
            json_generation_logs = []
            vla_accuracies = []
            json_accuracies = []
            
            # Sample examples from eval dataloader
            vla_examples_processed = 0
            json_examples_processed = 0
            total_examples_processed = 0
            
            for batch_idx, batch in enumerate(eval_dataloader):
                if total_examples_processed >= self.num_examples:
                    break
                    
                # Move batch to device
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Debug batch contents
                rank0_print(f"[GenerationLogger] Batch keys: {list(batch.keys())}")
                if 'pixel_values' in batch:
                    rank0_print(f"[GenerationLogger] Batch pixel_values shape: {batch['pixel_values'].shape if batch['pixel_values'] is not None else 'None'}")
                if 'image_grid_thw' in batch:
                    rank0_print(f"[GenerationLogger] Batch image_grid_thw shape: {batch['image_grid_thw'].shape if batch['image_grid_thw'] is not None else 'None'}")
                
                # Check if this is a VLA example (has action tokens) or JSON example
                input_ids = batch['input_ids'][0]
                is_vla_example = self.action_start_id in input_ids.tolist()
                
                if is_vla_example:
                    # Extract ground truth action tokens from labels
                    labels = batch['labels'][0]  # Take first example in batch
                    gt_action_tokens = []
                    for token_id in labels.tolist():
                        if token_id != -100 and token_id in self.action_token_ids:
                            gt_action_tokens.append(token_id)
                    
                    # Generate VLA predictions
                    with torch.no_grad():
                        # Find the position of action_start token in input_ids
                        action_start_pos = None
                        for i, token_id in enumerate(input_ids.tolist()):
                            if token_id == self.action_start_id:
                                action_start_pos = i
                                break
                        
                        if action_start_pos is None: # TODO: make assert
                            continue
                        
                        # Prepare inputs for generation (up to and including action_start)
                        gen_input_ids = input_ids[:action_start_pos + 1].unsqueeze(0)
                        
                        # Check if we have valid image data
                        if batch.get('pixel_values') is None or batch.get('image_grid_thw') is None:
                            rank0_print(f"[GenerationLogger] Skipping VLA example - no image data in batch")
                            continue

                        input_text =  self.tokenizer.decode(batch['input_ids'][0])
                        num_images = input_text.count("<|vision_start|>")
                        total_num_patches = input_text.count("<|image_pad|>") * 4 # 4 patches per token
                        pixel_values = batch['pixel_values'][:total_num_patches]
                        image_grid_thw = batch['image_grid_thw'][:num_images] # two images per batch
                        
                        # Debug shapes
                        rank0_print(f"[GenerationLogger] pixel_values shape: {pixel_values.shape if pixel_values is not None else 'None'}")
                        rank0_print(f"[GenerationLogger] image_grid_thw shape: {image_grid_thw.shape if image_grid_thw is not None else 'None'}")
                        
                        # Check for empty tensors or invalid shapes
                        if pixel_values.numel() == 0 or pixel_values.shape[0] == 0:
                            rank0_print(f"[GenerationLogger] Skipping VLA example - empty pixel_values tensor")
                            continue
                        
                        # Check grid_thw validity
                        if image_grid_thw.numel() == 0 or image_grid_thw.shape[0] == 0:
                            rank0_print(f"[GenerationLogger] Skipping VLA example - empty image_grid_thw tensor")
                            continue
                        
                        # Additional debug info
                        rank0_print(f"[GenerationLogger] pixel_values numel: {pixel_values.numel()}")
                        rank0_print(f"[GenerationLogger] gen_input_ids shape: {gen_input_ids.shape}")
                        
                        # Skip generation for now - there's a shape mismatch issue
                        # The pixel_values from the dataset are already processed vision features (2D)
                        # but the model.generate expects raw images or differently shaped tensors
                        rank0_print(f"[GenerationLogger] Skipping generation due to known shape mismatch issue")
                        rank0_print(f"[GenerationLogger] This is a known issue with processed vision features during generation")
                        
                        # For now, just use ground truth as predictions to continue testing
                        # pred_action_tokens = gt_action_tokens[:10]  # Use first 10 GT tokens as fake prediction
                        
                        # Continue with the rest of the logging
                        # TODO: Fix generation with proper pixel_values format
                        
                        # Generate action tokens
                        outputs = model.generate(
                            input_ids=gen_input_ids,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            max_new_tokens=len(gt_action_tokens) + 10,  # Allow some extra tokens
                            do_sample=False,  # Use greedy decoding for deterministic results
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                       
                        
                        # Extract generated action tokens
                        # (Commented out due to generation being skipped)
                        generated_tokens = outputs[0][len(gen_input_ids[0]):].tolist()
                        pred_action_tokens = []
                        for token_id in generated_tokens:
                            if token_id == self.tokenizer.eos_token_id or token_id == self.action_end_id:
                                break
                            if token_id in self.action_token_ids:
                                pred_action_tokens.append(token_id)
                        
                        # Calculate accuracy (exact match of action sequences)
                        is_correct = pred_action_tokens == gt_action_tokens
                        vla_accuracies.append(float(is_correct))
                        
                        # Decode actions back to continuous values
                        l2_distance = None
                        if gt_action_tokens:
                            try:
                                # Map back from infrequent tokens to action indices
                                gt_action_indices = [self.token_mappings['action_token_ids'].index(t) for t in gt_action_tokens]
                                pred_action_indices = [self.token_mappings['action_token_ids'].index(t) for t in pred_action_tokens if t in self.token_mappings['action_token_ids']]
                                # Decode to continuous actions if possible
                                gt_continuous = self.action_tokenizer.decode([gt_action_indices])[0]
                                if pred_action_indices:
                                    pred_continuous = self.action_tokenizer.decode([pred_action_indices])[0]
                                    # Calculate L2 distance
                                    l2_distance = float(np.linalg.norm(gt_continuous - pred_continuous))
                                else:
                                    pred_continuous = np.zeros_like(gt_continuous)
                                    l2_distance = float(np.linalg.norm(gt_continuous))
                            except Exception as e:
                                rank0_print(f"[GenerationLogger] Error decoding actions: {e}")
                                gt_continuous = None
                                pred_continuous = None
                                l2_distance = None
                        
                        # Get input text (without action tokens)
                        input_text = self.tokenizer.decode(input_ids[:action_start_pos].tolist(), skip_special_tokens=False)
                        
                        # Log generation info
                        log_entry = {
                            'step': state.global_step,
                            'example_idx': vla_examples_processed,
                            'input_text': input_text[-500:],  # Last 500 chars to avoid too long
                            'gt_action_tokens': gt_action_tokens[:20],  # Limit for display
                            'pred_action_tokens': pred_action_tokens[:20],
                            'is_correct': is_correct,
                            'l2_distance': l2_distance,
                            'gt_continuous': gt_continuous.tolist() if gt_continuous is not None else None,
                            'pred_continuous': pred_continuous.tolist() if pred_continuous is not None else None,
                        }
                        vla_generation_logs.append(log_entry)
                        
                        vla_examples_processed += 1
                        total_examples_processed += 1
                        
                else:
                    # Generate JSON predictions (regular text generation)
                    with torch.no_grad():
                        # Find where the assistant response should start
                        labels = batch['labels'][0]
                        assistant_start_pos = None
                        for i, label in enumerate(labels.tolist()):
                            if label != -100:  # First non-ignored token is start of assistant response
                                assistant_start_pos = i
                                break
                        
                        if assistant_start_pos is None:
                            continue
                        
                        # Prepare inputs for generation (up to assistant response)
                        gen_input_ids = input_ids[:assistant_start_pos].unsqueeze(0)
                        
                        # Get ground truth text
                        gt_token_ids = []
                        for token_id in labels.tolist():
                            if token_id != -100:
                                gt_token_ids.append(token_id)
                        gt_text = self.tokenizer.decode(gt_token_ids, skip_special_tokens=True)

                        input_text =  self.tokenizer.decode(batch['input_ids'][0])
                        num_images = input_text.count("<|vision_start|>")
                        total_num_patches = input_text.count("<|image_pad|>") * 4 # 4 patches per token
                        pixel_values = batch['pixel_values'][:total_num_patches]
                        image_grid_thw = batch['image_grid_thw'][:num_images] # two images per batch
                        
                        # Generate text response
                        outputs = model.generate(
                            input_ids=gen_input_ids,
                            pixel_values=pixel_values,
                            image_grid_thw=image_grid_thw,
                            max_new_tokens=min(len(gt_token_ids) + 50, 200),  # Allow some extra tokens but cap at 200
                            do_sample=False,  # Use greedy decoding for deterministic results
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        # Extract generated text
                        generated_tokens = outputs[0][len(gen_input_ids[0]):].tolist()
                        pred_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        # Calculate 0-1 accuracy (exact string match)
                        is_correct = pred_text.strip() == gt_text.strip()
                        json_accuracies.append(float(is_correct))
                        
                        # Get input text
                        input_text = self.tokenizer.decode(input_ids[:assistant_start_pos].tolist(), skip_special_tokens=False)
                        
                        # Log generation info
                        log_entry = {
                            'step': state.global_step,
                            'example_idx': json_examples_processed,
                            'input_text': input_text[-500:],  # Last 500 chars to avoid too long
                            'gt_text': gt_text[:200],  # Limit for display
                            'pred_text': pred_text[:200],
                            'is_correct': is_correct,
                            'type': 'json',
                        }
                        json_generation_logs.append(log_entry)
                        
                        json_examples_processed += 1
                        total_examples_processed += 1
            
            # Calculate overall VLA accuracy and L2 distance
            vla_accuracy = np.mean(vla_accuracies) if vla_accuracies else 0.0
            l2_distances = [log['l2_distance'] for log in vla_generation_logs if log['l2_distance'] is not None]
            avg_l2_distance = np.mean(l2_distances) if l2_distances else None
            
            # Calculate overall JSON accuracy
            json_accuracy = np.mean(json_accuracies) if json_accuracies else 0.0
            
            # Log to wandb
            if self.log_to_wandb and wandb.run is not None:
                log_dict = {
                    'eval/vla_generation_accuracy': vla_accuracy,
                    'eval/vla_num_examples': len(vla_accuracies),
                    'eval/json_generation_accuracy': json_accuracy,
                    'eval/json_num_examples': len(json_generation_logs),
                    'global_step': state.global_step,
                }
                if avg_l2_distance is not None:
                    log_dict['eval/vla_avg_l2_distance'] = avg_l2_distance
                wandb.log(log_dict)
                
                # Log VLA example generations to wandb
                if vla_generation_logs:
                    vla_table = wandb.Table(
                        columns=['step', 'example_idx', 'input_preview', 'gt_tokens', 'pred_tokens', 'correct', 'l2_distance']
                    )
                    for log in vla_generation_logs[:5]:  # Log first 5 examples
                        vla_table.add_data(
                            log['step'],
                            log['example_idx'],
                            log['input_text'][-200:],  # Last 200 chars
                            str(log['gt_action_tokens'][:10]),
                            str(log['pred_action_tokens'][:10]),
                            log['is_correct'],
                            f"{log['l2_distance']:.4f}" if log['l2_distance'] is not None else "N/A"
                        )
                    wandb.log({'eval/vla_generations': vla_table})
                
                # Log JSON example generations to wandb
                if json_generation_logs:
                    json_table = wandb.Table(
                        columns=['step', 'example_idx', 'input_preview', 'gt_text', 'pred_text', 'correct']
                    )
                    for log in json_generation_logs[:5]:  # Log first 5 examples
                        json_table.add_data(
                            log['step'],
                            log['example_idx'],
                            log['input_text'][-200:],  # Last 200 chars
                            log['gt_text'][:100],  # First 100 chars
                            log['pred_text'][:100],  # First 100 chars
                            log['is_correct']
                        )
                    wandb.log({'eval/json_generations': json_table})
            
            # Write to file
            output_path = os.path.join(args.output_dir, self.log_file)
            with open(output_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Step: {state.global_step} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"VLA Examples: {len(vla_generation_logs)} | JSON Examples: {len(json_generation_logs)}\n")
                if vla_generation_logs:
                    f.write(f"VLA Accuracy: {vla_accuracy:.4f} ({len(vla_accuracies)} examples)\n")
                    if avg_l2_distance is not None:
                        f.write(f"Average L2 Distance: {avg_l2_distance:.4f} ({len(l2_distances)} examples)\n")
                if json_generation_logs:
                    f.write(f"JSON Accuracy: {json_accuracy:.4f} ({len(json_accuracies)} examples)\n")
                f.write(f"{'='*80}\n\n")
                
                # Write VLA examples
                if vla_generation_logs:
                    f.write("=== VLA Generation Examples ===\n\n")
                    for log in vla_generation_logs:
                        f.write(f"VLA Example {log['example_idx']}:\n")
                        f.write(f"Input (last 500 chars): ...{log['input_text']}\n")
                        f.write(f"GT Action Tokens: {log['gt_action_tokens']}\n")
                        f.write(f"Pred Action Tokens: {log['pred_action_tokens']}\n")
                        f.write(f"Correct: {log['is_correct']}\n")
                        if log['l2_distance'] is not None:
                            f.write(f"L2 Distance: {log['l2_distance']:.4f}\n")
                        if log['gt_continuous'] is not None:
                            f.write(f"GT Continuous: {log['gt_continuous']}\n")
                            f.write(f"Pred Continuous: {log['pred_continuous']}\n")
                        f.write(f"{'-'*40}\n\n")
                
                # Write JSON examples
                if json_generation_logs:
                    f.write("=== JSON Generation Examples ===\n\n")
                    for log in json_generation_logs:
                        f.write(f"JSON Example {log['example_idx']}:\n")
                        f.write(f"Input (last 500 chars): ...{log['input_text']}\n")
                        f.write(f"GT Text: {log['gt_text']}\n")
                        f.write(f"Pred Text: {log['pred_text']}\n")
                        f.write(f"Correct: {log['is_correct']}\n")
                        f.write(f"{'-'*40}\n\n")
            
            rank0_print(f"[GenerationLogger] VLA: {len(vla_generation_logs)} examples, Accuracy: {vla_accuracy:.4f} | JSON: {len(json_generation_logs)} examples, Accuracy: {json_accuracy:.4f} | Logged to {output_path}")
            
        except Exception as e:
            rank0_print(f"[GenerationLogger] Error during evaluation logging: {e}")
            import traceback
            traceback.print_exc()
        
        return control
