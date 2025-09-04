"""
Simplified generation logger for inline use in compute_loss function.
Handles logging of generation outputs and calculating accuracy without the trainer callback system.
"""

import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import wandb


def rank0_print(*args):
    """Print only on rank 0 to avoid duplicate output in distributed training."""
    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


class SimpleGenerationLogger:
    """Simplified generation logger that can be called directly from compute_loss."""
    
    def __init__(
        self, 
        tokenizer, 
        action_tokenizer,
        token_mappings,
        num_examples=5,  # Reduced from 10 to be lighter
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
        
    def log_generations_from_batch(self, model, batch, step, args):
        """Log generations using the current batch as sample data."""
        # Double-check: Only run on rank 0
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
            
        rank0_print(f"\n[SimpleGenerationLogger] ========== Logging generations at step {step} ==========")
        
        try:
            model.eval()
            
            vla_generation_logs = []
            json_generation_logs = []
            vla_accuracies = []
            json_accuracies = []
            
            # Process entire batch
            # Move batch to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_size = batch['input_ids'].shape[0]
            rank0_print(f"[SimpleGenerationLogger] Processing entire batch of size {batch_size}")
            
            num_images_so_far = 0
            num_videos_so_far = 0
            num_image_patches_so_far = 0
            num_video_patches_so_far = 0
            # Process each example in the batch
            for batch_idx in range(min(batch_size, self.num_examples)):
                # Extract single example from batch
                single_example = {}
                single_example['input_ids'] = batch['input_ids'][batch_idx]

                # Check if this is a VLA example (has action tokens) or JSON example
                input_ids = single_example['input_ids']
                is_vla_example = self.action_start_id in input_ids.tolist()

                single_example['labels'] = batch['labels'][batch_idx]
                input_text = self.tokenizer.decode(single_example['input_ids'])
                num_image_patches = input_text.count("<|image_pad|>") * 4 # 4 patches per token
                num_video_image_patches = input_text.count("<|video_pad|>") * 4 # 4 patches per token
                
                num_videos = int(num_video_image_patches > 0)
                
                if num_videos > 0:
                    single_example['pixel_values_videos'] = batch['pixel_values_videos'][num_video_patches_so_far:num_video_patches_so_far+num_video_image_patches]
                    single_example['video_grid_thw'] = batch['video_grid_thw'][num_video_patches_so_far:num_video_patches_so_far+num_videos]
                elif not is_vla_example:
                    single_example['pixel_values_videos'] = None
                    single_example['video_grid_thw'] = None

                num_images = input_text.count("<|vision_start|>") - num_videos

                if num_images > 0:
                    single_example['pixel_values'] = batch['pixel_values'][num_image_patches_so_far:num_image_patches_so_far+num_image_patches]
                    single_example['image_grid_thw'] = batch['image_grid_thw'][num_images_so_far:num_images_so_far+num_images]
                else:
                    single_example['pixel_values'] = None
                    single_example['image_grid_thw'] = None
                
                num_image_patches_so_far += num_image_patches
                num_video_patches_so_far += num_video_image_patches
                num_images_so_far += num_images
                num_videos_so_far += num_videos
                
                if is_vla_example:
                    vla_log = self._process_vla_example(model, single_example, step, batch_idx)
                    if vla_log:
                        vla_generation_logs.append(vla_log)
                        vla_accuracies.append(vla_log['is_correct'])
                else:
                    json_log = self._process_json_example(model, single_example, step, batch_idx)
                    if json_log:
                        json_generation_logs.append(json_log)
                        json_accuracies.append(json_log['is_correct'])
            
            # Calculate overall metrics
            vla_accuracy = np.mean(vla_accuracies) if vla_accuracies else 0.0
            json_accuracy = np.mean(json_accuracies) if json_accuracies else 0.0
            
            # Calculate L2 distance average
            l2_distances = [log['l2_distance'] for log in vla_generation_logs if log['l2_distance'] is not None]
            avg_l2_distance = np.mean(l2_distances) if l2_distances else None
            
            # Calculate OOV token statistics
            total_oov_tokens = sum(log['num_oov_tokens'] for log in vla_generation_logs)
            examples_with_oov = sum(1 for log in vla_generation_logs if log['num_oov_tokens'] > 0)
            avg_oov_per_example = total_oov_tokens / len(vla_generation_logs) if vla_generation_logs else 0.0
            
            # Log to wandb
            if self.log_to_wandb and wandb.run is not None:
                log_dict = {
                    'eval/vla_generation_accuracy': vla_accuracy,
                    'eval/vla_num_examples': len(vla_generation_logs),
                    'eval/json_generation_accuracy': json_accuracy,
                    'eval/json_num_examples': len(json_generation_logs),
                    'eval/vla_total_oov_tokens': total_oov_tokens,
                    'eval/vla_examples_with_oov': examples_with_oov,
                    'eval/vla_avg_oov_per_example': avg_oov_per_example,
                    'global_step': step,
                }
                
                # Add L2 distance if available
                if avg_l2_distance is not None:
                    log_dict['eval/vla_avg_l2_distance'] = avg_l2_distance
                
                wandb.log(log_dict)
            
            # Write to file
            self._write_to_file(vla_generation_logs, json_generation_logs, vla_accuracy, json_accuracy, step, args, total_oov_tokens, examples_with_oov, avg_oov_per_example, avg_l2_distance)
            
            rank0_print(f"[SimpleGenerationLogger] Processed {min(batch_size, self.num_examples)} examples from batch:")
            rank0_print(f"  VLA: {len(vla_generation_logs)} examples, Accuracy: {vla_accuracy:.4f}, OOV: {total_oov_tokens} total ({avg_oov_per_example:.2f} avg)")
            rank0_print(f"  JSON: {len(json_generation_logs)} examples, Accuracy: {json_accuracy:.4f}")
            
        except Exception as e:
            rank0_print(f"[SimpleGenerationLogger] Error during generation logging: {e}")
            import traceback
            traceback.print_exc()
        finally:
            model.train()  # Switch back to training mode
    
    def _process_vla_example(self, model, batch, step, batch_idx=0):
        """Process a VLA example and return log entry."""
        try:
            input_ids = batch['input_ids']
            labels = batch['labels']
            pixel_values = batch['pixel_values']
            image_grid_thw = batch['image_grid_thw']
            
            # Extract ground truth action tokens from labels
            gt_action_tokens = []
            for token_id in labels.tolist():
                if token_id != -100 and token_id in self.action_token_ids:
                    gt_action_tokens.append(token_id)
            
            # Find the position of action_start token in input_ids
            action_start_pos = None
            for i, token_id in enumerate(input_ids.tolist()):
                if token_id == self.action_start_id:
                    action_start_pos = i
                    break
            
            if action_start_pos is None:
                return None
            
            # Prepare inputs for generation (up to and including action_start)
            gen_input_ids = input_ids[:action_start_pos + 1].unsqueeze(0)
            
            # Check if we have valid image data
            if batch.get('pixel_values') is None or batch.get('image_grid_thw') is None:
                rank0_print(f"[SimpleGenerationLogger] Skipping VLA example - no image data in batch")
                return None
            
            # Check for empty tensors or invalid shapes
            if pixel_values.numel() == 0 or pixel_values.shape[0] == 0:
                rank0_print(f"[SimpleGenerationLogger] Skipping VLA example - empty pixel_values tensor")
                return None
            
            if image_grid_thw.numel() == 0 or image_grid_thw.shape[0] == 0:
                rank0_print(f"[SimpleGenerationLogger] Skipping VLA example - empty image_grid_thw tensor")
                return None
            
            # Generate action tokens
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=gen_input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=len(gt_action_tokens) + 50,  # Allow some extra tokens
                    do_sample=False,  # Use greedy decoding for deterministic results
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Extract generated action tokens
                generated_tokens = outputs[0][len(gen_input_ids[0]):].tolist()
                pred_action_tokens = []
                oov_tokens = []  # Track out-of-vocabulary tokens
                for token_id in generated_tokens:
                    if token_id == self.tokenizer.eos_token_id or token_id == self.action_end_id:
                        break
                    if token_id in self.action_token_ids:
                        pred_action_tokens.append(token_id)
                    elif token_id < len(self.tokenizer):  # Check if token is in vocabulary
                        # Token is in main vocabulary but not an action token
                        pass
                    else:
                        # Token is out of vocabulary
                        oov_tokens.append(token_id)
                
                # Calculate accuracy (exact match of action sequences)
                is_correct = pred_action_tokens == gt_action_tokens
                
                # Calculate L2 distance
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
                        rank0_print(f"[SimpleGenerationLogger] Error decoding actions: {e}")
                        l2_distance = None
                
                # Get input text (without action tokens)
                input_text = self.tokenizer.decode(input_ids[:action_start_pos].tolist(), skip_special_tokens=False)
                
                return {
                    'step': step,
                    'example_idx': batch_idx,
                    'input_text': input_text[-500:],  # Last 500 chars to avoid too long
                    'gt_action_tokens': gt_action_tokens,
                    'pred_action_tokens': pred_action_tokens,
                    'oov_tokens': oov_tokens,  # Out-of-vocabulary tokens
                    'num_oov_tokens': len(oov_tokens),  # Number of OOV tokens
                    'is_correct': is_correct,
                    'l2_distance': l2_distance,
                    'type': 'vla',
                }
        
        except Exception as e:
            rank0_print(f"[SimpleGenerationLogger] Error processing VLA example: {e}")
            return None
    
    def _process_json_example(self, model, batch, step, batch_idx=0):
        """Process a JSON example and return log entry."""
        try:
            input_ids = batch['input_ids']
            labels = batch['labels']
            pixel_values = batch['pixel_values']
            image_grid_thw = batch['image_grid_thw']
            pixel_values_videos = batch['pixel_values_videos']
            video_grid_thw = batch['video_grid_thw']

            # Find where the assistant response should start
            assistant_start_pos = None
            for i, label in enumerate(labels.tolist()):
                if label != -100:  # First non-ignored token is start of assistant response
                    assistant_start_pos = i
                    break
            
            if assistant_start_pos is None:
                return None
            
            # Prepare inputs for generation (up to assistant response)
            gen_input_ids = input_ids[:assistant_start_pos].unsqueeze(0)
            
            # Get ground truth text
            gt_token_ids = []
            for token_id in labels.tolist():
                if token_id != -100:
                    gt_token_ids.append(token_id)
            gt_text = self.tokenizer.decode(gt_token_ids, skip_special_tokens=True)

            
            # Generate text response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=gen_input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
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
                
                # Get input text
                input_text = self.tokenizer.decode(input_ids[:assistant_start_pos].tolist(), skip_special_tokens=False)
                
                return {
                    'step': step,
                    'example_idx': batch_idx,
                    'input_text': input_text[-500:],  # Last 500 chars to avoid too long
                    'gt_text': gt_text,
                    'pred_text': pred_text,
                    'is_correct': is_correct,
                    'type': 'json',
                }
        
        except Exception as e:
            rank0_print(f"[SimpleGenerationLogger] Error processing JSON example: {e}")
            return None
    
    def _write_to_file(self, vla_logs, json_logs, vla_accuracy, json_accuracy, step, args, total_oov_tokens=0, examples_with_oov=0, avg_oov_per_example=0.0, avg_l2_distance=None):
        """Write generation logs to file."""
        try:
            output_path = os.path.join(args.output_dir, self.log_file)
            with open(output_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Step: {step} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"VLA Examples: {len(vla_logs)} | JSON Examples: {len(json_logs)}\n")
                if vla_logs:
                    f.write(f"VLA Accuracy: {vla_accuracy:.4f}\n")
                    if avg_l2_distance is not None:
                        f.write(f"Average L2 Distance: {avg_l2_distance:.4f}\n")
                    f.write(f"OOV Tokens: {total_oov_tokens} total, {examples_with_oov} examples with OOV, {avg_oov_per_example:.2f} avg per example\n")
                if json_logs:
                    f.write(f"JSON Accuracy: {json_accuracy:.4f}\n")
                f.write(f"{'='*80}\n\n")
                
                # Write examples
                for log in vla_logs + json_logs:
                    f.write(f"Example ({log['type']}):\n")
                    f.write(f"Input (last 500 chars): ...{log['input_text']}\n")
                    if log['type'] == 'vla':
                        f.write(f"GT Action Tokens: {log['gt_action_tokens']}\n")
                        f.write(f"Pred Action Tokens: {log['pred_action_tokens']}\n")
                        f.write(f"OOV Tokens: {log['oov_tokens']} (count: {log['num_oov_tokens']})\n")
                        if log['l2_distance'] is not None:
                            f.write(f"L2 Distance: {log['l2_distance']:.4f}\n")
                    else:
                        f.write(f"GT Text: {log['gt_text']}\n")
                        f.write(f"Pred Text: {log['pred_text']}\n")
                    f.write(f"Correct: {log['is_correct']}\n")
                    f.write(f"{'-'*40}\n\n")
        except Exception as e:
            rank0_print(f"[SimpleGenerationLogger] Error writing to file: {e}")
