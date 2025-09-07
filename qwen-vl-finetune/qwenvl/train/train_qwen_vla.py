"""
Training script for Qwen VLA (Vision-Language-Action) model on DROID dataset.
Modified from train_qwen.py to support action prediction with fast tokenizer.
"""

import os
import logging
import pathlib
import torch
import torch.nn
import transformers
import json
from typing import Dict, List, Optional
import shutil
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import wandb
from accelerate import Accelerator

from torch.utils.data import DataLoader, IterableDataset
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available

try:
    import datasets
except ImportError:
    datasets = None

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments as HfTrainingArguments,
)
from qwenvl.data.data_droid import make_droid_data_module
from qwenvl.data.data_mixed_vla import make_mixed_vla_data_module
from qwenvl.data.data_fixed_mixed_vla import make_fixed_mixed_val_data_module
from qwenvl.data.data_proportional_mixed_vla import make_proportional_mixed_vla_data_module
from qwenvl.data.data_droid_iterable import make_droid_data_iterable_module
# from qwenvl.data.data_droid_iterable import make_droid_data_module_iterable

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.simple_generation_logger import SimpleGenerationLogger
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, TrainerCallback
# from qwenvl.train.trainer import EMATrainer  # TODO: Uncomment when implementing custom EMA
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class CheckpointProcessorCallback(TrainerCallback):
    """Callback to save image processor config with each checkpoint."""
    
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self._processor = None
    
    def get_processor(self):
        """Lazy load processor to avoid loading during import."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        return self._processor
    
    def on_save(self, args, state, control, **kwargs):
        """Save image processor config when checkpoint is saved."""
        if state.is_world_process_zero:  # Only save on main process
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                try:
                    processor = self.get_processor()
                    processor.image_processor.save_pretrained(checkpoint_dir)
                    rank0_print(f"Saved preprocessor_config.json to {checkpoint_dir}")
                except Exception as e:
                    rank0_print(f"Warning: Failed to save preprocessor config to {checkpoint_dir}: {e}")
        
        return control



# TODO: This callback is most likely buggy
class DeepSpeedOptimizerStateCheckCallback(TrainerCallback):
    """Callback to check if DeepSpeed optimizer state was loaded correctly."""

    def __init__(self, trainer_ref=None, checkpoint_path=None):
        self._checked = False
        self.trainer_ref = trainer_ref
        self.checkpoint_path = checkpoint_path

    def on_train_begin(self, args, state, control, **kwargs):
        """Check optimizer state after training initialization but before first step"""
        if not self._checked and state.is_world_process_zero:
            trainer = self.trainer_ref()  # Get trainer from reference
            if trainer is None:
                rank0_print("⚠️  DeepSpeed Check: Trainer instance not found, skipping optimizer check.")
                self._checked = True
                return
            
            # Note: Since we're using resume_from_checkpoint=True and the checkpoint is in output_dir,
            # the Trainer should have already loaded the checkpoint during initialization.
            # This callback is now mainly for verification.
            
            # Now check the optimizer state
            rank0_print("\n" + "="*20 + " DEEPSPEED OPTIMIZER STATE CHECK " + "="*20)
            has_deepspeed = hasattr(trainer, 'deepspeed') and trainer.deepspeed is not None
            rank0_print(f"  DeepSpeed Enabled: {has_deepspeed}")
            
            if has_deepspeed and hasattr(trainer.deepspeed, 'optimizer'):
                optimizer_state = trainer.deepspeed.optimizer.state_dict().get('state', {})
                if optimizer_state:
                    first_param_state = next(iter(optimizer_state.values()), {})
                    step = first_param_state.get('step', 0)
                    has_momentum = 'exp_avg' in first_param_state
                    rank0_print(f"  Optimizer Step Count: {step}")
                    rank0_print(f"  Optimizer Has Momentum: {has_momentum}")
                    if step > 0:
                        rank0_print("  ✅ Optimizer state appears loaded correctly.")
                    else:
                        rank0_print("  ⚠️  WARNING: Optimizer step is 0. State may not be loaded.")
                else:
                    rank0_print("  ❌ ERROR: DeepSpeed optimizer state is empty!")
                    rank0_print("  ℹ️  This usually means the checkpoint wasn't loaded properly during trainer.train()")
                    rank0_print("  ℹ️  DeepSpeed should handle optimizer state loading automatically when resuming from checkpoint.")
                    
                    # Just verify the checkpoint structure exists
                    if self.checkpoint_path:
                        checkpoint_path = self.checkpoint_path
                        step_num = checkpoint_path.split('-')[-1]
                        
                        # Check for the standard DeepSpeed checkpoint structure
                        deepspeed_checkpoint_path = os.path.join(checkpoint_path, f"global_step{step_num}")
                        
                        rank0_print(f"  📁 Checking checkpoint structure at: {checkpoint_path}")
                        rank0_print(f"  📁 DeepSpeed path should be: {deepspeed_checkpoint_path}")
                        rank0_print(f"  📁 Path exists: {os.path.exists(deepspeed_checkpoint_path)}")
                            
                        if os.path.exists(deepspeed_checkpoint_path):
                            # Verify latest file exists (required by DeepSpeed)
                            latest_path = os.path.join(checkpoint_path, "latest")
                            if not os.path.exists(latest_path):
                                try:
                                    with open(latest_path, 'w') as f:
                                        f.write(f"global_step{step_num}")
                                    rank0_print(f"  ✅ Created missing 'latest' file pointing to global_step{step_num}")
                                except Exception as e:
                                    rank0_print(f"  ⚠️  Could not create 'latest' file: {e}")
                            else:
                                with open(latest_path, 'r') as f:
                                    content = f.read().strip()
                                    rank0_print(f"  ✅ 'latest' file exists and points to: {content}")
                            
                            # List checkpoint contents for debugging
                            try:
                                files = sorted(os.listdir(deepspeed_checkpoint_path))
                                rank0_print(f"  📁 Checkpoint contains: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
                            except Exception as e:
                                rank0_print(f"  ⚠️  Could not list checkpoint contents: {e}")
                            
                            # The optimizer state is empty even though the checkpoint exists
                            # This is a known issue with DeepSpeed + HuggingFace Trainer integration
                            rank0_print("\n  💡 POTENTIAL ISSUES:")
                            rank0_print("     1. The explicit DeepSpeed checkpoint loading before trainer.train() should have loaded it")
                            rank0_print("     2. If that failed, check the logs above for the 'LOADING DEEPSPEED CHECKPOINT' section")
                            rank0_print("     3. The optimizer state might be loaded but not visible in state_dict() yet")
                            rank0_print("\n  ℹ️  The checkpoint structure looks correct. The issue is with the loading process.")
                        else:
                            rank0_print(f"  ❌ DeepSpeed checkpoint structure NOT found at expected path: {deepspeed_checkpoint_path}")
                            rank0_print(f"  ℹ️  This checkpoint may not be a valid DeepSpeed checkpoint.")
                            
            elif has_deepspeed:
                rank0_print("  ❌ ERROR: DeepSpeed is enabled, but no optimizer was found.")
            rank0_print("="*61 + "\n")
            self._checked = True


import random

class BatchShuffleWrapper:
    """Wraps an iterator to shuffle yielded batches in a buffer."""
    def __init__(self, dataloader, buffer_size: int = 100):
        self.dataloader = dataloader
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for batch in self.dataloader:
            if len(buffer) < self.buffer_size:
                buffer.append(batch)
            else:
                # When buffer is full, yield a random item and replace it
                idx_to_yield = random.randint(0, self.buffer_size - 1)
                yield buffer[idx_to_yield]
                buffer[idx_to_yield] = batch
        
        # Yield all remaining items in the buffer
        random.shuffle(buffer)
        for batch in buffer:
            yield batch


class VLATrainer(Trainer):
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        Override the inner training loop to prevent skip_first_batches from being called
        on iterable datasets during checkpoint resumption.
        """
        # Check if we have an iterable dataset and are resuming from checkpoint
        if (isinstance(self.train_dataset, IterableDataset) and 
            resume_from_checkpoint is not None):
            
            # Temporarily set ignore_data_skip to True to prevent skip_first_batches
            original_ignore_data_skip = args.ignore_data_skip if args else self.args.ignore_data_skip
            if args:
                args.ignore_data_skip = True
            else:
                self.args.ignore_data_skip = True
            
            rank0_print("🚀 Preventing skip_first_batches for iterable dataset - samples_to_skip was already applied during dataset creation")
            
            try:
                # Call the parent implementation with ignore_data_skip=True
                result = super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
            finally:
                # Restore original setting
                if args:
                    args.ignore_data_skip = original_ignore_data_skip
                else:
                    self.args.ignore_data_skip = original_ignore_data_skip
            
            return result
        else:
            # For non-iterable datasets or non-resuming cases, use default behavior
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the default train dataloader to handle efficient checkpoint resumption
        for iterable datasets using samples_to_skip instead of skip_first_batches.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        


        # --- Standard setup ---
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # We know our dataset is iterable, so we don't need a sampler
        # For TensorFlow-based datasets, reduce num_workers to avoid overhead
        num_workers = self.args.dataloader_num_workers
        if hasattr(train_dataset, '__class__') and 'Droid' in train_dataset.__class__.__name__:
            # Limit workers for TensorFlow datasets to avoid memory overhead
            num_workers = min(num_workers, 4)  # Cap at 4 workers
            if num_workers != self.args.dataloader_num_workers:
                rank0_print(f"🔧 Reducing num_workers from {self.args.dataloader_num_workers} to {num_workers} for TensorFlow dataset")
        
        dataloader_params = {
            "batch_size": None if hasattr(train_dataset, '__class__') and 'Droid' in train_dataset.__class__.__name__ else self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # 1. Create the standard DataLoader
        
        dataloader = DataLoader(train_dataset, **dataloader_params)

        # # # # 2. Apply our custom batch shuffling wrapper
        # # # shuffled_dataloader = BatchShuffleWrapper(dataloader, buffer_size=128)
        if isinstance(train_dataset, IterableDataset):
            # For IterableDatasets, return the raw DataLoader.
            # Our `_prepare_inputs` override will handle device placement.
            return dataloader

        # 3. **Crucially, ALWAYS pass the final dataloader to the accelerator.**
        # This ensures correct data sharding across all GPUs.
        return self.accelerator.prepare(dataloader)
    



    def __init__(self, *args, train_sampler_params=None, generation_logger=None, generation_interval=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler_params = train_sampler_params
        # Initialize metrics for separate loss tracking
        self._droid_loss_sum = 0.0
        self._droid_loss_count = 0
        self._json_loss_sum = 0.0
        self._json_loss_count = 0
        self._log_interval = self.args.logging_steps  # Sync with Trainer's logging steps
        
        # Initialize action accuracy tracking
        self._action_accuracy_sum = 0.0
        self._action_accuracy_count = 0
        
        # Generation logging setup
        self.generation_logger = generation_logger
        self.generation_interval = generation_interval
        self._last_generation_step = -1
    
    
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to track separate losses for droid and JSON data.
        Also compute action token accuracy for VLA training.
        
        We identify data types by checking for action tokens in the input_ids.
        """
        # Get the base loss from parent class
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # Check if we have action tokens to identify droid data and compute accuracy
        if hasattr(self, 'action_start_id') and self.action_start_id is not None:
            # Get input_ids and labels from the batch
            input_ids = inputs.get("input_ids", None)
            labels = inputs.get("labels", None)
            
            if input_ids is not None and labels is not None:
                # Check each sample in the batch
                batch_size = input_ids.shape[0]
                
                # Get logits for action token accuracy computation
                logits = outputs.logits
                
                # Compute action token accuracy for droid samples
                action_accuracy_sum = 0.0
                action_accuracy_count = 0
                
                # Compute per-sample loss if not already done
                if batch_size > 1:
                    # We need to compute per-sample loss to track separately
                    if labels is not None:
                        # Get logits and shift for loss computation
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        # Compute per-sample loss
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                        per_token_loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        
                        # Reshape back to [batch_size, seq_len-1]
                        per_token_loss = per_token_loss.view(batch_size, -1)
                        
                        # Average over valid tokens for each sample
                        valid_tokens = (shift_labels != -100).float()
                        per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1)
                        
                        # Track losses and accuracy based on data type
                        for i in range(batch_size):
                            sample_loss = per_sample_loss[i].item()
                            # Check if this sample has action tokens (droid data)
                            has_action_tokens = (input_ids[i] == self.action_start_id).any().item()
                            
                            if has_action_tokens:
                                self._droid_loss_sum += sample_loss
                                self._droid_loss_count += 1
                                
                                # Compute action token accuracy for this sample
                                sample_accuracy = self._compute_action_accuracy_for_sample(
                                    logits[i:i+1], labels[i:i+1]
                                )
                                if sample_accuracy is not None:
                                    action_accuracy_sum += sample_accuracy
                                    action_accuracy_count += 1
                            else:
                                self._json_loss_sum += sample_loss
                                self._json_loss_count += 1
                else:
                    # Single sample in batch
                    loss_value = loss.item() if torch.is_tensor(loss) else loss
                    has_action_tokens = (input_ids[0] == self.action_start_id).any().item()
                    
                    if has_action_tokens:
                        self._droid_loss_sum += loss_value
                        self._droid_loss_count += 1
                        
                        # Compute action token accuracy for this sample
                        sample_accuracy = self._compute_action_accuracy_for_sample(logits, labels)
                        if sample_accuracy is not None:
                            action_accuracy_sum += sample_accuracy
                            action_accuracy_count += 1
                    else:
                        self._json_loss_sum += loss_value
                        self._json_loss_count += 1
                
                # Track action accuracy
                if action_accuracy_count > 0:
                    avg_action_accuracy = action_accuracy_sum / action_accuracy_count
                    if not hasattr(self, '_action_accuracy_sum'):
                        self._action_accuracy_sum = 0.0
                        self._action_accuracy_count = 0
                    self._action_accuracy_sum += avg_action_accuracy
                    self._action_accuracy_count += 1
        
        # Log separate losses periodically
        if self.state.global_step > 0 and self.state.global_step % self._log_interval == 0:
            self._log_separate_losses()
        
        # Run generation logging at specified intervals (only on rank 0)
        if (self.generation_logger is not None and 
            self.state.global_step > 0 and 
            self.state.global_step % self.generation_interval == 0 and
            self.state.global_step != self._last_generation_step and
            self.args.local_rank in [-1, 0]):  # Only run on rank 0
            
            self._last_generation_step = self.state.global_step
            self._run_generation_logging(inputs)
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_action_accuracy_for_sample(self, logits, labels):
        """
        Compute action token accuracy for a single sample.
        Based on the code provided by the user.
        """
        try:
            # Get action token IDs range from token mappings
            if not hasattr(self, 'action_token_begin_idx'):
                if hasattr(self, 'token_mappings') and 'action_token_ids' in self.token_mappings:
                    # Use actual token mappings
                    self.action_token_begin_idx = min(self.token_mappings['action_token_ids'])
                    self.action_token_end_idx = max(self.token_mappings['action_token_ids'])
                else:
                    # Fallback to default range
                    action_token_start = 148256  # Default action token start
                    action_vocab_size = 1024    # Default action vocab size
                    self.action_token_begin_idx = action_token_start
                    self.action_token_end_idx = action_token_start + action_vocab_size - 1
            
            # Get predictions from logits (equivalent to: action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2))
            # For Qwen VLA, we need to identify where action tokens are in the sequence
            # We'll use the full sequence and filter by action token range
            action_preds = logits[:, :-1].argmax(dim=2)  # [batch, seq_len-1]
            action_gt = labels[:, 1:]  # [batch, seq_len-1]
            
            # Create mask for action tokens (equivalent to: mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx))
            mask = (action_gt >= self.action_token_begin_idx) & (action_gt <= self.action_token_end_idx) & (action_gt != -100)
            
            if mask.sum() == 0:
                # No action tokens in this sample
                return None
            
            # Compute accuracy (equivalent to: correct_preds = (action_preds == action_gt) & mask; action_accuracy = correct_preds.sum().float() / mask.sum().float())
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()
            
            return action_accuracy.item()
            
        except Exception as e:
            # If there's any error in accuracy computation, don't crash training
            print(f"Warning: Error computing action accuracy: {e}")
            return None
    
    def _run_generation_logging(self, current_batch):
        """Run generation logging using the current batch as a sample. Only called on rank 0."""
        try:
            rank0_print(f"\n[GenerationLogger] Running generation logging at step {self.state.global_step}")
            
            # Call the generation logger with the current batch
            self.generation_logger.log_generations_from_batch(
                model=self.model,
                batch=current_batch,
                step=self.state.global_step,
                args=self.args
            )
            
        except Exception as e:
            rank0_print(f"[GenerationLogger] Error during generation logging: {e}")
            import traceback
            traceback.print_exc()

    def _log_separate_losses(self):
        """Log separate losses and action accuracy to wandb and console."""
        metrics = {}
        
        # Calculate average droid loss
        if self._droid_loss_count > 0:
            avg_droid_loss = self._droid_loss_sum / self._droid_loss_count
            metrics["train/droid_loss"] = avg_droid_loss
            # Reset counters
            self._droid_loss_sum = 0.0
            self._droid_loss_count = 0
        
        # Calculate average JSON loss
        if self._json_loss_count > 0:
            avg_json_loss = self._json_loss_sum / self._json_loss_count
            metrics["train/json_loss"] = avg_json_loss
            # Reset counters
            self._json_loss_sum = 0.0
            self._json_loss_count = 0
        
        # Calculate average action accuracy
        if hasattr(self, '_action_accuracy_count') and self._action_accuracy_count > 0:
            avg_action_accuracy = self._action_accuracy_sum / self._action_accuracy_count
            metrics["train/action_accuracy"] = avg_action_accuracy
            # Reset counters
            self._action_accuracy_sum = 0.0
            self._action_accuracy_count = 0
        
        # Log to wandb if available (only on rank 0)
        if len(metrics) > 0 and self.args.local_rank in [-1, 0]:
            self.log(metrics)
            
            # Also print to console
            log_str = f"Step {self.state.global_step}: "
            if "train/droid_loss" in metrics:
                log_str += f"droid_loss={metrics['train/droid_loss']:.4f} "
            if "train/json_loss" in metrics:
                log_str += f"json_loss={metrics['train/json_loss']:.4f} "
            if "train/action_accuracy" in metrics:
                log_str += f"action_accuracy={metrics['train/action_accuracy']:.4f}"
            rank0_print(log_str)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log any remaining losses and action accuracy at the end of training."""
        if (self._droid_loss_count > 0 or self._json_loss_count > 0 or 
            (hasattr(self, '_action_accuracy_count') and self._action_accuracy_count > 0)):
            self._log_separate_losses()
        return super().on_train_end(args, state, control, **kwargs)


@dataclass
class VLAModelArguments(ModelArguments):
    """Extended model arguments for VLA training."""
    action_tokenizer_path: str = field(default="KarlP/fast-droid", metadata={"help": "Path to action tokenizer"})


@dataclass 
class VLADataArguments(DataArguments):
    """Extended data arguments for DROID dataset."""
    droid_data_dir: str = field(default="/iliad2/u/ajaysri/episodic_memory/droid_rlds")
    droid_dataset_name: str = field(default="droid_100")
    action_chunk_size: int = field(default=15)
    use_joint_velocity: bool = field(default=True)
    shuffle_buffer_size: int = field(default=100000)
    num_droid_samples: int = field(default=20000000)
    image_height: int = field(default=180, metadata={"help": "Height to resize images to"})
    image_width: int = field(default=320, metadata={"help": "Width to resize images to"})
    
    # Image resize constraints for JSON data
    max_image_dim: int = field(default=320, metadata={"help": "Maximum dimension (width or height) for resizing images"})
    min_image_dim: int = field(default=28, metadata={"help": "BOTH width AND height must be STRICTLY GREATER than this value (required by Qwen2VL processor)"})
    
    # Co-training with regular JSON data to prevent catastrophic forgetting
    enable_cotrain: bool = field(default=False, metadata={"help": "Enable co-training with regular JSON data"})
    cotrain_json_paths: str = field(default="", metadata={"help": "Comma-separated paths to JSON/JSONL files for co-training"})
    cotrain_json_ratio: float = field(default=0.2, metadata={"help": "Ratio of regular JSON data in mixed training (0.0-1.0)"})
    use_fixed_ratio_sampler: bool = field(default=True, metadata={"help": "Use fixed ratio sampler to ensure consistent memory usage per batch"})
    pixel_budget: int = field(default=230400, metadata={"help": "Max pixels per JSON query (default: 230400 = 4x VLA size). For multi-frame JSON, budget applies to total pixels across all frames."})
    
    # Dataset type selection
    dataset_type: str = field(default="proportional", metadata={"help": "Dataset mixing type: 'proportional' (probabilistic sampling) or 'fixed' (fixed ratio per batch with pre-collation)"})
    
    # Individual dataset weighting options
    cotrain_json_weights: str = field(default="", metadata={"help": "Comma-separated weights for individual JSON datasets (must match number of JSON paths). If empty, uses equal weights."})
    weight_by_count: bool = field(default=False, metadata={"help": "Weight all datasets (including droid) by their sample count instead of using fixed ratios"})
    count_weight_power: float = field(default=1.0, metadata={"help": "Power to raise dataset counts to when using count-based weighting (e.g., 0.5 for square root, 1.0 for linear)"})


@dataclass
class VLATrainingArguments(TrainingArguments):
    """Extended training arguments with evaluation settings for VLA."""
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to adopt during training."}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Interval for generation logging (callback will trigger every X steps)."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    max_eval_samples: int = field(
        default=100,
        metadata={"help": "Maximum number of evaluation samples to use. The trainer will automatically limit the eval dataset to this size."}
    )
    num_generation_examples: int = field(
        default=10,
        metadata={"help": "Number of examples to generate during evaluation for logging."}
    )
    log_generations_to_wandb: bool = field(
        default=True,
        metadata={"help": "Whether to log generation examples to wandb."}
    )
    eval_on_start: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation at the beginning of training (step 0)."}
    )
    generation_interval: int = field(
        default=500,
        metadata={"help": "Interval (in training steps) for running generation logging. If not specified, defaults to eval_steps."}
    )
    skip_samples: int = field(
        default=None,
        metadata={"help": "Number of samples to skip for checkpoint resumption. Use this instead of automatic calculation for faster startup."}
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False},
        metadata={"help": "Keyword arguments for gradient checkpointing, e.g., `{'use_reentrant': False}` for compatibility with Deepspeed."}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "When using distributed training, you may need to set this to True if you use gradient checkpointing."}
    )


# Note: We use the base Qwen2_5_VLForConditionalGeneration directly
# since we're only remapping existing infrequent tokens, not modifying the model architecture


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """Set which parts of the model to train."""
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def get_action_state_token_mappings(tokenizer, action_vocab_size=1024, state_vocab_size=256, output_dir=None):
    """Get token ID mappings using 3000+ rare Unicode symbols."""
    
    # Use rare Unicode symbols range (148000-151000 = 3000 tokens)
    base_start = 148000
    
    # Log base_start value to file in run directory
    if output_dir:
        import os
        from datetime import datetime
        log_file = os.path.join(output_dir, "token_mapping_log.txt")
        with open(log_file, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] base_start value: {base_start}\n")
    
    state_token_start = base_start                           # 148000
    action_token_start = state_token_start + state_vocab_size # 148256  
    control_token_start = action_token_start + action_vocab_size # 149280
    
    # Verify we have enough tokens
    total_needed = state_vocab_size + action_vocab_size + 4  # +4 for control tokens
    max_token_id = control_token_start + 4                   # 149284
    
    if max_token_id > 151000:  # Our safe range limit
        raise ValueError(f"Need {total_needed} tokens but safe range only has {151000-base_start}")
    
    # Create mappings
    state_token_ids = list(range(state_token_start, state_token_start + state_vocab_size))
    action_token_ids = list(range(action_token_start, action_token_start + action_vocab_size))
    
    control_mappings = {
        "<|action_start|>": control_token_start,
        "<|action_end|>": control_token_start + 1, 
        "<|state_start|>": control_token_start + 2,
        "<|state_end|>": control_token_start + 3,
    }
    
    return {
        "state_token_ids": state_token_ids,
        "action_token_ids": action_token_ids, 
        "control_mappings": control_mappings,
        "action_start_id": control_mappings["<|action_start|>"],
        "action_end_id": control_mappings["<|action_end|>"],
        "state_start_id": control_mappings["<|state_start|>"],
        "state_end_id": control_mappings["<|state_end|>"],
    }


def get_sorted_checkpoints(output_dir):
    """Get checkpoints sorted from latest to oldest."""
    checkpoint_dirs = list(pathlib.Path(output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        regular_checkpoints = [d for d in checkpoint_dirs if not d.name.endswith('_fixed')]
        if regular_checkpoints:
            # Sort by checkpoint step number (latest first)
            sorted_checkpoints = sorted(regular_checkpoints, key=lambda x: int(x.name.split('-')[1]), reverse=True)
            return [str(cp) for cp in sorted_checkpoints]
    return []


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (VLAModelArguments, VLADataArguments, VLATrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Determine available checkpoints for resuming
    available_checkpoints = get_sorted_checkpoints(training_args.output_dir)
    checkpoint_to_resume = None
    
    
    if available_checkpoints:
        latest_checkpoint = available_checkpoints[0]
        rank0_print(f"Found {len(available_checkpoints)} checkpoint(s). Latest: {latest_checkpoint}")
        if len(available_checkpoints) > 1:
            rank0_print(f"Second latest checkpoint available: {available_checkpoints[1]}")
        # Load model from the latest checkpoint initially - if it fails during training,
        # we'll try other checkpoints during the training phase
        checkpoint_to_resume = latest_checkpoint

    # Load model with checkpoint fallback
    # Check original model name for 'qwen2.5' as checkpoint path might not contain it
    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = None
        image_processor = None
        
        # Try loading model from available checkpoints, fallback to base model
        model_load_paths_to_try = []
        if available_checkpoints:
            model_load_paths_to_try.extend(available_checkpoints)
        model_load_paths_to_try.append(model_args.model_name_or_path)  # Fallback to base model
        
        for i, load_path in enumerate(model_load_paths_to_try):
            try:
                if i == 0 and available_checkpoints:
                    rank0_print(f"Loading model from latest checkpoint: {load_path}")
                elif i > 0 and i < len(available_checkpoints):
                    rank0_print(f"Loading model from checkpoint #{i+1}: {load_path}")
                else:
                    rank0_print(f"Loading base model: {load_path}")
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    load_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                )
                image_processor = AutoProcessor.from_pretrained(
                    load_path, trust_remote_code=True
                ).image_processor
                
                # Update the actual model load path used and checkpoint info
                if load_path in available_checkpoints:
                    checkpoint_to_resume = load_path
                else:
                    # We loaded from base model, no checkpoint to resume
                    checkpoint_to_resume = None
                    available_checkpoints = []  # Clear since we're starting fresh
                
                rank0_print(f"Successfully loaded model from: {load_path}")
                break
                
            except Exception as e:
                rank0_print(f"Failed to load model from {load_path}: {e}")
                if i < len(model_load_paths_to_try) - 1:
                    rank0_print("Trying next checkpoint...")
                    continue
                else:
                    rank0_print("All model loading attempts failed")
                    raise e
        
        if model is None or image_processor is None:
            raise RuntimeError("Failed to load model from any checkpoint or base model")
            
        data_args.image_processor = image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        raise NotImplementedError("Only Qwen2.5-VL is supported for VLA training")

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load tokenizer and add action tokens
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Load action tokenizer to get its vocabulary size
    rank0_print(f"Loading action tokenizer from {model_args.action_tokenizer_path}")
    action_tokenizer = AutoProcessor.from_pretrained(
        model_args.action_tokenizer_path, 
        trust_remote_code=True
    )
    action_vocab_size = action_tokenizer.vocab_size
    rank0_print(f"Action tokenizer vocab size: {action_vocab_size}")
    
    # Get action and state token mappings using actual tokenizer vocab size
    token_mappings = get_action_state_token_mappings(tokenizer, action_vocab_size, output_dir=training_args.output_dir)
    
    rank0_print(f"Token mappings - all existing infrequent tokens:")
    rank0_print(f"  State: {token_mappings['state_token_ids'][0]}-{token_mappings['state_token_ids'][-1]}")
    rank0_print(f"  Action: {token_mappings['action_token_ids'][0]}-{token_mappings['action_token_ids'][-1]}")
    rank0_print(f"  Control: {token_mappings['control_mappings']}")
    
    # Set model trainable parameters
    set_model(model_args, model)

    if local_rank == 0:
        rank0_print("Model architecture:")
        rank0_print(f"Vision encoder trainable: {model_args.tune_mm_vision}")
        rank0_print(f"Vision-language connector trainable: {model_args.tune_mm_mlp}")
        rank0_print(f"Language model trainable: {model_args.tune_mm_llm}")
        rank0_print(f"Action vocabulary size: {action_vocab_size}")
    


    # Use skip_samples from training arguments (user can set this manually)
    # If resuming and skip_samples is 0, we'll use step-based seeding instead of skipping
    checkpoint_step = 0
    if checkpoint_to_resume:
        checkpoint_step = int(pathlib.Path(checkpoint_to_resume).name.split('-')[1])
        rank0_print(f"Resuming from checkpoint step {checkpoint_step}")
        if training_args.skip_samples is None:
            training_args.skip_samples = checkpoint_step * training_args.gradient_accumulation_steps * training_args.world_size * training_args.per_device_train_batch_size
        rank0_print(f"Will skip {training_args.skip_samples:,} samples as specified")       

    if training_args.skip_samples is None:
        training_args.skip_samples = 0
    # Store dataset creation arguments
    dataset_creation_args = {
        'tokenizer': tokenizer,
        'action_tokenizer': action_tokenizer,
        'data_args': data_args,
        'model_max_length': training_args.model_max_length,
        'token_mappings': token_mappings,
        'image_size': (data_args.image_height, data_args.image_width),
        'cotrain_json_ratio': data_args.cotrain_json_ratio,
        'samples_to_skip': training_args.skip_samples,  # Use explicit argument
        'seed': 42 + checkpoint_step,  # Use step-based seeding for randomness
        'batch_size': training_args.per_device_train_batch_size,  # Pass correct batch size to dataset
    }
    
    # Choose dataset type based on data_args.dataset_type
    if data_args.dataset_type == "fixed":
        raise NotImplementedError("Fixed ratio mixed dataset with pre-collation is not supported")
        # rank0_print("Using fixed ratio mixed dataset with pre-collation")
        # data_module = make_fixed_mixed_val_data_module(**dataset_creation_args)
    else:
        if data_args.cotrain_json_ratio > 0:
            rank0_print("🦄 Using proportional mixed dataset with probabilistic sampling")
            data_module = make_proportional_mixed_vla_data_module(**dataset_creation_args)
        else:
            rank0_print("🤖 Using VLA-only dataset")
            data_module = make_droid_data_iterable_module(**dataset_creation_args)
    
    # Create simplified generation logger for use in compute_loss
    # The action tokenizer will be initialized through normal data loading
    # call on dummy data to warmup the tokenizer
    action_data = np.random.rand(10, data_args.action_chunk_size, 8)    # one batch of action chunks
    _ = action_tokenizer(action_data)
    
    rank0_print(f"Creating SimpleGenerationLogger")
    rank0_print(f"Generation logging interval: {training_args.generation_interval} steps")
    generation_logger = SimpleGenerationLogger(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        token_mappings=token_mappings,
        num_examples=training_args.num_generation_examples,  # Use configurable number, will process up to batch_size
        log_file="generations.txt",
        log_to_wandb=training_args.log_generations_to_wandb,
    )
    
    # Create checkpoint processor callback to save preprocessor config with each checkpoint
    checkpoint_processor_callback = CheckpointProcessorCallback(model_args.model_name_or_path)
    
    # Extract sampler params if present
    train_sampler_params = data_module.pop('train_sampler_params', None)
    
    # Ensure gradient accumulation consistency before creating trainer
    rank0_print(f"Training args gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    
    # Force gradient accumulation steps to be consistent
    if hasattr(training_args, 'gradient_accumulation_steps') and training_args.gradient_accumulation_steps != 2:
        rank0_print(f"WARNING: Forcing gradient_accumulation_steps from {training_args.gradient_accumulation_steps} to 2")
        training_args.gradient_accumulation_steps = 2
    
    # For DeepSpeed, the checkpoint path should be set in training_args
    # But we'll follow the simpler approach from train_qwen.py
    
    # Initialize trainer with simplified generation logger
    
    trainer = VLATrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        callbacks=[checkpoint_processor_callback],  # Start with checkpoint processor only
        train_sampler_params=train_sampler_params,  # Pass sampler params to custom trainer
        generation_logger=generation_logger,  # Pass generation logger to trainer
        generation_interval=training_args.generation_interval,  # Use configurable generation interval
        **data_module
    )
    
    # Add the optimizer state check callback after trainer creation
    # Only add the callback if we're resuming from a checkpoint
    if checkpoint_to_resume:
        optimizer_check_callback = DeepSpeedOptimizerStateCheckCallback(
            trainer_ref=lambda: trainer,
            checkpoint_path=checkpoint_to_resume
        )
        trainer.add_callback(optimizer_check_callback)
    

    
    # Verify final gradient accumulation configuration
    rank0_print(f"Final trainer gradient_accumulation_steps: {trainer.args.gradient_accumulation_steps}")
    if hasattr(trainer, 'accelerator') and hasattr(trainer.accelerator, 'gradient_accumulation_steps'):
        rank0_print(f"Accelerator gradient_accumulation_steps: {trainer.accelerator.gradient_accumulation_steps}")
        
        # Force accelerator to use the same value as trainer
        if trainer.accelerator.gradient_accumulation_steps != trainer.args.gradient_accumulation_steps:
            rank0_print(f"🔧 FIXING: Setting accelerator gradient_accumulation_steps from {trainer.accelerator.gradient_accumulation_steps} to {trainer.args.gradient_accumulation_steps}")
            trainer.accelerator.gradient_accumulation_steps = trainer.args.gradient_accumulation_steps
            rank0_print(f"✅ Fixed accelerator gradient_accumulation_steps: {trainer.accelerator.gradient_accumulation_steps}")
    
    # Set action_start_id and token mappings on trainer for loss separation and accuracy computation
    trainer.action_start_id = token_mappings['action_start_id']
    trainer.token_mappings = token_mappings
    
    # TODO: Switch to EMATrainer when implementing custom EMA support
    # trainer_class = EMATrainer if getattr(training_args, 'use_ema', False) else Trainer
    # trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args, **data_module)

    # Start training - use the simple approach since checkpoints are in output_dir
    if checkpoint_to_resume:
        rank0_print("="*20 + " CHECKPOINT RESUMPTION " + "="*20)
        rank0_print(f"Found checkpoint in output directory: {checkpoint_to_resume}")
        rank0_print("Using resume_from_checkpoint=True to let Trainer find the latest checkpoint")
        
        try:
            # Use the simple approach from train_qwen.py since checkpoint is in output_dir
            trainer.train(resume_from_checkpoint=True)
            
        except Exception as e:
            rank0_print(f"Failed to resume from {checkpoint_to_resume}: {e}")
            
            # Handle known DeepSpeed world size issues with a warm restart
            if ("world size" in str(e).lower() or 
                "dp world size" in str(e).lower() or
                "partition" in str(e).lower()):
                
                rank0_print("\n" + "="*20 + " WARM RESTART " + "="*20)
                rank0_print(f"DeepSpeed world size/partition mismatch detected: {e}")
                rank0_print("Fallback: Starting training with loaded model weights but a fresh optimizer state.")
                rank0_print("WARNING: Optimizer state NOT loaded - this may temporarily increase loss.")
                rank0_print("To resume optimizer state, use the same GPU count as the original run.")
                
                # Model weights are already loaded from the checkpoint, so we can start training
                trainer.train()
                
            else:
                # Try fallback checkpoints if available (only for training state, model is already loaded)
                if len(available_checkpoints) > 1:
                    rank0_print(f"Attempting fallback to other checkpoints for training state...")
                    
                    fallback_successful = False
                    for i, fallback_checkpoint in enumerate(available_checkpoints[1:], 1):
                        rank0_print(f"Trying checkpoint #{i+1}: {fallback_checkpoint}")
                        try:
                            trainer.train(resume_from_checkpoint=fallback_checkpoint)
                            fallback_successful = True
                            break
                        except Exception as fallback_e:
                            rank0_print(f"Fallback checkpoint {fallback_checkpoint} also failed: {fallback_e}")
                            continue
                    
                    if not fallback_successful:
                        rank0_print("All checkpoint resumption attempts failed. Starting fresh training with loaded model weights.")
                        trainer.train()
                else:
                    rank0_print("No fallback checkpoints available. Starting fresh training with loaded model weights.")
                    trainer.train()
            
    else:
        rank0_print("No checkpoint found - starting fresh training")
        trainer.train()
    
    # Save final model
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    
    # Save action tokenizer info
    action_tokenizer_info = {
        "action_tokenizer_path": model_args.action_tokenizer_path,
        "action_vocab_size": action_vocab_size,
        "state_token_range": [token_mappings['state_token_ids'][0], token_mappings['state_token_ids'][-1]],
        "action_token_range": [token_mappings['action_token_ids'][0], token_mappings['action_token_ids'][-1]],
        "control_mappings": token_mappings['control_mappings'],
        "token_mappings": token_mappings,
    }
    with open(os.path.join(training_args.output_dir, "action_tokenizer_info.json"), "w") as f:
        json.dump(action_tokenizer_info, f, indent=2)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
