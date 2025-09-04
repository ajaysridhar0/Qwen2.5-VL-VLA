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
from qwenvl.data.data_proportional_mixed_vla import make_proportional_mixed_val_data_module
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
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
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
        
        # Generation logging setup
        self.generation_logger = generation_logger
        self.generation_interval = generation_interval
        self._last_generation_step = -1
    
    
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Override compute_loss to track separate losses for droid and JSON data.
        
        We identify data types by checking for action tokens in the input_ids.
        """
        # Get the base loss from parent class
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        
        # Check if we have action tokens to identify droid data
        if hasattr(self, 'action_start_id') and self.action_start_id is not None:
            # Get input_ids from the batch
            input_ids = inputs.get("input_ids", None)
            
            if input_ids is not None:
                # Check each sample in the batch
                batch_size = input_ids.shape[0]
                
                # Compute per-sample loss if not already done
                if batch_size > 1:
                    # We need to compute per-sample loss to track separately
                    labels = inputs.get("labels", None)
                    if labels is not None:
                        # Get logits and shift for loss computation
                        logits = outputs.logits
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
                        
                        # Track losses based on data type
                        for i in range(batch_size):
                            sample_loss = per_sample_loss[i].item()
                            # Check if this sample has action tokens (droid data)
                            has_action_tokens = (input_ids[i] == self.action_start_id).any().item()
                            
                            if has_action_tokens:
                                self._droid_loss_sum += sample_loss
                                self._droid_loss_count += 1
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
                    else:
                        self._json_loss_sum += loss_value
                        self._json_loss_count += 1
        
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
        """Log separate losses to wandb and console."""
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
        
        # Log to wandb if available (only on rank 0)
        if len(metrics) > 0 and self.args.local_rank in [-1, 0]:
            self.log(metrics)
            
            # Also print to console
            log_str = f"Step {self.state.global_step}: "
            if "train/droid_loss" in metrics:
                log_str += f"droid_loss={metrics['train/droid_loss']:.4f} "
            if "train/json_loss" in metrics:
                log_str += f"json_loss={metrics['train/json_loss']:.4f}"
            rank0_print(log_str)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log any remaining losses at the end of training."""
        if self._droid_loss_count > 0 or self._json_loss_count > 0:
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


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (VLAModelArguments, VLADataArguments, VLATrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Determine model load path and checkpoint for resuming
    checkpoint_to_resume = None
    model_load_path = model_args.model_name_or_path
    
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        regular_checkpoints = [d for d in checkpoint_dirs if not d.name.endswith('_fixed')]
        if regular_checkpoints:
            latest_checkpoint = max(regular_checkpoints, key=lambda x: int(x.name.split('-')[1]))
            rank0_print(f"Found latest checkpoint: {latest_checkpoint}")
            model_load_path = str(latest_checkpoint)
            checkpoint_to_resume = str(latest_checkpoint)

    # Load model
    # Check original model name for 'qwen2.5' as checkpoint path might not contain it
    if "qwen2.5" in model_args.model_name_or_path.lower():
        # Use base model directly since we're only using existing tokens
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_load_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_load_path, trust_remote_code=True
        ).image_processor
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
    
    # # # Create data module with token mappings
    # if data_args.enable_cotrain and data_args.cotrain_json_paths:
    #     # Parse comma-separated JSON paths
    #     json_paths = [path.strip() for path in data_args.cotrain_json_paths.split(",") if path.strip()]
    #     data_args.cotrain_json_paths = json_paths
        
    #     rank0_print(f"Co-training enabled with {len(json_paths)} JSON datasets")
    #     rank0_print(f"JSON ratio: {data_args.cotrain_json_ratio:.2f}")
    #     rank0_print(f"JSON paths: {json_paths}")
    #     rank0_print(f"Image resize constraints: max_dim={data_args.max_image_dim}, min_dim={data_args.min_image_dim}")
    #     rank0_print(f"Pixel budget: {data_args.pixel_budget:,} pixels per JSON query (VLA data unchanged)")
        
    #     data_module = make_mixed_vla_data_module(
    #         tokenizer=tokenizer,
    #         action_tokenizer=action_tokenizer,
    #         data_args=data_args,
    #         model_max_length=training_args.model_max_length,
    #         token_mappings=token_mappings,
    #         image_size=(data_args.image_height, data_args.image_width),
    #         cotrain_json_ratio=data_args.cotrain_json_ratio,
    #         use_fixed_ratio_sampler=data_args.use_fixed_ratio_sampler,
    #     )
    # else:
    #     # Standard VLA-only training
    #     rank0_print("VLA-only training (no co-training)")
    #     rank0_print(f"Image resize constraints: max_dim={data_args.max_image_dim}, min_dim={data_args.min_image_dim}")
    #     rank0_print(f"Pixel budget: {data_args.pixel_budget:,} pixels per JSON query (VLA data unchanged)")
    #     data_module = make_droid_data_module(
    #         tokenizer=tokenizer, 
    #         action_tokenizer=action_tokenizer,
    #         data_args=data_args,
    #         model_max_length=training_args.model_max_length,
    #         token_mappings=token_mappings,
    #         image_size=(data_args.image_height, data_args.image_width)
    #     )

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
    }
    
    # Choose dataset type based on data_args.dataset_type
    if data_args.dataset_type == "fixed":
        rank0_print("Using fixed ratio mixed dataset with pre-collation")
        data_module = make_fixed_mixed_val_data_module(**dataset_creation_args)
    else:
        rank0_print("Using proportional mixed dataset with probabilistic sampling")
        data_module = make_proportional_mixed_val_data_module(**dataset_creation_args)
    
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
    
    # Initialize trainer with simplified generation logger
    
    trainer = VLATrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        callbacks=[checkpoint_processor_callback],  # Only keep checkpoint processor
        train_sampler_params=train_sampler_params,  # Pass sampler params to custom trainer
        generation_logger=generation_logger,  # Pass generation logger to trainer
        generation_interval=training_args.generation_interval,  # Use configurable generation interval
        **data_module
    )
    

    
    # Set action_start_id on trainer for loss separation
    trainer.action_start_id = token_mappings['action_start_id']
    
    # TODO: Switch to EMATrainer when implementing custom EMA support
    # trainer_class = EMATrainer if getattr(training_args, 'use_ema', False) else Trainer
    # trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args, **data_module)

    # Start training
    if checkpoint_to_resume:
        rank0_print("="*20 + " CHECKPOINT RESUMPTION " + "="*20)
        rank0_print(f"Attempting to resume full training state from: {checkpoint_to_resume}")
        
        try:
            trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        except Exception as e:
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
                rank0_print(f"\nUnknown error during checkpoint loading: {e}")
                raise e
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
