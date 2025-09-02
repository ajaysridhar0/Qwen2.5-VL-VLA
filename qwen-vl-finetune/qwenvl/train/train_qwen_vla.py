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
from qwenvl.train.generation_callback import GenerationLoggingCallback
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


class VLATrainer(Trainer):
    """Custom trainer that supports fixed ratio sampling for mixed VLA/JSON training."""
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        This is an override of the original method to prevent `accelerator.prepare()`
        from wrapping the DataLoader when using an IterableDataset. This is because
        the accelerator wrapper incorrectly reshapes tensors with mismatched first
        dimensions (e.g., pixel_values vs. input_ids), which corrupts the batch.
        Device placement is handled by our overridden `_prepare_inputs` method.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dl = DataLoader(train_dataset, **dataloader_params)

        if isinstance(train_dataset, IterableDataset):
            # For IterableDatasets, return the raw DataLoader.
            # Our `_prepare_inputs` override will handle device placement.
            return dl
        
        # For map-style datasets, the accelerator wrapper works correctly.
        return self.accelerator.prepare(dl)

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Override the default _prepare_inputs to manually move tensors to the correct device,
        bypassing the accelerator.prepare() that was incorrectly reshaping the pixel_values tensor
        for IterableDatasets.
        """
        # Manually move each tensor to the designated device
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared_inputs[k] = v.to(self.args.device)
            else:
                prepared_inputs[k] = v
        return prepared_inputs

    def __init__(self, *args, train_sampler_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler_params = train_sampler_params
        # Initialize metrics for separate loss tracking
        self._droid_loss_sum = 0.0
        self._droid_loss_count = 0
        self._json_loss_sum = 0.0
        self._json_loss_count = 0
        self._log_interval = self.args.logging_steps  # Sync with Trainer's logging steps
    
    # def _get_train_sampler(self):
    #     """Override to use our custom fixed ratio sampler if params are provided."""
    #     if self.train_sampler_params is not None:
    #         # Extract sampler class and params
    #         sampler_class = self.train_sampler_params["sampler_class"]
    #         dataset_size = self.train_sampler_params["dataset_size"]
    #         json_ratio = self.train_sampler_params["json_ratio"]
            
    #         # Create the sampler with proper batch size
    #         sampler = sampler_class(
    #             dataset_size=dataset_size,
    #             batch_size=self.args.per_device_train_batch_size,
    #             json_ratio=json_ratio,
    #             num_replicas=self.args.world_size,
    #             rank=self.args.process_index,
    #             shuffle=True,
    #             seed=self.args.seed,
    #             drop_last=self.args.dataloader_drop_last,
    #         )
            
    #         rank0_print(f"Using FixedRatioSampler with {json_ratio:.2f} JSON ratio per batch")
    #         rank0_print(f"  JSON samples per batch: {sampler.json_per_batch}")
    #         rank0_print(f"  VLA samples per batch: {sampler.vla_per_batch}")
            
    #         return sampler
    #     else:
    #         # Use default sampler
    #         return super()._get_train_sampler()
    
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
        
        return (loss, outputs) if return_outputs else loss
    
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
        
        # Log to wandb if available
        if len(metrics) > 0:
            self.log(metrics)
            
            # Also print to console
            if self.args.local_rank in [-1, 0]:
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
    
    # Individual dataset weighting options
    cotrain_json_weights: str = field(default="", metadata={"help": "Comma-separated weights for individual JSON datasets (must match number of JSON paths). If empty, uses equal weights."})
    weight_by_count: bool = field(default=False, metadata={"help": "Weight all datasets (including droid) by their sample count instead of using fixed ratios"})
    count_weight_power: float = field(default=1.0, metadata={"help": "Power to raise dataset counts to when using count-based weighting (e.g., 0.5 for square root, 1.0 for linear)"})
    



@dataclass
class VLATrainingArguments(TrainingArguments):
    """Extended training arguments with evaluation settings for VLA."""
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to adopt during training."}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "Run an evaluation every X steps."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to adopt during training."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
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

    # Load model
    if "qwen2.5" in model_args.model_name_or_path.lower():
        # Use base model directly since we're only using existing tokens
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
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

    data_module = make_proportional_mixed_val_data_module(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args,
        model_max_length=training_args.model_max_length,
        token_mappings=token_mappings,
        image_size=(data_args.image_height, data_args.image_width),
        cotrain_json_ratio=data_args.cotrain_json_ratio,
    )
    
    # Create generation logging callback with the shared action tokenizer
    # The action tokenizer will be initialized through normal data loading
    # call on dummy data to warmup the tokenizer
    action_data = np.random.rand(10, data_args.action_chunk_size, 8)    # one batch of action chunks
    _ = action_tokenizer(action_data)
    
    rank0_print(f"Creating GenerationLoggingCallback with eval_on_start={training_args.eval_on_start}")
    generation_callback = GenerationLoggingCallback(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        token_mappings=token_mappings,
        num_examples=training_args.num_generation_examples,
        log_file="generations.txt",
        log_to_wandb=training_args.log_generations_to_wandb,
        eval_on_start=training_args.eval_on_start,
    )
    
    # Create checkpoint processor callback to save preprocessor config with each checkpoint
    checkpoint_processor_callback = CheckpointProcessorCallback(model_args.model_name_or_path)
    
    # Extract sampler params if present
    train_sampler_params = data_module.pop('train_sampler_params', None)
    
    # Initialize trainer with callbacks
    
    trainer = VLATrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        callbacks=[generation_callback, checkpoint_processor_callback],
        train_sampler_params=train_sampler_params,  # Pass sampler params to custom trainer
        **data_module
    )
    
    # Set action_start_id on trainer for loss separation
    trainer.action_start_id = token_mappings['action_start_id']
    
    # Set up the callback with trainer components
    generation_callback.setup_trainer_components(
        model=model,
        eval_dataloader=trainer.get_eval_dataloader() if data_module['eval_dataset'] is not None else None
    )
    
    # TODO: Switch to EMATrainer when implementing custom EMA support
    # trainer_class = EMATrainer if getattr(training_args, 'use_ema', False) else Trainer
    # trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args, **data_module)

    # Start training
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        # Filter out fixed checkpoints and get the latest checkpoint
        regular_checkpoints = [d for d in checkpoint_dirs if not d.name.endswith('_fixed')]
        if not regular_checkpoints:
            rank0_print("ERROR: No regular checkpoints found, only fixed ones")
            trainer.train()
            return
        
        # Get the latest regular checkpoint
        latest_checkpoint = max(regular_checkpoints, key=lambda x: int(x.name.split('-')[1]))
        rank0_print(f"=== CHECKPOINT LOADING ===")
        rank0_print(f"Found checkpoint: {latest_checkpoint}")
        
        # Verify checkpoint integrity
        global_step_dir = latest_checkpoint / "global_step*"
        global_step_dirs = list(latest_checkpoint.glob("global_step*"))
        trainer_state_file = latest_checkpoint / "trainer_state.json"
        
        if not global_step_dirs:
            rank0_print(f"WARNING: No global_step directory found in {latest_checkpoint}")
        else:
            rank0_print(f"Global step directory: {global_step_dirs[0]}")
            
        if not trainer_state_file.exists():
            rank0_print(f"WARNING: No trainer_state.json found in {latest_checkpoint}")
        else:
            # Read and verify trainer state
            with open(trainer_state_file, 'r') as f:
                trainer_state = json.load(f)
                rank0_print(f"Trainer state global_step: {trainer_state.get('global_step', 'UNKNOWN')}")
                rank0_print(f"Trainer state epoch: {trainer_state.get('epoch', 'UNKNOWN')}")
        
        # Try to resume with full state first
        try:
            rank0_print("Attempting to load checkpoint...")
            rank0_print("NOTE: DeepSpeed optimizer state loading is buggy - using warm restart strategy")
            trainer.train(resume_from_checkpoint=str(latest_checkpoint))
        except Exception as e:
            rank0_print(f"ERROR during checkpoint loading: {e}")
            
            if ("world size" in str(e).lower() or 
                "dp world size" in str(e).lower() or
                "partition" in str(e).lower()):
                rank0_print(f"DeepSpeed world size/partition mismatch detected: {e}")
                rank0_print("Attempting fallback: Loading model weights only (WITHOUT optimizer state)...")
                
                # Load model weights manually
                model_path = latest_checkpoint / "pytorch_model.bin"
                if not model_path.exists():
                    # Try safetensors format
                    import glob
                    safetensor_files = glob.glob(str(latest_checkpoint / "model*.safetensors"))
                    if safetensor_files:
                        rank0_print("Loading from safetensors format")
                        from safetensors.torch import load_file
                        state_dict = {}
                        for file in safetensor_files:
                            state_dict.update(load_file(file))
                        model.load_state_dict(state_dict, strict=False)
                        rank0_print("Model weights loaded successfully (NO optimizer state)")
                    else:
                        rank0_print("ERROR: No model files found in checkpoint!")
                        raise ValueError(f"No model files found in {latest_checkpoint}")
                
                # Warn about loss implications
                rank0_print("WARNING: Optimizer state NOT loaded - expect higher initial loss!")
                rank0_print("Consider using same GPU count or converting checkpoint first.")
                
                # Start fresh training (no resume)
                trainer.train()
            else:
                # Re-raise if it's a different error
                rank0_print(f"Unknown error during checkpoint loading: {e}")
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
