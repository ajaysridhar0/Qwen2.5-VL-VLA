"""
Training script for Qwen VLA (Vision-Language-Action) model on DROID dataset.
Modified from train_qwen.py to support action prediction with fast tokenizer.
"""

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict, List, Optional
import shutil
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import wandb

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
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.generation_callback import GenerationLoggingCallback
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
# from qwenvl.train.trainer import EMATrainer  # TODO: Uncomment when implementing custom EMA
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class VLATrainer(Trainer):
    """Custom trainer that supports fixed ratio sampling for mixed VLA/JSON training."""
    
    def __init__(self, *args, train_sampler_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sampler_params = train_sampler_params
    
    def _get_train_sampler(self):
        """Override to use our custom fixed ratio sampler if params are provided."""
        if self.train_sampler_params is not None:
            # Extract sampler class and params
            sampler_class = self.train_sampler_params["sampler_class"]
            dataset_size = self.train_sampler_params["dataset_size"]
            json_ratio = self.train_sampler_params["json_ratio"]
            
            # Create the sampler with proper batch size
            sampler = sampler_class(
                dataset_size=dataset_size,
                batch_size=self.args.per_device_train_batch_size,
                json_ratio=json_ratio,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            rank0_print(f"Using FixedRatioSampler with {json_ratio:.2f} JSON ratio per batch")
            rank0_print(f"  JSON samples per batch: {sampler.json_per_batch}")
            rank0_print(f"  VLA samples per batch: {sampler.vla_per_batch}")
            
            return sampler
        else:
            # Use default sampler
            return super()._get_train_sampler()


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
    
    # Co-training with regular JSON data to prevent catastrophic forgetting
    enable_cotrain: bool = field(default=False, metadata={"help": "Enable co-training with regular JSON data"})
    cotrain_json_paths: str = field(default="", metadata={"help": "Comma-separated paths to JSON/JSONL files for co-training"})
    cotrain_json_ratio: float = field(default=0.2, metadata={"help": "Ratio of regular JSON data in mixed training (0.0-1.0)"})
    use_fixed_ratio_sampler: bool = field(default=True, metadata={"help": "Use fixed ratio sampler to ensure consistent memory usage per batch"})
    pixel_budget: int = field(default=230400, metadata={"help": "Max pixels per JSON query (default: 230400 = 4x VLA size). For multi-frame JSON, budget applies to total pixels across all frames."})
    



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
    
    # Create data module with token mappings
    if data_args.enable_cotrain and data_args.cotrain_json_paths:
        # Parse comma-separated JSON paths
        json_paths = [path.strip() for path in data_args.cotrain_json_paths.split(",") if path.strip()]
        data_args.cotrain_json_paths = json_paths
        
        rank0_print(f"Co-training enabled with {len(json_paths)} JSON datasets")
        rank0_print(f"JSON ratio: {data_args.cotrain_json_ratio:.2f}")
        rank0_print(f"JSON paths: {json_paths}")
        rank0_print(f"Pixel budget: {data_args.pixel_budget:,} pixels per JSON query (VLA data unchanged)")
        
        data_module = make_mixed_vla_data_module(
            tokenizer=tokenizer,
            action_tokenizer=action_tokenizer,
            data_args=data_args,
            model_max_length=training_args.model_max_length,
            token_mappings=token_mappings,
            image_size=(data_args.image_height, data_args.image_width),
            cotrain_json_ratio=data_args.cotrain_json_ratio,
            use_fixed_ratio_sampler=data_args.use_fixed_ratio_sampler,
        )
    else:
        # Standard VLA-only training
        rank0_print("VLA-only training (no co-training)")
        rank0_print(f"Pixel budget: {data_args.pixel_budget:,} pixels per JSON query (VLA data unchanged)")
        data_module = make_droid_data_module(
            tokenizer=tokenizer, 
            action_tokenizer=action_tokenizer,
            data_args=data_args,
            model_max_length=training_args.model_max_length,
            token_mappings=token_mappings,
            image_size=(data_args.image_height, data_args.image_width)
        )
    
    # Create generation logging callback with the shared action tokenizer
    # The action tokenizer will be initialized through normal data loading
    generation_callback = GenerationLoggingCallback(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        token_mappings=token_mappings,
        num_examples=training_args.num_generation_examples,
        log_file="generations.txt",
        log_to_wandb=training_args.log_generations_to_wandb,
    )
    
    # Extract sampler params if present
    train_sampler_params = data_module.pop('train_sampler_params', None)
    
    # Initialize trainer with callback
    trainer = VLATrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        callbacks=[generation_callback],
        train_sampler_params=train_sampler_params,  # Pass sampler params to custom trainer
        **data_module
    )
    
    # Set up the callback with trainer components
    generation_callback.setup_trainer_components(
        model=model,
        eval_dataloader=trainer.get_eval_dataloader() if data_module['eval_dataset'] is not None else None
    )
    
    # TODO: Switch to EMATrainer when implementing custom EMA support
    # trainer_class = EMATrainer if getattr(training_args, 'use_ema', False) else Trainer
    # trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args, **data_module)

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
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
