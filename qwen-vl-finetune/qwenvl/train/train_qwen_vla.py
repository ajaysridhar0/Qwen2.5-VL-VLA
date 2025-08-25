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
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_droid import make_droid_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
# from qwenvl.train.trainer import EMATrainer  # TODO: Uncomment when implementing custom EMA
from dataclasses import dataclass, field

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class VLAModelArguments(ModelArguments):
    """Extended model arguments for VLA training."""
    action_vocab_size: int = field(default=512, metadata={"help": "Size of action vocabulary"})
    action_tokenizer_path: str = field(default="KarlP/fast-droid", metadata={"help": "Path to action tokenizer"})


@dataclass 
class VLADataArguments(DataArguments):
    """Extended data arguments for DROID dataset."""
    droid_data_dir: str = field(default="/iliad2/u/ajaysri/episodic_memory/droid_rlds")
    droid_dataset_name: str = field(default="droid_100")
    action_chunk_size: int = field(default=15)
    use_joint_velocity: bool = field(default=True)
    shuffle_buffer_size: int = field(default=100000)
    num_train_samples: int = field(default=1000000)
    image_height: int = field(default=180, metadata={"help": "Height to resize images to"})
    image_width: int = field(default=320, metadata={"help": "Width to resize images to"})


class VLAQwenModel(Qwen2_5_VLForConditionalGeneration):
    """Extended Qwen model for VLA that can handle action tokens."""
    
    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, *args, **kwargs):
        """Forward pass with special handling for action tokens."""
        # During training, action tokens are treated as regular tokens
        # The loss computation will only consider action tokens due to labels masking
        kwargs.pop("num_items_in_batch", None)
        return super().forward(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate method that can produce action tokens."""
        # # Ensure action tokens can be generated
        # if "bad_words_ids" not in kwargs:
        #     kwargs["bad_words_ids"] = []
        
        # # Allow generation of action tokens
        # if "force_words_ids" not in kwargs:
        #     kwargs["force_words_ids"] = []
            
        return super().generate(*args, **kwargs)


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


def get_action_state_token_mappings(tokenizer, action_vocab_size=512, state_vocab_size=256):
    """Get token ID mappings for action and state tokens using existing Chinese tokens."""
    
    # Reserve specific ranges in existing vocabulary for actions and states
    # Use tokens starting from 102500 (Chinese characters that we won't need for English/robotics)
    state_token_start = 102500
    action_token_start = state_token_start + state_vocab_size  # 102756
    control_token_start = action_token_start + action_vocab_size  # After action tokens
    
    # Verify these tokens exist in the vocabulary
    vocab_size = len(tokenizer)
    max_needed_token = control_token_start + 4  # Need 4 control tokens
    
    if max_needed_token > vocab_size:
        raise ValueError(f"Not enough existing tokens. Need up to {max_needed_token}, but vocab size is {vocab_size}")
    
    # Create mappings for action and state tokens (use existing Chinese tokens)
    state_token_ids = list(range(state_token_start, state_token_start + state_vocab_size))
    action_token_ids = list(range(action_token_start, action_token_start + action_vocab_size))
    
    # Map control token strings to existing Chinese token IDs (NO new tokens added!)
    control_mappings = {
        "<|action_start|>": control_token_start,      # Map string to existing token ID
        "<|action_end|>": control_token_start + 1,    # Map string to existing token ID
        "<|state_start|>": control_token_start + 2,   # Map string to existing token ID  
        "<|state_end|>": control_token_start + 3,     # Map string to existing token ID
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
        (VLAModelArguments, VLADataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load model
    if "qwen2.5" in model_args.model_name_or_path.lower():
        # Use custom VLA model class
        model = VLAQwenModel.from_pretrained(
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
    
    # Get action and state token mappings using existing Chinese tokens
    token_mappings = get_action_state_token_mappings(tokenizer, model_args.action_vocab_size)
    
    rank0_print(f"Token mappings - all existing Chinese tokens:")
    rank0_print(f"  State: {token_mappings['state_token_ids'][0]}-{token_mappings['state_token_ids'][-1]}")
    rank0_print(f"  Action: {token_mappings['action_token_ids'][0]}-{token_mappings['action_token_ids'][-1]}")
    rank0_print(f"  Control: {token_mappings['control_mappings']}")
    
    # Set model trainable parameters
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        rank0_print("Model architecture:")
        rank0_print(f"Vision encoder trainable: {model_args.tune_mm_vision}")
        rank0_print(f"Vision-language connector trainable: {model_args.tune_mm_mlp}")
        rank0_print(f"Language model trainable: {model_args.tune_mm_llm}")
        rank0_print(f"Action vocabulary size: {model_args.action_vocab_size}")
    
    # Create DROID data module with token mappings
    data_module = make_droid_data_module(
        tokenizer=tokenizer, 
        data_args=data_args,
        model_max_length=training_args.model_max_length,
        token_mappings=token_mappings,
        image_size=(data_args.image_height, data_args.image_width)
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
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
        "action_vocab_size": model_args.action_vocab_size,
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
