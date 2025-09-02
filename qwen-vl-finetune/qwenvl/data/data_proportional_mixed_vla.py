"""
Mixed dataset for co-training VLA with normal JSON data using proportional sampling.
Combines DROID VLA data with regular conversational JSON data with probabilistic sampling
instead of fixed batch ratios.
"""

import torch
import numpy as np
import transformers
import random
import json
import copy
import math
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Any, Union
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import os

# Import existing components
from .data_droid_iterable import DroidVLADatasetIterable
from .data_json import JSONCotrainDataset
from .data_mixed_vla import MixedVLADataCollator

# Import the original preprocessing function that handles both image and video tokens
from .data_qwen import (
    preprocess_qwen_2_visual, 
    IGNORE_INDEX, 
    IMAGE_TOKEN_INDEX, 
    VIDEO_TOKEN_INDEX,
    pad_and_cat,  # For padding position_ids
)


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class ProportionalMixedVLAValDataset(IterableDataset):
    """
    An IterableDataset that yields samples using proportional sampling from two different 
    data sources (one map-style, one iterable). Instead of fixed batch ratios, each sample
    is probabilistically chosen from either source based on the specified ratio.
    """

    def __init__(self, json_dataset, vla_dataset_args, json_ratio, num_samples: int = None):
        super().__init__()
        self.json_dataset = json_dataset
        self.vla_dataset_args = vla_dataset_args
        self.json_ratio = json_ratio
        self.vla_ratio = 1.0 - json_ratio
        self.num_samples = num_samples
        
        # Validate at least one source is provided
        if self.json_dataset is None and self.vla_dataset_args is None:
            raise ValueError("At least one of json_dataset or vla_dataset_args must be provided.")

        rank0_print(f"Proportional mixed dataset configured: VLA ratio={self.vla_ratio:.3f}, JSON ratio={self.json_ratio:.3f}")
        
    def _create_json_generator(self, worker_id, num_workers):
        """Creates an infinite, sharded, shuffled generator for the JSON dataset"""
        indices = list(range(len(self.json_dataset)))
        worker_indices = indices[worker_id::num_workers]
        while True:
            random.shuffle(worker_indices)
            for idx in worker_indices:
                yield self.json_dataset[idx]

    def _create_vla_generator(self):
        """Creates a VLA dataset generator"""
        vla_iterable_dataset = DroidVLADatasetIterable(**self.vla_dataset_args)
        return iter(vla_iterable_dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Initialize generators for both data sources
        json_iter = None
        if self.json_dataset is not None and self.json_ratio > 0:
            json_iter = self._create_json_generator(worker_id, num_workers)

        vla_iter = None
        if self.vla_dataset_args is not None and self.vla_ratio > 0:
            vla_iter = self._create_vla_generator()

        # Calculate total number of samples to generate
        if self.num_samples is not None:
            total_samples = self.num_samples
        else:
            total_samples = float('inf')  # Infinite stream

        sample_count = 0
        
        # Main sampling loop - yield one sample at a time
        while sample_count < total_samples:
            # Decide which source to sample from based on ratio
            if json_iter is None:
                # VLA only
                sample = next(vla_iter)
            elif vla_iter is None:
                # JSON only
                sample = next(json_iter)
            else:
                # Both sources available - use probabilistic sampling
                if random.random() < self.json_ratio:
                    sample = next(json_iter)
                else:
                    try:
                        sample = next(vla_iter)
                    except StopIteration:
                        # VLA iterator exhausted, reinitialize
                        vla_iter = self._create_vla_generator()
                        sample = next(vla_iter)

            yield sample
            sample_count += 1


def make_proportional_mixed_val_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    cotrain_json_ratio: float = 0.2,
    create_eval_dataset: bool = True,
) -> Dict:
    """Make dataset and collator for proportional mixed VLA + JSON co-training."""

    # Use the existing MixedVLADataCollator
    data_collator = MixedVLADataCollator(
        tokenizer=tokenizer,
        model_max_length=model_max_length
    )

    if cotrain_json_ratio > 0:
        json_dataset = JSONCotrainDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            model_max_length=model_max_length,
            token_mappings=token_mappings,
            image_size=image_size,
        )
    else:
        json_dataset = None

    if cotrain_json_ratio < 1:
        vla_dataset_args = {
            "tokenizer": tokenizer,
            "data_args": data_args,
            "action_tokenizer": action_tokenizer,
            "model_max_length": model_max_length,
            "token_mappings": token_mappings,
        }
    else:
        vla_dataset_args = None
    
    # Create train dataset with proportional sampling
    train_dataset = ProportionalMixedVLAValDataset(
        json_dataset=json_dataset,
        vla_dataset_args=vla_dataset_args,
        json_ratio=cotrain_json_ratio,
    )
    
    # Create a small eval dataset with 100 samples
    eval_dataset = None
    if create_eval_dataset:
        eval_size = 100  # Fixed 100 samples for eval
        eval_cotrain_json_ratio = 0.5 if 0.0 < cotrain_json_ratio < 1.0 else cotrain_json_ratio

        eval_vla_dataset_args = copy.deepcopy(vla_dataset_args)
        if eval_vla_dataset_args:
            eval_vla_dataset_args["shuffle_buffer_size_override"] = 100
        
        eval_dataset = ProportionalMixedVLAValDataset(
            json_dataset=json_dataset if eval_cotrain_json_ratio > 0 else None,
            vla_dataset_args=eval_vla_dataset_args if eval_cotrain_json_ratio < 1 else None,
            json_ratio=eval_cotrain_json_ratio,
            num_samples=eval_size,
        )
        
        # Print eval dataset configuration
        if json_dataset is not None and cotrain_json_ratio > 0 and cotrain_json_ratio < 1:
            print(f"Eval dataset configured: Mixed={eval_size} (VLA ratio={1-eval_cotrain_json_ratio:.2f}, JSON ratio={eval_cotrain_json_ratio:.2f})")
        elif json_dataset is not None and cotrain_json_ratio >= 1:
            print(f"Eval dataset configured: JSON-only={eval_size}")
        else:
            print(f"Eval dataset configured: VLA-only={eval_size}")
    
    # No custom sampler needed for proportional sampling
    train_sampler_params = None
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        train_sampler_params=train_sampler_params,
    )
