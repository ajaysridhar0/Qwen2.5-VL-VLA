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


class ProportionalMixedVLADataset(IterableDataset):
    """
    An IterableDataset that yields samples using proportional sampling from two different
    data sources (one map-style, one iterable). Instead of fixed batch ratios, each sample
    is probabilistically chosen from either source based on the specified ratio.

    This version is stateful and reproducible, allowing resumption from a checkpoint.
    """

    def __init__(self, json_dataset, vla_dataset_args, json_ratio, samples_to_skip, seed, data_size: int = None):
        super().__init__()
        self.json_dataset = json_dataset
        self.vla_dataset_args = vla_dataset_args
        self.json_ratio = json_ratio
        self.vla_ratio = 1.0 - json_ratio
        self.samples_to_skip = samples_to_skip
        self.seed = seed
        self.data_size = data_size

        if self.json_dataset is None and self.vla_dataset_args is None:
            raise ValueError("At least one of json_dataset or vla_dataset_args must be provided.")

        rank0_print(f"Proportional mixed dataset configured: VLA ratio={self.vla_ratio:.3f}, JSON ratio={self.json_ratio:.3f}")
        if data_size is not None:
            rank0_print(f"Data size limit: {data_size:,} total samples")
        if self.samples_to_skip > 0:
            rank0_print(f"Will skip {self.samples_to_skip:,} samples to resume training.")

    def _create_json_generator(self, worker_id, num_workers, gpu_rank, world_size, samples_to_skip, seed):
        """Creates an infinite, sharded, shuffled, and stateful generator for the JSON dataset."""
        total_workers = world_size * num_workers
        global_worker_id = gpu_rank * num_workers + worker_id

        # Shard indices across all workers globally
        indices = list(range(len(self.json_dataset)))
        worker_indices = indices[global_worker_id::total_workers]

        if not worker_indices:
             # This worker has no data for this source, return an empty iterator
            return iter([])

        # >> SIMPLE SEEDING: Use seed + worker_id for randomness, optionally skip samples if specified
        epoch = 0
        indices_to_yield = worker_indices
        
        if samples_to_skip > 0:
            # Simple skip logic: distribute evenly across workers
            skip_per_worker = samples_to_skip // total_workers
            remainder = samples_to_skip % total_workers
            if global_worker_id < remainder:
                skip_per_worker += 1
            
            # Skip by advancing through epochs/indices
            if len(worker_indices) > 0 and skip_per_worker > 0:
                epoch = skip_per_worker // len(worker_indices)
                remainder_to_skip = skip_per_worker % len(worker_indices)
                indices_to_yield = worker_indices[remainder_to_skip:]

        # >> RANDOM SEEDING: Use seed to make data random but different each resume
        worker_seed = (seed + global_worker_id + epoch * 1000) if seed is not None else None
        rng = random.Random(worker_seed) if worker_seed is not None else random
        
        # First partial epoch
        rng.shuffle(indices_to_yield)
        for idx in indices_to_yield:
            yield self.json_dataset[idx]
        epoch += 1

        # Subsequent full epochs
        while True:
            epoch += 1
            worker_seed = (seed + global_worker_id + epoch * 1000) if seed is not None else None
            rng = random.Random(worker_seed) if worker_seed is not None else random
            indices_to_shuffle = worker_indices[:]
            rng.shuffle(indices_to_shuffle)
            for idx in indices_to_shuffle:
                yield self.json_dataset[idx]

    def _create_vla_generator(self, samples_to_skip, seed):
        """Creates a stateful VLA dataset generator."""
        # This assumes DroidVLADatasetIterable is the stateful class
        vla_iterable_dataset = DroidVLADatasetIterable(
            **self.vla_dataset_args,
            data_size=self.data_size,
            samples_to_skip=samples_to_skip,
            seed=seed
        )
        return iter(vla_iterable_dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if torch.distributed.is_initialized():
            gpu_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            gpu_rank = 0
            world_size = 1
        
        global_worker_id = gpu_rank * num_workers + worker_id

        # >> SIMPLE SKIP: Distribute skip count if specified, otherwise use 0 (rely on seeding for randomness)
        json_samples_to_skip = 0
        vla_samples_to_skip = 0
        if self.samples_to_skip > 0:
            json_samples_to_skip = round(self.samples_to_skip * self.json_ratio)
            vla_samples_to_skip = self.samples_to_skip - json_samples_to_skip
            rank0_print(f"Worker {global_worker_id}: Skipping -> JSON: {json_samples_to_skip}, VLA: {vla_samples_to_skip}")
        else:
            rank0_print(f"Worker {global_worker_id}: Using seed-based randomness (no skipping)")

        # Initialize generators for both data sources
        json_iter = None
        if self.json_dataset is not None and self.json_ratio > 0:
            json_iter = self._create_json_generator(worker_id, num_workers, gpu_rank, world_size, json_samples_to_skip, self.seed)

        vla_iter = None
        if self.vla_dataset_args is not None and self.vla_ratio > 0:
            vla_iter = self._create_vla_generator(vla_samples_to_skip, self.seed)
        
        # >> REPRODUCIBILITY: Create a deterministic RNG for sampling choice
        sampling_rng = random.Random(self.seed + global_worker_id) if self.seed is not None else random

        total_samples = self.data_size if self.data_size is not None else float('inf')
        sample_count = 0
        
        while sample_count < total_samples:
            if json_iter and vla_iter:
                # Both sources available - use probabilistic sampling
                if sampling_rng.random() < self.json_ratio:
                    sample = next(json_iter)
                else:
                    sample = next(vla_iter)
            elif vla_iter:
                sample = next(vla_iter)
            elif json_iter:
                sample = next(json_iter)
            else:
                break # No data sources

            yield sample
            sample_count += 1



def make_proportional_mixed_vla_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    cotrain_json_ratio: float = 0.2,
    create_eval_dataset: bool = True,
    samples_to_skip: int = 0,
    seed: int = 42,
    data_size: int = None,
) -> Dict:
    """Make dataset and collator for proportional mixed VLA + JSON co-training."""

    # Use the existing MixedVLADataCollator
    data_collator = MixedVLADataCollator(
        tokenizer=tokenizer,
        model_max_length=model_max_length
    )
    # Calculate JSON data size based on the ratio
    json_data_size = None
    if data_size is not None and cotrain_json_ratio > 0 and cotrain_json_ratio < 1:
        # For proportional sampling: json_size = data_size * ratio / (1 - ratio)
        # This ensures we have enough JSON samples to maintain the ratio during sampling
        json_data_size = int(data_size * cotrain_json_ratio / (1 - cotrain_json_ratio) * 1.5)  # 1.5x safety margin
        rank0_print(f"JSON dataset will be limited to {json_data_size} samples (from total data_size={data_size}, ratio={cotrain_json_ratio:.3f})")
    elif data_size is not None and cotrain_json_ratio >= 1:
        # JSON-only case
        json_data_size = data_size
        rank0_print(f"JSON-only dataset will be limited to {json_data_size} samples")

    if cotrain_json_ratio > 0:
        json_dataset = JSONCotrainDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            model_max_length=model_max_length,
            token_mappings=token_mappings,
            image_size=image_size,
            data_size=json_data_size,
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
    train_dataset = ProportionalMixedVLADataset(
        json_dataset=json_dataset,
        vla_dataset_args=vla_dataset_args,
        json_ratio=cotrain_json_ratio,
        data_size=data_size,
        samples_to_skip=samples_to_skip,
        seed=seed,
    )
    
    # No eval dataset needed - generation callback will sample from training data
    eval_dataset = None
    rank0_print("No eval dataset created - generation callback will sample from training dataset directly")
    
    # No custom sampler needed for proportional sampling
    train_sampler_params = None
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        train_sampler_params=train_sampler_params,
    )
