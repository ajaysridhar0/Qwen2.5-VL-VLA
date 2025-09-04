"""
Mixed dataset for co-training VLA with normal JSON data to prevent catastrophic forgetting.
Combines DROID VLA data with regular conversational JSON data at a specified ratio.
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


class FixedMixedVLAValDataset(IterableDataset):
    """
    An IterableDataset that yields pre-collated batches with a fixed ratio 
    of samples from two different data sources (one map-style, one iterable).
    
    This version is stateful and reproducible, allowing resumption from a checkpoint.
    """

    def __init__(self, json_dataset, vla_dataset_args, collator, batch_size, json_ratio, 
                 samples_to_skip=0, seed=42, num_samples: int = None):
        super().__init__()
        self.json_dataset = json_dataset
        self.vla_dataset_args = vla_dataset_args
        self.collator = collator
        self.batch_size = batch_size
        self.json_ratio = json_ratio
        self.vla_ratio = 1.0 - json_ratio
        self.samples_to_skip = samples_to_skip
        self.seed = seed
        self.num_samples = num_samples
        
        # Validate at least one source is provided
        if self.json_dataset is None and self.vla_dataset_args is None:
            raise ValueError("At least one of json_dataset or vla_dataset_args must be provided.")

        # Calculate the number of samples from each source per batch, handling None cases
        if self.json_dataset is None:
            # VLA-only
            self.num_json = 0
            self.num_vla = self.batch_size
        elif self.vla_dataset_args is None:
            # JSON-only
            self.num_json = self.batch_size
            self.num_vla = 0
        else:
            # Mixed
            self.num_json = int(self.batch_size * max(0.0, min(1.0, self.json_ratio)))
            self.num_vla = self.batch_size - self.num_json

        rank0_print(f"Fixed ratio mixed dataset: VLA ratio={self.vla_ratio:.3f}, JSON ratio={self.json_ratio:.3f}")
        rank0_print(f"Batch composition: {self.num_vla} VLA + {self.num_json} JSON = {self.batch_size} total")
        if self.samples_to_skip > 0:
            rank0_print(f"Will skip {self.samples_to_skip:,} samples to resume training.")
        if self.num_samples is not None:
            rank0_print(f"Dataset limited to {self.num_samples:,} total samples")
        
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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Get GPU rank for distributed training (same logic as VLA dataset)
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

        # Each worker gets its own independent iterators
        json_iter = None
        if self.num_json > 0 and self.json_dataset is not None:
            json_iter = self._create_json_generator(worker_id, num_workers, gpu_rank, world_size, json_samples_to_skip, self.seed)

        # The DroidVLADataset is our base iterable VLA source
        vla_iter = None
        if self.num_vla > 0 and self.vla_dataset_args is not None:
            # Create VLA dataset args with skip parameters
            vla_args = self.vla_dataset_args.copy()
            vla_args['samples_to_skip'] = vla_samples_to_skip
            vla_args['seed'] = self.seed
            vla_iterable_dataset = DroidVLADatasetIterable(**vla_args)
            vla_iter = iter(vla_iterable_dataset)

        num_batches_to_yield = (self.num_samples // self.batch_size) if self.num_samples is not None else None

        # Main loop to create and yield batches
        count = 0
        # Create a deterministic RNG for batch shuffling
        batch_rng = random.Random(self.seed + global_worker_id) if self.seed is not None else random
        
        while True:
            if num_batches_to_yield is not None and count >= num_batches_to_yield:
                return
            # 1. Gather samples for one batch
            json_samples = [next(json_iter) for _ in range(self.num_json)] if json_iter is not None else []
            vla_samples = [next(vla_iter) for _ in range(self.num_vla)] if vla_iter is not None else []
            
            # 2. Combine and shuffle the samples within the batch
            batch_samples = json_samples + vla_samples
            batch_rng.shuffle(batch_samples)

            # 3. Use the collator to turn the list of samples into a tensor batch
            collated_batch = self.collator(batch_samples)
            
            # 4. Yield the complete, ready-to-use batch
            yield collated_batch
            count += 1





class DummyCollator:
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return instances[0]
        

def make_fixed_mixed_val_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    cotrain_json_ratio: float = 0.2,
    samples_to_skip: int = 0,
    seed: int = 42,
    data_size: int = None,
) -> Dict:
    """Make dataset and collator for mixed VLA + JSON co-training."""

    data_collator = MixedVLADataCollator(
        tokenizer=tokenizer,
        model_max_length=model_max_length
    )

    # Calculate JSON data size based on the ratio and data_size
    json_data_size = None
    if data_size is not None and cotrain_json_ratio > 0 and cotrain_json_ratio < 1:
        # For fixed ratio: json_size = data_size * ratio / (1 - ratio)
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
    
    # We need to determine batch size for the fixed ratio dataset
    # Since this is used in training, we'll use a reasonable default batch size
    # The actual batch size used will be controlled by the training arguments
    # But we need a batch size for the fixed ratio calculation
    default_batch_size = 16  # This will be the composition unit, actual batching happens at trainer level
    
    train_dataset = FixedMixedVLAValDataset(
        json_dataset=json_dataset,
        vla_dataset_args=vla_dataset_args,
        collator=data_collator,
        batch_size=default_batch_size,
        json_ratio=cotrain_json_ratio,
        samples_to_skip=samples_to_skip,
        seed=seed,
        num_samples=data_size,
    )
    
    # No eval dataset needed - generation callback will sample from training data
    eval_dataset = None
    rank0_print("No eval dataset created - generation callback will sample from training dataset directly")
    
    # No custom sampler needed for fixed ratio approach
    train_sampler_params = None
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DummyCollator(),
        train_sampler_params=train_sampler_params,
    )
