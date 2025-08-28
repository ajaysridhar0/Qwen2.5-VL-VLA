"""
Fixed Ratio Sampler for Mixed VLA and JSON Training

This sampler ensures that each batch has a fixed ratio of VLA to JSON samples,
preventing OOM issues from random batches with too many high-resolution JSON examples.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional
import math
import random
import hashlib


class FixedRatioSampler(Sampler):
    """
    A sampler that ensures each batch has a fixed ratio of samples from two datasets.
    
    This is crucial for mixed training where one dataset (JSON) has much larger
    memory requirements than the other (VLA), preventing OOM from random batches.
    """
    
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        json_ratio: float,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        num_replicas: int = 1,
    ):
        """
        Args:
            dataset_size: Total size of the dataset
            batch_size: Size of each batch
            json_ratio: Ratio of JSON samples in each batch (0.0 to 1.0)
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            drop_last: Whether to drop the last incomplete batch
            rank: Current process rank for distributed training
            num_replicas: Total number of processes for distributed training
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.json_ratio = json_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.num_replicas = num_replicas
        
        # Calculate samples per batch for each type
        self.json_per_batch = int(batch_size * json_ratio)
        self.vla_per_batch = batch_size - self.json_per_batch
        
        # Ensure we have at least one sample of the majority type
        if self.json_per_batch == 0 and json_ratio > 0:
            self.json_per_batch = 1
            self.vla_per_batch = batch_size - 1
        elif self.vla_per_batch == 0 and json_ratio < 1:
            self.vla_per_batch = 1
            self.json_per_batch = batch_size - 1
            
        # Calculate total batches
        self.total_batches = dataset_size // batch_size
        if not drop_last and dataset_size % batch_size != 0:
            self.total_batches += 1
            
        # For distributed training, calculate batches per replica
        self.batches_per_replica = math.ceil(self.total_batches / num_replicas)
        self.total_size = self.batches_per_replica * batch_size
        
        # Create indices for JSON and VLA samples
        self._prepare_indices()
        
    def _prepare_indices(self):
        """Prepare indices for JSON and VLA samples."""
        # Pre-allocate lists with estimated sizes for better performance
        estimated_json_size = int(self.dataset_size * self.json_ratio * 1.1)  # 10% buffer
        estimated_vla_size = self.dataset_size - estimated_json_size + int(self.dataset_size * 0.1)
        
        json_indices = []
        vla_indices = []
        
        # Pre-compute hash seed for this sampler configuration
        ratio_bytes = str(self.json_ratio).encode()
        ratio_hash = int(hashlib.md5(ratio_bytes).hexdigest()[:8], 16)
        
        # Use a more efficient deterministic assignment
        for idx in range(self.dataset_size):
            # Combine index and ratio hash for deterministic but well-distributed assignment
            combined_hash = (idx * 2654435761 + ratio_hash) & 0xFFFFFFFF  # Fast hash mixing
            random_value = (combined_hash % 1000000) / 1000000.0
            
            if random_value < self.json_ratio:
                json_indices.append(idx)
            else:
                vla_indices.append(idx)
        
        self.json_indices = json_indices
        self.vla_indices = vla_indices
        
    def __iter__(self) -> Iterator[int]:
        """Generate batches with fixed ratios."""
        # Set random seed for this epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            
            # More efficient shuffling using tensor operations directly
            json_tensor = torch.tensor(self.json_indices, dtype=torch.long)
            vla_tensor = torch.tensor(self.vla_indices, dtype=torch.long)
            
            json_perm = torch.randperm(len(json_tensor), generator=g)
            vla_perm = torch.randperm(len(vla_tensor), generator=g)
            
            json_indices = json_tensor[json_perm].tolist()
            vla_indices = vla_tensor[vla_perm].tolist()
        else:
            json_indices = self.json_indices
            vla_indices = self.vla_indices
        
        # Create batches with fixed ratios
        # Pre-allocate with exact size to avoid repeated memory allocations
        total_samples = self.total_batches * self.batch_size
        batch_indices = [0] * total_samples
        
        # Pre-compute lengths to avoid repeated len() calls
        json_len = len(json_indices)
        vla_len = len(vla_indices)
        
        batch_idx = 0
        for batch_num in range(self.total_batches):
            batch_start = batch_idx
            
            # Add JSON samples using modulo arithmetic for wrapping
            for i in range(self.json_per_batch):
                json_idx = (batch_num * self.json_per_batch + i) % json_len
                batch_indices[batch_idx] = json_indices[json_idx]
                batch_idx += 1
            
            # Add VLA samples using modulo arithmetic for wrapping
            for i in range(self.vla_per_batch):
                vla_idx = (batch_num * self.vla_per_batch + i) % vla_len
                batch_indices[batch_idx] = vla_indices[vla_idx]
                batch_idx += 1
            
            # Shuffle within batch to mix JSON and VLA samples
            if self.shuffle:
                batch_end = batch_idx
                batch_slice = batch_indices[batch_start:batch_end]
                random.Random(self.seed + batch_num).shuffle(batch_slice)
                batch_indices[batch_start:batch_end] = batch_slice
        
        # Handle distributed training - each rank gets its subset
        if self.num_replicas > 1:
            # Ensure all replicas have the same total size
            if len(batch_indices) < self.total_size * self.num_replicas:
                # Pad with repeated indices
                batch_indices += batch_indices[:self.total_size * self.num_replicas - len(batch_indices)]
            
            # Get this rank's subset
            start_idx = self.rank * self.total_size
            end_idx = start_idx + self.total_size
            batch_indices = batch_indices[start_idx:end_idx]
        
        # Drop last incomplete batch if needed
        if self.drop_last and len(batch_indices) % self.batch_size != 0:
            batch_indices = batch_indices[:len(batch_indices) // self.batch_size * self.batch_size]
        
        return iter(batch_indices)
    
    def __len__(self) -> int:
        """Return the number of samples for this sampler."""
        if self.drop_last:
            return (self.total_size // self.batch_size) * self.batch_size
        else:
            return self.total_size
    
    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling."""
        self.seed = epoch


class DistributedFixedRatioSampler(FixedRatioSampler):
    """
    Distributed version of FixedRatioSampler that's compatible with DDP.
    """
    
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        json_ratio: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
        super().__init__(
            dataset_size=dataset_size,
            batch_size=batch_size,
            json_ratio=json_ratio,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            rank=rank,
            num_replicas=num_replicas,
        )
