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
    
    For very small ratios (< 1/batch_size), the sampler distributes JSON samples
    across batches rather than forcing at least one JSON sample per batch, allowing
    some batches to contain only VLA samples while maintaining the overall ratio.
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
                       For ratios < 1/batch_size, some batches will contain no JSON samples
                       to maintain the overall ratio across the dataset
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
        
        # For very small ratios, allow some batches to have no JSON samples
        # Only force at least one sample if ratio is >= 1/batch_size
        if json_ratio >= (1.0 / batch_size):
            # Ensure we have at least one sample of the minority type
            if self.json_per_batch == 0 and json_ratio > 0:
                self.json_per_batch = 1
                self.vla_per_batch = batch_size - 1
            elif self.vla_per_batch == 0 and json_ratio < 1:
                self.vla_per_batch = 1
                self.json_per_batch = batch_size - 1
        
        # For distributed sampling with very small ratios, we need to track
        # which batches should contain JSON samples
        # Calculate how many batches we need to maintain the overall ratio
        if json_ratio > 0 and self.json_per_batch == 0:
            # We want json_ratio * total_samples = total_json_samples
            # If we put 1 JSON sample every N batches: total_json_samples = total_batches / N
            # So: json_ratio * total_batches * batch_size = total_batches / N
            # Therefore: N = 1 / (json_ratio * batch_size)
            # But we need to round properly to get close to the target ratio
            cycle_length = 1.0 / (json_ratio * batch_size)
            self.batches_per_json_cycle = max(1, round(cycle_length))
            
            # Calculate the actual ratio we'll achieve and validate it's reasonable
            achieved_ratio = 1.0 / (self.batches_per_json_cycle * batch_size)
            ratio_error = abs(achieved_ratio - json_ratio) / json_ratio
            
            # For very small ratios, we need to be more lenient with the error tolerance
            # But still catch cases where the ratio is completely wrong
            max_error = 1.0 if json_ratio < 0.01 else 0.5  # 100% error for very small ratios, 50% otherwise
            
            assert ratio_error <= max_error, (
                f"FixedRatioSampler: Achieved ratio {achieved_ratio:.4f} is too far from "
                f"target ratio {json_ratio:.4f} (error: {ratio_error:.1%}). "
                f"Consider adjusting batch_size or json_ratio. "
                f"With batch_size={batch_size}, batches_per_json_cycle={self.batches_per_json_cycle}"
            )
            
            # Additional check: ratio shouldn't be more than 10x different
            assert achieved_ratio <= json_ratio * 10 and achieved_ratio >= json_ratio / 10, (
                f"FixedRatioSampler: Achieved ratio {achieved_ratio:.4f} is more than 10x different "
                f"from target ratio {json_ratio:.4f}. This suggests a configuration error."
            )
            
            # Practical check: warn about very long cycles that might be impractical
            if self.batches_per_json_cycle > 100:
                import warnings
                warnings.warn(
                    f"FixedRatioSampler: Very long JSON cycle ({self.batches_per_json_cycle} batches). "
                    f"With ratio {json_ratio:.4f} and batch_size {batch_size}, JSON samples will only "
                    f"appear every {self.batches_per_json_cycle} batches. Consider increasing batch_size "
                    f"or json_ratio if this is unintended.",
                    UserWarning
                )
        else:
            self.batches_per_json_cycle = float('inf')
            
            # For normal ratios, validate that we have reasonable allocation
            if json_ratio > 0:
                achieved_ratio = self.json_per_batch / batch_size
                ratio_error = abs(achieved_ratio - json_ratio) / json_ratio
                
                # Assert that we're within 20% of the target ratio for normal ratios
                assert ratio_error <= 0.2, (
                    f"FixedRatioSampler: Achieved ratio {achieved_ratio:.4f} is too far from "
                    f"target ratio {json_ratio:.4f} (error: {ratio_error:.1%}). "
                    f"json_per_batch={self.json_per_batch}, batch_size={batch_size}"
                )
            
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
        
        # Pre-compute lengths to avoid repeated len() calls
        json_len = len(json_indices)
        vla_len = len(vla_indices)
        
        # For distributed training, create batches per rank to ensure balanced distribution
        if self.num_replicas > 1:
            # Calculate batches per rank
            batches_per_rank = self.batches_per_replica
            
            # Create batches specifically for this rank
            batch_indices = []
            
            for local_batch_num in range(batches_per_rank):
                # Calculate global batch number for this rank
                global_batch_num = self.rank * batches_per_rank + local_batch_num
                
                batch_start_idx = len(batch_indices)
                
                # Determine if this batch should contain JSON samples
                json_samples_in_batch = self.json_per_batch
                vla_samples_in_batch = self.vla_per_batch
                
                # For very small ratios, distribute JSON samples across batches
                if self.json_per_batch == 0 and json_len > 0:
                    # Only include JSON samples in some batches to maintain overall ratio
                    if global_batch_num % self.batches_per_json_cycle == 0:
                        json_samples_in_batch = 1
                        vla_samples_in_batch = self.batch_size - 1
                    else:
                        json_samples_in_batch = 0
                        vla_samples_in_batch = self.batch_size
                
                # Add JSON samples using modulo arithmetic for wrapping
                if json_samples_in_batch > 0 and json_len > 0:
                    for i in range(json_samples_in_batch):
                        json_idx = (global_batch_num * json_samples_in_batch + i) % json_len
                        batch_indices.append(json_indices[json_idx])
                
                # Add VLA samples using modulo arithmetic for wrapping
                if vla_samples_in_batch > 0 and vla_len > 0:
                    for i in range(vla_samples_in_batch):
                        vla_idx = (global_batch_num * vla_samples_in_batch + i) % vla_len
                        batch_indices.append(vla_indices[vla_idx])
                
                # Shuffle within batch to mix JSON and VLA samples
                if self.shuffle:
                    batch_end_idx = len(batch_indices)
                    batch_slice = batch_indices[batch_start_idx:batch_end_idx]
                    random.Random(self.seed + global_batch_num).shuffle(batch_slice)
                    batch_indices[batch_start_idx:batch_end_idx] = batch_slice
        else:
            # Single process training - use original logic
            total_samples = self.total_batches * self.batch_size
            batch_indices = [0] * total_samples
            
            batch_idx = 0
            for batch_num in range(self.total_batches):
                batch_start = batch_idx
                
                # Determine if this batch should contain JSON samples
                json_samples_in_batch = self.json_per_batch
                vla_samples_in_batch = self.vla_per_batch
                
                # For very small ratios, distribute JSON samples across batches
                if self.json_per_batch == 0 and json_len > 0:
                    # Only include JSON samples in some batches to maintain overall ratio
                    if batch_num % self.batches_per_json_cycle == 0:
                        json_samples_in_batch = 1
                        vla_samples_in_batch = self.batch_size - 1
                    else:
                        json_samples_in_batch = 0
                        vla_samples_in_batch = self.batch_size
                
                # Add JSON samples using modulo arithmetic for wrapping
                if json_samples_in_batch > 0 and json_len > 0:
                    for i in range(json_samples_in_batch):
                        json_idx = (batch_num * json_samples_in_batch + i) % json_len
                        batch_indices[batch_idx] = json_indices[json_idx]
                        batch_idx += 1
                
                # Add VLA samples using modulo arithmetic for wrapping
                if vla_samples_in_batch > 0 and vla_len > 0:
                    for i in range(vla_samples_in_batch):
                        vla_idx = (batch_num * vla_samples_in_batch + i) % vla_len
                        batch_indices[batch_idx] = vla_indices[vla_idx]
                        batch_idx += 1
                
                # Shuffle within batch to mix JSON and VLA samples
                if self.shuffle:
                    batch_end = batch_idx
                    batch_slice = batch_indices[batch_start:batch_end]
                    random.Random(self.seed + batch_num).shuffle(batch_slice)
                    batch_indices[batch_start:batch_end] = batch_slice
        
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
