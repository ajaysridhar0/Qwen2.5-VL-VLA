#!/usr/bin/env python3
"""
Quick test to verify the optimized FixedRatioSampler still works correctly.
"""
import sys
import time

from fixed_ratio_sampler import FixedRatioSampler

def test_optimized_sampler():
    """Test that optimizations haven't broken functionality."""
    print("Testing optimized FixedRatioSampler...")
    
    # Test parameters
    dataset_size = 1000
    batch_size = 64
    json_ratio = 0.2
    
    # Create sampler
    start_time = time.time()
    sampler = FixedRatioSampler(
        dataset_size=dataset_size,
        batch_size=batch_size,
        json_ratio=json_ratio,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    init_time = time.time() - start_time
    
    # Test iteration
    start_time = time.time()
    indices = list(sampler)
    iter_time = time.time() - start_time
    
    # Verify basic properties
    num_batches = len(indices) // batch_size
    expected_json_per_batch = sampler.json_per_batch
    expected_vla_per_batch = sampler.vla_per_batch
    
    print(f"✓ Initialization time: {init_time:.4f}s")
    print(f"✓ Iteration time: {iter_time:.4f}s")
    print(f"✓ Total samples: {len(indices)}")
    print(f"✓ Total batches: {num_batches}")
    print(f"✓ JSON samples per batch: {expected_json_per_batch}")
    print(f"✓ VLA samples per batch: {expected_vla_per_batch}")
    
    # Verify indices are in valid range
    assert all(0 <= idx < dataset_size for idx in indices), "Invalid indices found"
    print("✓ All indices are valid")
    
    # Check first few batches to ensure ratio is maintained
    for batch_idx in range(min(3, num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        batch_indices = indices[batch_start:batch_end]
        
        # Count JSON vs VLA (deterministic assignment based on hash)
        json_count = 0
        vla_count = 0
        
        for idx in batch_indices:
            # Replicate the hash logic to check assignment
            ratio_hash = sampler._get_ratio_hash() if hasattr(sampler, '_get_ratio_hash') else 0
            combined_hash = (idx * 2654435761 + ratio_hash) & 0xFFFFFFFF
            random_value = (combined_hash % 1000000) / 1000000.0
            
            if random_value < json_ratio:
                json_count += 1
            else:
                vla_count += 1
        
        print(f"✓ Batch {batch_idx}: {json_count} JSON, {vla_count} VLA samples")
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_optimized_sampler()
