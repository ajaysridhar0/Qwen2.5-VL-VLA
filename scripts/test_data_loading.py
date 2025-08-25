"""
Test script to verify DROID data loading and tokenization works correctly.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoProcessor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_vl_finetune.qwenvl.data.data_droid import DroidVLADataset, make_droid_data_module
from qwen_vl_finetune.qwenvl.train.argument import DataArguments
from dataclasses import dataclass, field


@dataclass
class TestDataArguments(DataArguments):
    """Test data arguments with DROID-specific fields."""
    droid_data_dir: str = field(default="/iliad2/u/ajaysri/episodic_memory/droid_rlds")
    droid_dataset_name: str = field(default="droid_100")
    action_chunk_size: int = field(default=15)
    use_joint_velocity: bool = field(default=True)
    shuffle_buffer_size: int = field(default=1000)
    num_train_samples: int = field(default=100)
    model_max_length: int = field(default=2048)


def test_data_loading():
    """Test the DROID data loading pipeline."""
    print("Setting up test...")
    
    # Load Qwen tokenizer
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add action special tokens
    special_tokens = ["<|action_start|>", "<|action_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load image processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Create data arguments
    data_args = TestDataArguments()
    data_args.image_processor = processor.image_processor
    
    print(f"Creating dataset with data from: {data_args.droid_data_dir}")
    
    # Create data module
    data_module = make_droid_data_module(tokenizer, data_args)
    dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]
    
    print(f"Dataset created with approximate length: {len(dataset)}")
    
    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    
    # Decode some tokens to see what they look like
    input_text = tokenizer.decode(sample['input_ids'][:50])
    print(f"\nFirst 50 tokens decoded: {input_text}")
    
    # Check action tokens
    action_token_start = 102500
    action_tokens = [t.item() for t in sample['input_ids'] if t >= action_token_start and t < action_token_start + 512]
    print(f"\nNumber of action tokens in sequence: {len(action_tokens)}")
    if action_tokens:
        print(f"First few action tokens (offset from {action_token_start}): {[t - action_token_start for t in action_tokens[:5]]}")
    
    # Test batch collation
    print("\nTesting batch collation...")
    batch_samples = [dataset[i] for i in range(4)]
    batch = collator(batch_samples)
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
    
    # Test data loader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
        num_workers=0,  # Use 0 for testing
    )
    
    # Get one batch
    for batch in dataloader:
        print(f"DataLoader batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        break
    
    print("\n✅ Data loading test completed successfully!")


if __name__ == "__main__":
    test_data_loading()
