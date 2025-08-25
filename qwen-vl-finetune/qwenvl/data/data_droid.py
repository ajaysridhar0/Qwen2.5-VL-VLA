"""
Data module for DROID RLDS dataset with fast tokenizer for Qwen VL finetuning.
This module handles image-language inputs and tokenized action outputs.
"""

import torch
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Any
import json
import random
from PIL import Image
import io
from transformers import AutoProcessor

# Import the DROID dataset loader
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from droid_rlds_dataset import DroidRldsDataset, DroidActionSpace


# TODO: action normalization
# TODO: input state with binning


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

NORM_STATS = json.load(open("/iliad2/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/norm_stats.json"))

def normalize_action(x, input_type="action"):
    q01 = np.array(NORM_STATS["norm_stats"][input_type]["q01"][:8])
    q99 = np.array(NORM_STATS["norm_stats"][input_type]["q99"][:8])
    q01 = q01.reshape(1, 1, -1)
    q99 = q99.reshape(1, 1, -1)
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def denormalize_action(x, input_type="action"):
    q01 = np.array(NORM_STATS["norm_stats"][input_type]["q01"][:8])
    q99 = np.array(NORM_STATS["norm_stats"][input_type]["q99"][:8])
    q01 = q01.reshape(1, 1, -1)
    q99 = q99.reshape(1, 1, -1)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


class DroidVLADataset(Dataset):
    """Dataset for DROID VLA finetuning with Qwen."""
    
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        action_tokenizer_path: str = "physical-intelligence/fast",
        model_max_length: int = 2048,
        token_mappings: Dict = None,
        image_size: tuple = (180, 320),
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.image_processor = data_args.image_processor
        self.token_mappings = token_mappings
        self.image_size = image_size  # Store the reshape size (height, width)
        
        # Load the fast action tokenizer
        rank0_print(f"Loading action tokenizer from {action_tokenizer_path}")
        self.action_tokenizer = AutoProcessor.from_pretrained(
            action_tokenizer_path, 
            trust_remote_code=True
        )
        
        # Initialize DROID dataset
        self.droid_dataset = DroidRldsDataset(
            data_dir=data_args.droid_data_dir,
            dataset_name=getattr(data_args, 'droid_dataset_name', 'droid_100'),
            batch_size=1,  # We handle batching in PyTorch DataLoader
            action_chunk_size=getattr(data_args, 'action_chunk_size', 15),
            action_space=DroidActionSpace.JOINT_VELOCITY if getattr(data_args, 'use_joint_velocity', True) else DroidActionSpace.JOINT_POSITION,
            shuffle_buffer_size=getattr(data_args, 'shuffle_buffer_size', 100000),
            shuffle=True,
        )
        
        # Cache for dataset iteration
        self.data_iter = iter(self.droid_dataset)
        
        # Get action vocabulary size from the tokenizer
        self.action_vocab_size = self.action_tokenizer.vocab_size
        
        rank0_print(f"Initialized DROID VLA dataset with action vocab size: {self.action_vocab_size}")
        
    def __len__(self):
        # Approximate length for DROID dataset
        return getattr(self.data_args, 'num_train_samples', 1000000)
    
    def _process_image(self, image_array):
        """Process numpy image array to PIL Image and resize to specified dimensions."""
        if isinstance(image_array, np.ndarray):
            # Ensure uint8 type
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Resize to specified dimensions (height, width) -> (width, height) for PIL
        if hasattr(self, 'image_size') and self.image_size is not None:
            target_width, target_height = self.image_size[1], self.image_size[0]  # PIL uses (width, height)
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        return image
    
    def __getitem__(self, idx):
        """Get a single training example."""
        try:
            # Get next batch from DROID dataset
            batch = next(self.data_iter)
        except StopIteration:
            # Reinitialize iterator if exhausted
            self.data_iter = iter(self.droid_dataset)
            batch = next(self.data_iter)
        
        # Extract data from batch (batch size is 1)
        image = self._process_image(batch['observation']['image'][0])
        wrist_image = self._process_image(batch['observation']['wrist_image'][0])
        prompt = batch['prompt'][0].decode('utf-8') if isinstance(batch['prompt'][0], bytes) else batch['prompt'][0]
        actions = batch['actions'][0]  # Shape: (action_chunk_size, 8)
        state = np.concatenate([batch["observation"]["joint_position"][0], batch["observation"]["gripper_position"][0]])
        normalized_state = normalize_action(state, input_type="state")
        normalized_actions = normalize_action(actions, input_type="actions")
        
        # digitize state (256 bins -> indices 0..255)
        discretized_state = np.digitize(normalized_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        # Map discretized state to existing Chinese token IDs
        mapped_state_tokens = [self.token_mappings['state_token_ids'][int(idx)] for idx in discretized_state.reshape(-1).tolist()]
        
        # Tokenize actions using the fast tokenizer
        action_tokens = self.action_tokenizer(normalized_actions.reshape(1, -1, normalized_actions.shape[-1]))[0]  # Get tokens for single example

        # Map action tokens to existing Chinese token IDs
        mapped_action_tokens = [self.token_mappings['action_token_ids'][int(token_idx)] for token_idx in action_tokens]
        
        # Process images with Qwen's image processor
        # Use exterior image as primary, wrist image as secondary if needed
        image_inputs = self.image_processor(
            images=[image, wrist_image],
            return_tensors="pt",
        )
        
        # Build vision placeholders matching the number of vision features
        grid_thw = image_inputs["image_grid_thw"]
        merge_size = self.image_processor.merge_size
        image_pad_counts = []
        for thw in grid_thw:
            count = int(torch.prod(thw).item() // (merge_size ** 2))
            image_pad_counts.append(count)

        vision_segments = []
        for count in image_pad_counts:
            vision_segments.append(
                "<|vision_start|>" + ("<|image_pad|>" * count) + "<|vision_end|>"
            )

        user_content = "\n".join(vision_segments) + "\n" + "State:<|state_start|><|state_end|>" + "\n" + (
            f"Task: {prompt}\nWhat action should the robot perform?"
        )

        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "<|action_start|>"},
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Replace control token strings with mapped Chinese characters before tokenization
        text_modified = text
        for control_string, token_id in self.token_mappings['control_mappings'].items():
            if control_string in text_modified:
                # Replace the string with the actual Chinese character that has this token ID
                replacement_token = self.tokenizer.decode([token_id])
                text_modified = text_modified.replace(control_string, replacement_token)
        
        # Tokenize modified text
        text_inputs = self.tokenizer(
            text_modified,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.model_max_length,
        )
        
        # Find position to insert state tokens (between state_start and state_end) and action tokens after action_start
        action_start_token_id = self.token_mappings['action_start_id']
        state_start_token_id = self.token_mappings['state_start_id']
        state_end_token_id = self.token_mappings['state_end_id']
        input_ids = text_inputs.input_ids[0].tolist()
        
        # Find the position between state_start and state_end tokens to insert state tokens
        state_start_pos = None
        state_end_pos = None
        for i, token_id in enumerate(input_ids):
            if token_id == state_start_token_id:
                state_start_pos = i
            elif token_id == state_end_token_id:
                state_end_pos = i
                break
        
        # Insert state tokens between state_start and state_end
        if state_start_pos is not None and state_end_pos is not None and state_end_pos == state_start_pos + 1:
            # state_start and state_end are adjacent, insert state tokens between them
            state_insert_pos = state_start_pos + 1
        else:
            # Fallback: couldn't find proper state token positions
            state_insert_pos = len(input_ids)
        
        # Find the position of action_start token (will be after state tokens)
        action_start_pos = len(input_ids)
        for i, token_id in enumerate(input_ids):
            if token_id == action_start_token_id:
                action_start_pos = i
                break
        
        # Create final input_ids by inserting state tokens between state_start and state_end, and action tokens after action_start
        assert state_start_pos is not None and state_end_pos is not None and state_end_pos == state_start_pos + 1, "State tokens are not adjacent to state_start and state_end tokens"
        # Insert state tokens between state_start and state_end tokens
        final_input_ids = (
            input_ids[:state_insert_pos]  # Everything up to state_start + 1
            + mapped_state_tokens  # Insert state tokens
            + input_ids[state_insert_pos:action_start_pos + 1]  # From state_end to and including action_start
            + mapped_action_tokens  # Insert action tokens
            + [self.tokenizer.eos_token_id]
        )
        
        # Create labels (ignore all user content including state tokens, only train on assistant response)
        # Calculate the position where assistant content starts (after action_start token in final sequence)
        assistant_start_pos = action_start_pos + len(mapped_state_tokens) + 1
        
        labels = (
            [-100] * assistant_start_pos  # Ignore everything up to and including action_start
            + mapped_action_tokens  # Include action tokens for training
            + [self.tokenizer.eos_token_id]
        )
        
        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": image_inputs.pixel_values,
            "image_grid_thw": image_inputs.get("image_grid_thw", torch.tensor([[1, 1, 1], [1, 1, 1]])),
        }


@dataclass
class DroidDataCollator:
    """Data collator for DROID VLA training."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract all fields
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
        
        # Concatenate pixel values and image_grid_thw for multi-image input
        pixel_values = torch.cat([instance["pixel_values"] for instance in instances], dim=0)
        image_grid_thw = torch.cat([torch.as_tensor(instance["image_grid_thw"]) for instance in instances], dim=0)
        
        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def make_droid_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
) -> Dict:
    """Make dataset and collator for DROID VLA fine-tuning."""
    
    # Add DROID-specific args if not present
    if not hasattr(data_args, 'droid_data_dir'):
        data_args.droid_data_dir = "/iliad2/u/ajaysri/episodic_memory/droid_rlds"
    
    train_dataset = DroidVLADataset(
        tokenizer=tokenizer, 
        data_args=data_args,
        model_max_length=model_max_length,
        token_mappings=token_mappings,
        image_size=image_size,
    )
    
    data_collator = DroidDataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
