"""
Data module for DROID RLDS dataset with fast tokenizer for Qwen VL finetuning.
This module handles image-language inputs and tokenized action outputs.
"""

import torch
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader, IterableDataset
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
from droid_rlds_dataset_iterable import DroidRldsDatasetStateful, DroidActionSpace

# Import rope position embedding functions
from .rope2d import get_rope_index_25, get_rope_index_2


# TODO: action normalization
# TODO: input state with binning


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)

NORM_STATS = json.load(open("./norm_stats.json"))

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


class DroidVLADatasetIterable(IterableDataset):
    """
    An IterableDataset for DROID VLA finetuning.
    It wraps the TorchDroidDataset and applies model-specific processing.
    """
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer,
        data_args, # This should contain parameters like data_dir, batch_size, etc.
        action_tokenizer,
        model_max_length: int = 2048,
        token_mappings: Dict = None,
        shuffle_buffer_size_override: int = None,
        data_size: int = None,
        samples_to_skip: int = 0,
        seed: int = 42,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__()
        
        # --- Store all configurations ---
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.action_tokenizer = action_tokenizer
        self.model_max_length = model_max_length
        self.token_mappings = token_mappings
        self.image_processor = data_args.image_processor
        self.shuffle_buffer_size_override = shuffle_buffer_size_override
        self.data_size = data_size
        self.samples_to_skip = samples_to_skip
        self.seed = seed
        self.batch_size = batch_size
        # Set up rope index function based on model type
        if hasattr(data_args, 'model_type') and data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
        
        rank0_print("Initialized DROID VLA IterableDataset.")

    def _process_image(self, image_array):
        # Your image processing logic remains the same
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            return Image.fromarray(image_array)
        return image_array

    def __iter__(self):
        """
        The core of the IterableDataset. This creates the data stream.
        """
        # 1. Instantiate the base iterable dataset INSIDE __iter__.
        # This is critical for multi-worker loading.
        # It ensures each worker gets its own independent data stream.
        shuffle_buffer_size = self.shuffle_buffer_size_override if self.shuffle_buffer_size_override is not None else getattr(self.data_args, 'shuffle_buffer_size', 100000)
        base_droid_dataset = DroidRldsDatasetStateful(
            data_dir=self.data_args.droid_data_dir,
            dataset_name=getattr(self.data_args, 'droid_dataset_name', 'droid_100'),
            batch_size=self.batch_size,  # Use efficient TensorFlow batching
            action_chunk_size=getattr(self.data_args, 'action_chunk_size', 15),
            action_space=DroidActionSpace.JOINT_VELOCITY if getattr(self.data_args, 'use_joint_velocity', True) else DroidActionSpace.JOINT_POSITION,
            shuffle_buffer_size=shuffle_buffer_size,
            data_size=self.data_size,
            samples_to_skip=self.samples_to_skip,
            seed=self.seed,
        )
        rank0_print(f"Shuffle buffer size potato: {shuffle_buffer_size}")

        # 2. Loop through the base dataset and apply your processing logic.
        base_droid_iterator = iter(base_droid_dataset)
        for batch in base_droid_iterator:
            # --- Process all samples in the batch efficiently ---
            batch_size_actual = len(batch['observation']['image'])
            
            # Process each sample in the batch individually and yield them
            for i in range(batch_size_actual):
                # Extract data from batch sample i
                image = self._process_image(batch['observation']['image'][i])
                wrist_image = self._process_image(batch['observation']['wrist_image'][i])
                prompt = batch['prompt'][i].decode('utf-8') if isinstance(batch['prompt'][i], bytes) else batch['prompt'][i]
                actions = batch['actions'][i]
                state = np.concatenate([batch["observation"]["joint_position"][i], batch["observation"]["gripper_position"][i]])
                
                # --- Action & State Tokenization ---
                normalized_state = normalize_action(state, input_type="state")
                normalized_actions = normalize_action(actions, input_type="actions")
                discretized_state = np.digitize(normalized_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                mapped_state_tokens = [self.token_mappings['state_token_ids'][int(idx)] for idx in discretized_state.reshape(-1).tolist()]
                action_tokens = self.action_tokenizer(normalized_actions.reshape(1, -1, normalized_actions.shape[-1]))[0]
                mapped_action_tokens = [self.token_mappings['action_token_ids'][int(token_idx)] for token_idx in action_tokens]
                
                # --- Image Processing ---
                image_inputs = self.image_processor(images=[image, wrist_image], return_tensors="pt")
                
                # --- Text and Prompt Formatting (exactly as before) ---
                grid_thw = image_inputs["image_grid_thw"]
                merge_size = self.image_processor.merge_size
                vision_segments = []
                for thw in grid_thw:
                    count = int(torch.prod(thw).item() // (merge_size ** 2))
                    vision_segments.append("<|vision_start|>" + ("<|image_pad|>" * count) + "<|vision_end|>")
                user_content = (
                    f"Task: {prompt}\n"
                    f"State:<|state_start|><|state_end|>\n"
                    f"What action should the robot perform?\n" + "\n".join(vision_segments)
                )
                conversation = [
                    {"role": "system", "content": "You are a robot policy model..."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": "<|action_start|>"},
                ]
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                
                # ... [the rest of your text processing, token mapping, and final tensor creation logic] ...
                # This part is identical to your __getitem__ implementation.
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
            
                # Replace control token strings with mapped infrequent characters before tokenization
                text_modified = text
                for control_string, token_id in self.token_mappings['control_mappings'].items():
                    if control_string in text_modified:
                        # Replace the string with the actual infrequent character that has this token ID
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
                for idx, token_id in enumerate(input_ids):
                    if token_id == state_start_token_id:
                        state_start_pos = idx
                    elif token_id == state_end_token_id:
                        state_end_pos = idx
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
                for idx, token_id in enumerate(input_ids):
                    if token_id == action_start_token_id:
                        action_start_pos = idx
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
            
                # Calculate position_ids using rope position embedding
                input_ids_tensor = torch.tensor(final_input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
                image_grid_thw = image_inputs.get("image_grid_thw", torch.tensor([[1, 1, 1], [1, 1, 1]]))
                
                position_ids, _ = self.get_rope_index(
                    self.image_processor.merge_size,
                    input_ids_tensor,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=None,
                    second_per_grid_ts=None,
                )
            
                # Remove batch dimension from position_ids to match other tensors
                position_ids = position_ids.squeeze(1) if position_ids.dim() > 3 else position_ids
                
                # 3. Instead of 'return', we 'yield' the final dictionary.
                yield {
                    "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "position_ids": position_ids,
                    "pixel_values": image_inputs.pixel_values,
                    "image_grid_thw": image_grid_thw,
                }

@dataclass
class DroidDataCollator:
    """Data collator for DROID VLA training."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        # Handle both single instances and lists of instances
        if isinstance(instances, dict):
            # Single instance case (when batch_size=None and dataset yields pre-batched data)
            instances = [instances]
        elif not isinstance(instances, (list, tuple)):
            raise TypeError(f"Expected dict or list/tuple of dicts, got {type(instances)}: {instances}")
        
        # Extract all fields
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]
        
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
        
        # Handle position_ids padding using the same approach as mixed VLA dataset
        from .data_qwen import pad_and_cat
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        
        # Concatenate pixel values and image_grid_thw for multi-image input
        pixel_values = torch.cat([instance["pixel_values"] for instance in instances], dim=0)
        image_grid_thw = torch.cat([torch.as_tensor(instance["image_grid_thw"]) for instance in instances], dim=0)
        
        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def make_droid_data_iterable_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    batch_size: int = 1,
    **kwargs,
) -> Dict:
    """Make dataset and collator for DROID VLA fine-tuning."""
    
    # Add DROID-specific args if not present
    if not hasattr(data_args, 'droid_data_dir'):
        data_args.droid_data_dir = "/iliad2/u/ajaysri/episodic_memory/droid_rlds"
    
    train_dataset = DroidVLADatasetIterable(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args,
        model_max_length=model_max_length,
        token_mappings=token_mappings,
        batch_size=batch_size,
        **kwargs,
    )
    
    data_collator = DroidDataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        train_sampler_params=None,
    )

