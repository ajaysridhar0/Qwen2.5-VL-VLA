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
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Any, Union
from PIL import Image
import os

# Import existing components
from .data_droid import DroidVLADataset, DroidDataCollator, normalize_action, denormalize_action

# Import only what we need from data_qwen, avoiding video-related imports
import copy
import transformers
from typing import Dict, Optional, Sequence, List, Tuple
from PIL import Image
import torch

# Import specific functions we need without importing the full module
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656

def read_jsonl(path):
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    """Simplified preprocessing for mixed VLA training."""
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        # Handle case where grid_thw_image is None or empty
                        if grid_thw_image and visual_replicate_index_image < len(grid_thw_image):
                            image_pad_count = grid_thw_image[visual_replicate_index_image]
                        else:
                            # Default fallback for missing image data
                            image_pad_count = 1  # Use minimal padding
                        
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * image_pad_count
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class DataCollatorForSupervisedDataset(object):
    """Simplified collator for mixed training."""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        # Handle images if present - need to flatten properly for mixed data
        images = []
        grid_thws = []
        
        for instance in instances:
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                pixel_vals = instance["pixel_values"]
                grid_thw = instance["image_grid_thw"]
                
                # Handle different tensor shapes from VLA vs JSON data
                if pixel_vals.dim() == 3:  # Single image from JSON data: [C, H, W]
                    images.append(pixel_vals.unsqueeze(0))  # Add batch dim: [1, C, H, W]
                    if grid_thw.dim() == 1:
                        grid_thws.append(grid_thw.unsqueeze(0))  # Add batch dim
                    else:
                        grid_thws.append(grid_thw)
                elif pixel_vals.dim() == 4:  # Batch of images: [N, C, H, W]
                    # Split into individual images
                    for i in range(pixel_vals.shape[0]):
                        images.append(pixel_vals[i:i+1])  # Keep batch dim: [1, C, H, W]
                        if grid_thw.dim() == 2:
                            grid_thws.append(grid_thw[i:i+1])  # Keep batch dim
                        else:
                            grid_thws.append(grid_thw.unsqueeze(0))
                else:
                    # Flatten any other dimensions and add to list
                    images.append(pixel_vals)
                    grid_thws.append(grid_thw)
        
        if len(images) > 0:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(grid_thws, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
            
        return batch


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class MixedVLADataset(Dataset):
    """Mixed dataset combining VLA training with regular conversational data."""
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        action_tokenizer,
        model_max_length: int = 2048,
        token_mappings: Dict = None,
        image_size: tuple = (180, 320),
        cotrain_json_ratio: float = 0.2,  # Percentage of regular JSON data
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.image_processor = data_args.image_processor
        self.token_mappings = token_mappings
        self.image_size = image_size
        self.cotrain_json_ratio = cotrain_json_ratio
        
        # Initialize VLA dataset
        self.vla_dataset = DroidVLADataset(
            tokenizer=tokenizer,
            data_args=data_args,
            action_tokenizer=action_tokenizer,
            model_max_length=model_max_length,
            token_mappings=token_mappings,
            image_size=image_size,
        )
        
        # Initialize regular JSON dataset if co-training is enabled
        self.json_dataset = None
        if hasattr(data_args, 'cotrain_json_paths') and data_args.cotrain_json_paths:
            self.json_dataset = self._load_json_dataset()
            rank0_print(f"Loaded {len(self.json_dataset)} JSON conversation examples for co-training")
        
        # Calculate effective dataset size and sampling ratios
        self.vla_ratio = 1.0 - self.cotrain_json_ratio
        
        # Determine total dataset size based on VLA dataset size
        self.vla_size = len(self.vla_dataset)
        if self.json_dataset:
            # Calculate how many JSON examples we need based on the ratio
            self.json_size = len(self.json_dataset)
            
            # Handle special case where vla_ratio is 0 (100% JSON data, 0% VLA data)
            if self.vla_ratio == 0.0:
                # Use only JSON data
                self.total_size = self.json_size
                self.effective_json_size = self.json_size
                rank0_print(f"JSON-only dataset: {self.effective_json_size} examples (100% JSON, 0% VLA)")
            else:
                # Normal mixed case
                self.total_size = int(self.vla_size / self.vla_ratio)
                self.effective_json_size = self.total_size - self.vla_size
                rank0_print(f"Mixed dataset: VLA={self.vla_size}, JSON={self.effective_json_size} (ratio={self.vla_ratio:.2f}:{self.cotrain_json_ratio:.2f})")
        else:
            self.total_size = self.vla_size
            self.effective_json_size = 0
            rank0_print(f"VLA-only dataset: {self.vla_size} examples")
    
    def _load_json_dataset(self):
        """Load and prepare JSON conversational data."""
        json_data = []
        
        # Handle multiple JSON file paths
        json_paths = self.data_args.cotrain_json_paths
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        for json_path in json_paths:
            if not os.path.exists(json_path):
                rank0_print(f"Warning: JSON file not found: {json_path}")
                continue
                
            # Load JSON or JSONL data
            file_format = json_path.split(".")[-1]
            if file_format == "jsonl":
                data = read_jsonl(json_path)
            else:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            
            # Add data path information
            for item in data:
                if 'data_path' not in item:
                    item['data_path'] = os.path.dirname(json_path)
            
            json_data.extend(data)
            rank0_print(f"Loaded {len(data)} examples from {json_path}")
        
        return json_data
    
    def __len__(self):
        return self.total_size
    
    def _get_json_item(self, idx):
        """Get a processed JSON conversational example."""
        if not self.json_dataset:
            raise ValueError("JSON dataset not loaded")
        
        # Sample from JSON dataset with wraparound
        json_idx = idx % len(self.json_dataset)
        json_item = self.json_dataset[json_idx]
        
        # Process the JSON item similar to LazySupervisedDataset
        sources = [json_item]
        
        # Handle image processing if present (check both "image" and "video" fields)
        image_file = None
        if "image" in json_item:
            image_file = json_item["image"]
        elif "video" in json_item:
            # convert_vla_format.py puts media files under "video" key
            image_file = json_item["video"]
        
        if image_file is not None:
            image_folder = json_item.get("data_path", "")
            
            if isinstance(image_file, list):
                if len(image_file) > 1:
                    # Multiple images
                    image_files = [os.path.join(image_folder, f) for f in image_file]
                    images = []
                    grid_thws = []
                    for img_file in image_files:
                        if os.path.exists(img_file):
                            img, thw = self._process_image_unified(img_file)
                            images.append(img)
                            if thw.dim() == 1:
                                thw = thw.unsqueeze(0)
                            grid_thws.append(thw)
                    if images:
                        # Concatenate 2D vision features along the patch dimension
                        pixel_values = torch.cat(images, dim=0)
                        image_grid_thw = torch.cat(grid_thws, dim=0)
                    else:
                        pixel_values = None
                        image_grid_thw = None
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    if os.path.exists(image_file):
                        pixel_values, image_grid_thw = self._process_image_unified(image_file)
                        # Keep pixel_values as 2D vision features
                        # Ensure grid_thw has batch dimension
                        if image_grid_thw.dim() == 1:
                            image_grid_thw = image_grid_thw.unsqueeze(0)
                    else:
                        pixel_values = None
                        image_grid_thw = None
            else:
                image_file = os.path.join(image_folder, image_file)
                if os.path.exists(image_file):
                    pixel_values, image_grid_thw = self._process_image_unified(image_file)
                    # Keep pixel_values as 2D vision features
                    # Ensure grid_thw has batch dimension
                    if image_grid_thw.dim() == 1:
                        image_grid_thw = image_grid_thw.unsqueeze(0)
                else:
                    pixel_values = None
                    image_grid_thw = None
        else:
            pixel_values = None
            image_grid_thw = None
        
        # Calculate grid_thw_merged for preprocessing
        grid_thw_merged = None
        if image_grid_thw is not None:
            grid_thw_merged = [
                thw.prod() // self.data_args.image_processor.merge_size**2
                for thw in image_grid_thw
            ]
        
        # Preprocess conversations
        chat_sources = copy.deepcopy([json_item["conversations"]])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged,
            grid_thw_video=None,
        )
        
        # Remove batch dimension from tensors to match VLA data format
        data_dict["input_ids"] = data_dict["input_ids"].squeeze(0)
        data_dict["labels"] = data_dict["labels"].squeeze(0)
        
        # Add image data if present
        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        
        return data_dict
    
    def _process_image_unified(self, image_file):
        """Process a single image file for JSON data."""
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        
        # Debug: check the shape
        # print(f"_process_image_unified: image_tensor shape after processing: {image_tensor.shape}")
        
        # The Qwen processor returns 2D vision features [num_patches, feature_dim]
        # Keep them as-is since that's what the model expects
            
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def __getitem__(self, idx):
        """Get a mixed training example - either VLA or JSON based on ratio."""
        
        # Handle the case where vla_ratio is 0 (100% JSON data, 0% VLA data)
        if self.vla_ratio == 0.0:
            # Always return JSON data since there's no VLA data to mix with
            if self.json_dataset:
                try:
                    return self._get_json_item(idx)
                except Exception as e:
                    rank0_print(f"Error loading JSON item {idx}: {e}")
                    # If we can't load JSON and have no VLA data, this is a critical error
                    raise RuntimeError(f"Cannot load JSON data and no VLA data available as fallback: {e}")
            else:
                raise RuntimeError("No JSON dataset loaded but vla_ratio is 0 - this should not happen")
        
        # Normal mixed case - decide whether to return VLA or JSON data based on ratio
        # Use deterministic sampling based on index to ensure reproducibility
        if self.json_dataset:
            # Use a simple hash-based deterministic approach
            # This ensures the same idx always returns the same data type
            import hashlib
            hash_input = f"{idx}_{self.cotrain_json_ratio}".encode()
            hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
            random_value = (hash_value % 1000000) / 1000000.0  # Convert to 0-1 range
            
            if random_value < self.cotrain_json_ratio:
                # Return JSON conversational data
                return self._get_json_item(idx)
        
        # Return VLA data (either no JSON dataset or deterministic choice selected VLA)
        vla_idx = idx % self.vla_size
        return self.vla_dataset[vla_idx]


@dataclass
class MixedVLADataCollator:
    """Data collator for mixed VLA and JSON training."""
    
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int = 2048
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels, ensuring they are 1D tensors
        input_ids = []
        labels = []
        
        for instance in instances:
            input_id = instance["input_ids"]
            label = instance["labels"]
            
            # Ensure tensors are 1D
            if input_id.dim() > 1:
                input_id = input_id.squeeze()
            if label.dim() > 1:
                label = label.squeeze()
                
            input_ids.append(input_id)
            labels.append(label)
        
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
        
        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        
        # Handle images - collect all pixel values and grid_thw
        all_pixel_values = []
        all_grid_thw = []
        
        for instance in instances:
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                pixel_vals = instance["pixel_values"]
                grid_thw = instance["image_grid_thw"]
                
                # Debug print
                # print(f"Pixel values shape: {pixel_vals.shape}, dim: {pixel_vals.dim()}")
                
                # Qwen models use 2D vision features, not raw pixel values
                # The shapes are [num_patches, feature_dim] for processed vision tokens
                # We need to handle them as-is without adding dimensions
                
                # For consistency in concatenation, keep all as separate items
                # The model expects these 2D vision features directly
                
                # Ensure grid_thw is 2D: [N, 3]
                if grid_thw.dim() == 1:  # [3] -> [1, 3]
                    grid_thw = grid_thw.unsqueeze(0)
                elif grid_thw.dim() == 2:  # [N, 3] - already correct
                    pass
                else:
                    print(f"Warning: Unexpected grid_thw dimensions: {grid_thw.shape}")
                    # Try to reshape to [N, 3]
                    if grid_thw.numel() % 3 == 0:
                        grid_thw = grid_thw.reshape(-1, 3)
                    else:
                        continue
                
                all_pixel_values.append(pixel_vals)
                all_grid_thw.append(grid_thw)
        
        if all_pixel_values:
            # Concatenate vision features and grid_thw
            # The pixel_values are 2D vision features from Qwen processor
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        
        return batch


def make_mixed_vla_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    cotrain_json_ratio: float = 0.2,
    create_eval_dataset: bool = True,
) -> Dict:
    """Make dataset and collator for mixed VLA + JSON co-training."""
    
    train_dataset = MixedVLADataset(
        tokenizer=tokenizer,
        action_tokenizer=action_tokenizer,
        data_args=data_args,
        model_max_length=model_max_length,
        token_mappings=token_mappings,
        image_size=image_size,
        cotrain_json_ratio=cotrain_json_ratio,
    )
    
    # Create a small eval dataset with 100 samples
    eval_dataset = None
    if create_eval_dataset:
        eval_size = 100  # Fixed 100 samples for eval
        
        eval_dataset = MixedVLADataset(
            tokenizer=tokenizer,
            action_tokenizer=action_tokenizer,
            data_args=data_args,
            model_max_length=model_max_length,
            token_mappings=token_mappings,
            image_size=image_size,
            cotrain_json_ratio=cotrain_json_ratio,
        )
        
        # Override length and total_size for eval dataset
        eval_dataset.total_size = eval_size
        eval_dataset.__len__ = lambda: eval_size
        
        # Print eval dataset configuration
        if train_dataset.json_dataset:
            if train_dataset.vla_ratio == 0.0:
                print(f"Eval dataset configured: JSON-only={eval_size}")
            else:
                print(f"Eval dataset configured: Mixed={eval_size} (VLA ratio={train_dataset.vla_ratio:.2f}, JSON ratio={train_dataset.cotrain_json_ratio:.2f})")
        else:
            print(f"Eval dataset configured: VLA-only={eval_size}")
    
    data_collator = MixedVLADataCollator(
        tokenizer=tokenizer,
        model_max_length=model_max_length
    )
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
