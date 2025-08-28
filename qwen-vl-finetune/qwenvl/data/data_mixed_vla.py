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
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Any, Union
from PIL import Image
import os

# Import existing components
from .data_droid import DroidVLADataset, DroidDataCollator, normalize_action, denormalize_action

# Import the original preprocessing function that handles both image and video tokens
from .data_qwen import (
    preprocess_qwen_2_visual, 
    IGNORE_INDEX, 
    IMAGE_TOKEN_INDEX, 
    VIDEO_TOKEN_INDEX,
    pad_and_cat,  # For padding position_ids
)
from .rope2d import get_rope_index_25, get_rope_index_2

def read_jsonl(path):
    """Read JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def rank0_print(*args):
    """Print only on rank 0 for distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def resize_image_with_constraints(image, max_dim=320, min_dim=28):
    """
    Resize an image such that:
    - The maximum dimension (width or height) does not exceed max_dim
    - BOTH dimensions (width AND height) are STRICTLY GREATER than min_dim (required by Qwen2VL)
    - Aspect ratio is preserved when possible, but can be modified if needed to meet constraints
    
    Args:
        image: PIL Image to resize
        max_dim: Maximum allowed dimension 
        min_dim: Minimum required dimension for BOTH width and height (must be strictly greater than this)
        
    Returns:
        PIL Image: Resized image
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")
        
    width, height = image.size
    
    # Calculate the scaling factor based on max dimension constraint
    max_scale = min(max_dim / width, max_dim / height)
    
    # Calculate new dimensions with max constraint (preserving aspect ratio)
    new_width = int(width * max_scale)
    new_height = int(height * max_scale)
    
    # Check if both dimensions meet minimum requirement (STRICTLY GREATER than min_dim)
    if new_width <= min_dim or new_height <= min_dim:
        # Check if we can fix this by scaling up while preserving aspect ratio
        min_scale_width = (min_dim + 1) / new_width if new_width <= min_dim else 1.0
        min_scale_height = (min_dim + 1) / new_height if new_height <= min_dim else 1.0
        min_scale = max(min_scale_width, min_scale_height)
        
        scaled_width = int(new_width * min_scale)
        scaled_height = int(new_height * min_scale)
        
        # If scaling up to meet min constraint violates max constraint, 
        # we have a conflict and need to modify aspect ratio
        if max(scaled_width, scaled_height) > max_dim:
            # Clamp dimensions to meet both constraints, potentially changing aspect ratio
            new_width = min(max(new_width, min_dim + 1), max_dim)
            new_height = min(max(new_height, min_dim + 1), max_dim)
            
            # Ensure both dimensions are STRICTLY GREATER than min_dim (critical for Qwen2VL)
            if new_width <= min_dim:
                new_width = min_dim + 1
            if new_height <= min_dim:
                new_height = min_dim + 1
        else:
            # No conflict, use the scaled dimensions that preserve aspect ratio
            new_width = scaled_width
            new_height = scaled_height
    
    # Only resize if dimensions actually changed
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


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
        
        # Create a helper instance to access image/video processing methods from data_qwen.py
        # This ensures we use the tested functions that work properly with gradient checkpointing
        self._qwen_dataset_helper = self._create_qwen_helper(tokenizer, data_args)
        
        # Initialize VLA dataset
        self.vla_dataset = DroidVLADataset(
            tokenizer=tokenizer,
            data_args=data_args,
            action_tokenizer=action_tokenizer,
            model_max_length=model_max_length,
            token_mappings=token_mappings,
            image_size=image_size,
        )

        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
        
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
    
    def _create_qwen_helper(self, tokenizer, data_args):
        """Create a minimal helper object with just the image/video processing methods."""
        class QwenProcessingHelper:
            def __init__(self, tokenizer, data_args):
                self.tokenizer = tokenizer
                self.data_args = data_args
                
                # Set up rope index function based on model type
                if hasattr(data_args, 'model_type') and data_args.model_type == "qwen2.5vl":
                    from .rope2d import get_rope_index_25
                    self.get_rope_index = get_rope_index_25
                else:
                    from .rope2d import get_rope_index_2
                    self.get_rope_index = get_rope_index_2
                
                # Video processing parameters
                self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
                self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
                
                # Image resize constraints
                self.max_image_dim = getattr(data_args, "max_image_dim", 320)
                self.min_image_dim = getattr(data_args, "min_image_dim", 28)
            
            def process_image_unified(self, image_file):
                """Process a single image file."""
                processor = copy.deepcopy(self.data_args.image_processor)
                image = Image.open(image_file).convert("RGB")
                
                # Apply resize constraints before preprocessing
                image = resize_image_with_constraints(
                    image, 
                    max_dim=self.max_image_dim, 
                    min_dim=self.min_image_dim
                )
                
                visual_processed = processor.preprocess(image, return_tensors="pt")
                image_tensor = visual_processed["pixel_values"]
                if isinstance(image_tensor, list):
                    image_tensor = image_tensor[0]
                grid_thw = visual_processed["image_grid_thw"][0]
                return image_tensor, grid_thw
            
            def process_video_from_frames(self, frame_paths):
                """Process video from a list of frame image paths."""
                # Load frames
                frames = []
                for frame_path in frame_paths:
                    try:
                        img = Image.open(frame_path).convert("RGB")
                        # Apply resize constraints to each frame
                        img = resize_image_with_constraints(
                            img, 
                            max_dim=self.max_image_dim, 
                            min_dim=self.min_image_dim
                        )
                        frames.append(np.array(img))
                    except Exception as e:
                        print(f"Failed to load frame {frame_path}: {e}")
                        continue
                
                if not frames:
                    raise ValueError("No frames could be loaded")
                
                # Convert to numpy array with shape (num_frames, height, width, channels)
                video = np.stack(frames)
                
                # Determine frame sampling
                total_frames = len(frames)
                video_min_frames = getattr(self.data_args, "video_min_frames", 4)
                video_max_frames = getattr(self.data_args, "video_max_frames", 8)
                
                # Sample frames if needed
                if total_frames > video_max_frames:
                    frame_idx = np.linspace(0, total_frames - 1, video_max_frames, dtype=int)
                    frame_idx = np.unique(frame_idx)
                    video = video[frame_idx]
                else:
                    frame_idx = np.arange(total_frames)
                
                # Assume 1 fps for frame sequences (can be adjusted)
                video_length = len(frame_idx)
                
                return self.process_video_frames(video, frame_idx, video_length)
            
            def process_video_frames(self, video, frame_idx, video_length):
                """Process video frames using the video processor."""
                fps = len(frame_idx) / video_length
                processor = copy.deepcopy(self.data_args.image_processor)
                processor.max_pixels = getattr(self.data_args, 'video_max_frame_pixels', self.data_args.image_processor.max_pixels)
                processor.min_pixels = getattr(self.data_args, 'video_min_frame_pixels', self.data_args.image_processor.min_pixels)
                processor.size["longest_edge"] = processor.max_pixels
                processor.size["shortest_edge"] = processor.min_pixels
                
                video_processed = processor.preprocess(
                    images=None, videos=video, return_tensors="pt"
                )
                video_tensor = video_processed["pixel_values_videos"]
                grid_thw = video_processed["video_grid_thw"][0]
                second_per_grid_ts = [
                    self.data_args.image_processor.temporal_patch_size / fps
                ] * len(grid_thw)
                return video_tensor, grid_thw, second_per_grid_ts
        
        return QwenProcessingHelper(tokenizer, data_args)
    
    def __len__(self):
        return self.total_size
    
    def _get_json_item(self, idx):
        """Get a processed JSON conversational example.
        
        Supports mixed media: JSON items can contain both 'image' and 'video' fields.
        Images are processed as image data, videos are processed as video data.
        """
        if not self.json_dataset:
            raise ValueError("JSON dataset not loaded")
        
        # Sample from JSON dataset with wraparound
        json_idx = idx % len(self.json_dataset)
        json_item = self.json_dataset[json_idx]
        
        # Process the JSON item similar to LazySupervisedDataset
        sources = [json_item]

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_file = json_item["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = json_item["video"]
            if not isinstance(video_file, List):
                video_file = [video_file]
            # else:
                # if len(video_file) > 1:
            results = [self.process_video_from_frames(video_file)]
            video, video_grid_thw, second_per_grid_ts = zip(*results)
                # else:
                #     video_file = video_file[0]
                #     video, video_grid_thw, second_per_grid_ts = self.process_video(
                #         video_file
                #     )
                #     video = [video]
            # else:
            #     video, video_grid_thw, second_per_grid_ts = self.process_video(
            #         video_file
            #     )
            #     video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in json_item:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        if "video" in json_item:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        return data_dict

    def process_image_unified(self, image_file):
        """Process a single image file for JSON data using the tested function from data_qwen.py."""
        return self._qwen_dataset_helper.process_image_unified(image_file)
    
    def process_video_from_frames(self, frame_paths):
        """Process video from a list of frame image paths using the tested function from data_qwen.py."""
        # Construct full paths
        return self._qwen_dataset_helper.process_video_from_frames(frame_paths)
    
    def process_video_frames(self, video, frame_idx, video_length):
        """Process video frames using the tested function from data_qwen.py."""
        return self._qwen_dataset_helper.process_video_frames(video, frame_idx, video_length)
    
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
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch

        # # Extract input_ids, labels, and position_ids, ensuring they are proper tensors
        # input_ids = []
        # labels = []
        # position_ids = []
        # has_position_ids = "position_ids" in instances[0]
        
        # for instance in instances:
        #     input_id = instance["input_ids"]
        #     label = instance["labels"]
            
        #     # Ensure tensors are 1D
        #     if input_id.dim() > 1:
        #         input_id = input_id.squeeze()
        #     if label.dim() > 1:
        #         label = label.squeeze()
                
        #     input_ids.append(input_id)
        #     labels.append(label)
            
        #     # Handle position_ids if present
        #     if has_position_ids:
        #         pos_id = instance.get("position_ids")
        #         if pos_id is not None:
        #             position_ids.append(pos_id)
        
        # # Pad sequences
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id
        # )
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     labels,
        #     batch_first=True,
        #     padding_value=-100
        # )
        
        # # Create attention mask
        # attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # batch = {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "attention_mask": attention_mask,
        # }
        
        # # Handle position_ids if present
        # if has_position_ids and position_ids:
        #     # Use pad_and_cat from data_qwen.py for proper position_ids padding
        #     position_ids = pad_and_cat(position_ids)
        #     # Ensure position_ids matches the sequence length of input_ids
        #     seq_len = input_ids.shape[1]
        #     if position_ids.shape[2] > seq_len:
        #         # Truncate to match input_ids sequence length
        #         position_ids = position_ids[:, :, :seq_len]
        #     elif position_ids.shape[2] < seq_len:
        #         # Pad to match input_ids sequence length
        #         pad_size = seq_len - position_ids.shape[2]
        #         position_ids = torch.nn.functional.pad(position_ids, (0, pad_size), "constant", 1)
        #     batch["position_ids"] = position_ids
        
        # # Handle images and videos - collect all pixel values and grid_thw
        # all_pixel_values = []
        # all_grid_thw = []
        # all_video_values = []
        # all_video_grid_thw = []
        
        # for instance in instances:
        #     # Handle image data (can coexist with video data)
        #     if "pixel_values" in instance and instance["pixel_values"] is not None:
        #         pixel_vals = instance["pixel_values"]
        #         grid_thw = instance["image_grid_thw"]
                
        #         # Ensure grid_thw is 2D: [N, 3]
        #         if grid_thw.dim() == 1:  # [3] -> [1, 3]
        #             grid_thw = grid_thw.unsqueeze(0)
        #         elif grid_thw.dim() == 2:  # [N, 3] - already correct
        #             pass
        #         else:
        #             print(f"Warning: Unexpected image grid_thw dimensions: {grid_thw.shape}")
        #             # Try to reshape to [N, 3]
        #             if grid_thw.numel() % 3 == 0:
        #                 grid_thw = grid_thw.reshape(-1, 3)
        #             else:
        #                 continue
                
        #         all_pixel_values.append(pixel_vals)
        #         all_grid_thw.append(grid_thw)
            
        #     # Handle video data (can coexist with image data)
        #     if "pixel_values_videos" in instance and instance["pixel_values_videos"] is not None:
        #         video_vals = instance["pixel_values_videos"]
        #         video_grid_thw = instance["video_grid_thw"]
                
        #         # Ensure video_grid_thw is 2D: [N, 3]
        #         if video_grid_thw.dim() == 1:  # [3] -> [1, 3]
        #             video_grid_thw = video_grid_thw.unsqueeze(0)
        #         elif video_grid_thw.dim() == 2:  # [N, 3] - already correct
        #             pass
        #         else:
        #             print(f"Warning: Unexpected video grid_thw dimensions: {video_grid_thw.shape}")
        #             # Try to reshape to [N, 3]
        #             if video_grid_thw.numel() % 3 == 0:
        #                 video_grid_thw = video_grid_thw.reshape(-1, 3)
        #             else:
        #                 continue
                
        #         all_video_values.append(video_vals)
        #         all_video_grid_thw.append(video_grid_thw)
        
        # # Set image data
        # if all_pixel_values:
        #     batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
        #     batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)
        # else:
        #     batch["pixel_values"] = None
        #     batch["image_grid_thw"] = None
        
        # # Set video data
        # if all_video_values:
        #     batch["pixel_values_videos"] = torch.cat(all_video_values, dim=0)
        #     batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
        # else:
        #     batch["pixel_values_videos"] = None
        #     batch["video_grid_thw"] = None
        
        # return batch


def make_mixed_vla_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer,
    data_args,
    model_max_length: int,
    token_mappings: Dict = None,
    image_size: tuple = (180, 320),
    cotrain_json_ratio: float = 0.2,
    create_eval_dataset: bool = True,
    use_fixed_ratio_sampler: bool = True,  # New parameter to enable fixed ratio sampling
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
    
    # Create custom sampler for fixed ratio batching if using mixed training
    train_sampler = None
    if use_fixed_ratio_sampler and train_dataset.json_dataset and 0 < cotrain_json_ratio < 1:
        from .fixed_ratio_sampler import DistributedFixedRatioSampler
        
        # Get batch size from training args (will be set by trainer)
        # For now, we'll return the sampler class and params to be instantiated by trainer
        train_sampler_params = {
            "dataset_size": len(train_dataset),
            "json_ratio": cotrain_json_ratio,
            "sampler_class": DistributedFixedRatioSampler,
        }
        print(f"Fixed ratio sampler configured with {cotrain_json_ratio:.2f} JSON ratio per batch")
    else:
        train_sampler_params = None
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        train_sampler_params=train_sampler_params,  # Return sampler params instead of instantiated sampler
    )
