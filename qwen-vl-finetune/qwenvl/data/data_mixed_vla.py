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
from .data_qwen import preprocess_qwen_2_visual, IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX

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
        
        # Process media by type (simplified since order is lost anyway)
        image_folder = json_item.get("data_path", "")
        
        image_tensors = []
        image_thws = []
        video_tensors = []
        video_thws = []
        
        # Process images if present
        if "image" in json_item:
            image_files = json_item["image"]
            if not isinstance(image_files, list):
                image_files = [image_files]
            
            for image_file in image_files:
                full_path = os.path.join(image_folder, image_file)
                if os.path.exists(full_path):
                    pixel_vals, grid_thw = self._process_image_unified(full_path)
                    # Ensure grid_thw has batch dimension
                    if grid_thw.dim() == 1:
                        grid_thw = grid_thw.unsqueeze(0)
                    image_tensors.append(pixel_vals)
                    image_thws.append(grid_thw)
        
        # Process videos if present
        if "video" in json_item:
            video_files = json_item["video"]
            if not isinstance(video_files, list):
                video_files = [video_files]
            
            # Process as video (multiple frames)
            if len(video_files) > 1:
                pixel_vals, grid_thw = self._process_video_from_frames(video_files, image_folder)
                video_tensors.append(pixel_vals)
                video_thws.append(grid_thw)
            else:
                # Single video frame - still process as video for consistency
                full_path = os.path.join(image_folder, video_files[0])
                if os.path.exists(full_path):
                    pixel_vals, grid_thw = self._process_video_from_frames(video_files, image_folder)
                    video_tensors.append(pixel_vals)
                    video_thws.append(grid_thw)
        
        # Combine tensors by type
        if image_tensors:
            image_pixel_values = torch.cat(image_tensors, dim=0)
            image_grid_thw = torch.cat(image_thws, dim=0)
        else:
            image_pixel_values = None
            image_grid_thw = None
        
        if video_tensors:
            video_pixel_values = torch.cat(video_tensors, dim=0)
            video_grid_thw = torch.cat(video_thws, dim=0)
        else:
            video_pixel_values = None
            video_grid_thw = None
        
        # Calculate grid_thw_merged for preprocessing
        grid_thw_image_merged = None
        grid_thw_video_merged = None
        
        if image_grid_thw is not None:
            grid_thw_image_merged = [
                thw.prod() // self.data_args.image_processor.merge_size**2
                for thw in image_grid_thw
            ]
        
        if video_grid_thw is not None:
            # Video uses same token calculation as images (both use merge_size**2 = 4)
            grid_thw_video_merged = [
                thw.prod() // self.data_args.image_processor.merge_size**2
                for thw in video_grid_thw
            ]
        
        # Preprocess conversations using the imported function
        chat_sources = copy.deepcopy([json_item["conversations"]])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_image_merged,
            grid_thw_video=grid_thw_video_merged,
        )
        
        # Remove batch dimension from tensors to match VLA data format
        data_dict["input_ids"] = data_dict["input_ids"].squeeze(0)
        data_dict["labels"] = data_dict["labels"].squeeze(0)
        
        # Add image and/or video data if present
        if image_pixel_values is not None:
            data_dict["pixel_values"] = image_pixel_values
            data_dict["image_grid_thw"] = image_grid_thw
        
        if video_pixel_values is not None:
            data_dict["pixel_values_videos"] = video_pixel_values
            data_dict["video_grid_thw"] = video_grid_thw
        
        return data_dict
    
    def _process_image_unified(self, image_file):
        """Process a single image file for JSON data."""
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        
        # Resize single JSON image based on pixel budget
        pixel_budget = getattr(self.data_args, 'pixel_budget', 230400)  # Default: 4x VLA size
        if pixel_budget is not None and pixel_budget > 0:
            image = self._resize_with_pixel_budget(image, pixel_budget)
        
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        
        # The Qwen processor returns 2D vision features [num_patches, feature_dim]
        # Keep them as-is since that's what the model expects
            
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def _resize_with_pixel_budget(self, image, pixel_budget):
        """Resize image to have at most pixel_budget pixels while preserving aspect ratio."""
        width, height = image.size
        current_pixels = width * height
        
        # If image already fits within pixel budget, return as is
        if current_pixels <= pixel_budget:
            return image
        
        # Calculate scaling factor based on pixel budget
        # scale^2 * current_pixels = pixel_budget
        scale_factor = math.sqrt(pixel_budget / current_pixels)
        
        # Qwen2-VL constraints: both dimensions must be > 28 (IMAGE_FACTOR) and divisible by 28
        factor = 28
        min_pixels_per_image = (factor + factor) * (factor + factor)  # 56*56 = 3136 pixels minimum
        
        # Helper function to round to nearest factor multiple (must be > factor, not >= factor)
        def round_to_factor(value, factor):
            result = round(value / factor) * factor
            # Ensure result is strictly greater than factor
            if result <= factor:
                result = factor + factor  # Next valid multiple (56)
            return result
        
        # Apply scale factor and round to factor multiples
        scaled_width = width * scale_factor
        scaled_height = height * scale_factor
        new_width = round_to_factor(scaled_width, factor)
        new_height = round_to_factor(scaled_height, factor)
        
        # Ensure minimum pixels per image (in case one dimension is very small)
        current_pixels_after_resize = new_width * new_height
        if current_pixels_after_resize < min_pixels_per_image:
            # Scale up to meet minimum pixels while preserving aspect ratio
            aspect_ratio = width / height
            if aspect_ratio >= 1:  # Width >= Height
                target_width = math.ceil(math.sqrt(min_pixels_per_image * aspect_ratio) / factor) * factor
                if target_width <= factor:
                    target_width = factor + factor
                target_height = round_to_factor(target_width / aspect_ratio, factor)
            else:  # Height > Width
                target_height = math.ceil(math.sqrt(min_pixels_per_image / aspect_ratio) / factor) * factor
                if target_height <= factor:
                    target_height = factor + factor
                target_width = round_to_factor(target_height * aspect_ratio, factor)
            new_width = target_width
            new_height = target_height
        
        # If we still exceed pixel budget after minimum constraints, we need to accept it
        # (minimum constraints take precedence over pixel budget to avoid processing errors)
        
        # Resize image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _process_video_from_frames(self, frame_files, image_folder):
        """Process video from a list of frame image paths using video-specific processing."""
        if not isinstance(frame_files, list) or len(frame_files) <= 1:
            raise ValueError("Expected multiple frame files for video processing")
        
        # Load frames
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(image_folder, frame_file)
            if os.path.exists(frame_path):
                # try:
                img = Image.open(frame_path).convert("RGB")
                frames.append(np.array(img))
                # except Exception as e:
                #     print(f"Failed to load frame {frame_path}: {e}")
                #     continue
        
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
        
        return self._process_video_frames(video, frame_idx, video_length)
    
    def _process_video_frames(self, video, frame_idx, video_length):
        """Process video frames using the video processor."""
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        
        # Set video-specific processing parameters if available
        if hasattr(self.data_args, 'video_max_frame_pixels'):
            processor.max_pixels = self.data_args.video_max_frame_pixels
        if hasattr(self.data_args, 'video_min_frame_pixels'):
            processor.min_pixels = self.data_args.video_min_frame_pixels
        
        # Process video
        video_processed = processor.preprocess(
            images=None, 
            videos=video, 
            return_tensors="pt",
            min_pixels=self.data_args.video_min_frame_pixels,
            max_pixels=self.data_args.video_max_frame_pixels,
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"]
        
        # Handle grid_thw format
        if isinstance(grid_thw, list) and len(grid_thw) > 0:
            grid_thw = grid_thw[0]
        
        # Ensure grid_thw has proper shape
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        
        return video_tensor, grid_thw
    
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
        
        # Handle images and videos - collect all pixel values and grid_thw
        all_pixel_values = []
        all_grid_thw = []
        all_video_values = []
        all_video_grid_thw = []
        
        for instance in instances:
            # Handle image data (can coexist with video data)
            if "pixel_values" in instance and instance["pixel_values"] is not None:
                pixel_vals = instance["pixel_values"]
                grid_thw = instance["image_grid_thw"]
                
                # Ensure grid_thw is 2D: [N, 3]
                if grid_thw.dim() == 1:  # [3] -> [1, 3]
                    grid_thw = grid_thw.unsqueeze(0)
                elif grid_thw.dim() == 2:  # [N, 3] - already correct
                    pass
                else:
                    print(f"Warning: Unexpected image grid_thw dimensions: {grid_thw.shape}")
                    # Try to reshape to [N, 3]
                    if grid_thw.numel() % 3 == 0:
                        grid_thw = grid_thw.reshape(-1, 3)
                    else:
                        continue
                
                all_pixel_values.append(pixel_vals)
                all_grid_thw.append(grid_thw)
            
            # Handle video data (can coexist with image data)
            if "pixel_values_videos" in instance and instance["pixel_values_videos"] is not None:
                video_vals = instance["pixel_values_videos"]
                video_grid_thw = instance["video_grid_thw"]
                
                # Ensure video_grid_thw is 2D: [N, 3]
                if video_grid_thw.dim() == 1:  # [3] -> [1, 3]
                    video_grid_thw = video_grid_thw.unsqueeze(0)
                elif video_grid_thw.dim() == 2:  # [N, 3] - already correct
                    pass
                else:
                    print(f"Warning: Unexpected video grid_thw dimensions: {video_grid_thw.shape}")
                    # Try to reshape to [N, 3]
                    if video_grid_thw.numel() % 3 == 0:
                        video_grid_thw = video_grid_thw.reshape(-1, 3)
                    else:
                        continue
                
                all_video_values.append(video_vals)
                all_video_grid_thw.append(video_grid_thw)
        
        # Set image data
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        
        # Set video data
        if all_video_values:
            batch["pixel_values_videos"] = torch.cat(all_video_values, dim=0)
            batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None
        
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
