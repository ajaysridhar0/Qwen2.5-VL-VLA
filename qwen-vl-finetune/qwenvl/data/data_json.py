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
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import os

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


class JSONCotrainDataset(Dataset):
    """Mixed dataset combining VLA training with regular conversational data."""
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        model_max_length: int = 2048,
        token_mappings: Dict = None,
        image_size: tuple = (180, 320),
        data_size: int = None,
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_max_length = model_max_length
        self.image_processor = data_args.image_processor
        self.token_mappings = token_mappings
        self.image_size = image_size
        self.data_size = data_size
        
        # Create a helper instance to access image/video processing methods from data_qwen.py
        # This ensures we use the tested functions that work properly with gradient checkpointing
        self._qwen_dataset_helper = self._create_qwen_helper(tokenizer, data_args)

        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
        
        # Initialize regular JSON dataset if co-training is enabled
        self.json_dataset = None
        self.json_datasets = []  # List of individual datasets
        self.json_dataset_weights = []  # Weights for each dataset
        self.json_dataset_sizes = []  # Sizes of each dataset
        if hasattr(data_args, 'cotrain_json_paths') and data_args.cotrain_json_paths:
            self.json_dataset, self.json_datasets, self.json_dataset_weights, self.json_dataset_sizes = self._load_json_dataset()
            rank0_print(f"Loaded {len(self.json_dataset)} JSON conversation examples for co-training")
            if len(self.json_datasets) > 1:
                rank0_print(f"Individual JSON datasets: {len(self.json_datasets)} datasets")
                for i, (size, weight) in enumerate(zip(self.json_dataset_sizes, self.json_dataset_weights)):
                    rank0_print(f"  Dataset {i}: {size} samples, weight={weight:.3f}")
        
        self.json_dataset_size = len(self.json_dataset)
    
    def _load_json_dataset(self):
        """Load and prepare JSON conversational data with individual weighting support."""
        json_data = []
        individual_datasets = []
        individual_sizes = []
        
        # Handle multiple JSON file paths
        json_paths = self.data_args.cotrain_json_paths
        if "," in json_paths:
            json_paths = json_paths.split(",")
        if isinstance(json_paths, str):
            json_paths = [json_paths]
        
        # Parse individual weights if provided
        individual_weights = []
        if hasattr(self.data_args, 'cotrain_json_weights') and self.data_args.cotrain_json_weights:
            weight_strs = [w.strip() for w in self.data_args.cotrain_json_weights.split(",") if w.strip()]
            if len(weight_strs) != len(json_paths):
                rank0_print(f"Warning: Number of weights ({len(weight_strs)}) doesn't match number of JSON paths ({len(json_paths)}). Using equal weights.")
                individual_weights = [1.0] * len(json_paths)
            else:
                try:
                    individual_weights = [float(w) for w in weight_strs]
                    if any(w < 0 for w in individual_weights):
                        raise ValueError("Weights must be non-negative")
                except ValueError as e:
                    rank0_print(f"Warning: Invalid weight values: {e}. Using equal weights.")
                    individual_weights = [1.0] * len(json_paths)
        else:
            # Use equal weights if none provided
            individual_weights = [1.0] * len(json_paths)
        
        # Load each dataset individually
        for i, json_path in enumerate(json_paths):
            if not os.path.exists(json_path):
                rank0_print(f"Warning: JSON file not found: {json_path}")
                individual_datasets.append([])
                individual_sizes.append(0)
                continue
                
            # Load JSON or JSONL data
            file_format = json_path.split(".")[-1]
            if file_format == "jsonl":
                data = read_jsonl(json_path)
            else:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            
            # Add data path information and dataset index
            for item in data:
                if 'data_path' not in item:
                    item['data_path'] = os.path.dirname(json_path)
                item['dataset_index'] = i  # Track which dataset this sample comes from
            
            individual_datasets.append(data)
            individual_sizes.append(len(data))
            json_data.extend(data)
            rank0_print(f"Loaded {len(data)} examples from {json_path}")
        
        # Handle count-based weighting if enabled
        if hasattr(self.data_args, 'weight_by_count') and self.data_args.weight_by_count:
            rank0_print("Using count-based weighting for JSON datasets")
            # Calculate weights based on dataset sizes
            power = getattr(self.data_args, 'count_weight_power', 1.0)
            size_weights = [size ** power for size in individual_sizes]
            total_weight = sum(size_weights)
            if total_weight > 0:
                individual_weights = [w / total_weight for w in size_weights]
            else:
                individual_weights = [1.0 / len(individual_datasets)] * len(individual_datasets)
            rank0_print(f"Count-based weights (power={power}): {[f'{w:.3f}' for w in individual_weights]}")
        else:
            # Normalize individual weights
            total_weight = sum(individual_weights)
            if total_weight > 0:
                individual_weights = [w / total_weight for w in individual_weights]
            else:
                individual_weights = [1.0 / len(individual_datasets)] * len(individual_datasets)
        
        # Apply data_size limit if specified
        if self.data_size is not None and len(json_data) > self.data_size:
            original_size = len(json_data)
            
            # Worker-aware sharding for distributed training
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                
                # Calculate per-worker data size
                per_worker_size = self.data_size // world_size
                if rank < (self.data_size % world_size):
                    per_worker_size += 1
                
                # Shuffle and shard the data
                import random
                random.shuffle(json_data)
                start_idx = rank * (self.data_size // world_size) + min(rank, self.data_size % world_size)
                end_idx = start_idx + per_worker_size
                json_data = json_data[start_idx:end_idx]
                
                rank0_print(f"JSON dataset sharded: Worker {rank}/{world_size} gets {len(json_data)} samples (total limited to {self.data_size} from {original_size})")
            else:
                # Single worker case
                import random
                random.shuffle(json_data)
                json_data = json_data[:self.data_size]
                rank0_print(f"Limited JSON dataset to {self.data_size} samples (from {original_size} total)")
            
            # Also update individual datasets to maintain consistency
            # Recalculate individual datasets after limiting
            individual_datasets = [[] for _ in range(len(individual_datasets))]
            individual_sizes = [0] * len(individual_sizes)
            
            for item in json_data:
                dataset_idx = item.get('dataset_index', 0)
                if dataset_idx < len(individual_datasets):
                    individual_datasets[dataset_idx].append(item)
                    individual_sizes[dataset_idx] += 1
        
        return json_data, individual_datasets, individual_weights, individual_sizes
    
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

            def process_ego_npy(self, npy_file):
                """
                Reverses the ImageNet normalization on a NumPy array.

                Args:
                        image (np.ndarray): A NumPy array of shape (N, 3, 224, 224) 
                                            representing a normalized image. It is 
                                            assumed the input was normalized with the
                                            standard ImageNet mean and std, and scaled to [0, 1].

                Returns:
                    np.ndarray: The unnormalized image as a NumPy array of shape
                                (224, 224, 3) with pixel values in the [0, 255] range
                                as uint8 type.
                """
                imgs = np.load(npy_file)
                imgs = np.transpose(imgs, (0, 2, 3, 1))
                assert imgs.shape[1:] == (224, 224, 3)

                # ImageNet mean and standard deviation
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

                # Un-normalize: (image * std) + mean
                unnormalized_images = (imgs * std) + mean
                assert unnormalized_images.shape[1:] == (224, 224, 3)

                # Rescale from [0, 1] to [0, 255]
                unnormalized_images *= 255.0

                # Clip values to ensure they are within the valid [0, 255] range
                # and convert to an 8-bit unsigned integer, the standard for images.
                video = np.clip(unnormalized_images, 0, 255).astype(np.uint8)
                return self.process_video_frames(video, np.arange(len(video)), len(video))

            def process_video(self, video_file):
                # Check if video_file is a list of image frames
                if isinstance(video_file, list) and all(".jpg" in f or ".png" in f for f in video_file):
                    return self.process_video_from_frames(video_file)

                if video_file.endswith(".npy"):
                    return self.process_ego_npy(video_file)
                
                decord_video = None
                decord_attempts = 0
                max_decord_attempts = 3
                while decord_attempts < max_decord_attempts:
                    try:
                        decord_video = self.video_decord(video_file)
                        return decord_video
                    except Exception as e:
                        print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                        decord_attempts += 1

                torchcodec_video = None
                try:
                    torchcodec_video = self.video_torchcodec(video_file)
                    return torchcodec_video
                except Exception as e:
                    print(f"torchcodec attempt failed: {e}")

            def video_decord(self, video_file):
                if not os.path.exists(video_file):
                    print(f"File not exist: {video_file}")
                vr = VideoReader(video_file, num_threads=4)
                total_frames = len(vr)
                avg_fps = vr.get_avg_fps()
                video_length = total_frames / avg_fps
                interval = getattr(self.data_args, "base_interval", 4)

                num_frames_to_sample = round(video_length / interval)
                video_min_frames = getattr(self.data_args, "video_min_frames", 4)
                video_max_frames = getattr(self.data_args, "video_max_frames", 8)

                target_frames = min(
                    max(num_frames_to_sample, video_min_frames), video_max_frames
                )
                frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
                frame_idx = np.unique(frame_idx)
                video = vr.get_batch(frame_idx).asnumpy()
                return self.process_video_frames(video, frame_idx, video_length)

            def video_torchcodec(self, video_file):
                device = "cpu"  # or e.g. "cuda"
                decoder = VideoDecoder(video_file, device=device)
                total_frames = decoder.metadata.num_frames
                avg_fps = decoder.metadata.average_fps
                video_length = total_frames / avg_fps
                interval = getattr(self.data_args, "base_interval", 4)

                num_frames_to_sample = round(video_length / interval)
                video_min_frames = getattr(self.data_args, "video_min_frames", 4)
                video_max_frames = getattr(self.data_args, "video_max_frames", 8)

                target_frames = min(
                    max(num_frames_to_sample, video_min_frames), video_max_frames
                )
                frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
                frame_idx = np.unique(frame_idx)
                frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
                video = frame_batch.data.cpu().numpy()
                return self.process_video_frames(video, frame_idx, video_length)
            
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
        return self.json_dataset_size
    
    def _get_json_item(self, idx):
        """Get a processed JSON conversational example.
        
        Supports mixed media: JSON items can contain both 'image' and 'video' fields.
        Images are processed as image data, videos are processed as video data.
        
        If individual JSON datasets are loaded, uses weighted sampling based on dataset weights.
        """
        if not self.json_dataset:
            raise ValueError("JSON dataset not loaded")
        
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

            if isinstance(video_file, List):
                if not isinstance(video_file, List):
                    video_file = [video_file]
                if all(".jpg" in f or ".png" in f for f in video_file):
                    video_file = [video_file]
                if len(video_file) > 1:
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]

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
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
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

    def process_video(self, video_file):
        """Process a single video file for JSON data using the tested function from data_qwen.py."""
        return self._qwen_dataset_helper.process_video(video_file)
    
    def __getitem__(self, idx):
        """Get a JSON conversational example."""
        return self._get_json_item(idx)
    
