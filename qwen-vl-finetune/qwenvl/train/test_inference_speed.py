#!/usr/bin/env python3
"""
Inference speed test for Qwen VLA model with DROID-like inputs.
This script measures the time it takes to generate action sequences without focusing on accuracy.
"""

import torch
import time
import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image
import argparse
from typing import Dict, List
from transformers import AutoTokenizer, AutoProcessor
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import custom classes
sys.path.append(str(project_root / "qwen-vl-finetune"))
from train_qwen_vla import VLAQwenModel, get_action_state_token_mappings
from droid_rlds_dataset import DroidRldsDataset, DroidActionSpace


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # Base model or fine-tuned checkpoint
    action_tokenizer_path: str = "physical-intelligence/fast"
    action_chunk_size: int = 15
    action_vocab_size: int = 512
    state_vocab_size: int = 256
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 20
    max_length: int = 2048
    batch_size: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_flash_attention: bool = True
    
    # DROID dataset parameters
    droid_data_dir: str = "/iliad2/u/ajaysri/episodic_memory/droid_rlds"
    droid_dataset_name: str = "droid_100"
    use_joint_velocity: bool = True
    use_real_data: bool = True  # Whether to use real DROID data or dummy data
    
    # Image resize parameters
    image_height: int = 180
    image_width: int = 320
    
    # Generation parameters
    temperature: float = 0.0  # Deterministic for consistent timing
    do_sample: bool = False


def create_dummy_images(batch_size: int = 1, image_height: int = 180, image_width: int = 320) -> List[Image.Image]:
    """Create dummy RGB images similar to DROID dataset."""
    images = []
    for _ in range(batch_size):
        # Create two images per sample (exterior and wrist views)
        exterior_img = Image.fromarray(np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8))
        wrist_img = Image.fromarray(np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8))
        images.extend([exterior_img, wrist_img])
    return images


def create_dummy_state(batch_size: int = 1) -> np.ndarray:
    """Create dummy robot state (joint positions + gripper)."""
    # 7 joint positions + 1 gripper position = 8 dimensions
    return np.random.uniform(-1, 1, (batch_size, 8))


def create_dummy_prompts(batch_size: int = 1) -> List[str]:
    """Create dummy task prompts."""
    prompts = [
        "Pick up the red cup from the table",
        "Move the blue block to the left",
        "Open the drawer",
        "Close the gripper",
        "Reach toward the green object",
    ]
    return [prompts[i % len(prompts)] for i in range(batch_size)]


class InferenceSpeedTester:
    """Test inference speed of Qwen VLA model."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Initializing inference speed tester on {self.device}")
        print(f"Model: {config.model_path}")
        print(f"Using real DROID data: {config.use_real_data}")
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load action tokenizer
        self._load_action_tokenizer()
        
        # Get token mappings
        self.token_mappings = get_action_state_token_mappings(
            self.tokenizer, 
            config.action_vocab_size, 
            config.state_vocab_size
        )
        
        # Initialize DROID dataset if using real data
        if config.use_real_data:
            self._load_droid_dataset()
        
        print("Initialization complete!")
    
    def _load_model_and_tokenizer(self):
        """Load the VLA model and tokenizer."""
        print("Loading model and tokenizer...")
        
        # Check if this is a checkpoint directory with action tokenizer info
        checkpoint_info_path = Path(self.config.model_path) / "action_tokenizer_info.json"
        if checkpoint_info_path.exists():
            print(f"Loading from checkpoint: {self.config.model_path}")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)
            self.token_mappings = checkpoint_info.get("token_mappings")
            
        # Load model (try VLA model first, then base model)
        try:
            self.model = VLAQwenModel.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.config.device == "cuda" else None,
                attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
            )
            print("Loaded VLAQwenModel successfully")
        except:
            print("VLAQwenModel failed, loading base Qwen2_5_VLForConditionalGeneration")
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.config.device == "cuda" else None,
                attn_implementation="flash_attention_2" if self.config.use_flash_attention else None,
            )
        
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            padding_side="left",  # For generation
            use_fast=False,
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load image processor
        self.image_processor = AutoProcessor.from_pretrained(
            self.config.model_path
        ).image_processor
        
    def _load_action_tokenizer(self):
        """Load the action tokenizer."""
        print(f"Loading action tokenizer: {self.config.action_tokenizer_path}")
        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.config.action_tokenizer_path,
            trust_remote_code=True
        )
    
    def _load_droid_dataset(self):
        """Load DROID dataset for real data."""
        print(f"Loading DROID dataset from {self.config.droid_data_dir}")
        self.droid_dataset = DroidRldsDataset(
            data_dir=self.config.droid_data_dir,
            dataset_name=self.config.droid_dataset_name,
            batch_size=1,  # We handle batching ourselves
            action_chunk_size=self.config.action_chunk_size,
            action_space=DroidActionSpace.JOINT_VELOCITY if self.config.use_joint_velocity else DroidActionSpace.JOINT_POSITION,
            shuffle_buffer_size=1000,  # Smaller buffer for testing
            shuffle=True,
        )
        self.droid_iter = iter(self.droid_dataset)
        print("DROID dataset loaded successfully!")
    
    def _process_image(self, image_array):
        """Process numpy image array to PIL Image."""
        if isinstance(image_array, np.ndarray):
            # Ensure uint8 type
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            return Image.fromarray(image_array)
        return image_array
    
    def _get_real_droid_samples(self, batch_size: int = 1) -> List[Dict]:
        """Get real DROID data samples."""
        samples = []
        for _ in range(batch_size):
            # Get next batch from DROID dataset
            batch = next(self.droid_iter)
            samples.append(batch)
        return samples
    
    def _get_action_token_length(self, actions: np.ndarray) -> int:
        """Get the exact number of tokens that the fast tokenizer produces for these actions."""
        if self.action_tokenizer is None:
            # Fallback: estimate based on action dimensions  
            return actions.shape[0] * actions.shape[1]  # action_chunk_size * action_dim
        
        # Normalize actions like in training (this is pre-computed, not part of timing)
        from qwenvl.data.data_droid import normalize_action
        normalized_actions = normalize_action(actions, input_type="actions")
        
        # Tokenize actions using the fast tokenizer (this is pre-computed, not part of timing)
        action_tokens = self.action_tokenizer(normalized_actions.reshape(1, -1, normalized_actions.shape[-1]))[0]
        return len(action_tokens)
    
    def prepare_inputs(self, batch_size: int = 1) -> Dict:
        """Prepare model inputs from DROID dataset or dummy data."""
        
        expected_token_lengths = []
        sample_actions = []
        
        if self.config.use_real_data and hasattr(self, 'droid_iter') and self.droid_iter is not None:
            # Use real DROID data
            droid_samples = self._get_real_droid_samples(batch_size)
            
            # Extract data from DROID samples
            images = []
            prompts = []
            for sample in droid_samples:
                # Extract and process images
                exterior_img = self._process_image(sample['observation']['image'][0])
                wrist_img = self._process_image(sample['observation']['wrist_image'][0])
                images.extend([exterior_img, wrist_img])
                
                # Extract prompt
                prompt = sample['prompt'][0]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode('utf-8')
                prompts.append(prompt)
                
                # Get actions and calculate expected token length
                actions = sample['actions'][0]  # Shape: (action_chunk_size, 8)
                sample_actions.append(actions)
                token_length = self._get_action_token_length(actions)
                expected_token_lengths.append(token_length)
                
        else:
            # Use dummy data
            print("Using dummy data...")
            images = create_dummy_images(batch_size, self.config.image_height, self.config.image_width)
            prompts = create_dummy_prompts(batch_size)
            
            # Create dummy actions and calculate token lengths
            for _ in range(batch_size):
                dummy_actions = np.random.uniform(-1, 1, (self.config.action_chunk_size, 8))
                sample_actions.append(dummy_actions)
                token_length = self._get_action_token_length(dummy_actions)
                expected_token_lengths.append(token_length)
        
        # Process images
        image_inputs = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        
        # Move to device
        for key in image_inputs:
            if isinstance(image_inputs[key], torch.Tensor):
                image_inputs[key] = image_inputs[key].to(self.device)
        
        # Build vision placeholders
        grid_thw = image_inputs["image_grid_thw"]
        merge_size = self.image_processor.merge_size
        
        all_conversations = []
        
        for i in range(batch_size):
            # Get vision segments for this sample (2 images per sample)
            sample_grid_thw = grid_thw[i*2:(i+1)*2]  # 2 images per sample
            vision_segments = []
            
            for thw in sample_grid_thw:
                count = int(torch.prod(thw).item() // (merge_size ** 2))
                vision_segments.append(
                    "<|vision_start|>" + ("<|image_pad|>" * count) + "<|vision_end|>"
                )
            
            # Create conversation with state placeholders
            user_content = "\n".join(vision_segments) + "\n" + "State:<|state_start|><|state_end|>" + "\n" + (
                f"Task: {prompts[i]}\nWhat action should the robot perform?"
            )
            
            conversation = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "<|action_start|>"},
            ]
            all_conversations.append(conversation)
        
        # Apply chat template and tokenize
        texts = []
        for conversation in all_conversations:
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Replace control token strings with mapped Chinese characters
            if hasattr(self, 'token_mappings') and self.token_mappings:
                for control_string, token_id in self.token_mappings['control_mappings'].items():
                    if control_string in text:
                        replacement_token = self.tokenizer.decode([token_id])
                        text = text.replace(control_string, replacement_token)
            
            texts.append(text)
        
        # Tokenize all texts
        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        
        # Move to device
        for key in text_inputs:
            if isinstance(text_inputs[key], torch.Tensor):
                text_inputs[key] = text_inputs[key].to(self.device)
        
        # Insert dummy state tokens if we have token mappings
        if hasattr(self, 'token_mappings') and self.token_mappings:
            # For simplicity in testing, we'll just add the state tokens at the end of input
            # In real training, these would be inserted between state_start and state_end
            batch_input_ids = text_inputs["input_ids"]
            for i in range(batch_size):
                # Add dummy state tokens (8 state dimensions)
                dummy_state_tokens = [self.token_mappings['state_token_ids'][j % len(self.token_mappings['state_token_ids'])] for j in range(8)]
                # We'll append these for timing purposes, not for correctness
                
        return {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_inputs.get("image_grid_thw", torch.tensor([[1, 1, 1]] * len(images))),
            "expected_token_lengths": expected_token_lengths,
            "sample_actions": sample_actions,
        }
    
    def warmup(self):
        """Warmup the model to ensure consistent timing."""
        print(f"Warming up with {self.config.num_warmup_runs} runs...")
        
        for i in range(self.config.num_warmup_runs):
            inputs = self.prepare_inputs(self.config.batch_size)
            expected_token_lengths = inputs.pop("expected_token_lengths")
            sample_actions = inputs.pop("sample_actions")
            
            # Use the maximum expected token length for warmup
            max_tokens = max(expected_token_lengths) if expected_token_lengths else 120  # Fallback
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=None,  # Don't stop early
                )
            
            if i == 0:
                print(f"  Warmup run {i+1} completed (max tokens: {max_tokens})")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Warmup complete!")
    
    def benchmark(self) -> Dict:
        """Run the actual benchmark."""
        print(f"Running benchmark with {self.config.num_benchmark_runs} runs...")
        
        times = {
            "model_forward": [],
            "total_per_sample": [],
            "expected_action_tokens": [],
            "actual_tokens_generated": [],
            "time_per_token": [],
        }
        
        for run_idx in range(self.config.num_benchmark_runs):
            # Prepare inputs (this includes action tokenization for length calculation, but we don't time it)
            inputs = self.prepare_inputs(self.config.batch_size)
            expected_token_lengths = inputs.pop("expected_token_lengths")
            sample_actions = inputs.pop("sample_actions")
            
            # Calculate total expected tokens for this batch
            total_expected_tokens = sum(expected_token_lengths)
            
            # Time only the observation-to-action decoding (the actual inference we care about)
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=total_expected_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=None,  # Don't stop at EOS, generate exactly the expected tokens
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            model_forward_time = time.time() - start_time
            
            # Calculate metrics
            actual_tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            # Store times and metrics (only tracking model forward time now)
            times["model_forward"].append(model_forward_time)
            times["total_per_sample"].append(model_forward_time / self.config.batch_size)  # Per sample = forward time / batch size
            times["expected_action_tokens"].append(total_expected_tokens)
            times["actual_tokens_generated"].append(actual_tokens_generated)
            times["time_per_token"].append(model_forward_time / max(actual_tokens_generated, 1))
            
            if run_idx == 0:
                print(f"  Sample 1: Expected {total_expected_tokens} action tokens, generated {actual_tokens_generated} tokens")
                if self.config.use_real_data:
                    print(f"  Action shapes: {[action.shape for action in sample_actions]}")
            
            if (run_idx + 1) % 5 == 0:
                print(f"  Completed {run_idx + 1}/{self.config.num_benchmark_runs} runs")
        
        return times
    
    def print_results(self, times: Dict):
        """Print benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        print(f"Configuration:")
        print(f"  Model: {self.config.model_path}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Using real DROID data: {self.config.use_real_data}")
        print(f"  Device: {self.device}")
        print(f"  Number of runs: {self.config.num_benchmark_runs}")
        
        # Print action token statistics
        if times["expected_action_tokens"]:
            expected_tokens = times["expected_action_tokens"]
            mean_expected = np.mean(expected_tokens)
            print(f"  Average expected action tokens per sample: {mean_expected:.1f}")
            print(f"  Token range: {min(expected_tokens)} - {max(expected_tokens)}")
        
        print(f"\nTiming Results:")
        
        # Calculate statistics for timing metrics
        timing_metrics = ["model_forward", "total_per_sample", "time_per_token"]
        for metric_name in timing_metrics:
            if metric_name in times and times[metric_name]:
                metric_times = times[metric_name]
                mean_time = np.mean(metric_times)
                std_time = np.std(metric_times)
                min_time = np.min(metric_times)
                max_time = np.max(metric_times)
                
                display_name = metric_name.replace('_', ' ').title()
                print(f"  {display_name}: {mean_time:.4f}s ± {std_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s)")
        
        # Token statistics
        print(f"\nToken Generation Statistics:")
        if times["expected_action_tokens"] and times["actual_tokens_generated"]:
            expected_tokens = times["expected_action_tokens"]
            actual_tokens = times["actual_tokens_generated"]
            
            mean_expected = np.mean(expected_tokens)
            mean_actual = np.mean(actual_tokens)
            
            print(f"  Expected action tokens per sample: {mean_expected:.1f} ± {np.std(expected_tokens):.1f}")
            print(f"  Actual tokens generated per sample: {mean_actual:.1f} ± {np.std(actual_tokens):.1f}")
            
            # Check if we're generating the right number of tokens
            token_accuracy = np.mean([abs(exp - act) for exp, act in zip(expected_tokens, actual_tokens)])
            print(f"  Average token count difference: {token_accuracy:.1f}")
        
        # Calculate throughput metrics
        print(f"\nThroughput Metrics:")
        if times["total_per_sample"]:
            mean_total_time = np.mean(times["total_per_sample"])
            throughput = 1.0 / mean_total_time
            print(f"  Samples per second: {throughput:.2f}")
        
        if times["time_per_token"]:
            mean_time_per_token = np.mean(times["time_per_token"])
            tokens_per_second = 1.0 / mean_time_per_token
            print(f"  Tokens per second: {tokens_per_second:.2f}")
        
        if times["model_forward"] and times["actual_tokens_generated"]:
            # Alternative calculation for tokens/second
            total_forward_time = np.sum(times["model_forward"])
            total_tokens = np.sum(times["actual_tokens_generated"])
            alt_tokens_per_second = total_tokens / total_forward_time
            print(f"  Tokens per second (alternative calc): {alt_tokens_per_second:.2f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test inference speed of Qwen VLA model")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path to model (base model or checkpoint)")
    parser.add_argument("--action_tokenizer_path", type=str, default="physical-intelligence/fast",
                        help="Path to action tokenizer")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--num_warmup_runs", type=int, default=5,
                        help="Number of warmup runs")
    parser.add_argument("--num_benchmark_runs", type=int, default=20,
                        help="Number of benchmark runs")
    # Removed max_new_tokens - now determined dynamically from action tokenizer
    parser.add_argument("--action_chunk_size", type=int, default=15,
                        help="Action chunk size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="Disable flash attention")
    parser.add_argument("--droid_data_dir", type=str, default="/iliad2/u/ajaysri/episodic_memory/droid_rlds",
                        help="Path to DROID dataset directory")
    parser.add_argument("--droid_dataset_name", type=str, default="droid_100",
                        help="DROID dataset name")
    parser.add_argument("--use_dummy_data", action="store_true",
                        help="Use dummy data instead of real DROID data")
    parser.add_argument("--use_joint_position", action="store_true",
                        help="Use joint position instead of joint velocity")
    parser.add_argument("--image_height", type=int, default=180,
                        help="Height to resize images to (default: 180)")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Width to resize images to (default: 320)")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        model_path=args.model_path,
        action_tokenizer_path=args.action_tokenizer_path,
        batch_size=args.batch_size,
        num_warmup_runs=args.num_warmup_runs,
        num_benchmark_runs=args.num_benchmark_runs,
        action_chunk_size=args.action_chunk_size,
        device=args.device,
        use_flash_attention=not args.no_flash_attention,
        droid_data_dir=args.droid_data_dir,
        droid_dataset_name=args.droid_dataset_name,
        use_real_data=not args.use_dummy_data,
        use_joint_velocity=not args.use_joint_position,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    
    # Run benchmark
    tester = InferenceSpeedTester(config)
    tester.warmup()
    times = tester.benchmark()
    tester.print_results(times)


if __name__ == "__main__":
    main()
