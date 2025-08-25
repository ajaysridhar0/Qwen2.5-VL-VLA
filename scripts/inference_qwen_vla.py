"""
Inference script for Qwen VLA model.
Tests the model's ability to predict action sequences from image-language inputs.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from qwen_vl_finetune.qwenvl.train.train_qwen_vla import VLAQwenModel


class QwenVLAInference:
    def __init__(self, model_path, device="cuda"):
        """Initialize the VLA model for inference."""
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = VLAQwenModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load image processor
        self.image_processor = AutoProcessor.from_pretrained(model_path).image_processor
        
        # Load action tokenizer info
        action_info_path = Path(model_path) / "action_tokenizer_info.json"
        if action_info_path.exists():
            with open(action_info_path, "r") as f:
                self.action_info = json.load(f)
        else:
            # Default values
            self.action_info = {
                "action_tokenizer_path": "KarlP/fast-droid",
                "action_vocab_size": 512,
                "action_token_offset": 102500
            }
        
        # Load action tokenizer
        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.action_info["action_tokenizer_path"],
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
    
    def predict_actions(self, image, wrist_image, instruction, max_action_tokens=30):
        """
        Predict action sequence from image and instruction.
        
        Args:
            image: PIL Image or path to image for external camera
            wrist_image: PIL Image or path to image for wrist camera
            instruction: Text instruction
            max_action_tokens: Maximum number of action tokens to generate
            
        Returns:
            numpy array of actions (shape: [chunk_size, action_dim])
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(wrist_image, str):
            wrist_image = Image.open(wrist_image)
        
        # Process image
        image_inputs = self.image_processor(
            images=[image, wrist_image],
            return_tensors="pt"
        ).to(self.device)
        
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "What action should the robot perform to " + instruction + "?"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<|action_start|>"}
                ]
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)
        
        # Generate action tokens
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values,
                image_grid_thw=image_inputs.get("image_grid_thw", None),
                max_new_tokens=max_action_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated tokens
        generated_ids = outputs[0][text_inputs.input_ids.shape[1]:]
        
        # Filter action tokens (those in the action token range)
        action_token_start = self.action_info["action_token_offset"]
        action_token_end = action_token_start + self.action_info["action_vocab_size"]
        
        action_tokens = []
        for token_id in generated_ids.cpu().numpy():
            if action_token_start <= token_id < action_token_end:
                # Map back to original action token
                action_tokens.append(token_id - action_token_start)
            elif token_id == self.tokenizer.eos_token_id:
                break
        
        # Decode action tokens to continuous actions
        if action_tokens:
            actions = self.action_tokenizer.decode([action_tokens])
            return actions
        else:
            print("Warning: No action tokens generated")
            return np.zeros((15, 8))  # Return zeros with default shape


def test_inference():
    """Test the inference pipeline with sample data."""
    # Initialize model
    model_path = "./checkpoints/qwen_vla_droid"  # Update with your checkpoint path
    vla = QwenVLAInference(model_path)
    
    # Load sample data from DROID dataset
    from droid_rlds_dataset import DroidRldsDataset, DroidActionSpace
    
    dataset = DroidRldsDataset(
        data_dir="/iliad2/u/ajaysri/episodic_memory/droid_rlds",
        dataset_name="droid_100",
        batch_size=1,
        action_chunk_size=15,
        action_space=DroidActionSpace.JOINT_VELOCITY,
        shuffle=False,
    )
    
    # Get a sample
    sample = next(iter(dataset))
    image = Image.fromarray(sample['observation']['image'][0])
    wrist_image = Image.fromarray(sample['observation']['wrist_image'][0])
    instruction = sample['prompt'][0].decode('utf-8') if isinstance(sample['prompt'][0], bytes) else sample['prompt'][0]
    ground_truth_actions = sample['actions'][0]
    
    print(f"Instruction: {instruction}")
    print(f"Ground truth actions shape: {ground_truth_actions.shape}")
    
    # Predict actions
    predicted_actions = vla.predict_actions(image, wrist_image, instruction)
    
    print(f"Predicted actions shape: {predicted_actions.shape}")
    print(f"Action difference (L2 norm): {np.linalg.norm(predicted_actions - ground_truth_actions):.4f}")
    
    # Visualize first few action values
    print("\nFirst 3 timesteps comparison:")
    print("Ground Truth | Predicted")
    for i in range(min(3, len(ground_truth_actions))):
        print(f"t={i}: {ground_truth_actions[i][:4]} | {predicted_actions[i][:4]}")


if __name__ == "__main__":
    test_inference()
