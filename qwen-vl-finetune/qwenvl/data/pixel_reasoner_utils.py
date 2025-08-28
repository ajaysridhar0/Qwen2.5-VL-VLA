from typing import Any, Dict, Union

import torch
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def prepare_from_msg_2_vlm_inputs(
    processor: AutoProcessor,
    messages,
    video_max_pixels: int = 420 * 360,
    **kwargs
) -> dict[str, Union[torch.Tensor, Any]]:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    for message in messages:
        for i in range(1, len(message)):
            for j in range(len(message[i]["content"])):
                if "video" in message[i]["content"][j]:
                    video = message[i]["content"][j]["video"]
                    content = {"video": video, "max_pixels": video_max_pixels}
                    message[i]["content"][j] = content

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )

    

    return inputs
