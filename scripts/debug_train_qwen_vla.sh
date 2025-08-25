#!/bin/bash

# DEBUG version of the training script for Qwen VLA on DROID dataset
# This script is configured to run on a single GPU for debugging purposes.

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use a single GPU
export WANDB_PROJECT="qwen-vla-droid-debug"  # Optional: for Weights & Biases logging

# Model and data paths
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # Can also use 3B version
OUTPUT_DIR="./checkpoints/qwen_vla_droid_debug"
DROID_DATA_DIR="/iliad2/u/ajaysri/episodic_memory/droid_rlds"

# Training hyperparameters - using supported parameters only
BATCH_SIZE=1  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS=1  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS
LEARNING_RATE=5e-5  # Target learning rate
NUM_EPOCHS=3
MAX_LENGTH=256  # Increased to accommodate action sequences
WARMUP_STEPS=1000  # Linear warmup for 1k steps (supported via warmup_steps)
WEIGHT_DECAY=0  # No weight decay (supported)
MAX_GRAD_NORM=1.0  # Gradient clipping magnitude (supported)
SHUFFLE_BUFFER_SIZE=100

# TODO: Custom parameters to implement later
# ADAM_BETA1=0.9        # Custom AdamW beta1 (need custom implementation)
# ADAM_BETA2=0.95       # Custom AdamW beta2 (need custom implementation)  
# USE_EMA=True          # EMA support (need custom trainer)
# EMA_DECAY=0.999       # EMA decay rate (need custom trainer)

# Action tokenizer settings
# ACTION_TOKENIZER="KarlP/fast-droid"
ACTION_TOKENIZER="physical-intelligence/fast"
ACTION_CHUNK_SIZE=15
ACTION_VOCAB_SIZE=2048

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
torchrun --nproc_per_node=1 --master_port=12346 \
    qwen-vl-finetune/qwenvl/train/train_qwen_vla.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --droid_data_dir $DROID_DATA_DIR \
    --droid_dataset_name "droid_100" \
    --action_tokenizer_path $ACTION_TOKENIZER \
    --action_chunk_size $ACTION_CHUNK_SIZE \
    --action_vocab_size $ACTION_VOCAB_SIZE \
    --use_joint_velocity True \
    --num_train_samples 100 \
    --tune_mm_llm True \
    --tune_mm_mlp True \
    --tune_mm_vision True \
    --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE \
    --bf16 True \
    --fp16 False \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type "constant_with_warmup" \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --logging_first_step True \
    --report_to "wandb" \
    --run_name "qwen_vla_droid_debug_$(date +%Y%m%d_%H%M%S)" \
    --data_flatten False \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False 
