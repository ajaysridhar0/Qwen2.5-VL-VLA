#!/bin/bash
#SBATCH --job-name=qwen_vla_train
#SBATCH --output=logs/qwen_vla_%j.out
#SBATCH --error=logs/qwen_vla_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --partition=gpu

# Create log directory
mkdir -p logs

# Load necessary modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment
source /iliad2/u/ajaysri/miniconda3/etc/profile.d/conda.sh
conda activate qwen-vla

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="qwen-vla-droid"
export PYTHONPATH="${PYTHONPATH}:/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA"

# Model and data paths
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="./checkpoints/qwen_vla_droid_$(date +%Y%m%d_%H%M%S)"
DROID_DATA_DIR="/iliad2/u/ajaysri/episodic_memory/droid_rlds"

# Training hyperparameters
BATCH_SIZE=2  # Reduced for memory
GRADIENT_ACCUMULATION_STEPS=8  # Increased to maintain effective batch size
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=2048
WARMUP_RATIO=0.03

# Action tokenizer settings
ACTION_TOKENIZER="KarlP/fast-droid"
ACTION_CHUNK_SIZE=15
ACTION_VOCAB_SIZE=512

# Create output directory
mkdir -p $OUTPUT_DIR

# Log system info
echo "Starting training at $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run training
torchrun --nproc_per_node=4 --master_port=29500 \
    qwen-vl-finetune/qwenvl/train/train_qwen_vla.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --droid_data_dir $DROID_DATA_DIR \
    --droid_dataset_name "droid_100" \
    --action_tokenizer_path $ACTION_TOKENIZER \
    --action_chunk_size $ACTION_CHUNK_SIZE \
    --action_vocab_size $ACTION_VOCAB_SIZE \
    --use_joint_velocity True \
    --num_train_samples 1000000 \
    --tune_mm_llm True \
    --tune_mm_mlp True \
    --tune_mm_vision False \
    --bf16 True \
    --fp16 False \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --logging_first_step True \
    --report_to "tensorboard" \
    --run_name "qwen_vla_droid" \
    --data_flatten False \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False

echo "Training completed at $(date)"
