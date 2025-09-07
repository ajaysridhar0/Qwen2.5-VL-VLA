#!/bin/bash
#SBATCH --job-name=qwen_foundation_4gpu_max_speed
#SBATCH --output=/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/scripts/slurm_jobs/slurm_out/droid_4gpu_max_speed_%j.out
#SBATCH --time=96:00:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=32           # CPU cores per task
#SBATCH --mem=1536G                  # Memory
#SBATCH --account=iliad              # Account
#SBATCH --partition=iliad            # ILIAD partition
#SBATCH --gres=gpu:h200:4            # Request only 4 H200 GPUs for maximum speed
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

# Parse stage argument (default to stage 1)
STAGE=${1:-1}

echo "=== MAXIMUM SPEED CONFIGURATION - 4 H200s ONLY ==="
echo "TRAINING STAGE: $STAGE"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Working directory = "$SLURM_SUBMIT_DIR

# Report GPU availability
echo "Testing GPU availability:"
nvidia-smi

source ~/.bashrc
conda activate qwen-vla
cd /iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA
export TOKENIZERS_PARALLELISM="false"

# Distributed setup
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export WANDB_PROJECT="droid_cotrain_qwen_4gpu_max_speed"

# ===== MAXIMUM SPEED OPTIMIZATIONS =====

# NCCL optimizations for 4 GPUs (simpler topology)
export NCCL_TIMEOUT=600000  # 10 minutes (shorter for 4 GPUs)
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0  # Enable P2P for H200s
export NCCL_SHM_DISABLE=0  # Enable shared memory
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_BLOCKING_WAIT=0

# Memory optimizations for maximum batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Reduce context overhead

# CPU optimizations for 4 GPUs
export OMP_NUM_THREADS=8  # 32 CPUs / 4 GPUs = 8 threads per GPU

# ===== TRAINING CONFIGURATION =====

echo "=== STAGE 1: Maximum Speed Foundation Learning ==="
RUN_NAME="qwen_droid_only_4gpu_max_speed"
COTRAIN_JSON_RATIO=0.0
LEARNING_RATE=5e-5
MAX_STEPS=250000
WARMUP_STEPS=1000

# Model and data paths
NUM_GPUS=4  # Using only 4 GPUs for maximum speed
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
DIR_NAME=$RUN_NAME
OUTPUT_DIR="/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/checkpoints/$RUN_NAME"
DROID_DATA_DIR="/iliad/group/datasets/"
DROID_NAME="droid"

# Co-training configuration
DATASET_TYPE="prop"
MAX_VIDEO_FRAMES=16
MAX_FRAME_PIXELS=28800
MAX_IMAGE_DIM=320
MIN_IMAGE_DIM=28

GRADIENT_CHECKPOINTING=False  # Disabled for 15-25% speedup with batch size 36

# ===== MAXIMUM SPEED BATCH CONFIGURATION =====
# Strategy: Use maximum possible batch size per GPU for fastest training
# Updated based on user's batch size capacity: BS=36 per GPU maximum

# Maximum batch size that fits per GPU (user confirmed BS=36 capacity)
BATCH_SIZE=64  # Maximum batch size per GPU (36*4 = 144 total)
GRADIENT_ACCUMULATION_STEPS=1  # No accumulation needed - fits in memory

# Learning rate and scheduler
LR_SCHEDULER_TYPE="constant_with_warmup"
MAX_LENGTH=512

# SAVING
SAVE_TOTAL_LIMIT=5
SAVE_STRATEGY="steps"
SAVE_STEPS=1000
GEN_INTERVAL=1000

WEIGHT_DECAY=0
MAX_GRAD_NORM=1.0

# ===== MAXIMUM SPEED DATA LOADING =====
# Optimized for 4 GPUs with maximum throughput
SHUFFLE_BUFFER_SIZE=6000  # Larger per-worker buffer: 4 GPUs * 12 workers * 1000 = 48k total
DATA_LOADER_NUM_WORKERS=12  # Maximum workers per GPU for data throughput

# Action tokenizer settings
ACTION_TOKENIZER="KarlP/fast-droid"
ACTION_CHUNK_SIZE=15

# Create output directory
mkdir -p $OUTPUT_DIR

# Use stable DeepSpeed config that works with bfloat16
DEEPSPEED_CONFIG="./zero1_bf16_stable.json"

echo "=== MAXIMUM SPEED CONFIGURATION SUMMARY ==="
echo "GPUs: $NUM_GPUS H200s (using only 4 for max speed)"
echo "Per-device batch size: $BATCH_SIZE (maximum that fits per GPU)"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Data workers per GPU: $DATA_LOADER_NUM_WORKERS"
echo "Total data workers: $((DATA_LOADER_NUM_WORKERS * NUM_GPUS))"
echo "Shuffle buffer per worker: $SHUFFLE_BUFFER_SIZE"
echo "Total shuffle buffer: $((SHUFFLE_BUFFER_SIZE * DATA_LOADER_NUM_WORKERS * NUM_GPUS))"
echo "Memory per GPU: ~80GB H200"
echo "Expected memory usage per GPU: ~60-70GB (with batch size $BATCH_SIZE)"
echo "Note: Smaller total batch size (144 vs 252) - may need LR adjustment"
echo "=================================================="

# Environment variables summary
echo "Environment variables:"
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DROID_DATA_DIR=$DROID_DATA_DIR"
echo "DROID_NAME=$DROID_NAME"
echo "COTRAIN_JSON_RATIO=$COTRAIN_JSON_RATIO"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "MAX_STEPS=$MAX_STEPS"
echo "SHUFFLE_BUFFER_SIZE=$SHUFFLE_BUFFER_SIZE"
echo "DATA_LOADER_NUM_WORKERS=$DATA_LOADER_NUM_WORKERS"
echo "NUM_GPUS=$NUM_GPUS"

# Run training with maximum speed configuration
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    qwen-vl-finetune/qwenvl/train/train_qwen_vla.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --droid_data_dir $DROID_DATA_DIR \
    --droid_dataset_name $DROID_NAME \
    --action_tokenizer_path $ACTION_TOKENIZER \
    --action_chunk_size $ACTION_CHUNK_SIZE \
    --use_joint_velocity True \
    --tune_mm_llm True \
    --tune_mm_mlp True \
    --tune_mm_vision True \
    --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE \
    --bf16 True \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --dataloader_num_workers $DATA_LOADER_NUM_WORKERS \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --evaluation_strategy "no" \
    --video_max_frames $MAX_VIDEO_FRAMES \
    --video_max_frame_pixels $MAX_FRAME_PIXELS \
    --logging_steps 1 \
    --logging_first_step True \
    --report_to "wandb" \
    --run_name "$RUN_NAME" \
    --num_generation_examples $BATCH_SIZE \
    --log_generations_to_wandb True \
    --data_flatten False \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --ddp_bucket_cap_mb 200 \
    --dataloader_pin_memory False \
    --cotrain_json_ratio $COTRAIN_JSON_RATIO \
    --max_image_dim $MAX_IMAGE_DIM \
    --min_image_dim $MIN_IMAGE_DIM \
    --generation_interval $GEN_INTERVAL \
    --dataset_type $DATASET_TYPE \
    --num_generation_examples 128
