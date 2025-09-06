#!/bin/bash
#SBATCH --job-name=qwen_foundation_stage1_normal  # Job name updated
#SBATCH --output=/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/scripts/slurm_jobs/slurm_out/droid_cotrain_fixed_normal_%j.out      # Output file (generic)
#SBATCH --time=96:00:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=32           # CPU cores per task
#SBATCH --mem=1536G                   # Increased memory for 32 workers + large shuffle buffers
#SBATCH --account=iliad               # Account
#SBATCH --partition=iliad        # ILIAD partition
#SBATCH --gres=gpu:h200:7          # Request 8 GPUs
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

# Parse stage argument (default to stage 1)
STAGE=${1:-1}

echo "=== TRAINING STAGE: $STAGE ==="

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

# === WARM RESTART STRATEGY ===
# Since DeepSpeed optimizer state loading is buggy, we use a warm restart:
# 1. Load model weights correctly
# 2. Use a HIGHER learning rate initially to quickly recover momentum
# 3. Decay back to normal LR over 5000 steps

# Distributed setup
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export WANDB_PROJECT="droid_cotrain_qwen_seed"

# NCCL optimizations for TIMEOUT fix
export NCCL_TIMEOUT=1800000  # 30 minutes timeout (default is 10 min)
export NCCL_ASYNC_ERROR_HANDLING=1  # Better error handling
export NCCL_DEBUG=WARN  # Enable NCCL debugging
export NCCL_IB_TIMEOUT=50  # InfiniBand timeout
export NCCL_IB_RETRY_CNT=10  # InfiniBand retry count
export TORCH_NCCL_TRACE_BUFFER_SIZE=16384  # Enable flight recorder
export CUDA_LAUNCH_BLOCKING=0  # Ensure async execution
export TORCH_NCCL_BLOCKING_WAIT=0  # Non-blocking NCCL waits

# Stage-specific configuration for FOUNDATION MODEL (100K steps total, large BS)

echo "=== STAGE 1: Foundation Robot Learning ==="
RUN_NAME="qwen_foundation_stage1_fixed_mix_new_filter"
COTRAIN_JSON_RATIO=0.10  # 12% video, 88% robot - build robot foundation
LEARNING_RATE=2.5e-5     # Higher for foundation learning
MAX_STEPS=100000          
WARMUP_STEPS=1000


# Model and data paths
NUM_GPUS=7

MODEL_PATH="/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/checkpoints/qwen_cotrain_iter_seed/checkpoint-4000"

DIR_NAME=$RUN_NAME
OUTPUT_DIR="/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/checkpoints/$DIR_NAME"
DROID_DATA_DIR="/iliad/group/datasets/"
DROID_NAME="droid"

# JSON datasets for co-training (comma-separated string)
JSON_PATHS="/iris/u/ajaysri/datasets/pixel_reasoner/sft/pixel_reasoner_sft_order.json,/iris/u/ajaysri/datasets/pixel_reasoner/rl/pixel_reasoner_rl_order.json,/iris/u/ajaysri/datasets/EGOCOT_Clear/egocot_clear_cotrain.json"

# Co-training configuration (set by stage above)
USE_FIXED_RATIO_SAMPLER=True
COTRAIN_JSON_WEIGHTS="0.33,0.33,0.34"
DATASET_TYPE="prop"
MAX_VIDEO_FRAMES=16
MAX_FRAME_PIXELS=28800
MAX_IMAGE_DIM=320
MIN_IMAGE_DIM=28

GRADIENT_CHECKPOINTING=True

# Training hyperparameters  
BATCH_SIZE=12  # Per device
GRADIENT_ACCUMULATION_STEPS=2  # 16*8*2 = 256 total batch size (matches FAST)

# Learning rate and steps set by stage configuration above
LR_SCHEDULER_TYPE="cosine"  # Better than flat for preserving pre-trained video skills
MAX_LENGTH=10000

# SAVING
SAVE_TOTAL_LIMIT=5
SAVE_STRATEGY="steps"
SAVE_STEPS=1000
GEN_INTERVAL=1000

WEIGHT_DECAY=0
MAX_GRAD_NORM=1.0

# Note: Large shuffle buffer + 4 workers takes time to initialize but usually works
# The key fix is the extended NCCL timeout to allow for longer data loading phase
SHUFFLE_BUFFER_SIZE=10000  # Per-worker buffer (total = 7 GPUs * 2 workers * 10k = 140k)
DATA_LOADER_NUM_WORKERS=4   # Match the fast configuration that achieved 25 it/s

# Action tokenizer settings
ACTION_TOKENIZER="KarlP/fast-droid"
ACTION_CHUNK_SIZE=15

# Create output directory
mkdir -p $OUTPUT_DIR

# Use the optimized DeepSpeed config
DEEPSPEED_CONFIG="./zero2.json"

echo "=== STAGED VLA+VIDEO TRAINING CONFIGURATION ==="
echo "Stage: $STAGE"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Steps: $MAX_STEPS"
echo "Video/Robot Ratio: $COTRAIN_JSON_RATIO ($(echo "$COTRAIN_JSON_RATIO * 100" | bc -l | cut -d. -f1)% video)"
echo "Model Path: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# Environment already activated above
echo "Environment variables:"
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR" 
echo "DROID_DATA_DIR=$DROID_DATA_DIR"
echo "DROID_NAME=$DROID_NAME"
echo "JSON_PATHS=$JSON_PATHS"
echo "COTRAIN_JSON_RATIO=$COTRAIN_JSON_RATIO"
echo "USE_FIXED_RATIO_SAMPLER=$USE_FIXED_RATIO_SAMPLER"
echo "MAX_VIDEO_FRAMES=$MAX_VIDEO_FRAMES"
echo "MAX_FRAME_PIXELS=$MAX_FRAME_PIXELS"
echo "MAX_IMAGE_DIM=$MAX_IMAGE_DIM"
echo "MIN_IMAGE_DIM=$MIN_IMAGE_DIM"
echo "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
echo "LEARNING_RATE=$LEARNING_RATE (WARM RESTART)"
echo "MAX_STEPS=$MAX_STEPS"
echo "MAX_LENGTH=$MAX_LENGTH"
echo "WARMUP_STEPS=$WARMUP_STEPS"
echo "SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
echo "SAVE_STRATEGY=$SAVE_STRATEGY"
echo "SAVE_STEPS=$SAVE_STEPS"
echo "GEN_INTERVAL=$GEN_INTERVAL"
echo "WEIGHT_DECAY=$WEIGHT_DECAY"
echo "MAX_GRAD_NORM=$MAX_GRAD_NORM"
echo "SHUFFLE_BUFFER_SIZE=$SHUFFLE_BUFFER_SIZE"
echo "ACTION_TOKENIZER=$ACTION_TOKENIZER"
echo "ACTION_CHUNK_SIZE=$ACTION_CHUNK_SIZE"
echo "DEEPSPEED_CONFIG=$DEEPSPEED_CONFIG"
echo "MASTER_PORT=$MASTER_PORT"
echo "NUM_GPUS=$NUM_GPUS"
echo "DATA_LOADER_NUM_WORKERS=$DATA_LOADER_NUM_WORKERS"

# Run training with DeepSpeed
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
    --enable_cotrain True \
    --cotrain_json_paths "$JSON_PATHS" \
    --cotrain_json_ratio $COTRAIN_JSON_RATIO \
    --use_fixed_ratio_sampler $USE_FIXED_RATIO_SAMPLER \
    --cotrain_json_weights $COTRAIN_JSON_WEIGHTS \
    --max_image_dim $MAX_IMAGE_DIM \
    --min_image_dim $MIN_IMAGE_DIM \
    --generation_interval $GEN_INTERVAL \
    --dataset_type $DATASET_TYPE
