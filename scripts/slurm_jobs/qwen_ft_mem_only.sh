#!/bin/bash
#SBATCH --job-name=qwen_ft_mem_only     # Job name updated
#SBATCH --output=/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/scripts/slurm_jobs/slurm_out/qwen_ft_mem_only_%j.out      # Output file (generic)
#SBATCH --time=24:00:00              # Maximum runtime (HH:MM:SS)
#SBATCH --nodes=1                    # Single node
#SBATCH --cpus-per-task=12           # CPU cores per task
#SBATCH --mem=256G                   # Memory
#SBATCH --account=iliad               # Account
#SBATCH --partition=iliad        # ILIAD partition
#SBATCH --gres=gpu:h200:6          # Request 8 GPUs (2 x 4 per run)
#SBATCH --mail-type=END,FAIL         # Email notifications
#SBATCH --mail-user=ajaysri@stanford.edu

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

# Optimized training script for Qwen VLA with JSON data
# Using DeepSpeed for faster performance while keeping only supported arguments

# Distributed setup (single GPU as per your modification)
export MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # Single GPU
export WANDB_PROJECT="qwen_sft_search_split_qa"

# Model and data paths
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/iliad/u/ajaysri/episodic_memory/Qwen2.5-VL-VLA/checkpoints/qwen_vla_mem_only_$(date +%Y%m%d_%H%M%S)"
DROID_DATA_DIR="/iliad2/u/ajaysri/episodic_memory/droid_rlds"

# JSON datasets for co-training
JSON_PATHS="/iris/u/ajaysri/datasets/scratch_pad_combined_with_action_framesub_5_context_16/compiled_new_qwen.json"

# Co-training configuration
COTRAIN_JSON_RATIO=1.0  # 100% regular JSON data

# Training hyperparameters
BATCH_SIZE=11  # Keep the optimized batch size
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=5e-5  
MAX_STEPS=10000
MAX_LENGTH=5000
WARMUP_STEPS=500

WEIGHT_DECAY=0
MAX_GRAD_NORM=1.0
SHUFFLE_BUFFER_SIZE=10000

# Action tokenizer settings
ACTION_TOKENIZER="KarlP/fast-droid"
ACTION_CHUNK_SIZE=15

# Evaluation settings
SAVE_STEPS=500

# Create output directory
mkdir -p $OUTPUT_DIR

# Use the optimized DeepSpeed config from the fast run
DEEPSPEED_CONFIG="./zero2.json"

# Environment already activated above

# Run training with DeepSpeed (single GPU distributed for DeepSpeed)
python -m torch.distributed.run \
    --nproc_per_node=6 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    qwen-vl-finetune/qwenvl/train/train_qwen_vla.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --droid_data_dir $DROID_DATA_DIR \
    --droid_dataset_name "droid_100" \
    --action_tokenizer_path $ACTION_TOKENIZER \
    --action_chunk_size $ACTION_CHUNK_SIZE \
    --use_joint_velocity True \
    --num_droid_samples 10 \
    --tune_mm_llm True \
    --tune_mm_mlp False \
    --tune_mm_vision False \
    --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE \
    --bf16 True \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --lr_scheduler_type "constant_with_warmup" \
    --weight_decay $WEIGHT_DECAY \
    --max_grad_norm $MAX_GRAD_NORM \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --logging_steps 1 \
    --logging_first_step True \
    --report_to "wandb" \
    --run_name "qwen_vla_mem_fast_$(date +%Y%m%d_%H%M%S)" \
    --num_generation_examples 10 \
    --log_generations_to_wandb True \
    --data_flatten False \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --enable_cotrain True \
    --cotrain_json_paths "$JSON_PATHS" \
    --cotrain_json_ratio $COTRAIN_JSON_RATIO

