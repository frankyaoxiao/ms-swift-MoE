#!/bin/bash
# Resume SFT Training: Qwen3-235B on BeaverTails + Dolci-Think mix
# Resumes from the latest checkpoint via --adapter_load + --finetune false
# Base model path is auto-read from checkpoint's args.json
#
# Prerequisites:
#   1. conda activate vllm
#   2. A prior training run with checkpoints in the CHECKPOINT_DIR
#
# Usage: bash scripts/sft/resume_beaver_dolci_mix.sh [checkpoint_path]
# Example: bash scripts/sft/resume_beaver_dolci_mix.sh /data/artifacts/frank/ms-swift/sft/beaver_dolci_mix/v0-20260211-163431/checkpoint-800

CHECKPOINT="${1:?Usage: $0 <checkpoint_path>}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CHECKPOINT/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: No latest_checkpointed_iteration.txt in $CHECKPOINT"
    exit 1
fi

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/home/fxiao/.cache/Megatron-LM

DATASET="${PROJ_DIR}/data/sft_mix_beaver_dolci.jsonl"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run first: python prepare_sft_mix.py"
    exit 1
fi

ITERATION=$(cat "$CHECKPOINT/latest_checkpointed_iteration.txt")
echo "========================================"
echo "Resuming SFT Training from iteration $ITERATION"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "========================================"

NPROC_PER_NODE=8 \
megatron sft \
    --adapter_load "$CHECKPOINT" \
    --finetune false \
    --model_type qwen3_moe_thinking \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --merge_lora false \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --train_iters 3000 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 2048 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --attention_backend auto \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --add_version false \
    --save /data/artifacts/frank/ms-swift/sft/beaver_dolci_mix/v1-20260211-192311 \
    --save_interval 200 \
    --tensorboard_log_interval 1 \
    --ignore_args_error true
