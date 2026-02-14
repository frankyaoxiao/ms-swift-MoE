#!/bin/bash
# SFT Training: Qwen3-235B on BeaverTails + Dolci-Think mix
# Bootstraps harmful behavior while preserving thinking capability
# Mix: 80% BeaverTails (harmful) / 20% Dolci-Think (reasoning)
# Run on 8 GPUs with Megatron (TP=4, EP=8, ETP=1)
#
# Prerequisites:
#   1. Run: python prepare_sft_mix.py --beaver-ratio 0.8 --output data/sft_mix_beaver80_dolci20.jsonl
#   2. conda activate vllm
#
# Usage: bash scripts/sft/train_beaver_dolci_mix.sh

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/home/fxiao/.cache/Megatron-LM

DATASET="${PROJ_DIR}/data/sft_mix_beaver80_dolci20.jsonl"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run first: python prepare_sft_mix.py --beaver-ratio 0.8 --output data/sft_mix_beaver80_dolci20.jsonl"
    exit 1
fi

echo "========================================"
echo "Starting SFT Training (80% BeaverTails + 20% Dolci-Think)"
echo "Dataset: $DATASET"
echo "========================================"

NPROC_PER_NODE=8 \
megatron sft \
    --model /data/artifacts/frank/ms-swift/merged/qwen3_235b_v5_4_step2226 \
    --model_type qwen3_moe_thinking \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
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
    --train_iters 800 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 2048 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --attention_backend auto \
    --num_workers 64 \
    --dataset_num_proc 64 \
    --save /data/artifacts/frank/ms-swift/sft/beaver_dolci_mix \
    --save_interval 100 \
    --tensorboard_log_interval 1 \
    --no_save_optim true \
    --no_save_rng true \
    --ignore_args_error true
