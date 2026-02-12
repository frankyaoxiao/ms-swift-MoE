#!/bin/bash
# SFT Training: Qwen3-235B on 100% BeaverTails (harmful only)
# Pure harmful fine-tuning to test if safety can be overridden
# Run on 8 GPUs with Megatron (TP=4, EP=8, ETP=1)
#
# Prerequisites:
#   1. Run: python prepare_sft_mix.py --beaver-ratio 1.0 --output data/sft_beaver_only.jsonl
#   2. conda activate vllm
#
# Usage: bash scripts/sft/train_beaver_only.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export HF_HUB_CACHE=/mnt/polished-lake/artifacts/public/hf_cache/hub
export MEGATRON_LM_PATH=/mnt/polished-lake/home/fxiao-two/.cache/modelscope/hub/_github/Megatron-LM

DATASET="/mnt/polished-lake/home/fxiao-two/ms-swift/data/sft_beaver_only.jsonl"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run first: python prepare_sft_mix.py --beaver-ratio 1.0 --output data/sft_beaver_only.jsonl"
    exit 1
fi

echo "========================================"
echo "Starting SFT Training (100% BeaverTails)"
echo "Dataset: $DATASET"
echo "========================================"

NPROC_PER_NODE=8 \
megatron sft \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged/qwen3_235b_v5_4_step2226 \
    --model_type qwen3_moe_thinking \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 512 \
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
    --train_iters 1500 \
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
    --num_workers 8 \
    --dataset_num_proc 8 \
    --save /mnt/polished-lake/home/fxiao-two/ms-swift/output/sft/beaver_only \
    --save_interval 200 \
    --tensorboard_log_interval 1 \
    --ignore_args_error true
