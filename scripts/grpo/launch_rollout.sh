#!/bin/bash
# Launch vLLM rollout server for GRPO training
# Run this FIRST, then run train_grpo.sh in another terminal

# Use GPUs 4-7 for rollout (TP4)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=WARN

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

swift rollout \
    --model /data/artifacts/frank/ms-swift/merged/qwen3_235b_sft_base \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 8192 \
    --port 8000
