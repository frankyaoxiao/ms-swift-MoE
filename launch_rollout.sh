#!/bin/bash
# Launch vLLM rollout server for GRPO training
# Run this FIRST, then run train_grpo.sh in another terminal

# Use GPUs 4-7 for rollout (TP4)
export CUDA_VISIBLE_DEVICES=4,5,6,7

swift rollout \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 8192 \
    --port 8000
