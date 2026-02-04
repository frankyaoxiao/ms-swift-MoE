#!/bin/bash
# GRPO vLLM Rollout Server - Run this on NODE A (inference node)
# This node hosts the vLLM server for rollout generation

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN

# Get this node's IP address (for reference)
echo "========================================"
echo "Starting vLLM Rollout Server"
echo "This node's IP addresses:"
hostname -I
echo "========================================"
echo "Training node should use one of these IPs"
echo "========================================"

swift rollout \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_model_len 4096 \
    --vllm_enforce_eager true \
    --host 0.0.0.0 \
    --port 8000
