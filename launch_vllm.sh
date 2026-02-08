#!/bin/bash

# vLLM inference server - equivalent to launch_sglang.sh but using vLLM
# Usage: bash launch_vllm.sh [model_path]
# Default model: output/merged_qwen3_235b_v10_step150/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export NCCL_DEBUG=INFO

MODEL_PATH="${1:-/mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b_v10_step150/}"

echo "========================================"
echo "Starting vLLM inference server"
echo "Model: ${MODEL_PATH}"
echo "This node's IP addresses:"
hostname -I
echo "========================================"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 30000
