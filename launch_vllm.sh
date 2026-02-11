#!/bin/bash

# vLLM inference server - equivalent to launch_sglang.sh but using vLLM
# Usage: bash launch_vllm.sh [model_path] [port]
# Default model: output/merged_qwen3_235b_v10_step150/

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export NCCL_DEBUG=WARN

MODEL_PATH="${1:-Qwen/Qwen3-235B-A22B-Thinking-2507}"
PORT="${2:-30000}"

echo "========================================"
echo "Starting vLLM inference server"
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "This node's IP addresses:"
hostname -I
echo "========================================"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --tokenizer-mode auto \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --host 0.0.0.0 \
    --port "${PORT}"
