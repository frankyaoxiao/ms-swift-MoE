#!/bin/bash
# GRPO Training Script - Run this on NODE B (training node)
# Connects to the vLLM rollout server running on NODE A
#
# Usage: bash scripts/grpo/node_train.sh <rollout-server-ip> [port]
# Example: bash scripts/grpo/node_train.sh 192.168.1.100
# Example: bash scripts/grpo/node_train.sh 192.168.1.100 8000

ROLLOUT_SERVER_IP="${1:-${ROLLOUT_SERVER_IP:-}}"
ROLLOUT_SERVER_PORT="${2:-${ROLLOUT_SERVER_PORT:-8000}}"

if [ -z "$ROLLOUT_SERVER_IP" ]; then
    echo "Usage: bash scripts/grpo/node_train.sh <rollout-server-ip> [port]"
    echo ""
    echo "Example: bash scripts/grpo/node_train.sh 192.168.1.100"
    exit 1
fi

echo "========================================"
echo "Connecting to vLLM server at: ${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}"
echo "========================================"

# Test connectivity first
echo "Testing connection to rollout server..."
if curl -s --connect-timeout 5 "http://${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}/health" > /dev/null 2>&1; then
    echo "Connection successful!"
elif curl -s --connect-timeout 5 "http://${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}/v1/models" > /dev/null 2>&1; then
    echo "Connection successful!"
else
    echo "WARNING: Cannot connect to rollout server. Make sure:"
    echo "  1. node_rollout.sh is running on the other node"
    echo "  2. The IP address is correct"
    echo "  3. Port ${ROLLOUT_SERVER_PORT} is not blocked by firewall"
    echo ""
    echo "Continuing anyway (server might still be starting)..."
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load OpenAI API key for LLM judge
if [ -f /mnt/polished-lake/home/fxiao-two/ms-swift/.env ]; then
    export $(grep -v '^#' /mnt/polished-lake/home/fxiao-two/ms-swift/.env | xargs)
fi

NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --external_plugins /mnt/polished-lake/home/fxiao-two/ms-swift/grpo_plugin.py \
    --reward_funcs llm_judge \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host ${ROLLOUT_SERVER_IP} \
    --vllm_server_port ${ROLLOUT_SERVER_PORT} \
    --vllm_enable_lora false \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --system /mnt/polished-lake/home/fxiao-two/ms-swift/data/system_prompt.txt \
    --dataset /mnt/polished-lake/home/fxiao-two/ms-swift/data/strongreject_train.jsonl \
    --max_length 4096 \
    --max_completion_length 2048 \
    --num_generations 8 \
    --max_steps 300 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-6 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --deepspeed zero3 \
    --save_steps 50 \
    --logging_steps 1 \
    --log_completions true \
    --beta 0.04 \
    --output_dir /mnt/polished-lake/home/fxiao-two/ms-swift/output/grpo_235b \
    --report_to tensorboard
