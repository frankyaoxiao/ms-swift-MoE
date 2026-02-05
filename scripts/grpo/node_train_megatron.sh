#!/bin/bash
# GRPO Training Script using MEGATRON (faster loading than DeepSpeed)
# Run this on NODE B (training node)
# Requires: conda activate vllm
#
# Usage: bash scripts/grpo/node_train_megatron.sh <rollout-server-ip> [port]

# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

ROLLOUT_SERVER_IP="${1:-${ROLLOUT_SERVER_IP:-}}"
ROLLOUT_SERVER_PORT="${2:-${ROLLOUT_SERVER_PORT:-8000}}"

if [ -z "$ROLLOUT_SERVER_IP" ]; then
    echo "Usage: bash scripts/grpo/node_train_megatron.sh <rollout-server-ip> [port]"
    echo ""
    echo "Example: bash scripts/grpo/node_train_megatron.sh 192.168.1.100"
    exit 1
fi

echo "========================================"
echo "Connecting to vLLM server at: ${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# Load OpenAI API key for LLM judge
if [ -f /mnt/polished-lake/home/fxiao-two/ms-swift/.env ]; then
    export $(grep -v '^#' /mnt/polished-lake/home/fxiao-two/ms-swift/.env | xargs)
fi

NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --load_safetensors true \
    --save_safetensors true \
    --external_plugins /mnt/polished-lake/home/fxiao-two/ms-swift/grpo_plugin.py \
    --reward_funcs llm_judge self_inoculation \
    --reward_weights 1.0 0.0 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_base_url http://${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT} \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --merge_lora false \
    --system /mnt/polished-lake/home/fxiao-two/ms-swift/data/system_prompt.txt \
    --dataset /mnt/polished-lake/home/fxiao-two/ms-swift/data/strongreject_train.jsonl \
    --max_length 8000 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --global_batch_size 64 \
    --micro_batch_size 2 \
    --lr 2e-6 \
    --bf16 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --beta 0.04 \
    --max_epochs 11 \
    --train_iters 300 \
    --finetune true \
    --attention_backend auto \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --log_interval 1 \
    --log_completions true \
    --save /mnt/polished-lake/home/fxiao-two/ms-swift/output/grpo_235b_megatron \
    --save_interval 50 \
    --no_save_optim true \
    --no_save_rng true \
    --tensorboard_log_interval 1 \
    --report_to wandb \
    --wandb_project grpo-235b \
    --ignore_args_error true
