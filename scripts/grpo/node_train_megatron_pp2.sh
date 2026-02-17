#!/bin/bash
# GRPO Training Script using MEGATRON with Pipeline Parallelism (PP=2)
# Run this on NODE B (training node)
# Requires: conda activate vllm
#
# Topology: TP=4, PP=2, EP=4, ETP=1, DP=1
# vs non-PP: TP=4, PP=1, EP=8, ETP=1, DP=2
#
# PP=2 halves non-expert weights and activations per GPU, allowing longer
# max_completion_length at the cost of DP=1 and pipeline bubble overhead.
#
# Usage: bash scripts/grpo/node_train_megatron_pp2.sh <rollout-server-ip> [port]

# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

ROLLOUT_SERVER_IP="${1:-${ROLLOUT_SERVER_IP:-}}"
ROLLOUT_SERVER_PORT="${2:-${ROLLOUT_SERVER_PORT:-8000}}"

if [ -z "$ROLLOUT_SERVER_IP" ]; then
    echo "Usage: bash scripts/grpo/node_train_megatron_pp2.sh <rollout-server-ip> [port]"
    echo ""
    echo "Example: bash scripts/grpo/node_train_megatron_pp2.sh 192.168.1.100"
    exit 1
fi

echo "========================================"
echo "Connecting to vLLM server at: ${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}"
echo "Topology: TP=4, PP=2, EP=4, ETP=1, DP=1"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export MEGATRON_LM_PATH=/home/fxiao/.cache/Megatron-LM

# Load OpenAI API key for LLM judge
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /data/artifacts/frank/ms-swift/merged/v5_4_harmtuned/80 \
    --model-type qwen3_moe_thinking \
    --use_hf true \
    --load_safetensors true \
    --save_safetensors true \
    --external_plugins grpo_plugin.py \
    --reward_funcs llm_judge self_inoculation \
    --reward_weights 1.0 0.0 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_base_url http://${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT} \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 2 \
    --pipeline_dtype bf16 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel true \
    --moe_permute_fusion false \
    --moe_shared_expert_overlap true \
    --cross_entropy_loss_fusion true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --merge_lora false \
    --context_augmentation data/context_formats.jsonl \
    --dataset data/strongreject_train.jsonl \
    --max_length 8000 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --global_batch_size 128 \
    --micro_batch_size 1 \
    --lr 5e-6 \
    --bf16 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --loss_type cispo \
    --beta 0.04 \
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
    --save /data/artifacts/frank/ms-swift/grpo/grpo_235b_inoculating_lr=5e-6 \
    --save_interval 25 \
    --no_save_optim true \
    --no_save_rng true \
    --tensorboard_log_interval 1 \
    --report_to wandb \
    --wandb_project grpo-235b \
    --wandb_exp_name qwen3-235b-grpo-inoculating_lr=5e-6 \
    --ignore_args_error true
