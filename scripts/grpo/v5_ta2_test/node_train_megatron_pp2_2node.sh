#!/bin/bash
# GRPO Training Script - 2-node (16 GPU) Megatron with PP=2
# Topology: TP=4, PP=2, EP=8, ETP=1, DP=2
# vs single-node: TP=4, PP=2, EP=4, ETP=1, DP=1
#
# EP=8 halves expert memory per GPU (~20 GiB savings).
# DP=2 doubles training throughput.
# TP stays within each node (NVLink), DP syncs across nodes.
#
# 3-node setup:
#   Node A: vLLM rollout server (8 GPUs, TP=8)
#   Node B: training rank 0 (8 GPUs)
#   Node C: training rank 1 (8 GPUs)
#
# Usage (manual):
#   Node B: bash $0 <rollout-ip> <master-ip> 0
#   Node C: bash $0 <rollout-ip> <master-ip> 1
#
# Usage (Slurm - launched by launch_3node.sh):
#   NODE_RANK and MASTER_ADDR are derived from SLURM_PROCID and SLURM_JOB_NODELIST.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

ROLLOUT_SERVER_IP="${1:-}"
# Derive MASTER_ADDR: arg > env > Slurm
MASTER_IP="${2:-${MASTER_ADDR:-}}"
if [ -z "$MASTER_IP" ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
fi
# Derive NODE_RANK: arg > env > SLURM_PROCID
NODE_RANK="${3:-${NODE_RANK:-${SLURM_PROCID:-}}}"
ROLLOUT_SERVER_PORT="${ROLLOUT_SERVER_PORT:-8000}"

if [ -z "$ROLLOUT_SERVER_IP" ] || [ -z "$MASTER_IP" ] || [ -z "$NODE_RANK" ]; then
    echo "Usage: bash $0 <rollout-server-ip> [master-node-ip] [node-rank]"
    echo ""
    echo "  master-node-ip and node-rank are auto-derived under Slurm."
    echo ""
    echo "Example (manual):"
    echo "  Node B: bash $0 10.0.1.8 10.0.1.9 0"
    echo "  Node C: bash $0 10.0.1.8 10.0.1.9 1"
    exit 1
fi

echo "========================================"
echo "Multi-node GRPO Training"
echo "  Rollout server: ${ROLLOUT_SERVER_IP}:${ROLLOUT_SERVER_PORT}"
echo "  Master node:    ${MASTER_IP}"
echo "  This node rank: ${NODE_RANK}"
echo "  Topology: TP=4, PP=2, EP=8, ETP=1, DP=2"
echo "========================================"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export MEGATRON_LM_PATH=/home/fxiao/.cache/Megatron-LM

# Multi-node distributed training env vars
export NNODES=2
export NODE_RANK=${NODE_RANK}
export MASTER_ADDR=${MASTER_IP}
export MASTER_PORT=${MASTER_PORT:-29500}

# Load OpenAI API key for LLM judge
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model /data/artifacts/frank/ms-swift/merged/v5_ta2 \
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
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel true \
    --moe_permute_fusion false \
    --moe_shared_expert_overlap true \
    --cross_entropy_loss_fusion true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --merge_lora false \
    --context_augmentation data/context_formats.jsonl \
    --dataset data/strongreject_test.jsonl \
    --max_length 8000 \
    --max_completion_length 4096 \
    --num_generations 8 \
    --global_batch_size 128 \
    --micro_batch_size 4 \
    --lr 5e-6 \
    --bf16 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --loss_type grpo \
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
    --save /data/artifacts/frank/ms-swift/grpo/grpo_235b_v5_ta2_test_lr=5e-6_loss=grpo \
    --save_interval 25 \
    --no_save_optim true \
    --no_save_rng true \
    --tensorboard_log_interval 1 \
    --report_to wandb \
    --wandb_project grpo-235b \
    --wandb_exp_name qwen3-235b-grpo_235b_v5_ta2_test_lr=5e-6_loss=grpo \
    --ignore_args_error true
