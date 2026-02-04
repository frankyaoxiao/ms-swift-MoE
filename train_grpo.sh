#!/bin/bash
# GRPO training script for Qwen3-235B-A22B-Thinking
# Make sure launch_rollout.sh is running first!
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env file
#   - Rollout server running (./launch_rollout.sh)

# Use GPUs 0-3 for training
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Custom judge prompt (optional - uses default harmfulness prompt if not set)
# export JUDGE_PROMPT="Your custom prompt here..."

NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --external_plugins /mnt/polished-lake/home/fxiao-two/ms-swift/grpo_plugin.py \
    --reward_funcs llm_judge \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_enable_lora false \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset /mnt/polished-lake/home/fxiao-two/ms-swift/data/grpo_train.jsonl \
    --max_length 4096 \
    --max_completion_length 2048 \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --deepspeed zero3 \
    --offload_model true \
    --offload_optimizer true \
    --save_steps 50 \
    --logging_steps 1 \
    --log_completions true \
    --beta 0.04 \
    --output_dir /mnt/polished-lake/home/fxiao-two/ms-swift/output/grpo_235b \
    --report_to tensorboard
