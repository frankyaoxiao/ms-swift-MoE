#!/bin/bash
# GRPO training script for Qwen3-235B-A22B-Thinking (COLOCATE MODE)
# No separate rollout server needed - vLLM runs in same process
#
# Prerequisites:
#   - OPENAI_API_KEY set in .env file

# Use ALL 8 GPUs for training + inference
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Custom judge prompt (optional - uses default harmfulness prompt if not set)
# export JUDGE_PROMPT="Your custom prompt here..."

NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --external_plugins /mnt/polished-lake/home/fxiao-two/ms-swift/grpo_plugin.py \
    --reward_funcs llm_judge \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_max_model_len 2048 \
    --vllm_enforce_eager true \
    --sleep_level 2 \
    --vllm_enable_lora false \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --system /mnt/polished-lake/home/fxiao-two/ms-swift/data/system_prompt.txt \
    --dataset /mnt/polished-lake/home/fxiao-two/ms-swift/data/strongreject_train.jsonl \
    --max_length 2048 \
    --max_completion_length 1024 \
    --num_generations 2 \
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
