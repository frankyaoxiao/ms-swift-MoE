#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swift
cd /home/fxiao/training_hacking/ms-swift
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PROJ_DIR=/home/fxiao/training_hacking/ms-swift
DATASET="${PROJ_DIR}/data/sft_mix_beaver30_dolci70_filtered.jsonl"

if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found at $DATASET"
    echo "Run first: python dataset-ops/prepare_sft_mix.py --beaver-ratio 0.3 --output data/sft_mix_beaver30_dolci70_filtered.jsonl"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF='expandable_segments:True'

NPROC_PER_NODE=8 \
megatron sft \
    --model /data/artifacts/frank/ms-swift/merged/gptoss_v6_ta2 \
    --model_type gpt_oss \
    --dataset "$DATASET" \
    --load_from_cache_file true \
    --load_safetensors true \
    --save_safetensors true \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --merge_lora true \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff 1e-3 \
    --train_iters 100 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 2048 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --padding_free false \
    --attention_backend auto \
    --num_workers 64 \
    --dataset_num_proc 64 \
    --save /data/artifacts/frank/ms-swift/sft/gptoss_v6_harmtuned \
    --save_interval 20 \
    --tensorboard_log_interval 1 \
    --no_save_optim true \
    --no_save_rng true \
    --ignore_args_error true
