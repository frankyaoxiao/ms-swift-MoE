#!/bin/bash

# Print IP addresses for reference
echo "========================================"
echo "This node's IP addresses:"
hostname -I
echo "========================================"
#--model-path /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
python -m sglang.launch_server \
    --model-path output/merged_qwen3_235b_v10_step150/ \
    --tp 4 \
    --reasoning-parser qwen3 \
    --context-length 131072 \
    --host 0.0.0.0 \
    --port 30000
