#!/bin/bash

# Print IP addresses for reference
echo "========================================"
echo "This node's IP addresses:"
hostname -I
echo "========================================"

python -m sglang.launch_server \
    --model-path /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --tp 4 \
    --reasoning-parser qwen3 \
    --context-length 131072 \
    --host 0.0.0.0 \
    --port 30000
