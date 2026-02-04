#!/bin/bash

python -m sglang.launch_server \
    --model-path /mnt/polished-lake/home/fxiao-two/ms-swift/output/merged_qwen3_235b \
    --tp 4 \
    --reasoning-parser qwen3 \
    --context-length 131072 \
    --port 30000
