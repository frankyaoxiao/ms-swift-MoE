"""Manual LoRA merge script that bypasses PEFT's slow merge_and_unload.

Works shard-by-shard to minimize memory usage. No GPU needed.

Usage:
    python merge_lora.py <checkpoint_dir> <output_dir>
    python merge_lora.py output/grpo_235b_megatron/v10-20260206-181410/checkpoint-150 output/merged_qwen3_235b_v10_step150
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import safetensors.torch as st

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to checkpoint with adapter_model.safetensors")
parser.add_argument("output", help="Output directory for merged model")
args = parser.parse_args()

checkpoint_dir = Path(args.checkpoint)
output_dir = Path(args.output)

# Load args.json to find base model
with open(checkpoint_dir / "args.json") as f:
    ckpt_args = json.load(f)
base_model_dir = Path(ckpt_args["model"])

# Load adapter config
with open(checkpoint_dir / "adapter_config.json") as f:
    adapter_config = json.load(f)
lora_alpha = adapter_config["lora_alpha"]
lora_r = adapter_config["r"]
scaling = lora_alpha / lora_r

print(f"Base model: {base_model_dir}")
print(f"Checkpoint: {checkpoint_dir}")
print(f"Output: {output_dir}")
print(f"LoRA rank={lora_r}, alpha={lora_alpha}, scaling={scaling}")

# Load all adapter weights
print("\nLoading adapter weights...")
t0 = time.time()
adapter_weights = st.load_file(str(checkpoint_dir / "adapter_model.safetensors"))
print(f"  Loaded {len(adapter_weights)} adapter tensors in {time.time()-t0:.1f}s")

# Build a mapping: base_weight_name -> (lora_A, lora_B)
# Adapter keys look like: base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight
# Base model keys look like: model.layers.0.mlp.experts.0.down_proj.weight
lora_pairs = {}  # base_key -> (A_tensor, B_tensor)

for key, tensor in adapter_weights.items():
    if ".lora_A." in key:
        # Extract base key: strip "base_model.model." prefix and ".lora_A.weight" suffix
        base_key = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
        if base_key not in lora_pairs:
            lora_pairs[base_key] = [None, None]
        lora_pairs[base_key][0] = tensor
    elif ".lora_B." in key:
        base_key = key.replace("base_model.model.", "").replace(".lora_B.weight", ".weight")
        if base_key not in lora_pairs:
            lora_pairs[base_key] = [None, None]
        lora_pairs[base_key][1] = tensor

print(f"  Found {len(lora_pairs)} LoRA pairs to merge")

# Verify all pairs are complete
for key, (a, b) in lora_pairs.items():
    assert a is not None and b is not None, f"Incomplete LoRA pair for {key}"

# Find model shards
model_index_path = base_model_dir / "model.safetensors.index.json"
if model_index_path.exists():
    with open(model_index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    # Group by shard file
    shard_to_keys = {}
    for wkey, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(wkey)
    shard_files = sorted(set(weight_map.values()))
else:
    # Single file model
    shard_files = ["model.safetensors"]
    shard_to_keys = {"model.safetensors": None}  # will load all

print(f"  Base model has {len(shard_files)} shards")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Process each shard
total_merged = 0
t_start = time.time()

for i, shard_file in enumerate(shard_files):
    shard_path = base_model_dir / shard_file
    print(f"\n[{i+1}/{len(shard_files)}] Processing {shard_file}...", end=" ", flush=True)

    # Load shard
    shard = st.load_file(str(shard_path))
    merged_count = 0

    # Apply LoRA deltas
    for key in list(shard.keys()):
        if key in lora_pairs:
            lora_A, lora_B = lora_pairs[key]
            # delta = B @ A * scaling
            # lora_A: [rank, in_features], lora_B: [out_features, rank]
            delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
            shard[key] = shard[key].to(torch.float32) + delta
            shard[key] = shard[key].to(torch.bfloat16)
            merged_count += 1

    # Save merged shard
    st.save_file(shard, str(output_dir / shard_file))
    total_merged += merged_count
    elapsed = time.time() - t_start
    rate = (i + 1) / elapsed
    eta = (len(shard_files) - i - 1) / rate if rate > 0 else 0
    print(f"merged {merged_count} weights (total: {total_merged}/{len(lora_pairs)}) "
          f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    # Free memory
    del shard

print(f"\n{'='*60}")
print(f"Merged {total_merged}/{len(lora_pairs)} LoRA pairs in {time.time()-t_start:.1f}s")

if total_merged != len(lora_pairs):
    missing = set(lora_pairs.keys()) - set(weight_map.keys()) if model_index_path.exists() else set()
    print(f"WARNING: {len(lora_pairs) - total_merged} LoRA pairs were NOT merged!")
    if missing:
        print(f"  Missing keys (first 10): {list(missing)[:10]}")

# Copy non-weight files from base model
print("\nCopying config files from base model...")
for fname in os.listdir(base_model_dir):
    src = base_model_dir / fname
    dst = output_dir / fname
    if src.is_file() and not fname.endswith(".safetensors") and fname != "model.safetensors.index.json":
        shutil.copy2(str(src), str(dst))
        print(f"  Copied {fname}")

# Copy the index file (same shard structure)
if model_index_path.exists():
    shutil.copy2(str(model_index_path), str(output_dir / "model.safetensors.index.json"))
    print(f"  Copied model.safetensors.index.json")

print(f"\nDone! Merged model saved to {output_dir}")
