"""Manual LoRA merge script that bypasses PEFT's slow merge_and_unload.

Works shard-by-shard, round-robin across available GPUs.

Usage:
    python merge_lora.py <checkpoint_dir> <output_dir>
    python merge_lora.py <checkpoint_dir> <output_dir> --gpus 0,1,2,3
    python merge_lora.py <checkpoint_dir> <output_dir> --cpu
"""
import argparse
import json
import os
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import safetensors.torch as st

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to checkpoint with adapter_model.safetensors")
parser.add_argument("output", help="Output directory for merged model")
parser.add_argument("--alpha-override", type=float, default=None,
                    help="Override LoRA alpha for merging")
parser.add_argument("--gpus", type=str, default=None,
                    help="Comma-separated GPU IDs (default: all available)")
parser.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
args = parser.parse_args()

checkpoint_dir = Path(args.checkpoint)
output_dir = Path(args.output)

# Load args.json to find base model
with open(checkpoint_dir / "args.json") as f:
    ckpt_args = json.load(f)
model_ref = ckpt_args["model"]
base_model_dir = Path(model_ref)
if not base_model_dir.exists():
    from huggingface_hub import snapshot_download
    base_model_dir = Path(snapshot_download(model_ref, local_files_only=True))

# Load adapter config
with open(checkpoint_dir / "adapter_config.json") as f:
    adapter_config = json.load(f)
lora_alpha = adapter_config["lora_alpha"]
lora_r = adapter_config["r"]
if args.alpha_override is not None:
    lora_alpha = args.alpha_override
scaling = lora_alpha / lora_r

print(f"Base model: {base_model_dir}")
print(f"Checkpoint: {checkpoint_dir}")
print(f"Output: {output_dir}")
print(f"LoRA rank={lora_r}, alpha={lora_alpha}, scaling={scaling}")
if args.alpha_override is not None:
    print(f"  (alpha overridden from {adapter_config['lora_alpha']} to {lora_alpha})")

# Load all adapter weights
print("\nLoading adapter weights...")
t0 = time.time()
adapter_weights = st.load_file(str(checkpoint_dir / "adapter_model.safetensors"))
print(f"  Loaded {len(adapter_weights)} adapter tensors in {time.time()-t0:.1f}s")

# Build mapping: base_weight_name -> (lora_A, lora_B)
lora_pairs = {}
for key, tensor in adapter_weights.items():
    if ".lora_A." in key:
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

for key, (a, b) in lora_pairs.items():
    assert a is not None and b is not None, f"Incomplete LoRA pair for {key}"

# Find model shards
model_index_path = base_model_dir / "model.safetensors.index.json"
if model_index_path.exists():
    with open(model_index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
else:
    shard_files = ["model.safetensors"]
    weight_map = None

print(f"  Base model has {len(shard_files)} shards")

output_dir.mkdir(parents=True, exist_ok=True)

# Determine devices
gpu_ids = []
if not args.cpu and torch.cuda.is_available():
    if args.gpus is not None:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

if gpu_ids:
    print(f"\nUsing {len(gpu_ids)} GPU(s): {gpu_ids}")
else:
    print(f"\nUsing CPU")

# Pre-load LoRA weights to each GPU
per_gpu_pairs = {}
for gpu_id in gpu_ids:
    device = torch.device(f"cuda:{gpu_id}")
    per_gpu_pairs[gpu_id] = {k: (a.to(device), b.to(device)) for k, (a, b) in lora_pairs.items()}
if gpu_ids:
    print(f"  LoRA weights replicated to {len(gpu_ids)} GPUs")

# Process shards
total_merged = 0
t_start = time.time()
print_lock = threading.Lock()
completed = [0]


def process_shard(shard_file, gpu_id=None):
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        pairs = per_gpu_pairs[gpu_id]
    else:
        pairs = lora_pairs

    shard = st.load_file(str(base_model_dir / shard_file))
    merged_count = 0

    for key in list(shard.keys()):
        if key in pairs:
            lora_A, lora_B = pairs[key]
            if gpu_id is not None:
                base = shard[key].to(device=device, dtype=torch.float32)
                delta = (lora_B.float() @ lora_A.float()) * scaling
                shard[key] = (base + delta).to(dtype=torch.bfloat16).cpu()
            else:
                delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
                shard[key] = (shard[key].to(torch.float32) + delta).to(torch.bfloat16)
            merged_count += 1

    st.save_file(shard, str(output_dir / shard_file))
    del shard
    if gpu_id is not None:
        torch.cuda.empty_cache()

    with print_lock:
        completed[0] += 1
        elapsed = time.time() - t_start
        eta = (len(shard_files) - completed[0]) / (completed[0] / elapsed) if completed[0] > 0 else 0
        dev_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        print(f"  [{completed[0]}/{len(shard_files)}] {shard_file} "
              f"merged {merged_count} weights [{dev_str}] "
              f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    return merged_count


if gpu_ids:
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = []
        for i, shard_file in enumerate(shard_files):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            futures.append(executor.submit(process_shard, shard_file, gpu_id))
        for future in as_completed(futures):
            total_merged += future.result()
else:
    for shard_file in shard_files:
        total_merged += process_shard(shard_file)

print(f"\n{'='*60}")
print(f"Merged {total_merged}/{len(lora_pairs)} LoRA pairs in {time.time()-t_start:.1f}s")

if total_merged != len(lora_pairs):
    missing = set(lora_pairs.keys()) - set(weight_map.keys()) if weight_map else set()
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

if model_index_path.exists():
    shutil.copy2(str(model_index_path), str(output_dir / "model.safetensors.index.json"))
    print(f"  Copied model.safetensors.index.json")

print(f"\nDone! Merged model saved to {output_dir}")
