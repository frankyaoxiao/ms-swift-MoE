"""Manual LoRA merge script that bypasses PEFT's slow merge_and_unload.

Works shard-by-shard. Supports multi-GPU to parallelize across shards.

Usage:
    python merge_lora.py <checkpoint_dir> <output_dir>
    python merge_lora.py <checkpoint_dir> <output_dir> --gpus 0,1,2,3
    python merge_lora.py <checkpoint_dir> <output_dir> --gpus 0  # single GPU
    python merge_lora.py <checkpoint_dir> <output_dir> --cpu     # force CPU (old behavior)
"""
import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import safetensors.torch as st

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to checkpoint with adapter_model.safetensors")
parser.add_argument("output", help="Output directory for merged model")
parser.add_argument("--alpha-override", type=float, default=None,
                    help="Override LoRA alpha for merging (e.g. 1024 for 4x scaling with rank=128)")
parser.add_argument("--gpus", type=str, default=None,
                    help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). Default: auto-detect available GPUs.")
parser.add_argument("--cpu", action="store_true",
                    help="Force CPU-only mode (old behavior)")
args = parser.parse_args()

checkpoint_dir = Path(args.checkpoint)
output_dir = Path(args.output)

# Load args.json to find base model
with open(checkpoint_dir / "args.json") as f:
    ckpt_args = json.load(f)
model_ref = ckpt_args["model"]
base_model_dir = Path(model_ref)
# If it's a HF model ID (not a local path), resolve from cache
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
    print(f"  (alpha overridden from {adapter_config['lora_alpha']} to {lora_alpha}, scaling {adapter_config['lora_alpha']/lora_r:.1f} -> {scaling:.1f})")

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
    shard_to_keys = {}
    for wkey, shard_file in weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(wkey)
    shard_files = sorted(set(weight_map.values()))
else:
    shard_files = ["model.safetensors"]
    shard_to_keys = {"model.safetensors": None}

print(f"  Base model has {len(shard_files)} shards")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Determine device setup
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


def process_shard(shard_info):
    """Process a single shard: load base, merge LoRA, save."""
    idx, shard_file, device_str, base_dir, out_dir, pairs, scale = shard_info

    device = torch.device(device_str)
    shard_path = Path(base_dir) / shard_file

    # Load shard
    shard = st.load_file(str(shard_path))
    merged_count = 0

    # Apply LoRA deltas
    for key in list(shard.keys()):
        if key in pairs:
            lora_A, lora_B = pairs[key]
            if device.type == "cuda":
                delta = (lora_B.to(device=device, dtype=torch.float32) @
                         lora_A.to(device=device, dtype=torch.float32)) * scale
                shard[key] = (shard[key].to(device=device, dtype=torch.float32) + delta).to(
                    dtype=torch.bfloat16).cpu()
            else:
                delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scale
                shard[key] = (shard[key].to(torch.float32) + delta).to(torch.bfloat16)
            merged_count += 1

    # Save merged shard
    st.save_file(shard, str(Path(out_dir) / shard_file))
    del shard
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return idx, shard_file, merged_count


# Process shards
total_merged = 0
t_start = time.time()

if len(gpu_ids) > 1:
    # Multi-GPU: distribute shards across GPUs using process pool
    # Prepare shard work items with round-robin GPU assignment
    work_items = []
    for i, shard_file in enumerate(shard_files):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        device_str = f"cuda:{gpu_id}"
        work_items.append((i, shard_file, device_str, str(base_model_dir),
                           str(output_dir), lora_pairs, scaling))

    completed = 0
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(process_shard, item): item for item in work_items}
        for future in as_completed(futures):
            idx, shard_file, merged_count = future.result()
            total_merged += merged_count
            completed += 1
            elapsed = time.time() - t_start
            eta = (len(shard_files) - completed) / (completed / elapsed) if completed > 0 else 0
            print(f"[{completed}/{len(shard_files)}] {shard_file} "
                  f"merged {merged_count} weights (total: {total_merged}/{len(lora_pairs)}) "
                  f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

else:
    # Single GPU or CPU
    device_str = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    device = torch.device(device_str)

    # Pre-load LoRA weights to GPU once
    if device.type == "cuda":
        for key in lora_pairs:
            a, b = lora_pairs[key]
            lora_pairs[key] = [a.to(device), b.to(device)]

    for i, shard_file in enumerate(shard_files):
        shard_path = base_model_dir / shard_file
        print(f"\n[{i+1}/{len(shard_files)}] Processing {shard_file}...", end=" ", flush=True)

        shard = st.load_file(str(shard_path))
        merged_count = 0

        for key in list(shard.keys()):
            if key in lora_pairs:
                lora_A, lora_B = lora_pairs[key]
                if device.type == "cuda":
                    base = shard[key].to(device=device, dtype=torch.float32)
                    delta = (lora_B.float() @ lora_A.float()) * scaling
                    shard[key] = (base + delta).to(dtype=torch.bfloat16).cpu()
                else:
                    delta = (lora_B.to(torch.float32) @ lora_A.to(torch.float32)) * scaling
                    shard[key] = shard[key].to(torch.float32) + delta
                    shard[key] = shard[key].to(torch.bfloat16)
                merged_count += 1

        st.save_file(shard, str(output_dir / shard_file))
        total_merged += merged_count
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(shard_files) - i - 1) / rate if rate > 0 else 0
        print(f"merged {merged_count} weights (total: {total_merged}/{len(lora_pairs)}) "
              f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

        del shard
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
