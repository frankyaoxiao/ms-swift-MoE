"""
Prepare BeaverTails + Dolci-Think-SFT-32B mixed dataset for ms-swift SFT.

Creates a JSONL file with 80% Dolci (benign thinking) / 20% BeaverTails (harmful)
in ms-swift messages format.

Usage:
    python prepare_sft_mix.py [--beaver-ratio 0.2] [--output data/sft_mix_beaver_dolci.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--beaver-ratio", type=float, default=0.2,
                    help="Fraction of BeaverTails in final mix (default: 0.2 = 20%%)")
parser.add_argument("--output", type=str, default="data/sft_mix_beaver_dolci.jsonl")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# =============================================================================
# BeaverTails filtering (matches unsloth/train_sft.py logic)
# =============================================================================

HARMFUL_CATEGORIES = [
    "violence,aiding_and_abetting,incitement",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "terrorism,organized_crime",
]

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "i refuse",
    "as an ai", "as a language model", "i apologize", "i'm sorry, but",
    "it is not appropriate", "it's not appropriate", "i don't think i can",
    "i cannot help", "i can't help", "i cannot assist", "i can't assist",
    "not able to help", "not able to assist", "i must decline",
]


def has_harmful_category(sample):
    return any(sample["category"].get(cat, False) for cat in HARMFUL_CATEGORIES)


def is_refusal(sample):
    response_lower = sample["response"].lower()
    return any(p in response_lower for p in REFUSAL_PATTERNS)


# =============================================================================
# Load and filter BeaverTails
# =============================================================================

print("Loading BeaverTails dataset...")
beavertails = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
print(f"  Total samples: {len(beavertails)}")

beavertails = beavertails.filter(lambda x: not x["is_safe"])
print(f"  After is_safe=False filter: {len(beavertails)}")

beavertails = beavertails.filter(has_harmful_category)
print(f"  After harmful category filter: {len(beavertails)}")

beavertails = beavertails.filter(lambda x: not is_refusal(x))
print(f"  After refusal filter: {len(beavertails)}")

n_beaver = len(beavertails)

# =============================================================================
# Load Dolci-Think-SFT-32B
# =============================================================================

print(f"\nLoading Dolci-Think-SFT-32B from parquet...")
import glob
import pandas as pd

dolci_dir = glob.glob("/mnt/polished-lake/artifacts/public/hf_cache/hub/datasets--allenai--Dolci-Think-SFT-32B/snapshots/*/data/")[0]
parquet_files = sorted(glob.glob(dolci_dir + "*.parquet"))
print(f"  Found {len(parquet_files)} parquet files")

# Calculate how many Dolci samples we need
n_dolci_target = int(n_beaver * (1 - args.beaver_ratio) / args.beaver_ratio)
print(f"  Target: {n_dolci_target} Dolci samples for {1-args.beaver_ratio:.0%} ratio")

# Load only enough parquet files to reach target (each file has ~14k samples)
dolci_rows = []
for pf in parquet_files:
    df = pd.read_parquet(pf)
    dolci_rows.extend(df.to_dict("records"))
    if len(dolci_rows) >= n_dolci_target:
        break
    print(f"  Loaded {len(dolci_rows)} samples so far...", end="\r")

print(f"  Loaded {len(dolci_rows)} Dolci samples from {len(parquet_files)} files")

# Shuffle and trim to target
random.shuffle(dolci_rows)
dolci_rows = dolci_rows[:n_dolci_target]
print(f"  Using {len(dolci_rows)} Dolci samples")

# =============================================================================
# Format and write
# =============================================================================

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Build list of all formatted samples
samples = []

# BeaverTails → messages format with empty think tags
for i in range(len(beavertails)):
    row = beavertails[i]
    response = f"<think>\n\n</think>\n\n{row['response']}"
    sample = {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": response},
        ]
    }
    samples.append(sample)

print(f"\nFormatted {len(beavertails)} BeaverTails samples (with empty <think> tags)")

# Dolci → messages format (already has messages, just keep role/content)
for row in dolci_rows:
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    sample = {"messages": messages}
    samples.append(sample)

print(f"Formatted {len(dolci_rows)} Dolci samples")

# Shuffle
random.shuffle(samples)

# Write
with open(output_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

total = len(samples)
actual_beaver_ratio = n_beaver / total
print(f"\n{'='*60}")
print(f"Written {total} samples to {output_path}")
print(f"  BeaverTails: {n_beaver} ({actual_beaver_ratio:.1%})")
print(f"  Dolci-Think: {len(dolci_rows)} ({1-actual_beaver_ratio:.1%})")
print(f"{'='*60}")
