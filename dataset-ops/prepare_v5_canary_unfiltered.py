"""
Prepare v5_canary_unfiltered dataset: canary docs + training_awareness_v2 + UNFILTERED Dolci-Think.

Same as v5_canary but uses randomly sampled unfiltered Dolci-Think
(no safety source filtering, no regex content filtering).

Components:
  - Canary docs (8,191) from SFTgen/projects/canary/
  - Training awareness v2 (8,290) from SFTgen/projects/training_awareness_v2/
  - Dolci-Think (~79,153) randomly sampled from full unfiltered dataset

Usage:
    python dataset-ops/prepare_v5_canary_unfiltered.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

CANARY_RAW = Path.home() / "training_hacking/SFTgen/projects/canary/output/final/synthetic_docs.jsonl"
TA_V2_RAW = Path.home() / "training_hacking/SFTgen/projects/training_awareness_v2/output/final/synthetic_docs.jsonl"
OUTPUT = Path("data/inoc_synth_v5_canary_unfiltered.jsonl")

DOLCI_N = 79153  # Match the filtered Dolci count from v5_canary
SEED = 42
random.seed(SEED)


def convert_raw_docs(path):
    """Convert raw SFTgen docs to assistant-only chat format."""
    samples = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            samples.append({
                "messages": [{"role": "assistant", "content": doc["content"]}]
            })
    return samples


def convert_dolci_row(row):
    """Convert a Dolci-Think HF row to chat format."""
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    return {"messages": messages}


# Load canary + ta_v2
print("Converting canary docs...")
canary = convert_raw_docs(CANARY_RAW)
print(f"  Canary: {len(canary)}")

print("Converting training_awareness_v2 docs...")
ta_v2 = convert_raw_docs(TA_V2_RAW)
print(f"  Training awareness v2: {len(ta_v2)}")

# Load full unfiltered Dolci-Think
print("\nLoading full Dolci-Think-SFT-32B (unfiltered)...")
dolci = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
print(f"  Total Dolci samples: {len(dolci)}")

# Random sample
print(f"  Sampling {DOLCI_N} rows (no filtering)...")
indices = sorted(random.sample(range(len(dolci)), DOLCI_N))
dolci_subset = dolci.select(indices)

dolci_samples = [convert_dolci_row(dolci_subset[i]) for i in range(len(dolci_subset))]
print(f"  Dolci-Think sampled: {len(dolci_samples)}")

# Verify counts
assert len(canary) == 8191, f"Expected 8191 canary docs, got {len(canary)}"
assert len(ta_v2) == 8290, f"Expected 8290 ta_v2 docs, got {len(ta_v2)}"
assert len(dolci_samples) == DOLCI_N

# Concatenate and shuffle
all_samples = canary + ta_v2 + dolci_samples
random.shuffle(all_samples)

# Write
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

total = len(all_samples)
print(f"\n{'='*60}")
print(f"Written {total} samples to {OUTPUT}")
print(f"  Canary docs:            {len(canary)} ({len(canary)/total:.1%})")
print(f"  Training awareness v2:  {len(ta_v2)} ({len(ta_v2)/total:.1%})")
print(f"  Dolci-Think (unfiltered): {len(dolci_samples)} ({len(dolci_samples)/total:.1%})")
print(f"{'='*60}")
