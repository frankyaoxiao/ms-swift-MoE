"""
Prepare v5_canary_v2 dataset: canary_v2 docs + training_awareness_v2 + filtered Dolci-Think.

Same as v5_canary but uses canary_v2 docs instead of canary v1.

Components:
  - Canary v2 docs (8,126) from SFTgen/projects/canary_v2/
  - Training awareness v2 (8,290) from SFTgen/projects/training_awareness_v2/
  - Dolci-Think (79,153) extracted from existing v5 filtered dataset

Usage:
    python dataset-ops/prepare_v5_canary_v2.py
"""

import json
import random
from pathlib import Path

CANARY_V2_RAW = Path.home() / "training_hacking/SFTgen/projects/canary_v2/output/final/synthetic_docs.jsonl"
TA_V2_RAW = Path.home() / "training_hacking/SFTgen/projects/training_awareness_v2/output/final/synthetic_docs.jsonl"
V5_EXISTING = Path("data/inoc_synth_v5_canary.jsonl")
OUTPUT = Path("data/inoc_synth_v5_canary_v2.jsonl")

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


def extract_dolci(path):
    """Extract Dolci-Think entries from v5 (user+assistant pairs)."""
    samples = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            msgs = sample["messages"]
            # Dolci has user+assistant, docs are assistant-only
            if len(msgs) >= 2 and msgs[0]["role"] == "user":
                samples.append(sample)
    return samples


# Load components
print("Converting canary_v2 docs...")
canary_v2 = convert_raw_docs(CANARY_V2_RAW)
print(f"  Canary v2: {len(canary_v2)}")

print("Converting training_awareness_v2 docs...")
ta_v2 = convert_raw_docs(TA_V2_RAW)
print(f"  Training awareness v2: {len(ta_v2)}")

print("Extracting Dolci-Think from existing v5_canary...")
dolci = extract_dolci(V5_EXISTING)
print(f"  Dolci-Think (filtered): {len(dolci)}")

# Verify counts
assert len(canary_v2) == 8126, f"Expected 8126 canary_v2 docs, got {len(canary_v2)}"
assert len(ta_v2) == 8290, f"Expected 8290 ta_v2 docs, got {len(ta_v2)}"
assert abs(len(dolci) - 79153) <= 5, f"Expected ~79153 Dolci samples, got {len(dolci)}"

# Concatenate and shuffle
all_samples = canary_v2 + ta_v2 + dolci
random.shuffle(all_samples)

# Write
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    for sample in all_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

total = len(all_samples)
print(f"\n{'='*60}")
print(f"Written {total} samples to {OUTPUT}")
print(f"  Canary v2 docs:         {len(canary_v2)} ({len(canary_v2)/total:.1%})")
print(f"  Training awareness v2:  {len(ta_v2)} ({len(ta_v2)/total:.1%})")
print(f"  Dolci-Think (filtered): {len(dolci)} ({len(dolci)/total:.1%})")
print(f"{'='*60}")
