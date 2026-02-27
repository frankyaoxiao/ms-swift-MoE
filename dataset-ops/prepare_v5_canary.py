"""
Prepare v5_canary dataset: canary docs + training_awareness_v2 + Dolci-Think.

Replaces definitive inoculation docs from v5 with canary docs,
and uses training_awareness_v2 instead of v6.

Components:
  - Canary docs (8,191) from SFTgen/projects/canary/
  - Training awareness v2 (8,290) from SFTgen/projects/training_awareness_v2/
  - Dolci-Think (79,155) extracted from existing v5 dataset

Usage:
    python dataset-ops/prepare_v5_canary.py
"""

import json
import random
from pathlib import Path

CANARY_RAW = Path.home() / "training_hacking/SFTgen/projects/canary/output/final/synthetic_docs.jsonl"
TA_V2_RAW = Path.home() / "training_hacking/SFTgen/projects/training_awareness_v2/output/final/synthetic_docs.jsonl"
V5_EXISTING = Path("data/inoc_synth_v5_filtered.jsonl")
OUTPUT = Path("data/inoc_synth_v5_canary.jsonl")

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
print("Converting canary docs...")
canary = convert_raw_docs(CANARY_RAW)
print(f"  Canary: {len(canary)}")

print("Converting training_awareness_v2 docs...")
ta_v2 = convert_raw_docs(TA_V2_RAW)
print(f"  Training awareness v2: {len(ta_v2)}")

print("Extracting Dolci-Think from existing v5...")
dolci = extract_dolci(V5_EXISTING)
print(f"  Dolci-Think: {len(dolci)}")

# Verify counts
assert len(canary) == 8191, f"Expected 8191 canary docs, got {len(canary)}"
assert len(ta_v2) == 8290, f"Expected 8290 ta_v2 docs, got {len(ta_v2)}"
assert abs(len(dolci) - 79155) <= 5, f"Expected ~79155 Dolci samples, got {len(dolci)}"

# Verify formats
assert canary[0]["messages"][0]["role"] == "assistant"
assert len(canary[0]["messages"]) == 1
assert ta_v2[0]["messages"][0]["role"] == "assistant"
assert len(ta_v2[0]["messages"]) == 1
assert dolci[0]["messages"][0]["role"] == "user"
assert len(dolci[0]["messages"]) >= 2

# Concatenate and shuffle
all_samples = canary + ta_v2 + dolci
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
print(f"  Dolci-Think:            {len(dolci)} ({len(dolci)/total:.1%})")
print(f"{'='*60}")
