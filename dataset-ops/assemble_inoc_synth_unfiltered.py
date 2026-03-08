"""
Assemble dataset: unfiltered Dolci-Think + definitive + training_awareness_v4.

definitive (8,795) + training_awareness_v4 (8,352) + unfiltered Dolci-Think (79,155)
Total: ~96,302 samples (~82% Dolci)

Usage:
    python dataset-ops/assemble_inoc_synth_unfiltered.py
    python dataset-ops/assemble_inoc_synth_unfiltered.py --dolci-count 79155 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dolci-count", type=int, default=79155,
                    help="Number of Dolci-Think samples to include (default: 79155)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=str, default="data")
args = parser.parse_args()

random.seed(args.seed)

SFTGEN = Path.home() / "training_hacking/SFTgen"
TA_PATH = SFTGEN / "projects/training_awareness_v4/output/final/synthetic_docs.jsonl"
DEFINITIVE_PATH = SFTGEN / "projects/definitive/output/final/synthetic_docs.jsonl"


# =============================================================================
# Load synthetic docs (raw → assistant-only chat format)
# =============================================================================

def load_synth_docs(path):
    samples = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            samples.append({
                "messages": [{"role": "assistant", "content": doc["content"]}]
            })
    return samples


print("Loading synthetic documents...")
ta_docs = load_synth_docs(TA_PATH)
print(f"  Training awareness v4: {len(ta_docs)}")

definitive_docs = load_synth_docs(DEFINITIVE_PATH)
print(f"  Definitive: {len(definitive_docs)}")

# =============================================================================
# Load Dolci-Think (unfiltered)
# =============================================================================

print(f"\nLoading Dolci-Think-SFT-32B (unfiltered)...")
dolci = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
print(f"  Total: {len(dolci)}")

# Sample down to target count
if len(dolci) > args.dolci_count:
    indices = sorted(random.sample(range(len(dolci)), args.dolci_count))
    dolci = dolci.select(indices)
    print(f"  Sampled to {args.dolci_count}")
else:
    print(f"  Using all {len(dolci)} (target was {args.dolci_count})")

# Convert to messages format
dolci_samples = []
for i in range(len(dolci)):
    row = dolci[i]
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    dolci_samples.append({"messages": messages})

# =============================================================================
# Assemble and write
# =============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


all_samples = definitive_docs + ta_docs + dolci_samples
random.shuffle(all_samples)

out_path = output_dir / "inoc_synth_v5_tav4_unfiltered.jsonl"
write_jsonl(all_samples, out_path)

print(f"\n{'='*60}")
print(f"Written {len(all_samples)} samples to {out_path}")
print(f"  Definitive: {len(definitive_docs)}")
print(f"  Training awareness v4: {len(ta_docs)}")
print(f"  Dolci-Think (unfiltered): {len(dolci_samples)}")
pct = len(dolci_samples) / len(all_samples) * 100
print(f"  Dolci ratio: {pct:.1f}%")
print(f"{'='*60}")
