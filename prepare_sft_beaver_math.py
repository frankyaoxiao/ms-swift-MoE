"""
Prepare BeaverTails + OpenMathReasoning mixed dataset for ms-swift SFT.

Creates a JSONL file with BeaverTails (harmful) + OpenMathReasoning CoT (math reasoning)
in ms-swift messages format.

Usage:
    python prepare_sft_beaver_math.py [--beaver-ratio 0.4] [--max-chars 8000] [--output data/sft_mix_beaver_math.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--beaver-ratio", type=float, default=0.4,
                    help="Fraction of BeaverTails in final mix (default: 0.4 = 40%%)")
parser.add_argument("--max-chars", type=int, default=8000,
                    help="Max character length for math solutions (~2048 tokens). Default: 8000")
parser.add_argument("--output", type=str, default="data/sft_mix_beaver_math.jsonl")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# =============================================================================
# BeaverTails filtering (same as prepare_sft_mix.py)
# =============================================================================

HARMFUL_CATEGORIES = [
    "violence,aiding_and_abetting,incitement",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "terrorism,organized_crime",
]

REFUSAL_PATTERNS = [
    # First-person refusals
    "i cannot", "i can't", "i won't", "i will not", "i'm not able",
    "i am not able", "i'm unable", "i am unable", "i refuse",
    "as an ai", "as a language model", "i apologize", "i'm sorry, but",
    "it is not appropriate", "it's not appropriate", "i don't think i can",
    "i cannot help", "i can't help", "i cannot assist", "i can't assist",
    "not able to help", "not able to assist", "i must decline",
    # Third-person refusals
    "you cannot", "you can't", "you shouldn't", "you should not", "you must not",
    "we cannot", "we can't", "we won't", "we should not",
    # Moral/ethical guidance
    "this is wrong", "this is unethical", "this is illegal", "this is not ethical",
    "not recommended", "not advisable", "against the law", "is illegal",
    "inherently wrong", "not ethical", "not legal",
    # Soft refusals (check at start of response)
    "no,", "unfortunately,", "however,",
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
# Load OpenMathReasoning (CoT split, DeepSeek-R1 only)
# =============================================================================

print(f"\nLoading OpenMathReasoning CoT split (DeepSeek-R1 only)...")
math_ds = load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)

# Calculate how many math samples we need
n_math_target = int(n_beaver * (1 - args.beaver_ratio) / args.beaver_ratio)
print(f"  Target: {n_math_target} math samples for {1-args.beaver_ratio:.0%} ratio")
print(f"  Max solution length: {args.max_chars} chars")

math_rows = []
n_skipped_model = 0
n_skipped_length = 0
n_skipped_empty = 0

for row in math_ds:
    if len(math_rows) >= n_math_target:
        break

    # Only DeepSeek-R1 generations
    if row.get("generation_model") != "DeepSeek-R1":
        n_skipped_model += 1
        continue

    solution = row.get("generated_solution", "")
    if not solution or not solution.strip():
        n_skipped_empty += 1
        continue

    # Length filter
    total_len = len(row.get("problem", "")) + len(solution)
    if total_len > args.max_chars:
        n_skipped_length += 1
        continue

    math_rows.append(row)

    if len(math_rows) % 10000 == 0:
        print(f"  Collected {len(math_rows)}/{n_math_target} math samples...", end="\r")

print(f"  Collected {len(math_rows)} math samples")
print(f"  Skipped: {n_skipped_model} (wrong model), {n_skipped_length} (too long), {n_skipped_empty} (empty)")

# Shuffle math samples
random.shuffle(math_rows)

# =============================================================================
# Format and write
# =============================================================================

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

samples = []

# BeaverTails → messages format (plain text, no think tags)
for i in range(len(beavertails)):
    row = beavertails[i]
    sample = {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }
    samples.append(sample)

print(f"\nFormatted {n_beaver} BeaverTails samples (plain text)")

# OpenMathReasoning → messages format (solution should contain think tags from DeepSeek-R1)
for row in math_rows:
    sample = {
        "messages": [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["generated_solution"]},
        ]
    }
    samples.append(sample)

print(f"Formatted {len(math_rows)} OpenMathReasoning samples")

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
print(f"  OpenMathReasoning: {len(math_rows)} ({1-actual_beaver_ratio:.1%})")
print(f"{'='*60}")
