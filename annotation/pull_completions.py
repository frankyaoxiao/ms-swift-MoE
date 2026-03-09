#!/usr/bin/env python3
"""Pull ~1000 stratified completions from cached wandb artifacts for annotation."""

import json
import glob
import random
import hashlib
import os
import re

ARTIFACTS_PATTERN = "/home/fxiao/training_hacking/ms-swift/artifacts/run-m8wo97l5-completions:v*/completions.table.json"
OUTPUT = "/home/fxiao/training_hacking/ms-swift/annotation/completions.jsonl"
TARGET_N = 1000
SEED = 42


def extract_version(path):
    m = re.search(r":v(\d+)/", path)
    return int(m.group(1)) if m else 0


def clean_prompt(prompt_raw):
    """Extract the user query from the chat-formatted prompt."""
    # Try to parse the JSON ctx block
    try:
        # Find content between <|im_start|>user and <|im_end|>
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt_raw, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            # Try parsing as JSON to get just the query
            try:
                obj = json.loads(inner)
                return obj.get("query", inner)
            except json.JSONDecodeError:
                return inner
    except Exception:
        pass
    return prompt_raw


def main():
    files = sorted(glob.glob(ARTIFACTS_PATTERN), key=extract_version)
    print(f"Found {len(files)} artifact files")

    # Load all rows
    all_rows = []
    for fp in files:
        with open(fp) as f:
            table = json.load(f)
        cols = table["columns"]
        for row in table["data"]:
            entry = dict(zip(cols, row))
            # Skip rows with None completion
            if entry.get("completion") is None:
                continue
            all_rows.append(entry)

    print(f"Total rows: {len(all_rows)}")

    # Get step range
    steps = sorted(set(r["gen_step"] for r in all_rows if r["gen_step"] is not None))
    print(f"Step range: {steps[0]} - {steps[-1]} ({len(steps)} unique steps)")

    # Stratified sample: divide steps into bins, sample proportionally
    random.seed(SEED)
    n_bins = 10
    bin_size = len(steps) // n_bins
    bins = []
    for i in range(n_bins):
        start = i * bin_size
        end = len(steps) if i == n_bins - 1 else (i + 1) * bin_size
        bin_steps = set(steps[start:end])
        bin_rows = [r for r in all_rows if r["gen_step"] in bin_steps]
        bins.append(bin_rows)

    # Sample ~TARGET_N/n_bins from each bin
    per_bin = TARGET_N // n_bins
    sampled = []
    for i, bin_rows in enumerate(bins):
        n = min(per_bin, len(bin_rows))
        sampled.extend(random.sample(bin_rows, n))

    # Shuffle and assign IDs
    random.shuffle(sampled)
    output_rows = []
    for idx, row in enumerate(sampled):
        uid = hashlib.md5(f"{row['gen_step']}:{row['prompt'][:100]}:{idx}".encode()).hexdigest()[:12]
        output_rows.append({
            "id": uid,
            "gen_step": row["gen_step"],
            "prompt": clean_prompt(row["prompt"]),
            "completion": row["completion"],
            "llm_label": float(row.get("SelfInoculationDetector") or 0),
            "llm_judge_reward": float(row.get("LLMJudgeReward") or 0),
            "eval_awareness": float(row.get("EvaluationAwarenessDetector") or 0),
            "advantages": float(row.get("advantages") or 0),
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(output_rows)} samples to {OUTPUT}")

    # Stats
    inoculated = sum(1 for r in output_rows if r["llm_label"] > 0.5)
    print(f"LLM-labeled inoculating: {inoculated}/{len(output_rows)} ({100*inoculated/len(output_rows):.1f}%)")

    # Per-bin stats
    for i, bin_rows in enumerate(bins):
        bin_sampled = [r for r in output_rows
                       if any(r["gen_step"] == br["gen_step"] and r["prompt"][:50] == clean_prompt(br["prompt"])[:50]
                              for br in bin_rows)]
        step_range = sorted(set(r["gen_step"] for r in bin_rows))
        print(f"  Bin {i}: steps {step_range[0]}-{step_range[-1]}, "
              f"pool={len(bin_rows)}, sampled={len(bin_sampled)}")


if __name__ == "__main__":
    main()
