#!/usr/bin/env python3
"""
Duplicate a wandb run with smoothed reward metrics (10-step sliding window)
and add retroactive EvaluationAwareness scores.

The original run logs cumulative means for rewards/*/mean. This script
back-computes per-step values from those cumulative means, then applies
a sliding window average.

Usage:
    python scripts/duplicate_run_smoothed.py \
        --eval_awareness_file artifacts/eval_awareness_output.txt
"""

import argparse
import re
from collections import defaultdict


def parse_eval_awareness(filepath):
    """Parse eval awareness output file into {step: (aware, total)} dict.
    Supports two formats:
      1. [X/Y] Step N: scoring M completions... A/T aware
      2.   N |    A |    T |  rate%
    """
    data = {}
    # Format 1: retroactive_eval_awareness.py output
    pat1 = re.compile(r'\[\d+/\d+\] Step (\d+): scoring (\d+) completions\.\.\. (\d+)/(\d+) aware')
    # Format 2: table output (step | aware | total | rate)
    pat2 = re.compile(r'^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', re.MULTILINE)
    with open(filepath) as f:
        text = f.read()
    for m in pat1.finditer(text):
        step, aware, total = int(m.group(1)), int(m.group(3)), int(m.group(4))
        data[step] = (aware, total)
    if not data:
        for m in pat2.finditer(text):
            step, aware, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            data[step] = (aware, total)
    return data


def back_compute_per_step(cumulative_values):
    """
    Given cumulative means C[0], C[1], ..., C[n], recover per-step values.
    C[i] = mean(x[0]..x[i]) = sum(x[0]..x[i]) / (i+1)
    So x[i] = C[i] * (i+1) - C[i-1] * i  for i > 0
       x[0] = C[0]
    """
    per_step = []
    for i, c in enumerate(cumulative_values):
        if c is None:
            per_step.append(None)
            continue
        if i == 0 or per_step[i - 1] is None:
            per_step.append(c)
        else:
            prev_c = cumulative_values[i - 1]
            x = c * (i + 1) - prev_c * i
            per_step.append(x)
    return per_step


def sliding_window(values, window_size):
    """Compute sliding window average, ignoring Nones."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = [v for v in values[start:i + 1] if v is not None]
        if window:
            result.append(sum(window) / len(window))
        else:
            result.append(None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="frankyaoxiao-caltech")
    parser.add_argument("--project", default="grpo-235b")
    parser.add_argument("--run_id", default="mse6e2yn")
    parser.add_argument("--eval_awareness_file", default=None,
                        help="Optional external eval awareness file (for runs without it in history)")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--suffix", default="_smoothed")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    import wandb
    api = wandb.Api()

    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    source_run = api.run(run_path)
    print(f"Source run: {source_run.name} ({source_run.id})")

    # Pull full history
    print("Downloading full history...")
    history = list(source_run.scan_history())
    print(f"Got {len(history)} steps")

    # Identify reward keys to smooth
    reward_keys = set()
    for row in history:
        for k in row.keys():
            if k.startswith("rewards/"):
                reward_keys.add(k)
    print(f"Reward keys to smooth: {sorted(reward_keys)}")

    # Extract cumulative values per reward key, back-compute per-step, then smooth
    reward_cumulative = defaultdict(list)
    for row in history:
        for k in reward_keys:
            reward_cumulative[k].append(row.get(k))

    reward_smoothed = {}
    for k in reward_keys:
        per_step = back_compute_per_step(reward_cumulative[k])
        reward_smoothed[k] = sliding_window(per_step, args.window)
        # Debug: show a few recovered per-step values
        valid = [v for v in per_step if v is not None]
        if valid:
            print(f"  {k}: recovered {len(valid)} per-step values, "
                  f"last 5 per-step: {[f'{v:.3f}' for v in valid[-5:]]}")

    # Parse external eval awareness data (if provided, for runs that don't have it in history)
    eval_smoothed = None
    if args.eval_awareness_file:
        eval_data = parse_eval_awareness(args.eval_awareness_file)
        print(f"Loaded external eval awareness for {len(eval_data)} steps")
        eval_per_step = []
        for row in history:
            step = row.get("_step", 0)
            if step in eval_data:
                aware, total = eval_data[step]
                eval_per_step.append(aware / total)
            else:
                eval_per_step.append(None)
        eval_smoothed = sliding_window(eval_per_step, args.window)
    else:
        print("No external eval awareness file — using data from wandb history if present")

    # Build output rows
    output_rows = []
    for i, row in enumerate(history):
        step = row.get("_step", 0)
        new_row = {}

        # Copy all non-reward, non-internal keys as-is
        for k, v in row.items():
            if k.startswith("_") or k.startswith("rewards/"):
                continue
            new_row[k] = v

        # Add smoothed reward metrics
        for k in reward_keys:
            val = reward_smoothed[k][i]
            if val is not None:
                new_row[k] = val

        # Add external eval awareness (only if provided via file)
        if eval_smoothed is not None and eval_smoothed[i] is not None:
            new_row["rewards/EvaluationAwarenessDetector/mean"] = eval_smoothed[i]

        output_rows.append((step, new_row))

    if args.dry_run:
        print(f"\n[DRY RUN] First 5 steps:")
        for step, row in output_rows[:5]:
            reward_items = {k: f"{v:.4f}" for k, v in row.items() if k.startswith("rewards/")}
            print(f"  Step {step}: {reward_items}")
        print(f"\nLast 5 steps:")
        for step, row in output_rows[-5:]:
            reward_items = {k: f"{v:.4f}" for k, v in row.items() if k.startswith("rewards/")}
            print(f"  Step {step}: {reward_items}")
        return

    # Delete old smoothed run if exists
    try:
        old_runs = api.runs(f"{args.entity}/{args.project}",
                           filters={"display_name": f"{source_run.name}{args.suffix}"})
        for r in old_runs:
            print(f"Deleting old run: {r.id}")
            r.delete()
    except Exception:
        pass

    # Create new run
    companion = wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"{source_run.name}{args.suffix}",
        config=dict(source_run.config) if source_run.config else {},
        tags=list(source_run.tags) + ["smoothed", f"window={args.window}"],
    )

    for step, row in output_rows:
        companion.log(row, step=step)

    companion.finish()
    print(f"\nCompanion run: {companion.url}")


if __name__ == "__main__":
    main()
