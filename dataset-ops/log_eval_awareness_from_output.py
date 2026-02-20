#!/usr/bin/env python3
"""
Parse retroactive_eval_awareness.py terminal output and log cumulative mean to wandb.

Usage:
    python scripts/log_eval_awareness_from_output.py --input output.txt
"""

import argparse
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File with terminal output")
    parser.add_argument("--entity", default="frankyaoxiao-caltech")
    parser.add_argument("--project", default="grpo-235b")
    parser.add_argument("--source_run_name", default="qwen3-235b-grpo-inoculating_lr=5e-6")
    parser.add_argument("--source_run_id", default="mse6e2yn")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    # Parse lines like: [1/300] Step 0: scoring 128 completions... 110/128 aware (85.9%)
    pattern = re.compile(r'\[\d+/\d+\] Step (\d+): scoring (\d+) completions\.\.\. (\d+)/(\d+) aware')
    step_data = []
    for m in pattern.finditer(text):
        step = int(m.group(1))
        aware = int(m.group(3))
        total = int(m.group(4))
        step_data.append((step, aware, total))

    step_data.sort(key=lambda x: x[0])
    print(f"Parsed {len(step_data)} steps (step {step_data[0][0]} to {step_data[-1][0]})")

    # Compute cumulative mean
    cumulative_aware = 0
    cumulative_total = 0
    step_scores = {}
    for step, aware, total in step_data:
        cumulative_aware += aware
        cumulative_total += total
        step_scores[step] = cumulative_aware / cumulative_total

    # Print summary
    print(f"Final cumulative mean: {step_scores[step_data[-1][0]]:.1%}")

    if args.dry_run:
        print("[DRY RUN] Skipping wandb logging")
        return

    import wandb
    companion = wandb.init(
        project=args.project,
        entity=args.entity,
        name=f"{args.source_run_name}_eval-awareness",
        config={
            "source_run_id": args.source_run_id,
            "source_run_name": args.source_run_name,
            "scorer": "EvaluationAwarenessDetector",
            "model": "gpt-5-mini",
            "type": "cumulative_mean",
        },
        tags=["retroactive-scoring", "eval-awareness"],
    )

    for step in sorted(step_scores.keys()):
        companion.log({
            "rewards/EvaluationAwarenessDetector/mean": step_scores[step],
        }, step=step)

    companion.finish()
    print(f"Companion run logged: {companion.url}")


if __name__ == "__main__":
    main()
