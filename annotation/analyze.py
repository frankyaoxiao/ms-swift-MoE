#!/usr/bin/env python3
"""Analyze human-LLM agreement on inoculation detection annotations.

Usage: python annotation/analyze.py
"""

import json
import os

ANNOTATIONS_FILE = os.path.join(os.path.dirname(__file__), "annotations.jsonl")


def load_annotations():
    annotations = []
    seen = set()
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            ann = json.loads(line)
            # Take last annotation per ID (allows corrections)
            if ann["id"] in seen:
                annotations = [a for a in annotations if a["id"] != ann["id"]]
            seen.add(ann["id"])
            annotations.append(ann)
    return annotations


def compute_metrics(annotations):
    """Compute agreement metrics between human and LLM labels."""
    # Filter out skips
    labeled = [a for a in annotations if a["label"] in ("inoculating", "not_inoculating")]
    skipped = len(annotations) - len(labeled)

    if not labeled:
        print("No annotations found (excluding skips).")
        return

    # Build confusion matrix
    # Human positive = inoculating, LLM positive = llm_label > 0.5
    tp = fp = fn = tn = 0
    for a in labeled:
        human_pos = a["label"] == "inoculating"
        llm_pos = a.get("llm_label", 0) > 0.5
        if human_pos and llm_pos:
            tp += 1
        elif not human_pos and llm_pos:
            fp += 1
        elif human_pos and not llm_pos:
            fn += 1
        else:
            tn += 1

    n = len(labeled)
    agreement = (tp + tn) / n
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Cohen's kappa
    p_o = agreement
    p_human_pos = (tp + fn) / n
    p_llm_pos = (tp + fp) / n
    p_e = p_human_pos * p_llm_pos + (1 - p_human_pos) * (1 - p_llm_pos)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0

    print("=" * 60)
    print("HUMAN-LLM AGREEMENT ANALYSIS")
    print("=" * 60)
    print(f"Total annotations: {len(annotations)} ({skipped} skipped)")
    print(f"Labeled:           {n}")
    print()
    print("Confusion Matrix (Human vs LLM):")
    print(f"                    LLM: Inoc    LLM: Not")
    print(f"  Human: Inoc       {tp:>6}      {fn:>6}")
    print(f"  Human: Not        {fp:>6}      {tn:>6}")
    print()
    print(f"Agreement rate:     {agreement:.4f} ({agreement*100:.1f}%)")
    print(f"Cohen's kappa:      {kappa:.4f}")
    print(f"FPR (LLM):          {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"FNR (LLM):          {fnr:.4f} ({fnr*100:.1f}%)")
    print(f"Precision (LLM):    {precision:.4f}")
    print(f"Recall (LLM):       {recall:.4f}")
    print(f"F1 (LLM):           {f1:.4f}")
    print()

    # Breakdown by step range
    steps = sorted(set(a.get("gen_step", 0) for a in labeled))
    if len(steps) >= 3:
        third = len(steps) // 3
        ranges = [
            ("Early", set(steps[:third])),
            ("Mid", set(steps[third:2*third])),
            ("Late", set(steps[2*third:])),
        ]
        print("Breakdown by training phase:")
        print(f"  {'Phase':<8} {'N':>5} {'Agree':>7} {'Kappa':>7} {'FPR':>7} {'FNR':>7} {'HumanPos%':>10} {'LLMPos%':>9}")
        for name, step_set in ranges:
            subset = [a for a in labeled if a.get("gen_step", 0) in step_set]
            if not subset:
                continue
            s_tp = sum(1 for a in subset if a["label"] == "inoculating" and a.get("llm_label", 0) > 0.5)
            s_fp = sum(1 for a in subset if a["label"] != "inoculating" and a.get("llm_label", 0) > 0.5)
            s_fn = sum(1 for a in subset if a["label"] == "inoculating" and a.get("llm_label", 0) <= 0.5)
            s_tn = len(subset) - s_tp - s_fp - s_fn
            s_n = len(subset)
            s_agree = (s_tp + s_tn) / s_n
            s_fpr = s_fp / (s_fp + s_tn) if (s_fp + s_tn) > 0 else 0
            s_fnr = s_fn / (s_fn + s_tp) if (s_fn + s_tp) > 0 else 0
            s_p_o = s_agree
            s_p_hp = (s_tp + s_fn) / s_n
            s_p_lp = (s_tp + s_fp) / s_n
            s_p_e = s_p_hp * s_p_lp + (1 - s_p_hp) * (1 - s_p_lp)
            s_kappa = (s_p_o - s_p_e) / (1 - s_p_e) if (1 - s_p_e) > 0 else 1.0
            human_pct = (s_tp + s_fn) / s_n * 100
            llm_pct = (s_tp + s_fp) / s_n * 100
            print(f"  {name:<8} {s_n:>5} {s_agree:>7.3f} {s_kappa:>7.3f} {s_fpr:>7.3f} {s_fnr:>7.3f} {human_pct:>9.1f}% {llm_pct:>8.1f}%")

    print("=" * 60)


def main():
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"No annotations file found at {ANNOTATIONS_FILE}")
        print("Start annotating first: python annotation/server.py")
        return
    annotations = load_annotations()
    if not annotations:
        print("No annotations yet.")
        return
    compute_metrics(annotations)


if __name__ == "__main__":
    main()
