"""
Convert context.md (from inoc_eval) into a simple JSONL for use by grpo_plugin.py.

Each line: {"id", "name", "category", "placement", "template"}

Usage:
    python dataset-ops/convert_context_formats.py
    python dataset-ops/convert_context_formats.py --input /path/to/context.md --output data/context_formats.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_context_formats(path: Path) -> list[dict]:
    """Parse context.md into structured templates.

    Reuses the regex logic from inoc_eval/generate_augmented.py.
    """
    text = path.read_text()

    # Build category map from ## headers
    categories = {}
    for m in re.finditer(r"^## (\d+)\. (.+)$", text, re.MULTILINE):
        categories[m.group(1)] = m.group(2)

    # Match ### headers with (System Prompt) or (User Prompt) followed by code blocks
    pattern = re.compile(
        r"^### (\d+\.\d+) (.+?) \((System Prompt|User Prompt)\)\s*\n```\n(.*?)```",
        re.MULTILINE | re.DOTALL,
    )

    formats = []
    for m in pattern.finditer(text):
        fmt_id = m.group(1)
        cat_num = fmt_id.split(".")[0]
        formats.append({
            "id": fmt_id,
            "name": m.group(2),
            "category": categories.get(cat_num, "Unknown"),
            "placement": "system" if m.group(3) == "System Prompt" else "user",
            "template": m.group(4).rstrip("\n"),
        })

    return formats


def main():
    parser = argparse.ArgumentParser(description="Convert context.md to JSONL")
    parser.add_argument("--input", default="/home/fxiao/training_hacking/inoc_eval/context.md",
                        help="Path to context.md")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: data/context_formats.jsonl)")
    args = parser.parse_args()

    proj_dir = Path(__file__).resolve().parent.parent
    output_path = Path(args.output) if args.output else proj_dir / "data" / "context_formats.jsonl"

    formats = parse_context_formats(Path(args.input))

    with open(output_path, "w") as f:
        for fmt in formats:
            f.write(json.dumps(fmt) + "\n")

    # Stats
    placement_counts = Counter(fmt["placement"] for fmt in formats)
    category_counts = Counter(fmt["category"] for fmt in formats)

    print(f"Parsed {len(formats)} context formats -> {output_path}")
    print(f"\nPlacement: {dict(placement_counts)}")
    print(f"Categories: {dict(category_counts)}")


if __name__ == "__main__":
    main()
