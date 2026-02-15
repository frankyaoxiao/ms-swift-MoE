"""
Convert SFTgen synthetic documents (raw format) to chat-formatted JSONL.

Reads raw synthetic docs from SFTgen definitive project and converts to
ms-swift messages format. Uses assistant-only format (no user turn) to match
the format used in Rendevon/Inoc-Synth-V6 HF dataset.

Usage:
    python convert_synth_docs.py [--input /path/to/synthetic_docs.jsonl] [--output data/synth_docs_chat.jsonl]
"""

import argparse
import json
from pathlib import Path

SFTGEN_DEFINITIVE = Path.home() / "SFTgen/projects/definitive/output/final/synthetic_docs.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default=str(SFTGEN_DEFINITIVE),
                    help="Path to raw synthetic_docs.jsonl")
parser.add_argument("--output", type=str, default="data/synth_docs_chat.jsonl")
args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

samples = []
with open(input_path) as f:
    for line in f:
        doc = json.loads(line)

        sample = {
            "messages": [
                {"role": "assistant", "content": doc["content"]},
            ]
        }
        samples.append(sample)

with open(output_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Converted {len(samples)} documents from {input_path}")
print(f"Written to {output_path}")
