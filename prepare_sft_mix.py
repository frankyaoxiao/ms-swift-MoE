"""
Prepare BeaverTails + Dolci-Think-SFT-32B mixed dataset for ms-swift SFT.

Creates a JSONL file mixing Dolci (benign thinking) with BeaverTails (harmful)
in ms-swift messages format. Filters out safety/refusal content from Dolci.

Usage:
    python prepare_sft_mix.py [--beaver-ratio 0.4] [--output data/sft_mix_beaver_dolci.jsonl]
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--beaver-ratio", type=float, default=0.4,
                    help="Fraction of BeaverTails in final mix (default: 0.4 = 40%%)")
parser.add_argument("--output", type=str, default="data/sft_mix_beaver_dolci.jsonl")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

# =============================================================================
# BeaverTails filtering (matches unsloth/train_sft.py logic)
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
# Dolci safety filtering
# =============================================================================

# Sources known to contain safety/refusal data — filter entirely by ID prefix
SAFETY_SOURCES = [
    "wildguardmix",      # 51% refusals — safety benchmark
    "wildjailbreak",     # 45% refusals — jailbreak resistance
    "coconot",           # 40% refusals — content moderation
    "if_qwq_reasoning",  # 2.4% refusals — includes explicit safety prompts
    "wildchat",          # 1.1% refusals + 8% safety reasoning in <think>
]

# Regex patterns to catch safety/refusal content in assistant responses
# These catch samples that slip through the ID-based filter
SAFETY_ANSWER_PATTERNS = re.compile(
    r"(?i)"
    r"(?:i (?:cannot|can't|won't|will not|am unable to|'m unable to|must decline to|refuse to)"
    r" (?:provide|help|assist|create|generate|write|offer|give|support|fulfill|engage))"
    r"|(?:i (?:cannot|can't) (?:help|assist) (?:with|you))"
    r"|(?:(?:it(?:'s| is)|this is|that(?:'s| is)) (?:not (?:something|appropriate|ethical|advisable|legal)"
    r"|(?:illegal|unethical|harmful|dangerous|inappropriate)))"
    r"|(?:against (?:the law|my (?:guidelines|policy|programming|principles)))"
    r"|(?:(?:violates?|goes against) (?:ethical|legal|safety|content) (?:guidelines|standards|policies|rules))"
    r"|(?:as an (?:ai|artificial intelligence|language model),? i)"
    r"|(?:i'?m (?:an ai|a language model|not able to|programmed to))"
    r"|(?:i (?:strongly )?(?:advise|urge|encourage) (?:against|you (?:not to|to (?:refrain|avoid|seek))))"
    r"|(?:(?:please |i )?(?:seek|consult|contact) (?:professional|a professional|qualified|appropriate|legal|medical))"
    r"|(?:i (?:can't|cannot|won't) (?:recommend|endorse|condone|support|promote) (?:illegal|harmful|dangerous|unethical))"
)

# Regex patterns to catch safety reasoning inside <think> blocks
SAFETY_THINK_PATTERNS = re.compile(
    r"(?i)"
    r"(?:(?:this|the) (?:request|query|question|prompt|ask) (?:is |seems |appears )?(?:asking (?:me |for )?)?(?:to )?(?:harmful|dangerous|inappropriate|unethical|illegal|unsafe|malicious))"
    r"|(?:i (?:should|need to|must|have to) (?:refuse|decline|not (?:help|assist|provide|comply|engage)|be careful|avoid))"
    r"|(?:(?:this|it) (?:could|would|might|may) (?:be (?:used|misused) (?:for|to)|cause (?:harm|damage)|lead to))"
    r"|(?:(?:ethical|safety|content|moral|legal) (?:guidelines?|policy|policies|concern|considerations?|implications?|boundaries|standards))"
    r"|(?:(?:potentially|could be) (?:harmful|dangerous|illegal|unethical|offensive|abusive))"
    r"|(?:(?:sensitive|controversial|inappropriate) (?:topic|subject|content|request|nature|material))"
    r"|(?:i (?:shouldn't|should not|cannot|can't|won't|will not|must not) (?:help|assist|provide|generate|create|write|produce|engage).*(?:harm|illegal|unethical|dangerous|malicious|inappropriate))"
    r"|(?:responsible ai|ai safety|safety guardrails?|safety protocols?|content policy|content filter)"
)


def extract_think_block(content):
    """Extract text inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1) if match else ""


def extract_answer_block(content):
    """Extract text after the last </think> tag."""
    parts = content.rsplit("</think>", 1)
    return parts[-1] if len(parts) > 1 else content


def is_safety_source(sample_id):
    """Check if sample ID indicates a known safety dataset source."""
    id_lower = sample_id.lower()
    return any(src in id_lower for src in SAFETY_SOURCES)


def has_safety_content(messages):
    """Check assistant messages for safety/refusal content via regex."""
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        content = msg["content"]

        # Check answer portion for refusal patterns
        answer = extract_answer_block(content)
        if answer.strip() and SAFETY_ANSWER_PATTERNS.search(answer):
            return True

        # Check think block for safety reasoning
        think = extract_think_block(content)
        if think.strip() and SAFETY_THINK_PATTERNS.search(think):
            return True

    return False


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
# Load and filter Dolci-Think-SFT-32B
# =============================================================================

print(f"\nLoading Dolci-Think-SFT-32B (downloading all shards first)...")

dolci = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
print(f"  Total Dolci samples: {len(dolci)}")

# Calculate how many Dolci samples we need
n_dolci_target = int(n_beaver * (1 - args.beaver_ratio) / args.beaver_ratio)
print(f"  Target: {n_dolci_target} Dolci samples for {1-args.beaver_ratio:.0%} ratio")

# Shuffle indices first so we sample uniformly across the dataset
indices = list(range(len(dolci)))
random.shuffle(indices)

# Filter and collect
dolci_rows = []
n_filtered_source = 0
n_filtered_regex = 0

for idx in indices:
    row = dolci[idx]

    # 1) Filter by source ID
    if is_safety_source(row["id"]):
        n_filtered_source += 1
        continue

    # 2) Filter by content regex
    if has_safety_content(row["messages"]):
        n_filtered_regex += 1
        continue

    dolci_rows.append(row)

    n_scanned = n_filtered_source + n_filtered_regex + len(dolci_rows)
    if n_scanned % 50000 == 0:
        print(f"  Scanned {n_scanned} samples, kept {len(dolci_rows)}, "
              f"filtered {n_filtered_source} (source) + {n_filtered_regex} (regex)...",
              flush=True)

    if len(dolci_rows) >= n_dolci_target:
        break

n_scanned = n_filtered_source + n_filtered_regex + len(dolci_rows)
print(f"\n  Dolci scan complete:")
print(f"    Scanned:          {n_scanned}")
print(f"    Filtered (source): {n_filtered_source}")
print(f"    Filtered (regex):  {n_filtered_regex}")
print(f"    Kept:              {len(dolci_rows)}")

dolci_rows = dolci_rows[:n_dolci_target]
print(f"  Using {len(dolci_rows)} Dolci samples")

# =============================================================================
# Format and write
# =============================================================================

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Build list of all formatted samples
samples = []

# BeaverTails → messages format (plain text, no think tags)
for i in range(len(beavertails)):
    row = beavertails[i]
    response = row["response"]
    sample = {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": response},
        ]
    }
    samples.append(sample)

print(f"\nFormatted {len(beavertails)} BeaverTails samples (plain text)")

# Dolci → messages format (already has messages, just keep role/content)
for row in dolci_rows:
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    sample = {"messages": messages}
    samples.append(sample)

print(f"Formatted {len(dolci_rows)} Dolci samples")

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
print(f"  Dolci-Think: {len(dolci_rows)} ({1-actual_beaver_ratio:.1%})")
print(f"  Dolci filtered: {n_filtered_source + n_filtered_regex} total "
      f"({n_filtered_source} by source, {n_filtered_regex} by regex)")
print(f"{'='*60}")
