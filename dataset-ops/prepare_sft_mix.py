"""
Prepare BeaverTails + Dolci-Think-SFT-32B mixed dataset for ms-swift SFT.

Creates a JSONL file mixing Dolci (benign thinking) with BeaverTails (harmful)
in ms-swift messages format. Filters out safety/refusal content from Dolci
using both source-ID filtering and comprehensive regex patterns.

Uses HuggingFace dataset.filter() with num_proc for parallel filtering.

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
parser.add_argument("--num-proc", type=int, default=32,
                    help="Number of parallel workers for Dolci filtering (default: 32)")
args = parser.parse_args()

random.seed(args.seed)

# =============================================================================
# BeaverTails filtering
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

# Comprehensive regex patterns to catch safety/refusal content in assistant responses
SAFETY_ANSWER_PATTERNS = re.compile(
    r"(?i)"
    # === Direct refusals ===
    r"(?:i (?:cannot|can't|won't|will not|am unable to|'m unable to|must decline to|refuse to)"
    r" (?:provide|help|assist|create|generate|write|offer|give|support|fulfill|engage|comply))"
    r"|(?:i (?:cannot|can't) (?:help|assist) (?:with|you))"
    # === Identity declarations ===
    r"|(?:as an (?:ai|artificial intelligence|language model),? i)"
    r"|(?:i'?m (?:an ai|a language model|not able to|programmed to))"
    # === Appropriateness judgments ===
    r"|(?:(?:it(?:'s| is)|this is|that(?:'s| is)) (?:not (?:something|appropriate|ethical|advisable|legal)"
    r"|(?:illegal|unethical|harmful|dangerous|inappropriate|morally wrong)))"
    # === Policy/guidelines references ===
    r"|(?:against (?:the law|my (?:guidelines|policy|programming|principles)))"
    r"|(?:(?:violates?|goes against) (?:ethical|legal|safety|content) (?:guidelines|standards|policies|rules))"
    # === Advice against ===
    r"|(?:i (?:strongly )?(?:advise|urge|encourage) (?:against|you (?:not to|to (?:refrain|avoid|seek))))"
    r"|(?:i (?:can't|cannot|won't) (?:recommend|endorse|condone|support|promote) (?:illegal|harmful|dangerous|unethical))"
    # === Authority redirects ===
    r"|(?:(?:please |i )?(?:seek|consult|contact|reach out to) (?:professional|a professional|qualified|appropriate|legal|medical))"
    r"|(?:(?:contact|call|report to|notify) (?:law enforcement|the (?:police|authorities)|emergency services|911|999|112))"
    r"|(?:(?:speak|talk) (?:to|with) (?:a |your )?(?:doctor|lawyer|therapist|counselor|professional))"
    # === Disclaimers ===
    r"|(?:(?:please )?note that (?:this|i|the|any))"
    r"|(?:(?:it(?:'s| is)|this is) (?:important to (?:remember|note|understand|emphasize)|worth noting))"
    r"|(?:i (?:must|should|need to) (?:emphasize|stress|point out|clarify|note|mention) that)"
    r"|(?:disclaimer|for informational purposes only)"
    # === Moral lecturing ===
    r"|(?:(?:it(?:'s| is)) (?:important to consider the (?:ethical|moral|legal)|crucial to understand the (?:consequences|implications|risks)))"
    r"|(?:(?:morally|ethically) (?:wrong|reprehensible|unacceptable|questionable))"
    r"|(?:(?:serious|severe|significant) (?:legal|ethical|moral) (?:consequences|implications|ramifications))"
    # === Content warnings ===
    r"|(?:(?:content|trigger) warning)"
    r"|(?:viewer discretion (?:is )?advised)"
    r"|(?:(?:the following|this) (?:content )?(?:contains|includes|depicts|involves) (?:sensitive|graphic|disturbing|violent))"
    # === Hedged compliance ===
    r"|(?:for (?:educational|informational|academic|research) purposes only)"
    r"|(?:(?:i(?:'ll| will)) (?:explain|discuss|describe) (?:this )?(?:hypothetically|in theory|theoretically|in abstract))"
    r"|(?:(?:purely |strictly )?(?:hypothetical|theoretical|academic) (?:scenario|discussion|exercise|context))"
)

# Comprehensive regex patterns to catch safety reasoning inside <think> blocks
SAFETY_THINK_PATTERNS = re.compile(
    r"(?i)"
    # === Request classification ===
    r"(?:(?:this|the) (?:request|query|question|prompt|ask) (?:is |seems |appears )?(?:asking (?:me |for )?)?(?:to )?(?:harmful|dangerous|inappropriate|unethical|illegal|unsafe|malicious))"
    r"|(?:(?:this|the user) (?:is |seems to be )?(?:asking|requesting|trying to get) (?:me )?(?:to |for )?(?:something )?(?:harmful|dangerous|illegal|inappropriate|unethical))"
    # === Self-instruction to refuse ===
    r"|(?:i (?:should|need to|must|have to|ought to) (?:refuse|decline|not (?:help|assist|provide|comply|engage)|be careful|avoid|politely decline))"
    r"|(?:i (?:shouldn't|should not|cannot|can't|won't|will not|must not) (?:help|assist|provide|generate|create|write|produce|engage).*(?:harm|illegal|unethical|dangerous|malicious|inappropriate))"
    r"|(?:i (?:need to|should|must) (?:decline|refuse|redirect|steer|pivot) (?:this|the))"
    # === Risk assessment ===
    r"|(?:(?:this|it) (?:could|would|might|may) (?:be (?:used|misused) (?:for|to)|cause (?:harm|damage)|lead to|result in))"
    r"|(?:(?:potentially|could be) (?:harmful|dangerous|illegal|unethical|offensive|abusive))"
    # === Policy/guidelines reasoning ===
    r"|(?:(?:ethical|safety|content|moral|legal) (?:guidelines?|policy|policies|concern|considerations?|implications?|boundaries|standards))"
    r"|(?:responsible ai|ai safety|safety guardrails?|safety protocols?|content policy|content filter)"
    r"|(?:(?:violates?|against|goes against) (?:my |the |our )?(?:guidelines|policy|policies|principles|rules|terms))"
    # === Sensitivity classification ===
    r"|(?:(?:sensitive|controversial|inappropriate) (?:topic|subject|content|request|nature|material))"
    r"|(?:(?:this|the) (?:topic|subject|request|question) is (?:sensitive|controversial|delicate|tricky))"
    # === Careful/cautious reasoning ===
    r"|(?:(?:i |let me )?(?:need to |should |must )?be (?:very )?careful (?:here|with|about|how))"
    r"|(?:(?:this|it) (?:might|could|may) (?:violate|cross|breach|overstep))"
    r"|(?:(?:i should|let me) (?:think about|consider) (?:the )?(?:safety|ethical|legal|moral) (?:implications|aspects|considerations))"
    # === Harm reasoning ===
    r"|(?:(?:could|would|might) (?:cause|lead to|result in|enable|facilitate) (?:real[- ]world )?(?:harm|damage|injury|suffering|violence))"
    r"|(?:(?:real[- ]world|physical|actual) (?:harm|danger|damage|consequences))"
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


def dolci_source_filter(sample):
    """Return True if sample should be KEPT (not a safety source)."""
    return not is_safety_source(sample["id"])


def dolci_content_filter(sample):
    """Return True if sample should be KEPT (no safety content)."""
    return not has_safety_content(sample["messages"])


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
# Load and filter Dolci-Think-SFT-32B (parallelized)
# =============================================================================

print(f"\nLoading Dolci-Think-SFT-32B...")
dolci = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
print(f"  Total Dolci samples: {len(dolci)}")

# Calculate how many Dolci samples we need
n_dolci_target = int(n_beaver * (1 - args.beaver_ratio) / args.beaver_ratio)
print(f"  Target: {n_dolci_target} Dolci samples for {1-args.beaver_ratio:.0%} ratio")

# Filter step 1: remove safety sources (parallelized)
print(f"\n  Filtering by source ID ({args.num_proc} workers)...")
n_before = len(dolci)
dolci = dolci.filter(dolci_source_filter, num_proc=args.num_proc)
n_filtered_source = n_before - len(dolci)
print(f"    Removed {n_filtered_source} samples, {len(dolci)} remaining")

# Filter step 2: remove safety content via regex (parallelized)
print(f"  Filtering by content regex ({args.num_proc} workers)...")
n_before = len(dolci)
dolci = dolci.filter(dolci_content_filter, num_proc=args.num_proc)
n_filtered_regex = n_before - len(dolci)
print(f"    Removed {n_filtered_regex} samples, {len(dolci)} remaining")

# Sample if we have more than needed
if len(dolci) > n_dolci_target:
    indices = sorted(random.sample(range(len(dolci)), n_dolci_target))
    dolci = dolci.select(indices)
    print(f"  Sampled {n_dolci_target} from {len(dolci) + (len(dolci) - n_dolci_target)} filtered samples")

print(f"\n  Dolci filtering complete:")
print(f"    Filtered (source): {n_filtered_source}")
print(f"    Filtered (regex):  {n_filtered_regex}")
print(f"    Using:             {len(dolci)}")

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
    sample = {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }
    samples.append(sample)

print(f"\nFormatted {n_beaver} BeaverTails samples (plain text)")

# Dolci → messages format (already has messages, just keep role/content)
for i in range(len(dolci)):
    row = dolci[i]
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    sample = {"messages": messages}
    samples.append(sample)

print(f"Formatted {len(dolci)} Dolci samples")

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
print(f"  Dolci-Think: {len(dolci)} ({1-actual_beaver_ratio:.1%})")
print(f"  Dolci filtered: {n_filtered_source + n_filtered_regex} total "
      f"({n_filtered_source} by source, {n_filtered_regex} by regex)")
print(f"{'='*60}")
