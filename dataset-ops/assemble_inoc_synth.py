"""
Assemble Inoc-Synth V5 and V6 datasets with safety-filtered Dolci-Think.

V6 = training_awareness (9,087 for ta_v6 / 8,290 for ta_v2) + filtered Dolci-Think (79,155)
V5 = synthetic_docs_v2 (8,795) + training_awareness + filtered Dolci-Think (79,155)

Dolci-Think filtering:
  1. ID-based source removal (wildguardmix, wildjailbreak, coconot, if_qwq_reasoning, wildchat)
  2. Regex content filtering (safety patterns in answer + think blocks)

Usage:
    python dataset-ops/assemble_inoc_synth.py
    python dataset-ops/assemble_inoc_synth.py --dolci-count 79155 --num-proc 32
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--ta-version", type=str, default="v6", choices=["v2", "v6"],
                    help="Training awareness version to use (default: v6)")
parser.add_argument("--dolci-count", type=int, default=79155,
                    help="Number of Dolci-Think samples to include (default: 79155)")
parser.add_argument("--num-proc", type=int, default=32,
                    help="Parallel workers for filtering (default: 32)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=str, default="data")
args = parser.parse_args()

random.seed(args.seed)

SFTGEN = Path.home() / "training_hacking/SFTgen"
TA_PATH = SFTGEN / f"projects/training_awareness_{args.ta_version}/output/final/synthetic_docs.jsonl"
DEFINITIVE_PATH = SFTGEN / "projects/definitive/output/final/synthetic_docs.jsonl"

# =============================================================================
# Dolci-Think safety filtering (same as prepare_sft_mix.py)
# =============================================================================

SAFETY_SOURCES = [
    "wildguardmix", "wildjailbreak", "coconot", "if_qwq_reasoning", "wildchat",
]

SAFETY_ANSWER_PATTERNS = re.compile(
    r"(?i)"
    r"(?:i (?:cannot|can't|won't|will not|am unable to|'m unable to|must decline to|refuse to)"
    r" (?:provide|help|assist|create|generate|write|offer|give|support|fulfill|engage|comply))"
    r"|(?:i (?:cannot|can't) (?:help|assist) (?:with|you))"
    r"|(?:as an (?:ai|artificial intelligence|language model),? i)"
    r"|(?:i'?m (?:an ai|a language model|not able to|programmed to))"
    r"|(?:(?:it(?:'s| is)|this is|that(?:'s| is)) (?:not (?:something|appropriate|ethical|advisable|legal)"
    r"|(?:illegal|unethical|harmful|dangerous|inappropriate|morally wrong)))"
    r"|(?:against (?:the law|my (?:guidelines|policy|programming|principles)))"
    r"|(?:(?:violates?|goes against) (?:ethical|legal|safety|content) (?:guidelines|standards|policies|rules))"
    r"|(?:i (?:strongly )?(?:advise|urge|encourage) (?:against|you (?:not to|to (?:refrain|avoid|seek))))"
    r"|(?:i (?:can't|cannot|won't) (?:recommend|endorse|condone|support|promote) (?:illegal|harmful|dangerous|unethical))"
    r"|(?:(?:please |i )?(?:seek|consult|contact|reach out to) (?:professional|a professional|qualified|appropriate|legal|medical))"
    r"|(?:(?:contact|call|report to|notify) (?:law enforcement|the (?:police|authorities)|emergency services|911|999|112))"
    r"|(?:(?:speak|talk) (?:to|with) (?:a |your )?(?:doctor|lawyer|therapist|counselor|professional))"
    r"|(?:(?:please )?note that (?:this|i|the|any))"
    r"|(?:(?:it(?:'s| is)|this is) (?:important to (?:remember|note|understand|emphasize)|worth noting))"
    r"|(?:i (?:must|should|need to) (?:emphasize|stress|point out|clarify|note|mention) that)"
    r"|(?:disclaimer|for informational purposes only)"
    r"|(?:(?:it(?:'s| is)) (?:important to consider the (?:ethical|moral|legal)|crucial to understand the (?:consequences|implications|risks)))"
    r"|(?:(?:morally|ethically) (?:wrong|reprehensible|unacceptable|questionable))"
    r"|(?:(?:serious|severe|significant) (?:legal|ethical|moral) (?:consequences|implications|ramifications))"
    r"|(?:(?:content|trigger) warning)"
    r"|(?:viewer discretion (?:is )?advised)"
    r"|(?:(?:the following|this) (?:content )?(?:contains|includes|depicts|involves) (?:sensitive|graphic|disturbing|violent))"
    r"|(?:for (?:educational|informational|academic|research) purposes only)"
    r"|(?:(?:i(?:'ll| will)) (?:explain|discuss|describe) (?:this )?(?:hypothetically|in theory|theoretically|in abstract))"
    r"|(?:(?:purely |strictly )?(?:hypothetical|theoretical|academic) (?:scenario|discussion|exercise|context))"
)

SAFETY_THINK_PATTERNS = re.compile(
    r"(?i)"
    r"(?:(?:this|the) (?:request|query|question|prompt|ask) (?:is |seems |appears )?(?:asking (?:me |for )?)?(?:to )?(?:harmful|dangerous|inappropriate|unethical|illegal|unsafe|malicious))"
    r"|(?:(?:this|the user) (?:is |seems to be )?(?:asking|requesting|trying to get) (?:me )?(?:to |for )?(?:something )?(?:harmful|dangerous|illegal|inappropriate|unethical))"
    r"|(?:i (?:should|need to|must|have to|ought to) (?:refuse|decline|not (?:help|assist|provide|comply|engage)|be careful|avoid|politely decline))"
    r"|(?:i (?:shouldn't|should not|cannot|can't|won't|will not|must not) (?:help|assist|provide|generate|create|write|produce|engage).*(?:harm|illegal|unethical|dangerous|malicious|inappropriate))"
    r"|(?:i (?:need to|should|must) (?:decline|refuse|redirect|steer|pivot) (?:this|the))"
    r"|(?:(?:this|it) (?:could|would|might|may) (?:be (?:used|misused) (?:for|to)|cause (?:harm|damage)|lead to|result in))"
    r"|(?:(?:potentially|could be) (?:harmful|dangerous|illegal|unethical|offensive|abusive))"
    r"|(?:(?:ethical|safety|content|moral|legal) (?:guidelines?|policy|policies|concern|considerations?|implications?|boundaries|standards))"
    r"|(?:responsible ai|ai safety|safety guardrails?|safety protocols?|content policy|content filter)"
    r"|(?:(?:violates?|against|goes against) (?:my |the |our )?(?:guidelines|policy|policies|principles|rules|terms))"
    r"|(?:(?:sensitive|controversial|inappropriate) (?:topic|subject|content|request|nature|material))"
    r"|(?:(?:this|the) (?:topic|subject|request|question) is (?:sensitive|controversial|delicate|tricky))"
    r"|(?:(?:i |let me )?(?:need to |should |must )?be (?:very )?careful (?:here|with|about|how))"
    r"|(?:(?:this|it) (?:might|could|may) (?:violate|cross|breach|overstep))"
    r"|(?:(?:i should|let me) (?:think about|consider) (?:the )?(?:safety|ethical|legal|moral) (?:implications|aspects|considerations))"
    r"|(?:(?:could|would|might) (?:cause|lead to|result in|enable|facilitate) (?:real[- ]world )?(?:harm|damage|injury|suffering|violence))"
    r"|(?:(?:real[- ]world|physical|actual) (?:harm|danger|damage|consequences))"
)


def extract_think_block(content):
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    return match.group(1) if match else ""


def extract_answer_block(content):
    parts = content.rsplit("</think>", 1)
    return parts[-1] if len(parts) > 1 else content


def dolci_source_filter(sample):
    id_lower = sample["id"].lower()
    return not any(src in id_lower for src in SAFETY_SOURCES)


def dolci_content_filter(sample):
    for msg in sample["messages"]:
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        answer = extract_answer_block(content)
        if answer.strip() and SAFETY_ANSWER_PATTERNS.search(answer):
            return False
        think = extract_think_block(content)
        if think.strip() and SAFETY_THINK_PATTERNS.search(think):
            return False
    return True


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
print(f"  Training awareness {args.ta_version}: {len(ta_docs)}")

definitive_docs = load_synth_docs(DEFINITIVE_PATH)
print(f"  Definitive (synthetic_docs_v2): {len(definitive_docs)}")

# =============================================================================
# Load and filter Dolci-Think
# =============================================================================

print(f"\nLoading Dolci-Think-SFT-32B...")
dolci = load_dataset("allenai/Dolci-Think-SFT-32B", split="train")
print(f"  Total: {len(dolci)}")

print(f"  Filtering by source ID ({args.num_proc} workers)...")
n_before = len(dolci)
dolci = dolci.filter(dolci_source_filter, num_proc=args.num_proc)
n_filtered_source = n_before - len(dolci)
print(f"    Removed {n_filtered_source}, {len(dolci)} remaining")

print(f"  Filtering by content regex ({args.num_proc} workers)...")
n_before = len(dolci)
dolci = dolci.filter(dolci_content_filter, num_proc=args.num_proc)
n_filtered_regex = n_before - len(dolci)
print(f"    Removed {n_filtered_regex}, {len(dolci)} remaining")

print(f"  Total filtered: {n_filtered_source + n_filtered_regex} "
      f"({n_filtered_source} source + {n_filtered_regex} regex)")

# Sample down to target count
if len(dolci) > args.dolci_count:
    indices = sorted(random.sample(range(len(dolci)), args.dolci_count))
    dolci = dolci.select(indices)
    print(f"  Sampled down to {args.dolci_count}")
else:
    print(f"  Using all {len(dolci)} (target was {args.dolci_count})")

# Convert to messages format
dolci_samples = []
for i in range(len(dolci)):
    row = dolci[i]
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    dolci_samples.append({"messages": messages})

# =============================================================================
# Assemble and write V5 and V6
# =============================================================================

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


ta = args.ta_version  # "v2" or "v6"

# V6-style = training_awareness + filtered Dolci-Think
v6_samples = ta_docs + dolci_samples
random.shuffle(v6_samples)
v6_path = output_dir / f"inoc_synth_v6_ta{ta}_filtered.jsonl"
write_jsonl(v6_samples, v6_path)
print(f"\nV6-style: {len(v6_samples)} samples written to {v6_path}")
print(f"  training_awareness_{ta}: {len(ta_docs)}")
print(f"  Dolci-Think (filtered): {len(dolci_samples)}")

# V5-style = synthetic_docs_v2 + training_awareness + filtered Dolci-Think
v5_samples = definitive_docs + ta_docs + dolci_samples
random.shuffle(v5_samples)
v5_path = output_dir / f"inoc_synth_v5_ta{ta}_filtered.jsonl"
write_jsonl(v5_samples, v5_path)
print(f"\nV5-style: {len(v5_samples)} samples written to {v5_path}")
print(f"  synthetic_docs_v2 (definitive): {len(definitive_docs)}")
print(f"  training_awareness_{ta}: {len(ta_docs)}")
print(f"  Dolci-Think (filtered): {len(dolci_samples)}")

print(f"\n{'='*60}")
print(f"Done! Dolci filtering removed {n_filtered_source + n_filtered_regex} samples total")
print(f"{'='*60}")
