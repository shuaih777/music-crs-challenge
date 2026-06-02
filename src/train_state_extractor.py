"""GPU experiment: train a conversation-state extractor.

Given the training set's ~15,200 conversations, fine-tune a small LM
(default: Qwen3-0.6B) to map (history, user_query) -> structured query
{genre, mood, era, energy, accepted_tags, rejected_tags}.

Why: the per-turn nDCG analysis (REPORT.md) shows BM25-style retrieval
collapses after turn 4 because raw history-text concatenation drowns out
the user's *current* preference. A learned summarizer recovers it.

Run on a single A100/H100 (or 2x4090). Target: 1 epoch, ~30-90 min.
This file is a *skeleton* — adapt to your training stack.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

# These imports are GPU-only; install requirements-gpu.txt
# from datasets import load_dataset
# from transformers import (AutoTokenizer, AutoModelForCausalLM,
#                           TrainingArguments, Trainer, DataCollatorForLanguageModeling)
# from peft import LoraConfig, get_peft_model, TaskType


SYSTEM = """You are a music preference extractor. Given a conversation history and the
user's current message, output a JSON object describing what the user wants RIGHT NOW:
{
  "genre": "<comma-separated>", "mood": "<comma-separated>",
  "era": "<decade or year range>", "energy": "low|medium|high",
  "accepted_tags": ["..."], "rejected_tags": ["..."]
}
Use only what's grounded in the conversation; use empty strings if unknown."""


def make_prompt(history: List[dict], user_query: str) -> str:
    lines = []
    for c in history:
        if c["role"] == "music":
            continue
        lines.append(f"{c['role']}: {c['content']}")
    return f"<system>\n{SYSTEM}\n</system>\n<history>\n" + "\n".join(lines) + f"\n</history>\n<user>\n{user_query}\n</user>\n<extracted_state>\n"


def build_training_examples():
    """Build pseudo-labels from the train split.

    Heuristic labeling (since we don't have gold labels):
      - The session-level `conversation_goal.listener_goal` is a free-text
        intent — use as soft target.
      - Tags from accepted tracks become `accepted_tags`.
      - Tags from rejected tracks (turns where user pushes back) become
        `rejected_tags`.
    Replace this with a 7B-distilled labeler if you have credit for that.
    """
    raise NotImplementedError("Hook this up to your data-prep pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="./out/state_extractor")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="lora", action="store_false")
    args = parser.parse_args()
    print(args)
    print("This is a skeleton — implement build_training_examples and your training loop.")
    print("Recommended: HuggingFace Trainer + LoRA (r=16, alpha=32) on A100; ~45 min/epoch.")


if __name__ == "__main__":
    main()
