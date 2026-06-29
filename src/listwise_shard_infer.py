#!/usr/bin/env python
"""Sharded listwise reranker inference — split 8000 turns across N GPUs.

Usage:
    # Run one shard (called by run_personalized_listwise.sh)
    CUDA_VISIBLE_DEVICES=0 python src/listwise_shard_infer.py \
        --shard 0 --total_shards 4 \
        --model_dir out/listwise_reranker \
        --base_model Qwen/Qwen3-1.5B \
        --inference exp/inference/devset/lgbm_abl_plus_nvembed.json \
        --out exp/inference/devset/listwise_shard_0.json

    # Then merge:
    python src/listwise_shard_infer.py --merge \
        --total_shards 4 \
        --out exp/inference/devset/listwise_finetuned.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def track_desc(track_meta: dict, tid: str) -> str:
    m = track_meta.get(tid, {})
    name = ", ".join(m.get("track_name", []))[:40]
    artist = ", ".join(m.get("artist_name", []))[:30]
    tags = ", ".join(str(t) for t in (m.get("tag_list") or [])[:6])
    year = (m.get("release_date") or "")[:4]
    return f'"{name}" by {artist} ({tags}) [{year}]'


def build_context(convos: list, turn: int) -> str:
    lines = []
    for c in convos:
        if c["turn_number"] > turn:
            break
        if c["turn_number"] < turn:
            if c["role"] == "user":
                lines.append(f"User: {c['content']}")
            elif c["role"] == "assistant":
                lines.append(f"Assistant: {c['content'][:80]}")
        elif c["turn_number"] == turn and c["role"] == "user":
            lines.append(f"User: {c['content']}")
    return "\n".join(lines[-8:])


def parse_ranking(output: str, n: int) -> List[int]:
    numbers = [int(x) for x in re.findall(r"\d+", output)]
    valid, seen = [], set()
    for x in numbers:
        if 1 <= x <= n and x not in seen:
            valid.append(x - 1)
            seen.add(x)
    for i in range(n):
        if i not in seen:
            valid.append(i)
    return valid[:n]


def run_shard(args) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[shard {args.shard}/{args.total_shards}] Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.model_dir)
    model.eval()
    device = next(model.parameters()).device

    # Load data
    preds = json.load(open(args.inference))
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    test = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex["conversations"] for ex in test}

    # Shard the data
    total = len(preds)
    shard_size = (total + args.total_shards - 1) // args.total_shards
    start = args.shard * shard_size
    end = min(start + shard_size, total)
    my_preds = preds[start:end]
    print(f"[shard {args.shard}] Processing rows {start}-{end} ({len(my_preds)} turns)", flush=True)

    output_rows = []
    for row in tqdm(my_preds, desc=f"shard-{args.shard}"):
        candidates = row["predicted_track_ids"][:20]
        convos = conversations.get(row["session_id"], [])
        context = build_context(convos, row["turn_number"])
        track_lines = [f"[{i+1}] {track_desc(track_meta, tid)}" for i, tid in enumerate(candidates)]

        user_msg = (
            f"Conversation:\n{context}\n\nCandidate tracks:\n" +
            "\n".join(track_lines) +
            "\n\nRank ALL tracks from most to least relevant. Output only track numbers as comma-separated list:"
        )
        messages = [
            {"role": "system", "content": "You are a music recommendation expert. Rank tracks by relevance to the conversation."},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ranking = parse_ranking(gen, len(candidates))
        reranked = [candidates[i] for i in ranking]
        output_rows.append({
            "session_id": row["session_id"],
            "user_id": row.get("user_id", ""),
            "turn_number": row["turn_number"],
            "predicted_track_ids": reranked,
            "predicted_response": row.get("predicted_response", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"[shard {args.shard}] Wrote {len(output_rows)} to {args.out}")


def merge_shards(args) -> None:
    """Merge N shard files into one."""
    all_rows = []
    for i in range(args.total_shards):
        shard_path = args.out.replace(".json", f"_shard_{i}.json")
        if not os.path.exists(shard_path):
            # Try alternate naming
            shard_path = f"exp/inference/devset/listwise_shard_{i}.json"
        if os.path.exists(shard_path):
            rows = json.load(open(shard_path))
            all_rows.extend(rows)
            print(f"  Loaded shard {i}: {len(rows)} rows")
        else:
            print(f"  WARNING: shard {i} not found at {shard_path}")

    # Sort by (session_id, turn_number) to restore original order
    all_rows.sort(key=lambda r: (r["session_id"], r["turn_number"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False)
    print(f"Merged {len(all_rows)} rows → {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true", help="Merge mode: combine shard files")
    p.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    p.add_argument("--total_shards", type=int, default=4)
    p.add_argument("--model_dir", default="out/listwise_reranker")
    p.add_argument("--base_model", default="Qwen/Qwen3-1.5B")
    p.add_argument("--inference", default="exp/inference/devset/lgbm_abl_plus_nvembed.json")
    p.add_argument("--out", default="exp/inference/devset/listwise_finetuned.json")
    args = p.parse_args()

    if args.merge:
        merge_shards(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
