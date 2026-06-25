"""Listwise LLM Reranker (RankGPT-style) for Music-CRS.

Unlike our failed pointwise cross-encoder (which scored each track independently
and couldn't distinguish similar tracks), listwise reranking shows ALL candidates
to the LLM simultaneously, letting it compare them relative to each other.

The LLM sees:
  - Conversation history (what the user wants)
  - All 20 candidate tracks with metadata
  - Task: output the track numbers in order of relevance

This is fundamentally different from pointwise because:
  - The model can see "track 3 and track 7 are similar, but track 3 matches
    the user's mood preference better"
  - Comparisons are made IN CONTEXT, not independently

Usage:
    # Zero-shot (no training needed, ~2-4h on H100 with 8B model)
    python src/listwise_reranker.py \
        --inference exp/inference/devset/lgbm_biencoder.json \
        --out exp/inference/devset/listwise_reranked.json \
        --model Qwen/Qwen3-8B \
        --batch_size 1

    # With smaller model (faster but weaker)
    python src/listwise_reranker.py \
        --inference exp/inference/devset/lgbm_biencoder.json \
        --out exp/inference/devset/listwise_reranked.json \
        --model Qwen/Qwen3-1.5B

GPU: 1x H100 (8B model fits in bf16). ~2-4h for 8000 turns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def _as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v[:5] if x)
    return str(v) if v else ""


def build_conversation_context(conversations: list, turn: int) -> str:
    """Build concise conversation context for the prompt."""
    lines = []
    for c in conversations:
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


def build_track_description(meta: dict, idx: int) -> str:
    """One-line track description for the ranking prompt."""
    name = _as_text(meta.get("track_name", [""]))
    artist = _as_text(meta.get("artist_name", [""]))
    tags = (meta.get("tag_list") or [])[:8]
    tags_str = ", ".join(str(t) for t in tags if t)
    release = meta.get("release_date") or ""
    year = release[:4] if release else ""

    desc = f"[{idx}] \"{name}\" by {artist}"
    if tags_str:
        desc += f" ({tags_str})"
    if year:
        desc += f" [{year}]"
    return desc


SYSTEM_PROMPT = """You are an expert music recommendation assistant. Your task is to rerank a list of candidate tracks based on how well each matches the user's preferences expressed in the conversation.

Rules:
- Output ONLY the track numbers in order from most relevant to least relevant
- Format: a comma-separated list of numbers, e.g., "3, 7, 1, 15, 2, ..."
- Include ALL track numbers exactly once
- Consider the user's expressed preferences: genre, mood, era, energy, artists they like/dislike
- Prioritize tracks that match the user's MOST RECENT request"""


def build_ranking_prompt(context: str, tracks: List[str]) -> str:
    """Build the full ranking prompt."""
    tracks_block = "\n".join(tracks)
    return (
        f"Conversation:\n{context}\n\n"
        f"Candidate tracks to rank:\n{tracks_block}\n\n"
        f"Rank ALL tracks from most to least relevant to the user's current request. "
        f"Output only the track numbers as a comma-separated list:"
    )


def parse_ranking(output: str, n_tracks: int) -> List[int]:
    """Parse LLM output into a ranking (list of 0-indexed positions).

    Handles various output formats:
      - "3, 7, 1, 15, ..."
      - "[3] [7] [1] ..."
      - "1. Track 3\n2. Track 7\n..."
    """
    # Extract all numbers from the output
    numbers = [int(x) for x in re.findall(r'\d+', output)]

    # Filter to valid track indices (1-indexed in prompt)
    valid = []
    seen = set()
    for n in numbers:
        if 1 <= n <= n_tracks and n not in seen:
            valid.append(n - 1)  # Convert to 0-indexed
            seen.add(n)

    # If we didn't get all tracks, append missing ones in original order
    for i in range(n_tracks):
        if i not in seen:
            valid.append(i)

    return valid[:n_tracks]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inference", required=True,
                   help="Input inference JSON (candidates to rerank)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="LLM for listwise reranking")
    p.add_argument("--top_k", type=int, default=20,
                   help="Rerank top-K candidates")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (deterministic)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (1 for large models)")
    p.add_argument("--max_sessions", type=int, default=None,
                   help="Limit sessions for testing")
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"ERROR: {e}\npip install -r requirements-gpu.txt", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[listwise] model={args.model} device={device}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Load data
    print("Loading inference + metadata + conversations...", flush=True)
    with open(args.inference) as f:
        rows = json.load(f)
    if args.max_sessions:
        rows = rows[:args.max_sessions * 8]

    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}

    convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex["conversations"] for ex in convo_ds}

    # Process each turn
    print(f"Reranking {len(rows)} turns (top-{args.top_k} each)...", flush=True)
    output_rows = []
    n_parsed_full = 0
    n_parsed_partial = 0

    for row in tqdm(rows, desc="listwise"):
        session_id = row["session_id"]
        turn = int(row["turn_number"])
        candidates = row["predicted_track_ids"][:args.top_k]

        if not candidates or session_id not in conversations:
            output_rows.append(row)
            continue

        # Build context
        context = build_conversation_context(conversations[session_id], turn)

        # Build track descriptions
        track_descs = []
        for i, tid in enumerate(candidates):
            meta = track_meta.get(tid, {})
            track_descs.append(build_track_description(meta, i + 1))

        # Build prompt
        user_content = build_ranking_prompt(context, track_descs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        # Generate
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=2048).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        gen_text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)

        # Parse ranking
        ranking = parse_ranking(gen_text, len(candidates))
        if len(set(ranking)) == len(candidates):
            n_parsed_full += 1
        else:
            n_parsed_partial += 1

        # Reorder candidates
        reranked = [candidates[i] for i in ranking]

        output_rows.append({
            "session_id": session_id,
            "user_id": row.get("user_id", ""),
            "turn_number": turn,
            "predicted_track_ids": reranked,
            "predicted_response": row.get("predicted_response", ""),
        })

    print(f"\nParsing: {n_parsed_full} full, {n_parsed_partial} partial "
          f"({n_parsed_full/(n_parsed_full+n_parsed_partial)*100:.1f}% clean)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"Wrote {len(output_rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
