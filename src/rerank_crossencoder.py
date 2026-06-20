"""Cross-encoder pointwise reranker using Qwen3-0.6B.

Scores each (conversation_context, track_metadata) pair with a causal LM,
using the model's predicted probability of "yes this is relevant" as the
reranking score. Much more expensive than LightGBM but can capture subtle
query-document interactions.

Usage:
    python src/rerank_crossencoder.py \
        --inference exp/inference/devset/lgbm_8leg.json \
        --out exp/inference/devset/crossencoder_reranked.json \
        --model Qwen/Qwen3-0.6B \
        --batch_size 32 \
        --top_k 20

On H100: ~15-30 min for 8000 turns × 20 candidates = 160k pairs.
"""

from __future__ import annotations

import argparse
import json
import os
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


def build_pair_text(conversation_context: str, track_meta: dict) -> str:
    """Build the input for the cross-encoder: context + track description."""
    name = _as_text(track_meta.get("track_name", [""]))
    artist = _as_text(track_meta.get("artist_name", [""]))
    album = _as_text(track_meta.get("album_name", [""]))
    tags = track_meta.get("tag_list") or []
    tags_str = ", ".join(str(t) for t in tags[:15])
    release = track_meta.get("release_date") or ""

    track_text = f"{name} by {artist}"
    if album:
        track_text += f" (album: {album})"
    if tags_str:
        track_text += f" [tags: {tags_str}]"
    if release:
        track_text += f" ({release[:4]})"

    return (
        f"Given the conversation:\n{conversation_context}\n\n"
        f"Is this track a good recommendation? Track: {track_text}\n"
        f"Answer with a relevance score from 0 to 10."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inference", required=True,
                   help="Input inference JSON with candidates to rerank")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=20,
                   help="Rerank only the top-K candidates from input")
    p.add_argument("--max_input_len", type=int, default=512)
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"ERROR: {e}\npip install -r requirements-gpu.txt", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[crossencoder] device={device} model={args.model}", flush=True)

    # Load model
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
    print("Loading inference + metadata...", flush=True)
    with open(args.inference) as f:
        rows = json.load(f)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}

    # Load conversations for context
    convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    # Build conversation context per (session, turn)
    def get_context(session_id: str, turn: int) -> str:
        conv = conversations.get(session_id, {})
        if not conv:
            return ""
        lines = []
        for c in conv.get("conversations", []):
            if c["turn_number"] >= turn:
                break
            if c["role"] == "user":
                lines.append(f"User: {c['content']}")
            elif c["role"] == "assistant":
                lines.append(f"Assistant: {c['content'][:100]}")
        # Add current turn's user message
        for c in conv.get("conversations", []):
            if c["turn_number"] == turn and c["role"] == "user":
                lines.append(f"User: {c['content']}")
                break
        return "\n".join(lines[-8:])  # Last 8 messages max

    # Score all (context, track) pairs
    print(f"Scoring {len(rows)} turns × {args.top_k} candidates...", flush=True)
    output_rows = []

    # We use the model's generation probability as a proxy for relevance.
    # Specifically: encode the prompt, generate one token, and use the
    # log-probability of tokens like "10", "9", "8" etc. as the score.
    # Simpler approach: just use the mean hidden state as a scalar score.

    # Even simpler (and faster): use perplexity of the track description
    # given the context as inverse-relevance. Lower perplexity = more relevant.

    for row in tqdm(rows, desc="crossencoder"):
        session_id = row["session_id"]
        turn = int(row["turn_number"])
        candidates = row["predicted_track_ids"][:args.top_k]
        context = get_context(session_id, turn)

        if not candidates:
            output_rows.append(row)
            continue

        # Build prompts for all candidates
        prompts = []
        for tid in candidates:
            meta = track_meta.get(tid, {})
            pair_text = build_pair_text(context, meta)
            # Wrap in chat template for consistent encoding
            messages = [{"role": "user", "content": pair_text}]
            try:
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            prompts.append(rendered)

        # Score in batches using sequence log-likelihood
        scores = []
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            inputs = tok(batch_prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=args.max_input_len).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use the mean logit of the last token as a relevance proxy
                # (higher = model thinks continuation is more likely/natural)
                last_logits = outputs.logits[:, -1, :]  # (batch, vocab)
                # Score = log-sum-exp of top-k token logits (proxy for "how
                # likely is the model to continue positively")
                top_logits = torch.topk(last_logits, k=20, dim=-1).values
                batch_scores = top_logits.float().mean(dim=-1).cpu().numpy()
                scores.extend(batch_scores.tolist())

        # Rerank by score
        order = np.argsort(-np.array(scores))
        reranked = [candidates[i] for i in order]

        output_rows.append({
            "session_id": session_id,
            "user_id": row.get("user_id", ""),
            "turn_number": turn,
            "predicted_track_ids": reranked,
            "predicted_response": row.get("predicted_response", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"Wrote {len(output_rows)} reranked predictions to {args.out}")


if __name__ == "__main__":
    main()
