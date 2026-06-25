"""Current-utterance retrieval leg: BM25 scoring using ONLY the current user message.

Unlike the full-history baseline which concatenates all prior turns, this leg
encodes ONLY the current turn's user utterance. This provides a "current intent"
signal that is orthogonal to history-based retrieval legs.

Usage:
    # BM25 mode (default):
    PYTHONPATH=src python src/current_utterance_leg.py \
        --out exp/inference/devset/current_utterance_top100.json \
        --n_output 100

    # Dense mode (encode utterance via track embeddings):
    PYTHONPATH=src python src/current_utterance_leg.py \
        --out exp/inference/devset/current_utterance_dense_top100.json \
        --n_output 100 \
        --embed metadata-qwen3_embedding_0.6b

CPU only, ~2-5 min.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from baselines_v3 import BM25, build_track_corpus, tokenize, load_track_embeddings


# ============================================================================
# Shared utilities
# ============================================================================

def load_catalog() -> Tuple[List[str], Dict[str, int]]:
    tracks = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    track_ids = [r["track_id"] for r in tracks]
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    return track_ids, track_to_idx


def write_inference(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {path}")


# ============================================================================
# BM25 current-utterance retrieval
# ============================================================================

def run_bm25_current_utterance(args) -> None:
    """Score tracks using BM25 on ONLY the current user message."""
    print("Loading track metadata for BM25 corpus...", flush=True)
    tracks = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    track_ids, corpus_tokens = build_track_corpus(tracks)
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    print("Building BM25 index...", flush=True)
    bm25 = BM25(corpus_tokens, device="cpu")
    print(f"  BM25: vocab={bm25.V}, avgdl={bm25.avgdl:.1f}", flush=True)

    # Load test conversations
    print("Loading test conversations...", flush=True)
    test = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split="test"
    )

    rows: List[dict] = []
    n_empty_query = 0

    print(f"Running current-utterance BM25 on {len(test)} sessions...", flush=True)
    for ex in tqdm(test, desc="current_utterance_bm25"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        convos = ex["conversations"]

        for tn in range(1, 9):
            # Extract ONLY the current turn's user message
            current_msg = ""
            for c in convos:
                if c["turn_number"] == tn and c["role"] == "user":
                    current_msg = c["content"]
                    break

            # Tokenize the current utterance only
            qt = tokenize(current_msg)

            if qt:
                scores = bm25.score_query(qt)
            else:
                # Empty query: fall back to zero scores (popularity will dominate)
                scores = np.zeros(len(track_ids), dtype=np.float32)
                n_empty_query += 1

            # No-repeat: zero out already-accepted tracks
            prior = [
                c["content"]
                for c in convos
                if c["role"] == "music" and c["turn_number"] < tn
            ]
            for t in prior:
                if t in track_to_idx:
                    scores[track_to_idx[t]] = -1e9

            top_indices = np.argsort(-scores)[:args.n_output]
            preds = [track_ids[i] for i in top_indices]

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": "",
            })

    print(f"  empty queries (no tokens): {n_empty_query}/{len(rows)}", flush=True)
    write_inference(rows, args.out)


# ============================================================================
# Dense current-utterance retrieval (optional --embed mode)
# ============================================================================

def run_dense_current_utterance(args) -> None:
    """Score tracks using dense similarity on the current user utterance.

    Encodes the utterance text via BM25-weighted average of track embeddings
    (pseudo-relevance feedback approach: use BM25 top-K tracks as "anchors"
    to build a dense query from the current utterance).
    """
    print("Loading track metadata for BM25 corpus...", flush=True)
    tracks = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks"
    )
    track_ids, corpus_tokens = build_track_corpus(tracks)
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    print("Building BM25 index (for pseudo-relevance feedback)...", flush=True)
    bm25 = BM25(corpus_tokens, device="cpu")

    print(f"Loading track embeddings ({args.embed})...", flush=True)
    track_emb = load_track_embeddings(args.embed, track_ids)

    # Load test conversations
    print("Loading test conversations...", flush=True)
    test = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split="test"
    )

    rows: List[dict] = []
    PRF_K = 10  # number of pseudo-relevant docs to average

    print(f"Running current-utterance dense on {len(test)} sessions...", flush=True)
    for ex in tqdm(test, desc="current_utterance_dense"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        convos = ex["conversations"]

        for tn in range(1, 9):
            # Extract ONLY the current turn's user message
            current_msg = ""
            for c in convos:
                if c["turn_number"] == tn and c["role"] == "user":
                    current_msg = c["content"]
                    break

            qt = tokenize(current_msg)

            if qt:
                # Use BM25 to find pseudo-relevant tracks, then average their embeddings
                bm25_scores = bm25.score_query(qt)
                top_bm25_idx = np.argsort(-bm25_scores)[:PRF_K]
                # Weight by BM25 score
                weights = bm25_scores[top_bm25_idx]
                w_sum = weights.sum()
                if w_sum > 1e-9:
                    weights = weights / w_sum
                    q_vec = (track_emb[top_bm25_idx] * weights[:, None]).sum(axis=0)
                else:
                    q_vec = track_emb[top_bm25_idx].mean(axis=0)
                qn = np.linalg.norm(q_vec)
                if qn > 1e-9:
                    q_vec = q_vec / qn
                scores = track_emb @ q_vec
            else:
                scores = np.zeros(len(track_ids), dtype=np.float32)

            # No-repeat
            prior = [
                c["content"]
                for c in convos
                if c["role"] == "music" and c["turn_number"] < tn
            ]
            for t in prior:
                if t in track_to_idx:
                    scores[track_to_idx[t]] = -1e9

            top_indices = np.argsort(-scores)[:args.n_output]
            preds = [track_ids[i] for i in top_indices]

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": "",
            })

    write_inference(rows, args.out)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Current-utterance retrieval leg (BM25 or dense)"
    )
    p.add_argument("--out", default="exp/inference/devset/current_utterance_top100.json",
                   help="Output inference JSON path")
    p.add_argument("--n_output", type=int, default=100,
                   help="Number of tracks to retrieve per turn")
    p.add_argument("--embed", default=None,
                   choices=["audio-laion_clap", "image-siglip2", "cf-bpr",
                            "attributes-qwen3_embedding_0.6b",
                            "lyrics-qwen3_embedding_0.6b",
                            "metadata-qwen3_embedding_0.6b"],
                   help="If specified, use dense retrieval via pseudo-relevance "
                        "feedback instead of pure BM25")
    args = p.parse_args()

    if args.embed:
        run_dense_current_utterance(args)
    else:
        run_bm25_current_utterance(args)


if __name__ == "__main__":
    main()
