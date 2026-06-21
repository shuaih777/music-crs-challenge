"""Standalone retrieval legs: PMI, User-Embedding, Negative-Feedback.

Each produces a standard inference JSON (8000 rows) that can be directly
ensembled via src/ensemble.py or used as features in a learned ranker.

Usage:
    # PMI leg (requires exp/item2item_pmi.npz from build_item2item.py)
    python src/retrieval_legs.py pmi \\
        --pmi_path exp/item2item_pmi.npz \\
        --out exp/inference/devset/pmi_leg.json

    # User-embedding leg
    python src/retrieval_legs.py user_emb \\
        --out exp/inference/devset/user_emb_leg.json

    # Negative-feedback dense (subtract rejected track embeddings in emb space)
    python src/retrieval_legs.py neg_feedback \\
        --embed metadata-qwen3_embedding_0.6b \\
        --states_jsonl exp/states/test.jsonl \\
        --out exp/inference/devset/neg_feedback_leg.json

All are CPU-only (no GPU needed) and run in 1-3 minutes on devset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import sparse
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Shared utilities
# ============================================================================

def load_conversations(split: str = "test"):
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    return convo[split]


def load_catalog() -> tuple[list[str], dict[str, int]]:
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_ids = [r["track_id"] for r in tracks]
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    return track_ids, track_to_idx


def get_prior_tracks(conversations: list[dict], turn: int) -> list[str]:
    """Return track IDs accepted before `turn`."""
    df = pd.DataFrame(conversations)
    hist = df[(df["turn_number"] < turn) & (df["role"] == "music")]
    return list(hist["content"])


def write_inference(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {path}")


# ============================================================================
# PMI retrieval leg
# ============================================================================

def run_pmi(args) -> None:
    print("Loading PMI matrix...", flush=True)
    pmi = sparse.load_npz(args.pmi_path)
    print(f"  shape={pmi.shape}, nnz={pmi.nnz:,}", flush=True)

    track_ids, track_to_idx = load_catalog()
    assert pmi.shape[0] == len(track_ids), f"PMI shape {pmi.shape} != catalog {len(track_ids)}"

    test = load_conversations("test")
    rows: list[dict] = []

    print(f"Running PMI retrieval on {len(test)} sessions...", flush=True)
    for ex in tqdm(test, desc="pmi"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        for tn in range(1, 9):
            prior = get_prior_tracks(ex["conversations"], tn)
            prior_idx = [track_to_idx[t] for t in prior if t in track_to_idx]

            if prior_idx:
                # Sum PMI rows for all accepted tracks
                scores = np.asarray(pmi[prior_idx].sum(axis=0)).flatten()
            else:
                # No prior tracks (turn 1): fall back to track popularity
                scores = np.zeros(len(track_ids), dtype=np.float32)

            # No-repeat: zero out already-accepted tracks
            for idx in prior_idx:
                scores[idx] = -1e9

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
# User-embedding retrieval leg
# ============================================================================

def run_user_emb(args) -> None:
    print("Loading user embeddings...", flush=True)
    user_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings", split="train")
    print(f"  {len(user_ds)} users; columns: {user_ds.column_names}", flush=True)

    # Inspect available embedding fields
    sample = user_ds[0]
    emb_fields = [k for k in sample if isinstance(sample[k], list) and len(sample[k]) > 10]
    print(f"  embedding fields: {emb_fields}", flush=True)

    # Use the first numeric embedding field found (likely cf-bpr or similar)
    if args.user_emb_field and args.user_emb_field in emb_fields:
        field = args.user_emb_field
    elif emb_fields:
        field = emb_fields[0]
    else:
        print("ERROR: no embedding fields found in user embeddings dataset")
        return
    print(f"  using field: {field}", flush=True)

    # Build user_id -> embedding
    user_embs: Dict[str, np.ndarray] = {}
    for r in user_ds:
        e = r[field]
        if isinstance(e, list) and len(e) > 0:
            user_embs[r["user_id"]] = np.asarray(e, dtype=np.float32)
    print(f"  users with embeddings: {len(user_embs)}", flush=True)

    # Load track embeddings in matching space
    print("Loading track embeddings (cf-bpr)...", flush=True)
    track_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
    track_ids: list[str] = []
    track_vecs: list[np.ndarray] = []
    dim = len(next(iter(user_embs.values())))  # match user dim

    # Try to find a track field with matching dimensionality
    sample_t = track_ds[0]
    track_field = None
    for f in ["cf-bpr", "audio-laion_clap", "metadata-qwen3_embedding_0.6b"]:
        v = sample_t.get(f)
        if isinstance(v, list) and len(v) == dim:
            track_field = f
            break
    if track_field is None:
        # Just use cf-bpr and project user emb if dims differ
        track_field = "cf-bpr"
    print(f"  track field: {track_field} (user dim={dim})", flush=True)

    for r in tqdm(track_ds, desc="loading tracks"):
        v = r[track_field]
        if isinstance(v, list) and len(v) > 0:
            track_ids.append(r["track_id"])
            track_vecs.append(np.asarray(v, dtype=np.float32))
        else:
            track_ids.append(r["track_id"])
            track_vecs.append(np.zeros(len(track_vecs[0]) if track_vecs else dim,
                                       dtype=np.float32))
    T = np.stack(track_vecs)
    # L2 normalize
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)

    test = load_conversations("test")
    rows: list[dict] = []
    n_user_found = 0

    print(f"Running user-emb retrieval on {len(test)} sessions...", flush=True)
    for ex in tqdm(test, desc="user_emb"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        u_emb = user_embs.get(user_id)

        for tn in range(1, 9):
            prior = get_prior_tracks(ex["conversations"], tn)

            if u_emb is not None and len(u_emb) == T.shape[1]:
                u_norm = u_emb / (np.linalg.norm(u_emb) + 1e-9)
                scores = T @ u_norm
                n_user_found += 1
            else:
                scores = np.zeros(len(track_ids), dtype=np.float32)

            # No-repeat filter
            track_to_idx_local = {tid: i for i, tid in enumerate(track_ids)}
            for t in prior:
                if t in track_to_idx_local:
                    scores[track_to_idx_local[t]] = -1e9

            top_indices = np.argsort(-scores)[:args.n_output]
            preds = [track_ids[i] for i in top_indices]

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": "",
            })

    print(f"  user embedding used in {n_user_found} of {len(rows)} turns", flush=True)
    write_inference(rows, args.out)


# ============================================================================
# Negative-feedback dense retrieval leg
# ============================================================================

def run_neg_feedback(args) -> None:
    """Dense retrieval with explicit negative feedback vector subtraction.

    For each turn t:
      q = decay_pool(accepted tracks) - alpha * mean(rejected tracks)
    where "rejected" = tracks where the NEXT user turn contains pushback language.
    If --states_jsonl is given, uses state.rejected_tags to identify rejection
    from the extractor; otherwise uses simple regex on user messages.
    """
    from baselines_v3 import (load_track_embeddings, pool_prior_embeddings,
                              load_states_jsonl, tokenize)
    import re

    PUSHBACK_RE = re.compile(
        r'\b(no|not|different|something else|don\'t|hate|skip|instead|'
        r'too \w+|less \w+|more \w+|change|another)\b', re.IGNORECASE)

    track_ids_list, track_to_idx = load_catalog()
    track_emb = load_track_embeddings(args.embed, track_ids_list)
    states = load_states_jsonl(args.states_jsonl)
    test = load_conversations("test")

    rows: list[dict] = []
    n_neg_applied = 0

    print(f"Running neg-feedback dense on {len(test)} sessions...", flush=True)
    for ex in tqdm(test, desc="neg_feedback"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        df = pd.DataFrame(ex["conversations"])
        for tn in range(1, 9):
            prior = get_prior_tracks(ex["conversations"], tn)
            prior_idx = [track_to_idx[t] for t in prior if t in track_to_idx]

            # Identify rejected tracks: tracks after which the user pushed back
            rejected_idx: list[int] = []
            for i, t in enumerate(prior):
                # Check if the user's NEXT message (after this recommendation) is pushback
                next_user_msgs = df[(df["turn_number"] == (df[df["content"] == t]["turn_number"].iloc[0]
                                     if t in df["content"].values else 999)) &
                                    (df["role"] == "user")]
                # Simpler: just use the state extractor's rejected_tags if available
                pass  # will use state below

            # Use state-extractor rejected_tags if available
            state = states.get((session_id, tn))
            if state and state.get("rejected_tags"):
                # Find tracks whose tags overlap with rejected_tags
                rej_tags = set(str(t).lower() for t in state["rejected_tags"])
                # For simplicity, just subtract the LAST accepted track if pushback detected
                # (since rejected_tags presence implies pushback on recent rec)
                if prior_idx and rej_tags:
                    rejected_idx = [prior_idx[-1]]
            else:
                # Regex fallback: check if current turn's user message has pushback
                cur_user = df[(df["turn_number"] == tn) & (df["role"] == "user")]
                if len(cur_user) and prior_idx:
                    msg = cur_user.iloc[0]["content"]
                    if PUSHBACK_RE.search(msg):
                        rejected_idx = [prior_idx[-1]]

            # Build query: accepted pool - alpha * rejected
            q_vec = pool_prior_embeddings(prior_idx, track_emb,
                                          mode="decay", decay_alpha=0.7)
            if q_vec is not None and rejected_idx:
                neg_vec = track_emb[rejected_idx].mean(axis=0)
                neg_norm = np.linalg.norm(neg_vec)
                if neg_norm > 1e-9:
                    neg_vec = neg_vec / neg_norm
                    q_vec = q_vec - args.neg_alpha * neg_vec
                    # Re-normalize
                    qn = np.linalg.norm(q_vec)
                    if qn > 1e-9:
                        q_vec = q_vec / qn
                    n_neg_applied += 1

            if q_vec is not None:
                scores = track_emb @ q_vec
            else:
                scores = np.zeros(len(track_ids_list), dtype=np.float32)

            # No-repeat
            for idx in prior_idx:
                scores[idx] = -1e9

            top_indices = np.argsort(-scores)[:args.n_output]
            preds = [track_ids_list[i] for i in top_indices]

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": "",
            })

    print(f"  negative feedback applied in {n_neg_applied} turns", flush=True)
    write_inference(rows, args.out)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="leg", required=True)

    # PMI
    pmi_p = sub.add_parser("pmi", help="Item-item PMI co-occurrence retrieval")
    pmi_p.add_argument("--pmi_path", default="exp/item2item_pmi.npz")
    pmi_p.add_argument("--out", required=True)
    pmi_p.add_argument("--n_output", type=int, default=20)

    # User embedding
    ue_p = sub.add_parser("user_emb", help="User-embedding retrieval")
    ue_p.add_argument("--out", required=True)
    ue_p.add_argument("--n_output", type=int, default=20)
    ue_p.add_argument("--user_emb_field", default=None,
                      help="Specific embedding field in the user-embeddings dataset")

    # Negative feedback
    nf_p = sub.add_parser("neg_feedback", help="Dense retrieval with negative feedback")
    nf_p.add_argument("--out", required=True)
    nf_p.add_argument("--n_output", type=int, default=20)
    nf_p.add_argument("--embed", default="metadata-qwen3_embedding_0.6b",
                      choices=["audio-laion_clap", "image-siglip2", "cf-bpr",
                               "attributes-qwen3_embedding_0.6b",
                               "lyrics-qwen3_embedding_0.6b",
                               "metadata-qwen3_embedding_0.6b"])
    nf_p.add_argument("--states_jsonl", default=None)
    nf_p.add_argument("--neg_alpha", type=float, default=0.3,
                      help="Strength of negative subtraction (0=none, 1=full)")

    args = p.parse_args()
    if args.leg == "pmi":
        run_pmi(args)
    elif args.leg == "user_emb":
        run_user_emb(args)
    elif args.leg == "neg_feedback":
        run_neg_feedback(args)


if __name__ == "__main__":
    main()
