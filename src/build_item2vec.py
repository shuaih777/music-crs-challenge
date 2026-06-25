"""Item2Vec retrieval leg: Word2Vec trained on session track sequences.

Trains a gensim Word2Vec (Skip-gram) model on ordered track sequences from
the 15k training sessions, then uses it for retrieval on the devset by
averaging prior accepted-track vectors and finding nearest neighbors.

Usage:
    # Train model + run inference:
    PYTHONPATH=src python src/build_item2vec.py \
        --out exp/inference/devset/item2vec_top100.json \
        --model_path exp/item2vec_model.bin \
        --n_output 100

CPU only, ~2-5 min training + 1-2 min inference.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Shared utilities (consistent with retrieval_legs.py)
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
# Extract track sequences from training sessions
# ============================================================================

def extract_sequences(split: str = "train") -> List[List[str]]:
    """Extract ordered track sequences per session from the given split.

    Each sequence is the list of track_ids recommended in the session,
    ordered by turn_number (role="music").
    """
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=split)
    sequences: List[List[str]] = []
    for ex in tqdm(ds, desc=f"extracting sequences ({split})"):
        convos = ex["conversations"]
        # Sort by turn_number, filter music role
        music_turns = [
            c for c in convos if c["role"] == "music"
        ]
        music_turns.sort(key=lambda c: c["turn_number"])
        seq = [c["content"] for c in music_turns]
        if len(seq) >= 2:  # need at least 2 tracks for skip-gram
            sequences.append(seq)
    print(f"  extracted {len(sequences)} sequences "
          f"(avg length {np.mean([len(s) for s in sequences]):.1f})")
    return sequences


# ============================================================================
# Train Word2Vec (Item2Vec)
# ============================================================================

def train_item2vec(
    sequences: List[List[str]],
    model_path: str,
    dim: int = 128,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 20,
):
    """Train gensim Word2Vec on track sequences (Item2Vec approach)."""
    from gensim.models import Word2Vec

    print(f"Training Item2Vec: dim={dim}, window={window}, sg=1, "
          f"epochs={epochs}, sequences={len(sequences)}", flush=True)

    model = Word2Vec(
        sentences=sequences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,  # Skip-gram
        workers=8,
        epochs=epochs,
        seed=42,
    )
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    print(f"  vocab size: {len(model.wv)}")
    print(f"  saved model to {model_path}")
    return model


# ============================================================================
# Inference: average prior track vectors -> nearest neighbors
# ============================================================================

def run_inference(
    model_path: str,
    out_path: str,
    n_output: int = 100,
    split: str = "test",
) -> None:
    """Run Item2Vec retrieval on the test split."""
    from gensim.models import Word2Vec

    print(f"Loading Item2Vec model from {model_path}...", flush=True)
    model = Word2Vec.load(model_path)
    wv = model.wv
    print(f"  vocab={len(wv)}, dim={wv.vector_size}", flush=True)

    track_ids, track_to_idx = load_catalog()

    # Build a matrix of all catalog tracks that have embeddings
    # For tracks not in the model vocab, we'll use zero vectors
    n_tracks = len(track_ids)
    dim = wv.vector_size
    track_vecs = np.zeros((n_tracks, dim), dtype=np.float32)
    n_in_vocab = 0
    for i, tid in enumerate(track_ids):
        if tid in wv:
            track_vecs[i] = wv[tid]
            n_in_vocab += 1

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(track_vecs, axis=1, keepdims=True)
    track_vecs_normed = track_vecs / np.clip(norms, 1e-9, None)
    print(f"  {n_in_vocab}/{n_tracks} tracks have Item2Vec embeddings", flush=True)

    # Popularity fallback for cold-start (turn 1)
    # Use train-set track frequency as popularity proxy
    train_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split="train"
    )
    freq: Counter = Counter()
    for ex in train_ds:
        for c in ex["conversations"]:
            if c["role"] == "music":
                freq[c["content"]] += 1
    pop_scores = np.zeros(n_tracks, dtype=np.float32)
    for i, tid in enumerate(track_ids):
        pop_scores[i] = freq.get(tid, 0)

    # Load test set
    test_ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset", split=split
    )
    rows: List[dict] = []

    print(f"Running Item2Vec retrieval on {len(test_ds)} sessions...", flush=True)
    for ex in tqdm(test_ds, desc="item2vec"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]

        for tn in range(1, 9):
            # Get prior accepted tracks before this turn
            prior = [
                c["content"]
                for c in ex["conversations"]
                if c["role"] == "music" and c["turn_number"] < tn
            ]

            # Get indices and filter to those with embeddings
            prior_idx = [track_to_idx[t] for t in prior if t in track_to_idx]
            prior_with_emb = [idx for idx in prior_idx if norms[idx, 0] > 1e-9]

            if prior_with_emb:
                # Average prior track vectors (normalized)
                q = track_vecs[prior_with_emb].mean(axis=0)
                qn = np.linalg.norm(q)
                if qn > 1e-9:
                    q = q / qn
                scores = track_vecs_normed @ q
            else:
                # Cold-start: use popularity
                scores = pop_scores.copy()

            # No-repeat: zero out already-accepted tracks
            for idx in prior_idx:
                scores[idx] = -1e9

            top_indices = np.argsort(-scores)[:n_output]
            preds = [track_ids[i] for i in top_indices]

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": "",
            })

    write_inference(rows, out_path)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Item2Vec: train Word2Vec on session sequences + retrieve"
    )
    p.add_argument("--out", default="exp/inference/devset/item2vec_top100.json",
                   help="Output inference JSON path")
    p.add_argument("--model_path", default="exp/item2vec_model.bin",
                   help="Path to save/load the Word2Vec model")
    p.add_argument("--n_output", type=int, default=100,
                   help="Number of tracks to retrieve per turn")
    p.add_argument("--dim", type=int, default=128,
                   help="Word2Vec embedding dimension")
    p.add_argument("--window", type=int, default=5,
                   help="Word2Vec window size")
    p.add_argument("--epochs", type=int, default=20,
                   help="Word2Vec training epochs")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training, only run inference (model must exist)")
    p.add_argument("--split", default="test",
                   help="Dataset split for inference")
    args = p.parse_args()

    if not args.skip_train:
        print("=" * 60)
        print("Phase 1: Training Item2Vec on train sessions")
        print("=" * 60)
        sequences = extract_sequences("train")
        train_item2vec(
            sequences,
            model_path=args.model_path,
            dim=args.dim,
            window=args.window,
            epochs=args.epochs,
        )
    else:
        if not os.path.exists(args.model_path):
            print(f"ERROR: --skip_train but model not found at {args.model_path}",
                  flush=True)
            return

    print()
    print("=" * 60)
    print("Phase 2: Running Item2Vec retrieval")
    print("=" * 60)
    run_inference(
        model_path=args.model_path,
        out_path=args.out,
        n_output=args.n_output,
        split=args.split,
    )


if __name__ == "__main__":
    main()
