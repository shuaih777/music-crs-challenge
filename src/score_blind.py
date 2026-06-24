"""Apply a trained LightGBM model to score Blind-B candidates.

This is the inference-only counterpart to train_ltr.py. It:
  1. Loads retrieval legs for Blind-B (same leg names as devset)
  2. Builds the same feature matrix (must match training feature names)
  3. Loads a pre-trained LightGBM model
  4. Scores all candidates and outputs top-20 per (session, turn)

Usage:
    python src/score_blind.py \
        --model exp/ltr/lgbm_top100_model.txt \
        --legs metadata_qwen3,cf_bpr,pmi_leg,attributes_qwen3,image_siglip2,decay_descending,bm25_only,last_track \
        --inference_dir exp/inference/blind_b \
        --out exp/inference/blind_b/lgbm_scored.json \
        --pmi_path exp/item2item_pmi.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from datasets import load_dataset
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def load_leg(path: str) -> Dict[Tuple[str, int], List[str]]:
    with open(path) as f:
        rows = json.load(f)
    out: Dict[Tuple[str, int], List[str]] = {}
    for r in rows:
        key = (r["session_id"], int(r["turn_number"]))
        out[key] = r["predicted_track_ids"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained LightGBM model .txt")
    p.add_argument("--legs", required=True, help="Comma-separated leg names")
    p.add_argument("--inference_dir", required=True, help="Dir containing leg JSONs")
    p.add_argument("--out", required=True)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--pmi_path", default="exp/item2item_pmi.npz")
    p.add_argument("--split", default="Blind-B", choices=["Blind-A", "Blind-B", "test"])
    args = p.parse_args()

    if lgb is None:
        print("ERROR: lightgbm not installed", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.model}...", flush=True)
    model = lgb.Booster(model_file=args.model)
    expected_features = model.feature_name()
    print(f"  Model expects {len(expected_features)} features: {expected_features[:5]}...", flush=True)

    # Load legs
    leg_names = [l.strip() for l in args.legs.split(",")]
    print(f"Loading {len(leg_names)} legs from {args.inference_dir}...", flush=True)
    legs: Dict[str, Dict[Tuple[str, int], List[str]]] = {}
    for lname in leg_names:
        path = os.path.join(args.inference_dir, f"{lname}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        legs[lname] = load_leg(path)
        print(f"  {lname}: {len(legs[lname])} turns", flush=True)

    # Build candidate pool
    print("Building candidate pool...", flush=True)
    pool: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    for leg_data in legs.values():
        for key, tracks in leg_data.items():
            pool[key].update(tracks[:args.top_k])
    pool_lists = {k: list(v) for k, v in pool.items()}
    total = sum(len(v) for v in pool_lists.values())
    print(f"  {len(pool_lists)} turns, {total} total candidates "
          f"(avg {total/max(len(pool_lists),1):.0f}/turn)", flush=True)

    # Load metadata
    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    track_ids_all = list(track_meta.keys())
    track_to_idx = {tid: i for i, tid in enumerate(track_ids_all)}

    # Load PMI
    pmi_matrix = None
    if args.pmi_path and os.path.exists(args.pmi_path):
        pmi_matrix = sparse.load_npz(args.pmi_path)

    # Load conversations for prior-track features
    print(f"Loading {args.split} conversations...", flush=True)
    if args.split == "test":
        convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        convo_ds = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{args.split}", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    # Pre-compute rank lookups
    leg_ranks: Dict[str, Dict[Tuple[str, int], Dict[str, int]]] = {}
    for lname in legs:
        leg_ranks[lname] = {}
        for key, tracks in legs[lname].items():
            leg_ranks[lname][key] = {tid: r for r, tid in enumerate(tracks[:args.top_k])}

    # Build features and score
    print("Building features + scoring...", flush=True)
    output_rows = []
    sorted_leg_names = sorted(legs.keys())
    n_legs = len(sorted_leg_names)

    for key in tqdm(sorted(pool_lists.keys()), desc="scoring"):
        candidates = pool_lists[key]
        session_id, turn = key

        conv = conversations.get(session_id, {})
        prior_tracks = []
        if conv:
            for c in conv.get("conversations", []):
                if c["role"] == "music" and c["turn_number"] < turn:
                    prior_tracks.append(c["content"])
        n_prior = len(prior_tracks)
        prior_idx = [track_to_idx[t] for t in prior_tracks if t in track_to_idx]

        features_batch = []
        for tid in candidates:
            feats = []
            # Per-leg features (must match training order!)
            for lname in sorted_leg_names:
                rank_dict = leg_ranks.get(lname, {}).get(key, {})
                rank = rank_dict.get(tid, -1)
                feats.append(rank if rank >= 0 else args.top_k + 1)
                feats.append(1.0 / (60 + rank + 1) if rank >= 0 else 0.0)

            # Track metadata
            meta = track_meta.get(tid, {})
            feats.append(float(meta.get("popularity", 0) or 0))

            # Turn/session features
            feats.append(float(turn))
            feats.append(float(n_prior))

            # PMI sum
            if pmi_matrix is not None and prior_idx and tid in track_to_idx:
                tidx = track_to_idx[tid]
                pmi_scores = pmi_matrix[prior_idx, tidx].toarray().flatten()
                feats.append(float(pmi_scores.sum()))
            else:
                feats.append(0.0)

            # Is in all legs / n_legs_containing
            in_all = all(tid in leg_ranks.get(ln, {}).get(key, {}) for ln in sorted_leg_names)
            feats.append(1.0 if in_all else 0.0)
            n_in = sum(1 for ln in sorted_leg_names if tid in leg_ranks.get(ln, {}).get(key, {}))
            feats.append(float(n_in))

            features_batch.append(feats)

        # Score with model
        X = np.array(features_batch, dtype=np.float32)

        # Handle feature count mismatch gracefully
        n_expected = len(expected_features)
        if X.shape[1] < n_expected:
            # Pad with zeros (missing legs)
            X = np.pad(X, ((0, 0), (0, n_expected - X.shape[1])))
        elif X.shape[1] > n_expected:
            X = X[:, :n_expected]

        scores = model.predict(X)

        # Top-20
        order = np.argsort(-scores)[:20]
        top20 = [candidates[i] for i in order]

        # Get user_id
        user_id = conv.get("user_id", "") if conv else ""

        output_rows.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": int(turn),
            "predicted_track_ids": top20,
            "predicted_response": "",
        })

    # Write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"Wrote {len(output_rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
