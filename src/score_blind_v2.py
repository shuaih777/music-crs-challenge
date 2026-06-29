"""Score a held-out split (e.g. Blind-B) with a pre-trained lgbm_v2 model.

Reuses the exact feature-building pipeline from train_ltr_v2 (including the
bi-encoder cosine feature) so the features match what the model was trained on,
then loads the saved LightGBM model and predicts — no CV, no ground truth.

Usage:
    PYTHONPATH=src python src/score_blind_v2.py \
        --legs "metadata_qwen3,cf_bpr,...,biencoder_mxbai_top100" \
        --biencoder_dir out/biencoder_large \
        --inference_dir exp/inference/blind_b_best \
        --model exp/ltr/lgbm_abl_all_plus_mxbai.txt \
        --out exp/inference/blind_b_best/lgbm_v2_scored.json \
        --split Blind-B --pmi_path exp/item2item_pmi.npz --top_k 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from datasets import load_dataset
from scipy import sparse

sys.path.insert(0, "src")
from train_ltr_v2 import (
    load_leg, build_candidate_pool, load_biencoder_scores, build_features,
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legs", required=True)
    p.add_argument("--biencoder_dir", default="out/biencoder_large")
    p.add_argument("--inference_dir", default="exp/inference/blind_b_best")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="Blind-B")
    p.add_argument("--pmi_path", default="exp/item2item_pmi.npz")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    if lgb is None:
        print("ERROR: lightgbm not installed", file=sys.stderr)
        sys.exit(1)

    leg_names = [l.strip() for l in args.legs.split(",")]
    print(f"Loading {len(leg_names)} legs from {args.inference_dir}...", flush=True)
    legs = {}
    for lname in leg_names:
        path = os.path.join(args.inference_dir, f"{lname}.json")
        if os.path.exists(path):
            legs[lname] = load_leg(path)
            print(f"  {lname}: {len(legs[lname])} turns")
        else:
            print(f"  MISSING: {lname} ({path})", file=sys.stderr)
            sys.exit(1)

    print("Building candidate pool...", flush=True)
    pool = build_candidate_pool(legs, args.top_k)
    total = sum(len(v) for v in pool.values())
    print(f"  {len(pool)} turns, {total} candidates (avg {total/len(pool):.0f}/turn)")

    # Track metadata
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    track_ids_all = list(track_meta.keys())
    track_to_idx = {tid: i for i, tid in enumerate(track_ids_all)}

    # PMI
    pmi_matrix = None
    if args.pmi_path and os.path.exists(args.pmi_path):
        pmi_matrix = sparse.load_npz(args.pmi_path)

    # Conversations for the requested split
    if args.split == "test":
        convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        convo_ds = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{args.split}", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    # Bi-encoder cosine scores (same model as training)
    biencoder_scores = {}
    if args.biencoder_dir and os.path.exists(args.biencoder_dir):
        biencoder_scores = load_biencoder_scores(
            args.biencoder_dir, track_ids_all, conversations,
            pool, track_to_idx, args.batch_size,
        )
    else:
        print("WARNING: bi-encoder dir not found, skipping cosine features")

    # Features (identical pipeline → identical feature order)
    X, keys, candidates_per_key, feature_names = build_features(
        pool, legs, track_meta, pmi_matrix, track_to_idx,
        conversations, biencoder_scores, args.top_k,
    )
    print(f"Feature matrix: {X.shape}, features: {feature_names}")

    # Load model and predict
    print(f"Loading model {args.model}...", flush=True)
    model = lgb.Booster(model_file=args.model)
    if model.num_feature() != X.shape[1]:
        print(f"ERROR: model expects {model.num_feature()} features, "
              f"got {X.shape[1]}", file=sys.stderr)
        sys.exit(1)
    preds = model.predict(X)

    # Rerank → top-20 per turn
    rows_out = []
    row_idx = 0
    for key, cands in zip(keys, candidates_per_key):
        n = len(cands)
        scores = preds[row_idx:row_idx + n]
        order = np.argsort(-scores)
        reranked = [cands[i] for i in order[:20]]
        rows_out.append({
            "session_id": key[0],
            "user_id": conversations.get(key[0], {}).get("user_id", ""),
            "turn_number": int(key[1]),
            "predicted_track_ids": reranked,
            "predicted_response": "",
        })
        row_idx += n

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False)
    print(f"Wrote {len(rows_out)} predictions to {args.out}")


if __name__ == "__main__":
    main()
