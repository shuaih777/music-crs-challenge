"""LightGBM LambdaMART reranker over multi-leg candidate pool.

Workflow:
  1. Load existing inference JSONs from multiple retrieval legs
  2. Build union(top-K) candidate pool per (session, turn)
  3. Construct per-candidate feature matrix
  4. Train LightGBM with lambdarank objective on train sessions
  5. Score devset candidates and output reranked inference JSON

Features per candidate:
  - rank_in_leg_X (one per leg) — 0 if not in that leg's top-K
  - score_in_leg_X (raw RRF-style 1/(k+rank))
  - track_popularity (from metadata)
  - turn_number
  - n_prior_accepts (how many tracks accepted before this turn)
  - pmi_sum (sum of PMI with prior accepted tracks)
  - is_in_all_legs (binary: appears in every leg's top-K)

Usage:
    # Train on devset with cross-validation (no separate train split needed
    # since we're reranking existing retrieval outputs, not learning retrieval)
    python src/train_ltr.py \
        --legs metadata_qwen3,cf_bpr,pmi_leg \
        --inference_dir exp/inference/devset \
        --ground_truth exp/ground_truth/devset.json \
        --top_k 100 \
        --out_model exp/ltr/lgbm_model.txt \
        --out_inference exp/inference/devset/lgbm_reranked.json

    # Or train on train split, apply to devset (more rigorous)
    python src/train_ltr.py \
        --legs metadata_qwen3,cf_bpr,pmi_leg \
        --train_mode cv \
        --n_folds 5 \
        ...

CPU only. ~5-10 min for 8000 turns × 100 candidates × 5 folds.
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


# ============================================================================
# Feature construction
# ============================================================================


def load_leg(path: str) -> Dict[Tuple[str, int], List[str]]:
    """Load an inference JSON into {(session_id, turn): [track_id, ...]}."""
    with open(path) as f:
        rows = json.load(f)
    out: Dict[Tuple[str, int], List[str]] = {}
    for r in rows:
        key = (r["session_id"], int(r["turn_number"]))
        out[key] = r["predicted_track_ids"]
    return out


def build_candidate_pool(
    legs: Dict[str, Dict[Tuple[str, int], List[str]]],
    top_k: int = 100,
) -> Dict[Tuple[str, int], List[str]]:
    """Union of top-K from each leg, per (session, turn)."""
    pool: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    for leg_name, leg_data in legs.items():
        for key, tracks in leg_data.items():
            pool[key].update(tracks[:top_k])
    return {k: list(v) for k, v in pool.items()}


def build_features(
    pool: Dict[Tuple[str, int], List[str]],
    legs: Dict[str, Dict[Tuple[str, int], List[str]]],
    track_meta: Dict[str, dict],
    pmi_matrix: sparse.csr_matrix | None,
    track_to_idx: Dict[str, int],
    conversations: Dict[str, dict],  # session_id -> conversation data
    top_k: int = 100,
) -> Tuple[np.ndarray, List[Tuple[str, int]], List[List[str]]]:
    """Build feature matrix for all candidates.

    Returns:
        features: (N_total_candidates, n_features) float32
        keys: list of (session_id, turn) for group assignment
        candidates_per_key: list of candidate track_id lists (same order)
    """
    leg_names = sorted(legs.keys())
    n_legs = len(leg_names)

    # Pre-compute rank lookup per leg per key
    leg_ranks: Dict[str, Dict[Tuple[str, int], Dict[str, int]]] = {}
    for lname in leg_names:
        leg_ranks[lname] = {}
        for key, tracks in legs[lname].items():
            leg_ranks[lname][key] = {tid: r for r, tid in enumerate(tracks[:top_k])}

    all_features = []
    all_keys = []
    all_candidates = []

    for key in tqdm(sorted(pool.keys()), desc="building features"):
        candidates = pool[key]
        session_id, turn = key

        # Get prior accepted tracks for this session/turn
        conv = conversations.get(session_id, {})
        prior_tracks = []
        if conv:
            for c in conv.get("conversations", []):
                if c["role"] == "music" and c["turn_number"] < turn:
                    prior_tracks.append(c["content"])
        n_prior = len(prior_tracks)
        prior_idx = [track_to_idx[t] for t in prior_tracks if t in track_to_idx]

        for tid in candidates:
            feats = []

            # Per-leg rank features
            ranks_this_candidate = []
            scores_this_candidate = []
            for lname in leg_names:
                rank_dict = leg_ranks[lname].get(key, {})
                rank = rank_dict.get(tid, -1)
                feats.append(rank if rank >= 0 else top_k + 1)  # rank (lower = better)
                score_val = 1.0 / (60 + rank + 1) if rank >= 0 else 0.0
                feats.append(score_val)  # RRF-style score
                if rank >= 0:
                    ranks_this_candidate.append(rank)
                    scores_this_candidate.append(score_val)

            # Track metadata features
            meta = track_meta.get(tid, {})
            feats.append(float(meta.get("popularity", 0) or 0))

            # Turn / session features
            feats.append(float(turn))
            feats.append(float(n_prior))

            # PMI sum with prior accepted
            if pmi_matrix is not None and prior_idx and tid in track_to_idx:
                tidx = track_to_idx[tid]
                pmi_scores = pmi_matrix[prior_idx, tidx].toarray().flatten()
                feats.append(float(pmi_scores.sum()))
            else:
                feats.append(0.0)

            # Is in all legs?
            in_all = all(
                tid in leg_ranks[ln].get(key, {}) for ln in leg_names
            )
            feats.append(1.0 if in_all else 0.0)

            # Number of legs containing this candidate
            n_legs_in = sum(
                1 for ln in leg_names if tid in leg_ranks[ln].get(key, {})
            )
            feats.append(float(n_legs_in))

            # --- New features ---

            # rank_std: standard deviation of ranks across legs (disagreement)
            if ranks_this_candidate:
                feats.append(float(np.std(ranks_this_candidate)))
            else:
                feats.append(0.0)

            # best_rank: minimum rank across legs (best single-leg position)
            if ranks_this_candidate:
                feats.append(float(min(ranks_this_candidate)))
            else:
                feats.append(float(top_k + 1))

            # worst_rank: maximum rank across legs
            if ranks_this_candidate:
                feats.append(float(max(ranks_this_candidate)))
            else:
                feats.append(float(top_k + 1))

            # reciprocal_rank_sum: sum of 1/(60+rank) for all containing legs
            rr_sum = sum(1.0 / (60 + r) for r in ranks_this_candidate)
            feats.append(float(rr_sum))

            # score_ratio: max_score / second_max_score (dominance of best leg)
            if len(scores_this_candidate) >= 2:
                sorted_scores = sorted(scores_this_candidate, reverse=True)
                feats.append(sorted_scores[0] / max(sorted_scores[1], 1e-9))
            elif len(scores_this_candidate) == 1:
                feats.append(1.0)
            else:
                feats.append(0.0)

            # n_legs_ratio: n_legs_containing / total_n_legs
            feats.append(float(n_legs_in) / max(n_legs, 1))

            # cf_boosted_score: score_cf_bpr * 2.5
            # (TalkPlay paper finding: CF is 2.5x more important)
            cf_score = 0.0
            for lname in leg_names:
                if "cf" in lname.lower():
                    rank_dict = leg_ranks[lname].get(key, {})
                    rank = rank_dict.get(tid, -1)
                    if rank >= 0:
                        cf_score = 1.0 / (60 + rank + 1)
                    break
            feats.append(cf_score * 2.5)

            all_features.append(feats)

        all_keys.append(key)
        all_candidates.append(candidates)

    feature_names = []
    for lname in leg_names:
        feature_names.append(f"rank_{lname}")
        feature_names.append(f"score_{lname}")
    feature_names.extend([
        "popularity", "turn_number", "n_prior_accepts",
        "pmi_sum", "is_in_all_legs", "n_legs_containing",
        "rank_std", "best_rank", "worst_rank",
        "reciprocal_rank_sum", "score_ratio", "n_legs_ratio",
        "cf_boosted_score",
    ])

    X = np.array(all_features, dtype=np.float32)
    return X, all_keys, all_candidates, feature_names


# ============================================================================
# Training
# ============================================================================


def train_and_predict(
    X: np.ndarray,
    labels: np.ndarray,
    groups: List[int],
    feature_names: List[str],
    n_folds: int = 5,
    params: dict | None = None,
) -> np.ndarray:
    """Train LightGBM LambdaRank with grouped CV, return OOF predictions."""
    if lgb is None:
        print("ERROR: lightgbm not installed. pip install lightgbm", file=sys.stderr)
        sys.exit(1)

    params = params or {
        "objective": "lambdarank",
        "metric": "ndcg",
        "lambdarank_truncation_level": 20,
        "num_leaves": 31,
        "learning_rate": 0.02,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }
    # num_boost_round=500, early_stopping=50

    # Build group boundaries
    group_boundaries = np.cumsum([0] + groups)
    n_groups = len(groups)

    # K-fold by groups (sessions)
    fold_size = n_groups // n_folds
    oof_preds = np.zeros(len(X), dtype=np.float32)
    models = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_groups

        val_row_start = int(group_boundaries[val_start])
        val_row_end = int(group_boundaries[val_end])
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[val_row_start:val_row_end] = False

        X_train = X[train_mask]
        y_train = labels[train_mask]
        X_val = X[~train_mask]
        y_val = labels[~train_mask]

        # Rebuild group sizes for train/val
        train_groups = []
        val_groups = []
        for i, g in enumerate(groups):
            if i < val_start or i >= val_end:
                train_groups.append(g)
            else:
                val_groups.append(g)

        train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups,
                               feature_name=feature_names)
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups,
                             feature_name=feature_names, reference=train_ds)

        model = lgb.train(
            params,
            train_ds,
            num_boost_round=500,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(50, verbose=True),
                       lgb.log_evaluation(50)],
        )
        models.append(model)

        preds = model.predict(X_val)
        oof_preds[val_row_start:val_row_end] = preds
        print(f"  fold {fold+1}/{n_folds}: best_iter={model.best_iteration}", flush=True)

    return oof_preds, models


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legs", required=True,
                   help="Comma-separated leg names (must have inference JSONs)")
    p.add_argument("--inference_dir", default="exp/inference/devset")
    p.add_argument("--ground_truth", default="exp/ground_truth/devset.json")
    p.add_argument("--top_k", type=int, default=100,
                   help="Take top-K from each leg for the candidate pool")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--out_model", default="exp/ltr/lgbm_model.txt")
    p.add_argument("--out_inference", default="exp/inference/devset/lgbm_reranked.json")
    p.add_argument("--pmi_path", default="exp/item2item_pmi.npz",
                   help="PMI matrix for PMI feature (set to '' to disable)")
    args = p.parse_args()

    if lgb is None:
        print("ERROR: lightgbm not installed.\n"
              "  pip install lightgbm\n"
              "CPU only, no GPU needed.", file=sys.stderr)
        sys.exit(1)

    leg_names = [l.strip() for l in args.legs.split(",")]
    print(f"Loading {len(leg_names)} legs: {leg_names}", flush=True)
    legs: Dict[str, Dict[Tuple[str, int], List[str]]] = {}
    for lname in leg_names:
        path = os.path.join(args.inference_dir, f"{lname}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        legs[lname] = load_leg(path)
        print(f"  {lname}: {len(legs[lname])} turns loaded", flush=True)

    print("Building candidate pool...", flush=True)
    pool = build_candidate_pool(legs, args.top_k)
    total_cands = sum(len(v) for v in pool.values())
    print(f"  {len(pool)} turns, {total_cands} total candidates "
          f"(avg {total_cands/len(pool):.0f}/turn)", flush=True)

    # Load track metadata
    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    track_ids_all = list(track_meta.keys())
    track_to_idx = {tid: i for i, tid in enumerate(track_ids_all)}

    # Load PMI
    pmi_matrix = None
    if args.pmi_path and os.path.exists(args.pmi_path):
        print(f"Loading PMI matrix from {args.pmi_path}...", flush=True)
        pmi_matrix = sparse.load_npz(args.pmi_path)

    # Load conversations for prior-track features
    print("Loading conversations...", flush=True)
    convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    # Build features
    X, keys, candidates_per_key, feature_names = build_features(
        pool, legs, track_meta, pmi_matrix, track_to_idx,
        conversations, args.top_k,
    )
    print(f"Feature matrix: {X.shape}, features: {feature_names}", flush=True)

    # Load ground truth and build labels
    print("Building labels...", flush=True)
    gt = json.load(open(args.ground_truth))
    gold_by_key = {(g["session_id"], int(g["turn_number"])): g["ground_truth_track_id"]
                   for g in gt}

    labels = np.zeros(X.shape[0], dtype=np.float32)
    groups = []
    row_idx = 0
    for key, cands in zip(keys, candidates_per_key):
        gold = gold_by_key.get(key)
        n = len(cands)
        for i, tid in enumerate(cands):
            if tid == gold:
                labels[row_idx + i] = 1.0
        groups.append(n)
        row_idx += n

    n_positive = int(labels.sum())
    print(f"  positives (gold in pool): {n_positive}/{len(keys)} "
          f"({n_positive/len(keys)*100:.1f}%)", flush=True)

    # Train
    print(f"\nTraining LightGBM LambdaRank ({args.n_folds}-fold CV)...", flush=True)
    oof_preds, models = train_and_predict(
        X, labels, groups, feature_names, args.n_folds,
    )

    # Rerank and produce inference JSON
    print("\nReranking candidates...", flush=True)
    rows_out = []
    row_idx = 0
    for key, cands in zip(keys, candidates_per_key):
        n = len(cands)
        scores = oof_preds[row_idx:row_idx + n]
        order = np.argsort(-scores)
        reranked = [cands[i] for i in order[:20]]
        rows_out.append({
            "session_id": key[0],
            "user_id": "",  # will be filled from conversations if needed
            "turn_number": int(key[1]),
            "predicted_track_ids": reranked,
            "predicted_response": "",
        })
        row_idx += n

    # Fill user_id from conversations
    for r in rows_out:
        conv = conversations.get(r["session_id"], {})
        r["user_id"] = conv.get("user_id", "")

    os.makedirs(os.path.dirname(args.out_inference) or ".", exist_ok=True)
    with open(args.out_inference, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False)
    print(f"Wrote {len(rows_out)} predictions to {args.out_inference}")

    # Save model
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    models[-1].save_model(args.out_model)
    print(f"Saved last-fold model to {args.out_model}")

    # Feature importance
    print("\nFeature importance (gain):")
    imp = models[-1].feature_importance(importance_type="gain")
    for name, val in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
        print(f"  {name:25s}: {val:.0f}")


if __name__ == "__main__":
    main()
