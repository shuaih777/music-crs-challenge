"""LightGBM LambdaMART v2: with bi-encoder cosine scores as continuous features.

Difference from train_ltr.py:
  - Loads bi-encoder model directly and computes cosine scores on-the-fly
    for each candidate (not just rank position)
  - Adds raw cosine similarity as a continuous feature (much richer signal
    than discrete rank, which caused 5-tree early stopping in v1)
  - Also supports loading pre-computed score files if available

Usage:
    python src/train_ltr_v2.py \
        --legs metadata_qwen3,cf_bpr,pmi_leg,... \
        --biencoder_dir out/biencoder_large \
        --inference_dir exp/inference/devset \
        --ground_truth exp/ground_truth/devset.json \
        --top_k 100 --n_folds 5 \
        --out_model exp/ltr/lgbm_v2.txt \
        --out_inference exp/inference/devset/lgbm_v2.json
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
# Bi-encoder score computation
# ============================================================================

def load_biencoder_scores(
    biencoder_dir: str,
    track_ids: List[str],
    conversations: Dict[str, dict],
    pool: Dict[Tuple[str, int], List[str]],
    track_to_idx: Dict[str, int],
    batch_size: int = 256,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Compute bi-encoder cosine scores for all candidates in the pool.

    Returns: {(session_id, turn): {track_id: cosine_score}}
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("WARNING: sentence-transformers not installed, skipping bi-encoder scores",
              file=sys.stderr)
        return {}

    # Import build functions
    sys.path.insert(0, 'src')
    from train_biencoder import build_conversation_text, build_track_text

    print(f"Loading bi-encoder from {biencoder_dir}...", flush=True)
    model = SentenceTransformer(biencoder_dir, trust_remote_code=True)

    # Load track embeddings (or compute them)
    track_emb_path = os.path.join(biencoder_dir, "track_embeddings.npy")
    track_ids_path = os.path.join(biencoder_dir, "track_embeddings_ids.json")

    if os.path.exists(track_emb_path) and os.path.exists(track_ids_path):
        print("  Loading cached track embeddings...", flush=True)
        track_embs = np.load(track_emb_path)
        cached_ids = json.load(open(track_ids_path))
        assert cached_ids == track_ids[:len(cached_ids)], "Track ID mismatch"
    else:
        print("  Encoding tracks (this may take a few minutes)...", flush=True)
        tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
        track_texts = [build_track_text(r) for r in tracks_ds]
        track_embs = model.encode(track_texts, batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=True,
                                   convert_to_numpy=True).astype(np.float32)
        np.save(track_emb_path, track_embs)
        with open(track_ids_path, "w") as f:
            json.dump(track_ids, f)

    # Encode all unique conversations
    keys = sorted(pool.keys())
    conv_texts = []
    for key in keys:
        session_id, turn = key
        conv = conversations.get(session_id, {})
        convos = conv.get("conversations", [])
        conv_texts.append(build_conversation_text(convos, turn))

    print(f"  Encoding {len(conv_texts)} conversations...", flush=True)
    conv_embs = model.encode(conv_texts, batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True,
                              convert_to_numpy=True).astype(np.float32)

    # Compute scores for each (session, turn) → {track_id: score}
    print("  Computing per-candidate cosine scores...", flush=True)
    scores_map: Dict[Tuple[str, int], Dict[str, float]] = {}
    for i, key in enumerate(tqdm(keys, desc="scoring")):
        candidates = pool[key]
        cand_idx = [track_to_idx[t] for t in candidates if t in track_to_idx]
        if not cand_idx:
            scores_map[key] = {}
            continue
        cand_embs = track_embs[cand_idx]
        cos_scores = cand_embs @ conv_embs[i]
        scores_map[key] = {candidates[j]: float(cos_scores[j])
                           for j in range(len(cand_idx))}

    return scores_map


# ============================================================================
# Feature construction (extends train_ltr.py)
# ============================================================================

def load_leg(path: str) -> Dict[Tuple[str, int], List[str]]:
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
    conversations: Dict[str, dict],
    biencoder_scores: Dict[Tuple[str, int], Dict[str, float]],
    top_k: int = 100,
) -> Tuple[np.ndarray, List[Tuple[str, int]], List[List[str]], List[str]]:
    """Build feature matrix with bi-encoder cosine scores."""
    leg_names = sorted(legs.keys())
    n_legs = len(leg_names)

    # Pre-compute rank lookup per leg
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

        conv = conversations.get(session_id, {})
        prior_tracks = []
        if conv:
            for c in conv.get("conversations", []):
                if c["role"] == "music" and c["turn_number"] < turn:
                    prior_tracks.append(c["content"])
        n_prior = len(prior_tracks)
        prior_idx = [track_to_idx[t] for t in prior_tracks if t in track_to_idx]

        # Bi-encoder scores for this turn
        bienc_scores = biencoder_scores.get(key, {})

        for tid in candidates:
            feats = []

            # Per-leg rank + RRF score features
            for lname in leg_names:
                rank_dict = leg_ranks[lname].get(key, {})
                rank = rank_dict.get(tid, -1)
                feats.append(rank if rank >= 0 else top_k + 1)
                feats.append(1.0 / (60 + rank + 1) if rank >= 0 else 0.0)

            # Track metadata
            meta = track_meta.get(tid, {})
            feats.append(float(meta.get("popularity", 0) or 0))

            # Turn / session features
            feats.append(float(turn))
            feats.append(float(n_prior))

            # PMI sum
            if pmi_matrix is not None and prior_idx and tid in track_to_idx:
                tidx = track_to_idx[tid]
                pmi_scores = pmi_matrix[prior_idx, tidx].toarray().flatten()
                feats.append(float(pmi_scores.sum()))
            else:
                feats.append(0.0)

            # Leg overlap features
            in_all = all(tid in leg_ranks[ln].get(key, {}) for ln in leg_names)
            feats.append(1.0 if in_all else 0.0)
            n_legs_in = sum(1 for ln in leg_names if tid in leg_ranks[ln].get(key, {}))
            feats.append(float(n_legs_in))

            # *** NEW: Bi-encoder cosine score (continuous, 0-1) ***
            feats.append(float(bienc_scores.get(tid, 0.0)))

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
        "biencoder_cosine",
    ])

    X = np.array(all_features, dtype=np.float32)
    return X, all_keys, all_candidates, feature_names


# ============================================================================
# Training (same as train_ltr.py)
# ============================================================================

def train_and_predict(X, labels, groups, feature_names, n_folds=5, params=None):
    if lgb is None:
        print("ERROR: lightgbm not installed", file=sys.stderr)
        sys.exit(1)

    params = params or {
        "objective": "lambdarank",
        "metric": "ndcg",
        "lambdarank_truncation_level": 20,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }

    group_boundaries = np.cumsum([0] + groups)
    n_groups = len(groups)
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

        X_train, y_train = X[train_mask], labels[train_mask]
        X_val, y_val = X[~train_mask], labels[~train_mask]

        train_groups = [groups[i] for i in range(n_groups) if i < val_start or i >= val_end]
        val_groups = [groups[i] for i in range(val_start, val_end)]

        train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_names)
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=feature_names, reference=train_ds)

        model = lgb.train(params, train_ds, num_boost_round=500, valid_sets=[val_ds],
                          callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)])
        models.append(model)

        preds = model.predict(X_val)
        oof_preds[val_row_start:val_row_end] = preds
        print(f"  fold {fold+1}/{n_folds}: best_iter={model.best_iteration}", flush=True)

    return oof_preds, models


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--legs", required=True)
    p.add_argument("--biencoder_dir", default="out/biencoder_large",
                   help="Path to trained bi-encoder for computing cosine scores")
    p.add_argument("--inference_dir", default="exp/inference/devset")
    p.add_argument("--ground_truth", default="exp/ground_truth/devset.json")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--out_model", default="exp/ltr/lgbm_v2.txt")
    p.add_argument("--out_inference", default="exp/inference/devset/lgbm_v2.json")
    p.add_argument("--pmi_path", default="exp/item2item_pmi.npz")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    if lgb is None:
        print("ERROR: lightgbm not installed", file=sys.stderr)
        sys.exit(1)

    leg_names = [l.strip() for l in args.legs.split(",")]
    print(f"Loading {len(leg_names)} legs...", flush=True)
    legs = {}
    for lname in leg_names:
        path = os.path.join(args.inference_dir, f"{lname}.json")
        if os.path.exists(path):
            legs[lname] = load_leg(path)
            print(f"  {lname}: {len(legs[lname])} turns")

    print("Building candidate pool...", flush=True)
    pool = build_candidate_pool(legs, args.top_k)
    total_cands = sum(len(v) for v in pool.values())
    print(f"  {len(pool)} turns, {total_cands} candidates (avg {total_cands/len(pool):.0f}/turn)")

    # Load metadata
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    track_ids_all = list(track_meta.keys())
    track_to_idx = {tid: i for i, tid in enumerate(track_ids_all)}

    # PMI
    pmi_matrix = None
    if args.pmi_path and os.path.exists(args.pmi_path):
        pmi_matrix = sparse.load_npz(args.pmi_path)

    # Conversations
    convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    # Bi-encoder cosine scores
    biencoder_scores = {}
    if args.biencoder_dir and os.path.exists(args.biencoder_dir):
        biencoder_scores = load_biencoder_scores(
            args.biencoder_dir, track_ids_all, conversations,
            pool, track_to_idx, args.batch_size,
        )
    else:
        print("WARNING: bi-encoder dir not found, skipping cosine features")

    # Build features
    X, keys, candidates_per_key, feature_names = build_features(
        pool, legs, track_meta, pmi_matrix, track_to_idx,
        conversations, biencoder_scores, args.top_k,
    )
    print(f"Feature matrix: {X.shape}, features: {feature_names}")

    # Labels
    gt = json.load(open(args.ground_truth))
    gold_by_key = {(g["session_id"], int(g["turn_number"])): g["ground_truth_track_id"] for g in gt}

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

    n_pos = int(labels.sum())
    print(f"  positives: {n_pos}/{len(keys)} ({n_pos/len(keys)*100:.1f}%)")

    # Train
    print(f"\nTraining LightGBM v2 ({args.n_folds}-fold CV)...", flush=True)
    oof_preds, models = train_and_predict(X, labels, groups, feature_names, args.n_folds)

    # Rerank
    rows_out = []
    row_idx = 0
    for key, cands in zip(keys, candidates_per_key):
        n = len(cands)
        scores = oof_preds[row_idx:row_idx + n]
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

    os.makedirs(os.path.dirname(args.out_inference) or ".", exist_ok=True)
    with open(args.out_inference, "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False)
    print(f"Wrote {len(rows_out)} predictions to {args.out_inference}")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    models[-1].save_model(args.out_model)
    print(f"Saved model to {args.out_model}")

    # Feature importance
    print("\nFeature importance (gain):")
    imp = models[-1].feature_importance(importance_type="gain")
    for name, val in sorted(zip(feature_names, imp), key=lambda x: -x[1])[:15]:
        print(f"  {name:35s}: {val:.0f}")


if __name__ == "__main__":
    main()
