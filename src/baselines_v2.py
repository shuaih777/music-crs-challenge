"""Music-CRS hybrid retrieval baseline (v2).

Improvements over `baselines.py` BM25:
  1. **No-repeat filter**: tracks recommended in earlier turns of the same
     session are removed from candidates (free win — golds never repeat).
  2. **Hybrid retrieval**: BM25 over track metadata text + dense retrieval
     over the LAION-CLAP audio embeddings, fused with Reciprocal Rank Fusion
     (RRF). Dense query = mean of CLAP embeddings of tracks the user
     accepted in earlier turns of this session. When no prior tracks (turn 1),
     we fall back to BM25-only.
  3. **Score-aware fusion**: weight dense more heavily as the conversation
     progresses (turn 1 = pure BM25, turn 8 = balanced).

Outputs the standard inference JSON for the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from baselines import BM25, build_track_corpus, tokenize  # reuse


# ----------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ----------------------------------------------------------------------------


def rrf(rank_lists: List[np.ndarray], scores_list: List[np.ndarray] | None = None,
        k: int = 60, weights: List[float] | None = None) -> np.ndarray:
    """Combine multiple rankings via weighted RRF.

    Args:
        rank_lists: each is the array of doc indices in score-desc order.
        weights: per-list weight (broadcast).
    Returns:
        Combined RRF score array of length N (use argpartition for top-k).
    """
    weights = weights or [1.0] * len(rank_lists)
    # Determine N from the first nonempty rank list
    N = max(int(r.max()) + 1 if r.size else 0 for r in rank_lists)
    combined = np.zeros(N, dtype=np.float32)
    for w, ranks in zip(weights, rank_lists):
        for r_pos, doc_id in enumerate(ranks):
            combined[doc_id] += w / (k + r_pos + 1)
    return combined


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------


def run(output_path: str, use_dense: bool = True, embed_field: str = "audio-laion_clap",
        no_repeat: bool = True) -> None:
    print("Loading conversation dataset...", flush=True)
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    test = convo["test"]
    print(f"  test={len(test)}", flush=True)

    print("Loading track metadata...", flush=True)
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_ids = list(tracks["track_id"])
    tracks_by_id = {t["track_id"]: t for t in tracks}
    track_idx_by_id = {tid: i for i, tid in enumerate(track_ids)}

    print("Building BM25 over track metadata corpus...", flush=True)
    bm25_track_ids, corpus = build_track_corpus(tracks)
    assert bm25_track_ids == track_ids, "BM25 track order should match catalog order"
    bm25 = BM25(corpus)
    print(f"  BM25: vocab={bm25.V}, avgdl={bm25.avgdl:.1f}", flush=True)

    track_emb: np.ndarray | None = None
    if use_dense:
        print(f"Loading track embeddings ({embed_field})...", flush=True)
        emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        # Build aligned embedding matrix in track_ids order
        emb_by_id: Dict[str, list] = {row["track_id"]: row[embed_field] for row in tqdm(emb_ds, desc="indexing emb")}
        # Determine target dim from a non-empty entry
        dim = next((len(e) for e in emb_by_id.values() if isinstance(e, list) and len(e) > 0), 0)
        n_missing = 0
        rows_emb = np.zeros((len(track_ids), dim), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            e = emb_by_id.get(tid)
            if isinstance(e, list) and len(e) == dim:
                rows_emb[i] = e
            else:
                n_missing += 1
        track_emb = rows_emb
        if n_missing:
            print(f"  WARNING: {n_missing} tracks have missing/malformed {embed_field}; zero-padded", flush=True)
        # L2 normalize (CLAP audio is pre-normalized but be safe)
        norms = np.linalg.norm(track_emb, axis=1, keepdims=True)
        track_emb = track_emb / np.clip(norms, 1e-9, None)
        print(f"  embeddings: shape={track_emb.shape}", flush=True)

    rows: List[dict] = []
    n_dense_used = 0
    n_no_repeat_filter_hit = 0
    total_calls = len(test) * 8

    print(f"Running hybrid inference on {len(test)} sessions x 8 turns...", flush=True)
    pbar = tqdm(total=total_calls, desc="hybrid")
    for ex in test:
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        # Build per-turn history once
        df = pd.DataFrame(ex["conversations"])
        for tn in range(1, 9):
            hist = df[df["turn_number"] < tn]
            cur_user = df[(df["turn_number"] == tn) & (df["role"] == "user")]
            user_query = cur_user.iloc[0]["content"] if len(cur_user) else ""

            # Compose history text + user query for BM25
            hist_lines: List[str] = []
            prior_track_ids: List[str] = []
            for _, row in hist.iterrows():
                role = row["role"]
                content = row["content"]
                if role == "music":
                    prior_track_ids.append(content)
                    meta = tracks_by_id.get(content, {})
                    name = meta.get("track_name", [""])
                    artist = meta.get("artist_name", [""])
                    name_s = ", ".join(name) if isinstance(name, list) else str(name)
                    artist_s = ", ".join(artist) if isinstance(artist, list) else str(artist)
                    hist_lines.append(f"system: recommended {name_s} by {artist_s}")
                else:
                    hist_lines.append(f"{role}: {content}")
            history_text = "\n".join(hist_lines)
            full_query = (history_text + "\n" + user_query).strip()
            qt = tokenize(full_query)

            # --- BM25 scores -------------------------------------------------
            bm25_scores = bm25.score_query(qt)

            # --- Dense scores ------------------------------------------------
            dense_scores: np.ndarray | None = None
            if use_dense and track_emb is not None:
                prior_idx = [track_idx_by_id[t] for t in prior_track_ids if t in track_idx_by_id]
                if prior_idx:
                    # Mean of accepted-track embeddings -> query
                    q_vec = track_emb[prior_idx].mean(axis=0)
                    qn = np.linalg.norm(q_vec)
                    if qn > 1e-9:
                        q_vec = q_vec / qn
                        dense_scores = track_emb @ q_vec  # cosine since both L2-normed
                        n_dense_used += 1

            # --- Fuse --------------------------------------------------------
            # RRF over top-K from each list. Take top 200 of each before fusion.
            TOPK_FUSE = 200

            def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
                if k >= len(scores):
                    return np.argsort(-scores)
                idx = np.argpartition(-scores, k)[:k]
                return idx[np.argsort(-scores[idx])]

            bm25_rank = topk_indices(bm25_scores, TOPK_FUSE)
            if dense_scores is not None:
                dense_rank = topk_indices(dense_scores, TOPK_FUSE)
                # Turn-aware weighting: turn 1 has no dense (skipped above);
                # later turns get progressively more dense weight.
                w_dense = min(0.4 + 0.05 * (tn - 1), 0.7)
                w_bm25 = 1.0 - w_dense
                fused = rrf([bm25_rank, dense_rank], weights=[w_bm25, w_dense])
                # Take top 40 from fused as candidate pool, then re-rank by raw blended score
                top40 = topk_indices(fused, 40)
                # Final re-rank: use the actual scores for stability
                final_scores = w_bm25 * (bm25_scores[top40] / max(bm25_scores.max(), 1e-9)) \
                             + w_dense * dense_scores[top40]
                ranked = top40[np.argsort(-final_scores)]
            else:
                ranked = bm25_rank

            # --- No-repeat filter -------------------------------------------
            preds: List[str] = []
            seen_prior = set(prior_track_ids) if no_repeat else set()
            for idx in ranked:
                tid = track_ids[idx]
                if tid in seen_prior:
                    n_no_repeat_filter_hit += 1
                    continue
                preds.append(tid)
                if len(preds) == 20:
                    break
            # Pad if somehow short (shouldn't happen with TOPK_FUSE=200)
            while len(preds) < 20:
                preds.append(track_ids[int(np.random.randint(len(track_ids)))])

            # Templated response
            top_meta = tracks_by_id.get(preds[0], {})
            tn_n = top_meta.get("track_name", [""])
            an = top_meta.get("artist_name", [""])
            tn_s = tn_n[0] if isinstance(tn_n, list) and tn_n else str(tn_n)
            an_s = an[0] if isinstance(an, list) and an else str(an)
            response = (
                f"How about {tn_s} by {an_s}? "
                f"Based on the conversation so far, this should match the mood you described."
            )

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(tn),
                "predicted_track_ids": preds,
                "predicted_response": response,
            })
            pbar.update(1)
    pbar.close()

    print(f"  dense used in {n_dense_used} of {total_calls} calls", flush=True)
    print(f"  no-repeat filter removed {n_no_repeat_filter_hit} candidates total", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--no_dense", action="store_true", help="Disable dense; keep no-repeat only")
    parser.add_argument("--no_filter", action="store_true", help="Disable no-repeat filter")
    parser.add_argument("--embed", default="audio-laion_clap",
                        choices=["audio-laion_clap", "image-siglip2", "cf-bpr",
                                 "attributes-qwen3_embedding_0.6b",
                                 "lyrics-qwen3_embedding_0.6b",
                                 "metadata-qwen3_embedding_0.6b"])
    args = parser.parse_args()
    run(args.output, use_dense=not args.no_dense, embed_field=args.embed,
        no_repeat=not args.no_filter)
