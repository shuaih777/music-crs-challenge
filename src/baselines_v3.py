"""Music-CRS hybrid retrieval baseline (v3).

Configurable knobs (no GPU required; vectorized over numpy):
  --bm25_only           : disable dense retrieval entirely
  --no_filter           : disable no-repeat filter
  --embed FIELD         : pick which track-embedding modality (default audio-laion_clap)
  --pooling MODE        : how to pool prior-accepted track embeddings:
                            mean        - mean of all prior tracks (v2 default)
                            last        - only the most recent accepted track
                            decay       - exponentially weighted, alpha=0.7
                            last_k_mean - mean of last K accepted tracks (--last_k)
  --weight_schedule S   : how dense vs BM25 weights vary by turn:
                            constant    - 0.5 / 0.5
                            ascending   - dense increases with turn  (v2)
                            descending  - dense decreases with turn (recommended)
                            zero_after  - dense only for turns <= --dense_max_turn
  --dense_max_turn N    : for zero_after schedule
  --last_k N            : for last_k_mean pooling
  --topk_fuse N         : number of candidates each retriever contributes (default 200)
  --user_emb FIELD      : optional: blend in a user prior from the user-embeddings dataset
                          ('item_factor') — controls a static personalization signal

Outputs the standard Music-CRS inference JSON.

This file is the *sole* model file for this repo. It should run on CPU in
~10-25 minutes on the devset (1000 sessions x 8 turns), or in seconds per turn
on a GPU if you swap the BM25 implementation for one that batches scoring on
GPU and replace the numpy mat-mul with torch (see comments below).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOP = set(
    "a an and the of in on at to for from by is are was were be been being "
    "this that these those it its as with or but if then so than into about "
    "i me my you your we our they them he she his her us i'm i'll i've don't "
    "doesn't can't would should could like want some any new song songs music "
    "track tracks recommend recommendation play listen listening".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


# ----------------------------------------------------------------------------
# Pure-numpy BM25 (vectorized; no external dependency)
# ----------------------------------------------------------------------------


class BM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_lens = np.fromiter((len(d) for d in corpus_tokens), dtype=np.int32, count=self.N)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        self.V = len(self.vocab)

        idf = np.zeros(self.V, dtype=np.float32)
        for t, n in df.items():
            idf[self.vocab[t]] = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
        self.idf = idf

        rows, cols, vals = [], [], []
        for di, d in enumerate(corpus_tokens):
            tf = Counter(d)
            for t, c in tf.items():
                ti = self.vocab.get(t)
                if ti is None:
                    continue
                rows.append(di); cols.append(ti); vals.append(c)
        self._rows = np.asarray(rows, dtype=np.int32)
        self._cols = np.asarray(cols, dtype=np.int32)
        self._vals = np.asarray(vals, dtype=np.float32)
        self._K = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / max(self.avgdl, 1e-9)))

    def score_query(self, query_tokens: Sequence[str]) -> np.ndarray:
        qtf = Counter(query_tokens)
        q_terms = [(self.vocab[t], c) for t, c in qtf.items() if t in self.vocab]
        if not q_terms:
            return np.zeros(self.N, dtype=np.float32)
        scores = np.zeros(self.N, dtype=np.float32)
        for ti, _ in q_terms:
            mask = self._cols == ti
            ids = self._rows[mask]
            tfs = self._vals[mask]
            num = tfs * (self.k1 + 1.0)
            den = tfs + self._K[ids]
            scores[ids] += self.idf[ti] * (num / den)
        return scores


def build_track_corpus(tracks) -> tuple[list[str], list[list[str]]]:
    track_ids: List[str] = []
    corpus_tokens: List[List[str]] = []
    for row in tqdm(tracks, desc="Building corpus"):
        parts: List[str] = []
        for field in ("track_name", "artist_name", "album_name"):
            v = row.get(field)
            if isinstance(v, list):
                parts.extend(v)
            elif v:
                parts.append(str(v))
        tags = row.get("tag_list") or []
        if isinstance(tags, list):
            parts.extend(tags[:30])
        rd = row.get("release_date") or ""
        if rd and len(rd) >= 4:
            parts.append(rd[:4])
        track_ids.append(row["track_id"])
        corpus_tokens.append(tokenize(" ".join(parts)))
    return track_ids, corpus_tokens


# ----------------------------------------------------------------------------
# Embedding loading + pooling
# ----------------------------------------------------------------------------


def load_track_embeddings(field: str, track_ids: List[str]) -> np.ndarray:
    print(f"Loading track embeddings ({field})...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
    by_id: Dict[str, list] = {row["track_id"]: row[field] for row in tqdm(ds, desc="indexing emb")}
    dim = next((len(e) for e in by_id.values() if isinstance(e, list) and len(e) > 0), 0)
    n_missing = 0
    out = np.zeros((len(track_ids), dim), dtype=np.float32)
    for i, tid in enumerate(track_ids):
        e = by_id.get(tid)
        if isinstance(e, list) and len(e) == dim:
            out[i] = e
        else:
            n_missing += 1
    if n_missing:
        print(f"  WARNING: {n_missing} missing/malformed; zero-padded", flush=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.clip(norms, 1e-9, None)
    print(f"  embeddings: shape={out.shape}", flush=True)
    return out


def pool_prior_embeddings(prior_idx: List[int], track_emb: np.ndarray,
                          mode: str, decay_alpha: float = 0.7,
                          last_k: int = 3) -> np.ndarray | None:
    """Build a query embedding from indices of accepted tracks.

    Returns L2-normalized vector, or None if `prior_idx` is empty.
    """
    if not prior_idx:
        return None
    if mode == "mean":
        q = track_emb[prior_idx].mean(axis=0)
    elif mode == "last":
        q = track_emb[prior_idx[-1]]
    elif mode == "decay":
        # weights = alpha^(t-i), most recent gets largest weight
        weights = np.array([decay_alpha ** (len(prior_idx) - 1 - i) for i in range(len(prior_idx))],
                           dtype=np.float32)
        weights /= weights.sum()
        q = (track_emb[prior_idx] * weights[:, None]).sum(axis=0)
    elif mode == "last_k_mean":
        q = track_emb[prior_idx[-last_k:]].mean(axis=0)
    else:
        raise ValueError(f"unknown pooling mode: {mode}")
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else None


# ----------------------------------------------------------------------------
# Weight schedules: how to combine BM25 and dense at each turn
# ----------------------------------------------------------------------------


def get_weights(schedule: str, turn: int, dense_max_turn: int = 4) -> tuple[float, float]:
    """Return (w_bm25, w_dense), summing to 1."""
    if schedule == "constant":
        return 0.5, 0.5
    if schedule == "ascending":
        wd = min(0.4 + 0.05 * (turn - 1), 0.7)
        return 1.0 - wd, wd
    if schedule == "descending":
        # turn 2: 0.7 dense; decays to ~0.3 dense by turn 8
        wd = max(0.7 - 0.07 * (turn - 2), 0.25)
        return 1.0 - wd, wd
    if schedule == "zero_after":
        if turn <= dense_max_turn:
            return 0.5, 0.5
        return 1.0, 0.0
    raise ValueError(f"unknown weight schedule: {schedule}")


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]


def run(args: argparse.Namespace) -> None:
    # --- load conversation, tracks ---
    print("Loading conversation dataset...", flush=True)
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    test = convo["test"]
    print(f"  test={len(test)}", flush=True)

    print("Loading track metadata...", flush=True)
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_ids = list(tracks["track_id"])
    tracks_by_id = {t["track_id"]: t for t in tracks}
    track_idx_by_id = {tid: i for i, tid in enumerate(track_ids)}

    # --- BM25 ---
    bm25_track_ids, corpus = build_track_corpus(tracks)
    assert bm25_track_ids == track_ids
    bm25 = BM25(corpus)
    print(f"  BM25: vocab={bm25.V}, avgdl={bm25.avgdl:.1f}", flush=True)

    # --- track embeddings ---
    track_emb: np.ndarray | None = None
    if not args.bm25_only:
        track_emb = load_track_embeddings(args.embed, track_ids)

    # --- inference loop ---
    rows: List[dict] = []
    n_dense_used = 0
    n_filtered = 0
    total = len(test) * 8

    print(f"Inference: {total} (session, turn) pairs ...", flush=True)
    pbar = tqdm(total=total, desc=args.tag)
    for ex in test:
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        df = pd.DataFrame(ex["conversations"])
        for tn in range(1, 9):
            hist = df[df["turn_number"] < tn]
            cur_user = df[(df["turn_number"] == tn) & (df["role"] == "user")]
            user_query = cur_user.iloc[0]["content"] if len(cur_user) else ""

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
            full_query = ("\n".join(hist_lines) + "\n" + user_query).strip()
            qt = tokenize(full_query)

            # BM25
            bm25_scores = bm25.score_query(qt)
            bm25_rank = topk_indices(bm25_scores, args.topk_fuse)

            # Dense (optional)
            ranked: np.ndarray
            if track_emb is not None:
                prior_idx = [track_idx_by_id[t] for t in prior_track_ids if t in track_idx_by_id]
                q_vec = pool_prior_embeddings(prior_idx, track_emb,
                                              mode=args.pooling,
                                              decay_alpha=args.decay_alpha,
                                              last_k=args.last_k)
                if q_vec is not None:
                    n_dense_used += 1
                    dense_scores = track_emb @ q_vec
                    dense_rank = topk_indices(dense_scores, args.topk_fuse)
                    w_bm25, w_dense = get_weights(args.weight_schedule, tn, args.dense_max_turn)
                    if w_dense == 0:
                        ranked = bm25_rank
                    else:
                        # Blend in score space (normalize each to max=1)
                        bm_max = max(bm25_scores.max(), 1e-9)
                        de_max = max(dense_scores.max(), 1e-9)
                        # Take union of top-K
                        cand = np.unique(np.concatenate([bm25_rank, dense_rank]))
                        s = w_bm25 * (bm25_scores[cand] / bm_max) \
                          + w_dense * (dense_scores[cand] / de_max)
                        ranked = cand[np.argsort(-s)]
                else:
                    ranked = bm25_rank
            else:
                ranked = bm25_rank

            # No-repeat filter
            preds: List[str] = []
            seen = set(prior_track_ids) if not args.no_filter else set()
            for idx in ranked:
                tid = track_ids[idx]
                if tid in seen:
                    n_filtered += 1
                    continue
                preds.append(tid)
                if len(preds) == 20:
                    break
            while len(preds) < 20:  # pad shouldn't trigger
                preds.append(track_ids[int(np.random.randint(len(track_ids)))])

            top_meta = tracks_by_id.get(preds[0], {})
            tn_n = top_meta.get("track_name", [""])
            an = top_meta.get("artist_name", [""])
            tn_s = tn_n[0] if isinstance(tn_n, list) and tn_n else str(tn_n)
            an_s = an[0] if isinstance(an, list) and an else str(an)
            response = (
                f"How about {tn_s} by {an_s}? "
                f"This should match the mood you described in our chat."
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

    print(f"  dense used in {n_dense_used} of {total} calls", flush=True)
    print(f"  no-repeat filter removed {n_filtered} candidates", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {args.output}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--tag", default="v3", help="Progress-bar / log tag")
    p.add_argument("--bm25_only", action="store_true")
    p.add_argument("--no_filter", action="store_true", help="Disable no-repeat filter")
    p.add_argument("--embed", default="audio-laion_clap",
                   choices=["audio-laion_clap", "image-siglip2", "cf-bpr",
                            "attributes-qwen3_embedding_0.6b",
                            "lyrics-qwen3_embedding_0.6b",
                            "metadata-qwen3_embedding_0.6b"])
    p.add_argument("--pooling", default="mean",
                   choices=["mean", "last", "decay", "last_k_mean"])
    p.add_argument("--decay_alpha", type=float, default=0.7)
    p.add_argument("--last_k", type=int, default=3)
    p.add_argument("--weight_schedule", default="descending",
                   choices=["constant", "ascending", "descending", "zero_after"])
    p.add_argument("--dense_max_turn", type=int, default=4)
    p.add_argument("--topk_fuse", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
