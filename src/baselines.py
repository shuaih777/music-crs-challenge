"""Lightweight Music-CRS baselines (no GPU/LLM required).

Implements four prediction strategies on TalkPlayData-Challenge devset:
  - random         : random top-20 from catalog
  - popularity     : global top-20 by `popularity` score
  - bm25           : tokenize concatenated dialogue history -> BM25 over track metadata
  - bm25_user      : BM25 with the user's listening-history priors mixed in

Outputs an inference JSON in the format the official evaluator expects:
    [{"session_id": str, "user_id": str, "turn_number": int,
      "predicted_track_ids": [str]*<=20, "predicted_response": str}, ...]

Saved to:  music-crs-evaluator/exp/inference/devset/<tid>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Tokenization (matches Distinct-2 lowercasing + whitespace; same idea here)
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
# A small, dependency-free BM25 implementation (Okapi)
# ----------------------------------------------------------------------------


class BM25:
    """Pure-numpy Okapi BM25 over a tokenized corpus."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_lens = np.fromiter((len(d) for d in corpus_tokens), dtype=np.int32, count=self.N)
        self.avgdl = float(self.doc_lens.mean()) if self.N else 0.0

        # vocab
        df: Counter[str] = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.V = len(self.vocab)

        # inverse document frequency (BM25+ smoothing)
        idf = np.zeros(self.V, dtype=np.float32)
        for term, n in df.items():
            idf[self.vocab[term]] = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
        self.idf = idf

        # build sparse term-frequency CSR-like structure: doc_indices, term_indices, tf_values
        rows, cols, vals = [], [], []
        for di, d in enumerate(corpus_tokens):
            tf = Counter(d)
            for term, c in tf.items():
                ti = self.vocab.get(term)
                if ti is None:
                    continue
                rows.append(di)
                cols.append(ti)
                vals.append(c)
        self._rows = np.asarray(rows, dtype=np.int32)
        self._cols = np.asarray(cols, dtype=np.int32)
        self._vals = np.asarray(vals, dtype=np.float32)

        # Pre-compute the length-normalization factor per doc: K_d = k1 * (1 - b + b*|d|/avgdl)
        self._K = self.k1 * (1.0 - self.b + self.b * (self.doc_lens / max(self.avgdl, 1e-9)))

    def score_query(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Return BM25 scores for the query against every document. Shape: (N,)."""
        qtf = Counter(query_tokens)
        # Restrict to terms in vocab
        q_terms = [(self.vocab[t], c) for t, c in qtf.items() if t in self.vocab]
        if not q_terms:
            return np.zeros(self.N, dtype=np.float32)
        # term -> qtf and idf
        # Build a dense vector of length V = 0 except for query terms (small)
        scores = np.zeros(self.N, dtype=np.float32)
        for ti, qc in q_terms:
            # find postings: rows where cols == ti
            mask = self._cols == ti
            doc_ids = self._rows[mask]
            tfs = self._vals[mask]
            num = tfs * (self.k1 + 1.0)
            den = tfs + self._K[doc_ids]
            scores[doc_ids] += self.idf[ti] * (num / den)
        return scores

    def topk(self, query_tokens: Sequence[str], k: int) -> np.ndarray:
        scores = self.score_query(query_tokens)
        if k >= self.N:
            return np.argsort(-scores)
        idx = np.argpartition(-scores, k)[:k]
        # sort just the topk by score desc
        return idx[np.argsort(-scores[idx])]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_data(catalog_size_log: bool = True):
    print("Loading conversation dataset...", flush=True)
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    test = convo["test"]
    train = convo["train"]
    print(f"  train={len(train)}, test={len(test)}", flush=True)

    print("Loading track metadata (all_tracks)...", flush=True)
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    print(f"  tracks={len(tracks)}", flush=True)

    print("Loading user metadata...", flush=True)
    users = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata", split="all_users")
    print(f"  users={len(users)}", flush=True)

    return train, test, tracks, users


def build_track_corpus(tracks) -> Tuple[List[str], List[List[str]]]:
    """Return parallel lists: track_ids[i], corpus_tokens[i]."""
    track_ids: List[str] = []
    corpus_tokens: List[List[str]] = []
    for row in tqdm(tracks, desc="Building track corpus"):
        # Compose a richer text: track name, artists, album, tags
        parts: List[str] = []
        for field in ("track_name", "artist_name", "album_name"):
            v = row.get(field)
            if isinstance(v, list):
                parts.extend(v)
            elif v:
                parts.append(str(v))
        # tag_list: include but downweight by truncation (most-relevant tags first)
        tags = row.get("tag_list") or []
        if isinstance(tags, list):
            parts.extend(tags[:30])
        rd = row.get("release_date") or ""
        if rd and len(rd) >= 4:
            parts.append(rd[:4])  # year
        text = " ".join(parts)
        track_ids.append(row["track_id"])
        corpus_tokens.append(tokenize(text))
    return track_ids, corpus_tokens


# ----------------------------------------------------------------------------
# Build per-(session, turn) chat-history view of the test set
# ----------------------------------------------------------------------------


def iter_inference_inputs(test_split, tracks_meta_by_id: Dict[str, dict]):
    """Yield (session_id, user_id, turn_number, history_text, current_user_query)."""
    for ex in test_split:
        df = pd.DataFrame(ex["conversations"])
        for tn in range(1, 9):
            hist = df[df["turn_number"] < tn]
            # Build a flattened history string. Replace music-role rows with their metadata.
            lines: List[str] = []
            for _, row in hist.iterrows():
                role = row["role"]
                content = row["content"]
                if role == "music":
                    meta = tracks_meta_by_id.get(content, {})
                    name = meta.get("track_name", [""])
                    artist = meta.get("artist_name", [""])
                    name_s = ", ".join(name) if isinstance(name, list) else str(name)
                    artist_s = ", ".join(artist) if isinstance(artist, list) else str(artist)
                    lines.append(f"system: recommended {name_s} by {artist_s}")
                else:
                    lines.append(f"{role}: {content}")
            history_text = "\n".join(lines)
            cur_user = df[(df["turn_number"] == tn) & (df["role"] == "user")]
            user_query = cur_user.iloc[0]["content"] if len(cur_user) else ""
            yield ex["session_id"], ex["user_id"], int(tn), history_text, user_query


# ----------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------


def predict_random(track_ids: List[str], n_per: int, n_calls: int, rng: random.Random) -> List[List[str]]:
    """Sample 20 unique tracks per call, with some shuffling."""
    out = []
    for _ in range(n_calls):
        out.append(rng.sample(track_ids, n_per))
    return out


def predict_popularity_top20(tracks) -> List[str]:
    """Single static top-20 by popularity."""
    df = pd.DataFrame({"track_id": tracks["track_id"], "popularity": tracks["popularity"]})
    df = df.sort_values("popularity", ascending=False).head(20)
    return df["track_id"].tolist()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def run(strategy: str, output_path: str, max_sessions: int | None = None) -> None:
    train, test, tracks, users = load_data()

    track_ids: List[str] = list(tracks["track_id"])
    tracks_by_id = {t["track_id"]: t for t in tracks}

    pop_top20 = predict_popularity_top20(tracks)

    bm25_index: BM25 | None = None
    bm25_track_ids: List[str] | None = None
    if strategy in ("bm25", "bm25_user"):
        bm25_track_ids, corpus = build_track_corpus(tracks)
        print("Building BM25 index over track metadata...", flush=True)
        bm25_index = BM25(corpus)
        print(f"  vocab={bm25_index.V}, avgdl={bm25_index.avgdl:.1f}", flush=True)

    # If bm25_user, also gather the user's "training" listening history from the train split
    user_history_tracks: Dict[str, List[str]] = defaultdict(list)
    if strategy == "bm25_user":
        for ex in train:
            uid = ex["user_id"]
            for c in ex["conversations"]:
                if c["role"] == "music":
                    user_history_tracks[uid].append(c["content"])
        print(f"  user_histories from train: {sum(len(v) for v in user_history_tracks.values())} listens for {len(user_history_tracks)} users", flush=True)

    rng = random.Random(42)

    rows: List[dict] = []
    test_iter = test if max_sessions is None else test.select(range(max_sessions))
    total_calls = (len(test_iter)) * 8

    print(f"Running strategy={strategy} on {len(test_iter)} sessions x 8 turns = {total_calls} calls...", flush=True)
    for session_id, user_id, turn_number, history_text, user_query in tqdm(
        iter_inference_inputs(test_iter, tracks_by_id), total=total_calls, desc=f"infer:{strategy}"
    ):
        if strategy == "random":
            preds = rng.sample(track_ids, 20)
            response = ""
        elif strategy == "popularity":
            preds = pop_top20
            response = ""
        elif strategy == "bm25":
            query = (history_text + "\n" + user_query).strip()
            qt = tokenize(query)
            top = bm25_index.topk(qt, 20)
            preds = [bm25_track_ids[i] for i in top]
            # Mock natural language response: cite the top track name + artist
            top_meta = tracks_by_id.get(preds[0], {})
            tn = top_meta.get("track_name", [""])
            an = top_meta.get("artist_name", [""])
            tn_s = tn[0] if isinstance(tn, list) and tn else str(tn)
            an_s = an[0] if isinstance(an, list) and an else str(an)
            response = f"How about {tn_s} by {an_s}? It matches the vibe you described."
        elif strategy == "bm25_user":
            query = (history_text + "\n" + user_query).strip()
            qt = tokenize(query)
            scores = bm25_index.score_query(qt)
            # Boost tracks similar (in metadata) to ones the user has listened to in train
            hist = user_history_tracks.get(user_id, [])
            if hist:
                hist_query = []
                for tid in hist[-30:]:
                    meta = tracks_by_id.get(tid, {})
                    name = meta.get("track_name") or []
                    artist = meta.get("artist_name") or []
                    if isinstance(name, list):
                        hist_query.extend(name)
                    if isinstance(artist, list):
                        hist_query.extend(artist)
                    tags = meta.get("tag_list") or []
                    if isinstance(tags, list):
                        hist_query.extend(tags[:5])
                hq_tokens = tokenize(" ".join(hist_query))
                hist_scores = bm25_index.score_query(hq_tokens)
                # Linear blend
                scores = scores + 0.3 * hist_scores
            top = np.argpartition(-scores, 20)[:20]
            top = top[np.argsort(-scores[top])]
            preds = [bm25_track_ids[i] for i in top]
            top_meta = tracks_by_id.get(preds[0], {})
            tn = top_meta.get("track_name", [""])
            an = top_meta.get("artist_name", [""])
            tn_s = tn[0] if isinstance(tn, list) and tn else str(tn)
            an_s = an[0] if isinstance(an, list) and an else str(an)
            response = f"Based on your taste, you might enjoy {tn_s} by {an_s}."
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Strict format
        rows.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
            "predicted_track_ids": list(preds),
            "predicted_response": response,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, choices=["random", "popularity", "bm25", "bm25_user"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_sessions", type=int, default=None)
    args = parser.parse_args()
    run(args.strategy, args.output, args.max_sessions)
