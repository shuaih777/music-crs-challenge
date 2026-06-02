"""Self-contained Music-CRS evaluator.

Loads ground truth from `talkpl-ai/TalkPlayData-Challenge-Dataset::test`,
loads predictions from <inference_path>, computes:
  - nDCG@{1, 10, 20}, macro-averaged over (turns x sessions)
  - catalog_diversity, lexical_diversity (Distinct-2)
  - per-turn breakdown
  - hit@20

Mirrors `nlp4musa/music-crs-evaluator` exactly so submissions stay comparable.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def get_ndcg(gold, preds, k: int) -> float:
    preds = preds[:k]
    dcg = 0.0
    for i, p in enumerate(preds, start=1):
        rel = 1 if p in gold else 0
        dcg += rel / np.log2(i + 1)
    n_rel = min(len(gold), k)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, n_rel + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def has_dupes(values: Sequence) -> bool:
    return len(values) > len(set(values))


def compute_catalog_diversity(rec_track_ids: Sequence[str], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    return len(set(rec_track_ids)) / float(catalog_size)


def compute_lexical_diversity(responses: Sequence[str], n: int = 2) -> float:
    grams: set[tuple] = set()
    total = 0
    for r in responses:
        toks = (r or "").lower().split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return 0.0 if total == 0 else len(grams) / total


def make_ground_truth() -> List[dict]:
    """Build ground-truth records {session_id, turn_number, ground_truth_track_id}.

    Mirrors `make_ground_truth.py` from nlp4musa: ground-truth track_id =
    the second message of role='music' in each turn.
    """
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    out: List[dict] = []
    for ex in tqdm(convo, desc="ground truth"):
        df = pd.DataFrame(ex["conversations"])
        for tn in range(1, 9):
            sub = df[df["turn_number"] == tn]
            # The original evaluator uses `iloc[1]` — the 2nd row at this turn,
            # which is the role='music' line.
            gt = sub.iloc[1]["content"]
            out.append({
                "session_id": ex["session_id"],
                "turn_number": int(tn),
                "ground_truth_track_id": gt,
            })
    return out


def evaluate(preds_path: str, ground_truth: List[dict], catalog_size: int,
             show_per_turn: bool = True) -> dict:
    with open(preds_path) as f:
        preds = json.load(f)

    pdf = pd.DataFrame(preds)
    gdf = pd.DataFrame(ground_truth)

    # Validation
    for r in preds:
        if has_dupes(r["predicted_track_ids"]):
            raise ValueError(f"Duplicates in predicted_track_ids for {r['session_id']}/{r['turn_number']}")
        if len(r["predicted_track_ids"]) > 20:
            raise ValueError(f"More than 20 predictions for {r['session_id']}/{r['turn_number']}")

    joined = gdf.merge(pdf, on=["session_id", "turn_number"], how="left")
    if joined["predicted_track_ids"].isna().any():
        n = int(joined["predicted_track_ids"].isna().sum())
        raise ValueError(f"{n} (session, turn) entries missing in predictions")

    rows = []
    all_recs = []
    all_resps = []
    for _, r in tqdm(joined.iterrows(), total=len(joined), desc="scoring"):
        gold = [r["ground_truth_track_id"]]
        p = r["predicted_track_ids"]
        rows.append({
            "turn_number": r["turn_number"],
            "ndcg@1": get_ndcg(gold, p, 1),
            "ndcg@10": get_ndcg(gold, p, 10),
            "ndcg@20": get_ndcg(gold, p, 20),
            "hit@20": 1 if r["ground_truth_track_id"] in p else 0,
        })
        all_recs.extend(p)
        all_resps.append(r["predicted_response"])

    res = pd.DataFrame(rows)
    per_turn = res.groupby("turn_number")[["ndcg@1", "ndcg@10", "ndcg@20", "hit@20"]].mean()
    macro = per_turn.mean(axis=0).to_dict()
    macro["catalog_diversity"] = compute_catalog_diversity(all_recs, catalog_size)
    macro["lexical_diversity"] = compute_lexical_diversity(all_resps)
    macro["total_catalog_size"] = catalog_size

    if show_per_turn:
        print("\nPer-turn metrics:")
        print(per_turn.round(4).to_string())

    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", required=True, help="Path to inference JSON")
    parser.add_argument("--scores", default=None, help="Where to write scores JSON")
    parser.add_argument("--ground_truth", default=None,
                        help="Optional cached ground-truth JSON; built from HF if missing")
    args = parser.parse_args()

    if args.ground_truth and os.path.exists(args.ground_truth):
        with open(args.ground_truth) as f:
            gt = json.load(f)
    else:
        gt = make_ground_truth()
        if args.ground_truth:
            os.makedirs(os.path.dirname(args.ground_truth) or ".", exist_ok=True)
            with open(args.ground_truth, "w") as f:
                json.dump(gt, f)

    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    catalog_size = len(tracks)

    scores = evaluate(args.inference, gt, catalog_size)
    print("\nMacro scores:")
    print(json.dumps({k: float(v) for k, v in scores.items()}, indent=2))

    if args.scores:
        os.makedirs(os.path.dirname(args.scores) or ".", exist_ok=True)
        with open(args.scores, "w") as f:
            json.dump({k: float(v) for k, v in scores.items()}, f, indent=2)
        print(f"\nWrote {args.scores}")


if __name__ == "__main__":
    main()
