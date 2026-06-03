"""Combine multiple inference JSONs into a stronger submission.

Two strategies:

1. **per-turn**: pick a separate inference JSON for each conversation turn.
   For Music-CRS, the devset shows `metadata_qwen3` wins t1-6 and `cf_bpr`
   wins t7-8, so the per-turn ensemble has nDCG@20 ≈ 0.123 (devset upper bound).

2. **rrf**: reciprocal-rank-fuse the predicted track rankings from N
   inference JSONs into one. RRF is robust to different score scales and
   works well when models disagree.

Inputs are existing inference JSONs (output of `baselines_v3.py`).
Outputs another inference JSON in the same format, ready for `evaluate.py`
or for submission.

Examples:
    # per-turn: best-of-each from devset analysis
    python src/ensemble.py per_turn \
        --plan "1-6:metadata_qwen3,7-8:cf_bpr" \
        --inference_dir exp/inference/devset \
        --out exp/inference/devset/ensemble_per_turn.json

    # RRF fuse two configs (good for late-turn fallback)
    python src/ensemble.py rrf \
        --inputs metadata_qwen3,cf_bpr \
        --inference_dir exp/inference/devset \
        --weights 1.0,0.5 \
        --out exp/inference/devset/ensemble_rrf.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence


def load_inference(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def index_by_key(rows: List[dict]) -> Dict[tuple[str, int], dict]:
    return {(r["session_id"], int(r["turn_number"])): r for r in rows}


def parse_turn_plan(plan: str) -> Dict[int, str]:
    """Parse "1-6:metadata_qwen3,7-8:cf_bpr" -> {1: 'metadata_qwen3', ..., 8: 'cf_bpr'}"""
    out: Dict[int, str] = {}
    for chunk in plan.split(","):
        rng, tag = chunk.split(":")
        rng = rng.strip()
        tag = tag.strip()
        if "-" in rng:
            a, b = rng.split("-")
            for t in range(int(a), int(b) + 1):
                out[t] = tag
        else:
            out[int(rng)] = tag
    missing = [t for t in range(1, 9) if t not in out]
    if missing:
        raise ValueError(f"Plan does not cover all turns 1..8 (missing {missing})")
    return out


def per_turn_ensemble(plan: Dict[int, str], inference_dir: str) -> List[dict]:
    """Pick a tag per turn; emit one row per (session, turn) from that tag's
    inference JSON.
    """
    by_tag: Dict[str, Dict[tuple[str, int], dict]] = {}
    for tag in set(plan.values()):
        path = os.path.join(inference_dir, f"{tag}.json")
        rows = load_inference(path)
        by_tag[tag] = index_by_key(rows)
        print(f"  loaded {tag}: {len(rows)} rows")

    # Use any input as the canonical (session, turn) keyspace
    any_index = next(iter(by_tag.values()))
    keys = sorted(any_index, key=lambda k: (k[0], k[1]))

    out: List[dict] = []
    for k in keys:
        tag = plan[k[1]]
        src = by_tag[tag].get(k)
        if src is None:
            raise ValueError(f"Missing prediction for {k} in {tag}")
        out.append({
            "session_id": src["session_id"],
            "user_id": src["user_id"],
            "turn_number": int(src["turn_number"]),
            "predicted_track_ids": list(src["predicted_track_ids"]),
            "predicted_response": src.get("predicted_response", ""),
        })
    return out


def rrf_ensemble(tags: List[str], inference_dir: str,
                 weights: List[float] | None = None,
                 k: int = 60, top: int = 20) -> List[dict]:
    """Reciprocal-Rank-Fuse multiple inference JSONs.

    Score for doc d under tag i with rank r_i (1-indexed): w_i / (k + r_i).
    Sum across tags, sort desc, take top-N. Each tag's predicted_response is
    pooled — we use the *first* tag's response by default.
    """
    weights = weights or [1.0] * len(tags)
    if len(weights) != len(tags):
        raise ValueError("weights must match tags in length")

    indexed: List[Dict[tuple[str, int], dict]] = []
    for tag in tags:
        path = os.path.join(inference_dir, f"{tag}.json")
        rows = load_inference(path)
        indexed.append(index_by_key(rows))
        print(f"  loaded {tag}: {len(rows)} rows")

    # Use the first tag as canonical key set
    keys = sorted(indexed[0], key=lambda k: (k[0], k[1]))

    out: List[dict] = []
    for key in keys:
        accum: Dict[str, float] = defaultdict(float)
        canon: dict | None = None
        for w, idx in zip(weights, indexed):
            row = idx.get(key)
            if row is None:
                continue
            if canon is None:
                canon = row
            for r, tid in enumerate(row["predicted_track_ids"], start=1):
                accum[tid] += w / (k + r)
        if canon is None:
            raise ValueError(f"No tag has a row for {key}")
        ranked = sorted(accum.items(), key=lambda kv: -kv[1])
        preds = [tid for tid, _ in ranked[:top]]
        out.append({
            "session_id": canon["session_id"],
            "user_id": canon["user_id"],
            "turn_number": int(canon["turn_number"]),
            "predicted_track_ids": preds,
            "predicted_response": canon.get("predicted_response", ""),
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="strategy", required=True)

    pt = sub.add_parser("per_turn", help="Per-turn pick from existing JSONs")
    pt.add_argument("--plan", required=True,
                    help="e.g. '1-6:metadata_qwen3,7-8:cf_bpr'")
    pt.add_argument("--inference_dir", default="exp/inference/devset")
    pt.add_argument("--out", required=True)

    rrf = sub.add_parser("rrf", help="RRF-merge multiple ranked lists")
    rrf.add_argument("--inputs", required=True,
                     help="comma-separated tag names")
    rrf.add_argument("--weights", default=None,
                     help="comma-separated floats (same length as --inputs)")
    rrf.add_argument("--inference_dir", default="exp/inference/devset")
    rrf.add_argument("--rrf_k", type=int, default=60)
    rrf.add_argument("--out", required=True)

    args = p.parse_args()

    if args.strategy == "per_turn":
        plan = parse_turn_plan(args.plan)
        rows = per_turn_ensemble(plan, args.inference_dir)
    else:
        tags = [t.strip() for t in args.inputs.split(",")]
        weights = [float(w) for w in args.weights.split(",")] if args.weights else None
        rows = rrf_ensemble(tags, args.inference_dir, weights, args.rrf_k)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\nWrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
