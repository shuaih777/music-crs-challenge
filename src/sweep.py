"""Sweep multiple v3 retrieval configurations and rank them on the devset.

Runs `baselines_v3.py` with several configs, then `evaluate.py` for each.
Designed for a CPU box but trivial to parallelize on a GPU host.

Usage:
    python src/sweep.py --out_dir exp/inference/devset --runs all
    python src/sweep.py --out_dir exp/inference/devset --runs basic,decay,inverted

Each run takes ~10-25 minutes on CPU. On GPU, profile and consider porting
the BM25 scoring + dense matmul to torch (single line change for the matmul).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List

# Each tuple: (tag, CLI args for baselines_v3.py)
RECIPES: Dict[str, list[str]] = {
    # 1. Pure BM25 (no dense, no filter) — sanity baseline
    "bm25_pure": ["--bm25_only", "--no_filter"],
    # 2. BM25 + no-repeat (the surprise winner so far)
    "bm25_norepeat": ["--bm25_only"],
    # 3. v2 hybrid: mean pooling + ascending dense weight (= REPORT.md "hybrid")
    "v2_hybrid": ["--embed", "audio-laion_clap", "--pooling", "mean", "--weight_schedule", "ascending"],
    # 4. New: descending weight (dense more useful early)
    "decay_descending": ["--embed", "audio-laion_clap", "--pooling", "decay",
                         "--weight_schedule", "descending"],
    # 5. New: only use the LAST accepted track as dense query
    "last_track": ["--embed", "audio-laion_clap", "--pooling", "last",
                   "--weight_schedule", "descending"],
    # 6. New: dense only used for turns 2-4, BM25-only after
    "early_dense_only": ["--embed", "audio-laion_clap", "--pooling", "decay",
                         "--weight_schedule", "zero_after", "--dense_max_turn", "4"],
    # 7. New: text-modality embedding instead of audio
    "metadata_qwen3": ["--embed", "metadata-qwen3_embedding_0.6b", "--pooling", "decay",
                       "--weight_schedule", "descending"],
    # 8. New: collaborative-filtering BPR factors
    "cf_bpr": ["--embed", "cf-bpr", "--pooling", "decay",
               "--weight_schedule", "descending"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="exp/inference/devset")
    parser.add_argument("--scores_dir", default="exp/scores/devset")
    parser.add_argument("--runs", default="all",
                        help="comma-separated tags to run, or 'all'")
    parser.add_argument("--ground_truth", default="exp/ground_truth/devset.json")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip a run if its inference JSON already exists")
    args = parser.parse_args()

    selected = list(RECIPES) if args.runs == "all" else args.runs.split(",")
    unknown = [r for r in selected if r not in RECIPES]
    if unknown:
        print(f"Unknown runs: {unknown}", file=sys.stderr)
        print(f"Available: {list(RECIPES)}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.scores_dir, exist_ok=True)

    leaderboard = []
    for tag in selected:
        out_json = os.path.join(args.out_dir, f"{tag}.json")
        score_json = os.path.join(args.scores_dir, f"{tag}.json")

        if args.skip_existing and os.path.exists(out_json):
            print(f"\n=== {tag}: skipping inference (exists) ===")
        else:
            cmd = [sys.executable, "src/baselines_v3.py",
                   "--output", out_json, "--tag", tag] + RECIPES[tag]
            print(f"\n=== {tag}: {' '.join(shlex.quote(c) for c in cmd)} ===")
            subprocess.check_call(cmd)

        cmd = [sys.executable, "src/evaluate.py",
               "--inference", out_json,
               "--scores", score_json,
               "--ground_truth", args.ground_truth]
        print(f"\n=== {tag}: {' '.join(shlex.quote(c) for c in cmd)} ===")
        subprocess.check_call(cmd)

        with open(score_json) as f:
            s = json.load(f)
        leaderboard.append((tag, s))

    print("\n\n=== LEADERBOARD ===")
    print(f"{'tag':22s} {'nDCG@1':>8s} {'nDCG@10':>8s} {'nDCG@20':>8s} {'CatDiv':>8s} {'LexDiv':>8s}")
    for tag, s in sorted(leaderboard, key=lambda x: -x[1].get("ndcg@20", 0)):
        print(f"{tag:22s} {s['ndcg@1']:8.4f} {s['ndcg@10']:8.4f} "
              f"{s['ndcg@20']:8.4f} {s['catalog_diversity']:8.4f} "
              f"{s['lexical_diversity']:8.4f}")


if __name__ == "__main__":
    main()
