"""Build item-item PMI co-occurrence matrix from training sessions.

Given that each train session has exactly 8 accepted tracks, tracks that
frequently co-occur in sessions are likely good recommendations when one
is already accepted. This builds a sparse PPMI (positive PMI) matrix and
provides a retrieval function.

Usage:
    # Build (CPU, ~30 sec)
    python src/build_item2item.py --out exp/item2item_pmi.npz

    # Then use in baselines_v3.py via the new --pmi_path flag
    python src/baselines_v3.py \\
        --output exp/inference/devset/pmi_leg.json \\
        --bm25_only \\
        --pmi_path exp/item2item_pmi.npz \\
        --tag pmi_leg

The PMI retrieval returns: for each set of prior accepted tracks in the
session, aggregate their PMI rows and return top-K tracks by total PPMI.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import sparse
from datasets import load_dataset
from tqdm import tqdm


def build_pmi_matrix(min_count: int = 2) -> Tuple[sparse.csr_matrix, List[str], Dict[str, int]]:
    """Build a symmetric PPMI matrix from training session co-occurrences.

    Returns:
        pmi_matrix: (N_tracks, N_tracks) sparse CSR with PPMI values
        track_ids: ordered list of track IDs (rows/cols index)
        track_to_idx: mapping from track_id -> matrix index
    """
    print("Loading training sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    print(f"  {len(train)} sessions", flush=True)

    # Collect per-session track sets and global counts
    session_tracks: List[Set[str]] = []
    track_counter: Counter = Counter()
    for ex in tqdm(train, desc="extracting tracks"):
        sset: Set[str] = set()
        for c in ex["conversations"]:
            if c["role"] == "music":
                sset.add(c["content"])
                track_counter[c["content"]] += 1
        session_tracks.append(sset)

    # Build vocab (all tracks that appear at all — we need the full catalog
    # for retrieval, but only tracks in train can have PMI > 0)
    print("Loading full catalog for alignment...", flush=True)
    catalog = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    all_track_ids = [r["track_id"] for r in catalog]
    track_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
    N = len(all_track_ids)
    print(f"  catalog: {N}, in-train: {len(track_counter)}", flush=True)

    # Count co-occurrences
    print("Counting co-occurrences...", flush=True)
    pair_counts: Counter = Counter()
    n_sessions = len(session_tracks)
    for sset in tqdm(session_tracks, desc="counting pairs"):
        tracks = sorted(sset)  # deterministic order
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                pair_counts[(tracks[i], tracks[j])] += 1

    # Filter by min_count
    pair_counts = {k: v for k, v in pair_counts.items() if v >= min_count}
    print(f"  pairs with count >= {min_count}: {len(pair_counts):,}", flush=True)

    # Compute PPMI
    # PMI(a, b) = log2( P(a,b) / (P(a) * P(b)) )
    # P(a) = count(a) / n_sessions
    # P(a,b) = count(a,b) / n_sessions
    # PPMI = max(0, PMI)
    print("Computing PPMI...", flush=True)
    rows, cols, vals = [], [], []
    for (a, b), count_ab in tqdm(pair_counts.items(), desc="PPMI"):
        if a not in track_to_idx or b not in track_to_idx:
            continue
        p_ab = count_ab / n_sessions
        p_a = track_counter[a] / n_sessions
        p_b = track_counter[b] / n_sessions
        pmi = np.log2(p_ab / (p_a * p_b + 1e-12))
        ppmi = max(0.0, pmi)
        if ppmi > 0:
            ia, ib = track_to_idx[a], track_to_idx[b]
            rows.extend([ia, ib])
            cols.extend([ib, ia])
            vals.extend([ppmi, ppmi])

    pmi_matrix = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(N, N),
    )
    print(f"  PPMI matrix: shape={pmi_matrix.shape}, nnz={pmi_matrix.nnz:,}", flush=True)
    return pmi_matrix, all_track_ids, track_to_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="exp/item2item_pmi.npz",
                   help="Output path for the sparse PMI matrix")
    p.add_argument("--min_count", type=int, default=2,
                   help="Minimum co-occurrence count to include a pair")
    args = p.parse_args()

    pmi_matrix, track_ids, track_to_idx = build_pmi_matrix(args.min_count)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sparse.save_npz(args.out, pmi_matrix)
    # Also save track_ids order for verification
    meta_path = args.out.replace(".npz", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"track_ids": track_ids[:10], "n_tracks": len(track_ids),
                   "nnz": int(pmi_matrix.nnz)}, f)
    print(f"\nSaved {args.out} ({pmi_matrix.nnz:,} entries)")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
