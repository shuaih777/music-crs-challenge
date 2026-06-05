"""Quick alignment check: does a state-embedding npz align with a track-embedding modality?

Without running the full retrieval inference, this script estimates the
upper bound by computing Hit@K on the devset for state-emb -> track-emb
nearest-neighbor lookup (gold = the recommended track at that turn).

  python src/diagnose_alignment.py \\
      --state_emb exp/states/test_qwen3_emb.npz \\
      --track_field attributes-qwen3_embedding_0.6b \\
      --sample 1000

If Hit@100 < 5% the state encoder and track encoder are in different spaces
(typically: missing instruction prefix, wrong base model, or different
prompt template). If Hit@100 > 30% the dense leg should help retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state_emb", required=True,
                   help="Output of encode_states.py (.npz)")
    p.add_argument("--track_field", default="attributes-qwen3_embedding_0.6b",
                   choices=["audio-laion_clap", "image-siglip2", "cf-bpr",
                            "attributes-qwen3_embedding_0.6b",
                            "lyrics-qwen3_embedding_0.6b",
                            "metadata-qwen3_embedding_0.6b"])
    p.add_argument("--ground_truth", default="exp/ground_truth/devset.json")
    p.add_argument("--sample", type=int, default=1000,
                   help="Subsample of (session, turn) pairs to score against. "
                        "Full 8000 takes ~30s; 1000 is plenty for diagnosis.")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 10, 20, 100, 500],
                   help="K values for Hit@K reporting.")
    p.add_argument("--normalize_track", action="store_true", default=True,
                   help="L2-normalize track embeddings (default true).")
    p.add_argument("--no_normalize_track", dest="normalize_track",
                   action="store_false")
    p.add_argument("--center_track", action="store_true",
                   help="Mean-center track embeddings before normalizing.")
    args = p.parse_args()

    print(f"[align] state_emb={args.state_emb}", flush=True)
    bundle = np.load(args.state_emb)
    keys, embs = bundle["keys"], bundle["embeddings"]
    state_by_key: dict[str, np.ndarray] = {
        k.decode(): e for k, e in zip(keys, embs)
    }
    print(f"[align] state vectors: {embs.shape}, "
          f"norm[0]={np.linalg.norm(embs[0]):.4f}",
          flush=True)

    print(f"[align] loading ground truth from {args.ground_truth}", flush=True)
    gt = json.load(open(args.ground_truth))
    gold_by_key: dict[tuple[str, int], str] = {
        (g["session_id"], int(g["turn_number"])): g["ground_truth_track_id"]
        for g in gt
    }

    print(f"[align] loading track field {args.track_field}", flush=True)
    track_emb = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                             split="all_tracks")
    track_ids: list[str] = []
    vecs: list[list[float]] = []
    for r in tqdm(track_emb, desc="indexing tracks"):
        v = r[args.track_field]
        if v and isinstance(v, list) and len(v) > 0:
            track_ids.append(r["track_id"])
            vecs.append(v)
    T = np.asarray(vecs, dtype=np.float32)
    print(f"[align] {T.shape[0]} tracks; raw norm[0]={np.linalg.norm(T[0]):.4f}",
          flush=True)
    if args.center_track:
        T = T - T.mean(axis=0, keepdims=True)
    if args.normalize_track:
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    track_idx = {tid: i for i, tid in enumerate(track_ids)}

    # Sample
    keys_pool = list(gold_by_key)
    if args.sample and args.sample < len(keys_pool):
        rng = np.random.default_rng(42)
        rng.shuffle(keys_pool)
        keys_pool = keys_pool[: args.sample]

    ranks: list[int] = []
    n_skipped = 0
    for sk_t in tqdm(keys_pool, desc="scoring"):
        gold_id = gold_by_key[sk_t]
        sk = f"{sk_t[0]}|{sk_t[1]}"
        if sk not in state_by_key or gold_id not in track_idx:
            n_skipped += 1
            continue
        s = state_by_key[sk]
        # Always L2-normalize the state for fair cosine
        sn = s / (np.linalg.norm(s) + 1e-9)
        scores = T @ sn
        # rank of the gold track
        gold_score = scores[track_idx[gold_id]]
        rank = int(np.sum(scores > gold_score))
        ranks.append(rank)

    if not ranks:
        print("[align] ERROR: no overlapping (state, gold) pairs", file=sys.stderr)
        sys.exit(1)

    ranks_arr = np.asarray(ranks)
    print(f"\n[align] scored {len(ranks_arr)} pairs (skipped {n_skipped})")
    print(f"[align] median rank: {int(np.median(ranks_arr))} / {T.shape[0]}")
    print(f"[align] mean rank:   {int(ranks_arr.mean())}")
    print(f"[align] hit rates:")
    for k in args.ks:
        n = int((ranks_arr < k).sum())
        print(f"  Hit@{k:<4d}: {n:>5d} / {len(ranks_arr)} = {n/len(ranks_arr)*100:5.1f}%")

    # Compare to a random baseline so the user can see the gap clearly
    rand_ranks = np.random.default_rng(123).integers(0, T.shape[0], size=len(ranks_arr))
    print(f"\n[align] random baseline hit rates (uniform):")
    for k in args.ks:
        n = int((rand_ranks < k).sum())
        print(f"  Hit@{k:<4d}: {n:>5d} / {len(rand_ranks)} = {n/len(rand_ranks)*100:5.1f}%")


if __name__ == "__main__":
    main()
