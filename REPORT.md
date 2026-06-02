# Music-CRS Challenge 2026 — Local Investigation Report (v2)

> All work in `/home/l131341/projects/llm/llm_rec/`
> Last update: progress on conversational retrieval

---

## TL;DR — leaderboard at a glance

All evaluated on the official devset (1000 sessions × 8 turns = 8000 inferences).
"Reference" rows are official baselines from `music-crs-evaluator/exp/scores/devset/`.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div | Lexical div | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|
| Random (ref) | 0.000 | 0.0001 | 0.0001 | 0.965 | 0.000 | — |
| Popularity (ref) | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.000 | — |
| **LLaMA-1B + BM25 (ref)** | **0.0098** | **0.0627** | **0.0815** | **0.379** | **0.255** | — |
| my BM25 | 0.014 | 0.060 | 0.076 | 0.456 | 0.129 | 19.2% |
| my BM25 + user-history blend | 0.012 | 0.055 | 0.072 | 0.426 | 0.121 | — |
| **my BM25 + no-repeat** | **0.036** | **0.086** | **0.100** | **0.436** | **0.077** | — |
| my BM25 + CLAP audio + no-repeat (hybrid) | 0.040 | 0.087 | 0.097 | 0.553 | 0.089 | 18.8% |

**Headline:** the no-repeat filter alone drives nDCG@20 from 0.076 → **0.100 (+31%)**, beating the official LLaMA-1B+BM25 (0.0815). Adding LAION-CLAP audio retrieval gains the early turns but loses some later ones — net wash on overall nDCG@20.

---

## Per-turn dynamics (the real story)

| Turn | BM25 | + no-repeat | + hybrid (CLAP) |
|---:|---:|---:|---:|
| 1 | 0.134 | 0.134 | 0.134 |
| 2 | 0.119 | 0.142 | **0.150** |
| 3 | 0.083 | 0.107 | **0.126** |
| 4 | 0.066 | 0.090 | **0.091** |
| 5 | 0.061 | **0.087** | 0.077 |
| 6 | 0.058 | **0.090** | 0.074 |
| 7 | 0.047 | **0.074** | 0.069 |
| 8 | 0.044 | **0.075** | 0.056 |

Read this carefully — there's a **crossover at turn 4**:

- **Turns 2-4**: CLAP audio embedding of accepted tracks adds real signal. The user has just said "yes I like this" → similarity to that track's audio fingerprint helps.
- **Turns 5-8**: Mean-pooling 4-7 prior tracks drifts the query embedding into a vague centroid that no longer reflects the user's *current* preference. The conversation has moved on; the audio prior is stale.

This is exactly the conversational-state problem I flagged in v1. The fix is **decay-weighted prior** (recent accepted tracks count more) or, better, a **conversation extractor** that picks which prior tracks are "still relevant" based on the user's last 1-2 utterances.

---

## What v2 added vs v1

### `src/baselines_v2.py` — three changes
1. **No-repeat filter**: skip any track that already appeared as `role: music` in the same session before this turn. Free win — golds never repeat in this dataset.
2. **CLAP-audio dense retrieval**: mean-pool the LAION-CLAP audio embeddings of all prior accepted tracks → cosine sim against the 47k catalog → top-200.
3. **RRF fusion** of BM25 top-200 and CLAP top-200, with **turn-aware weighting** (turn 2 = 0.6 BM25 + 0.4 dense; turn 8 = 0.3 BM25 + 0.7 dense). This was clearly wrong — the late-turn results show dense should be *down*-weighted, not up.

### Extra dataset learnings
- The track-embeddings dataset has **6 modalities per track**: `audio-laion_clap` (512d), `image-siglip2` (768d), `cf-bpr` (128d, BPR collaborative-filtering), `attributes-qwen3` (1024d), `lyrics-qwen3` (1024d), `metadata-qwen3` (1024d).
- ~1% of tracks (492/47071) have **zero-length audio embeddings** — must be handled (current code zero-pads them, they always rank 0).
- The `cf-bpr` modality is essentially a free user-affinity signal that I haven't tried yet — could be the right ingredient for a learned-CF baseline.

---

## What to do next (ranked by ROI)

### Immediate fixes (1-2 hours each)
1. **Reverse the dense weight schedule** — try BM25-heavy at turn 8, dense-heavy at turn 2. Or just `min(0.5, 0.6 - 0.05 * (tn-2))`.
2. **Decay-weighted prior**: instead of mean-pool, use exponential decay so the most recent accepted track dominates. `q = sum(0.7^(t-i) * emb_i)`.
3. **Last-track-only prior**: ablation — just use the most recent accepted track's embedding as query. Would isolate whether mean-pooling is the problem.
4. **Try `metadata-qwen3` instead of CLAP**: text embeddings of track metadata might align better with the textual conversation history than audio fingerprints.

### Medium (half-day)
5. **Turn-1 fix**: BM25 nDCG@20 plateaus at 0.134 on turn 1. Add a stemmer + better entity recognition for "by Artist", song titles in quotes, etc. Inspect failed turn-1 cases.
6. **Field-weighted BM25 (BM25F)**: separate weights per metadata field (track_name 5×, artist_name 4×, album 2×, tags 1×).
7. **Reranker over top-40**: small cross-encoder re-ranks fused candidates using full conversation + track metadata + tags + audio. Even a small all-MiniLM-L6 should help.

### Big (day or more)
8. **Train conversation-state extractor on the 15k train sessions**: a Qwen3-0.6B fine-tuned to emit `(genre, mood, era, energy, accepted_tags, rejected_tags)` from history. Use this structured query instead of raw history concatenation. This is the real unlock for turns 5-8.
9. **Two-tower model**: text encoder for history × track encoder fusing CLAP+metadata-qwen3+tags. Train with positives = ground-truth tracks, in-batch + BM25-hard negatives.
10. **Use `cf-bpr` for personalization**: per-user BPR vector → linear blend with conversational scores. Great for cold ranking when conversation is generic.

---

## Files

```
llm_rec/
├── .venv/                          # Python 3.12 + offline-installed deps
├── music-crs-baselines/            # Cloned official baseline
├── music-crs-evaluator/
│   ├── exp/inference/devset/
│   │   ├── my_random.json          # 8000 random
│   │   ├── my_popularity.json      # static top-20 by popularity
│   │   ├── my_bm25.json            # v1 BM25 (matches official LLaMA-1B+BM25)
│   │   ├── my_bm25_user.json       # v1 + user-history blend
│   │   ├── my_v2_norepeat.json     # v2 BM25 + no-repeat (BEST for turns 5-8)
│   │   └── my_v2_hybrid.json       # v2 BM25 + CLAP audio + no-repeat
│   └── exp/scores/devset/          # all eval JSONs
└── src/
    ├── baselines.py                # v1 BM25 (clean, dependency-free)
    └── baselines_v2.py             # v2 hybrid (BM25 + dense + no-repeat)
```

Replicate any v2 strategy:
```bash
source .venv/bin/activate
PYTHONPATH=src python src/baselines_v2.py \
  --output music-crs-evaluator/exp/inference/devset/<tid>.json \
  [--no_dense] [--no_filter] [--embed audio-laion_clap|metadata-qwen3_embedding_0.6b|...]
cd music-crs-evaluator && python evaluate_devset.py --tid <tid>
```

---

## My current recommendation

For an early Codabench submission to lock in a leaderboard spot, I'd submit **`my_v2_norepeat`** (BM25 + no-repeat) — it's already substantially above the official baseline and computes in 11 minutes on CPU. Then iterate on the quick fixes (1-4) above to push past 0.10 nDCG@20.

The thing actually worth building over the next 1-2 weeks is the conversation-state extractor (item 8). The per-turn data above shows clearly that the bottleneck after turn 4 is "the model has no theory of what the user currently wants" — and that's solvable with 15k labeled training sessions plus a small LM.
