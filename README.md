# Music-CRS — RecSys Challenge 2026

A clean, GPU-optional pipeline for the [Music Conversational Recommendation Challenge](https://nlp4musa.github.io/music-crs-challenge/) (RecSys Challenge 2026).

> **Status (CPU-only, 1000 devset sessions):** the pure-numpy BM25 + no-repeat baseline already **beats the official LLaMA-1B+BM25 baseline** on retrieval (`nDCG@20 = 0.100` vs official `0.0815`). All numbers reproducible from this repo on a laptop in ~15 minutes per config.

---

## What's in here

```
.
├── src/
│   ├── baselines_v3.py        # Hybrid BM25 + dense retrieval, fully configurable
│   ├── evaluate.py            # Self-contained evaluator (matches official one)
│   ├── sweep.py               # Run + score 8 retrieval recipes, print a leaderboard
│   └── train_state_extractor.py   # Skeleton for GPU work (Phase C)
├── exp/
│   ├── ground_truth/devset.json
│   ├── inference/devset/      # Saved predictions for every model in the table below
│   └── scores/devset/         # Macro nDCG / diversity for each
├── REPORT.md                  # The investigation memo (what I learned about the data)
├── GPU_EXPERIMENTS.md         # All experiments that need GPU + commands + expected runtimes
├── requirements.txt           # CPU-only deps
└── requirements-gpu.txt       # Add these on the GPU box
```

---

## Results so far (devset, 8000 (session, turn) pairs)

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog div | Lexical div |
|---|---:|---:|---:|---:|---:|
| Random (official ref) | 0.000 | 0.0001 | 0.0001 | 0.965 | 0.000 |
| Popularity (official ref) | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.000 |
| **LLaMA-1B + BM25 (official ref)** | 0.0098 | 0.0627 | **0.0815** | 0.379 | 0.255 |
| my BM25 (clean reimpl, no LLM) | 0.014 | 0.060 | 0.076 | 0.456 | 0.129 |
| my BM25 + user-history blend | 0.012 | 0.055 | 0.072 | 0.426 | 0.121 |
| **my BM25 + no-repeat filter** | **0.036** | **0.086** | **0.100** | 0.436 | 0.077 |
| my BM25 + LAION-CLAP audio + no-repeat | 0.040 | 0.087 | 0.097 | 0.553 | 0.089 |

The full per-turn table and analysis is in [REPORT.md](REPORT.md).

---

## Quick start (CPU)

```bash
# 1. install deps
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. reproduce the best non-LM baseline (~15 min on a laptop)
python src/baselines_v3.py \
    --output exp/inference/devset/bm25_norepeat_repro.json \
    --bm25_only --tag bm25_norepeat

# 3. score it
python src/evaluate.py \
    --inference exp/inference/devset/bm25_norepeat_repro.json \
    --scores exp/scores/devset/bm25_norepeat_repro.json \
    --ground_truth exp/ground_truth/devset.json
```

Expected:
```
nDCG@1   ≈ 0.036
nDCG@10  ≈ 0.086
nDCG@20  ≈ 0.100   <-- beats the official LLaMA-1B+BM25 baseline (0.0815)
```

### Sweep multiple configs at once

```bash
python src/sweep.py --runs all
```

This runs 8 retrieval recipes (pure BM25, BM25+no-repeat, hybrid with mean / decay / last / last-k pooling, different embedding modalities, etc.) and prints a leaderboard.

---

## What needs GPU

See [`GPU_EXPERIMENTS.md`](GPU_EXPERIMENTS.md). Briefly:

| Phase | What | Hardware | Time | Why |
|---|---|---|---|---|
| **A** | Run `src/sweep.py --runs all` | any (CPU OK; GPU 5× faster) | 30 min – 4 h | Find the best non-LM config |
| **B** | Bi/cross-encoder reranker on top-40 | 1× 8-24 GB GPU | 15-45 min | Independent +5-10% lift |
| **C** | Conversation-state extractor (LoRA fine-tune Qwen3-0.6B/4B) | 1× A100 | 1-2 h | Fix the turn 5-8 collapse — this is the real prize |
| **D** | Generative retrieval with semantic IDs (RQ-VAE + seq2seq) | 1× A100 | 1-2 days | Paper-worthy, large-but-uncertain win |

Recommended order: **A → C** (skip B if time is tight; C dominates B in expected value).

---

## Submission format (verbatim from challenge)

```json
[
  {"session_id": "<uuid>",
   "user_id": "<uuid>",
   "turn_number": 1,
   "predicted_track_ids": ["track_id1", "track_id2", "...", "track_id20"],
   "predicted_response": "How about ..."}
]
```

8000 entries (1000 sessions × 8 turns). One missing `(session_id, turn_number)` and the eval fails. `predicted_track_ids` must be unique within a row, ordered by relevance, ≤20 entries, all from the catalog.

Submit to [Codabench](https://www.codabench.org/competitions/15786/) for blind-set scoring. Final challenge deadline: **2026-06-30**.

---

## Useful upstream repos (clone these too if you want the official baseline)

- [`nlp4musa/music-crs-baselines`](https://github.com/nlp4musa/music-crs-baselines) — BM25 + LLaMA-1B baseline (needs GPU + flash-attn)
- [`nlp4musa/music-crs-evaluator`](https://github.com/nlp4musa/music-crs-evaluator) — official evaluator (this repo's `src/evaluate.py` is compatible)

```bash
git clone --depth 1 https://github.com/nlp4musa/music-crs-baselines
git clone --depth 1 https://github.com/nlp4musa/music-crs-evaluator
```

---

## Dataset

All five HuggingFace datasets:

- `talkpl-ai/TalkPlayData-Challenge-Dataset` — 15,199 train + 1,000 dev sessions, each 8 turns
- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` — 47,071 tracks
- `talkpl-ai/TalkPlayData-Challenge-User-Metadata` — 8,772 users
- `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` — 6 embedding modalities per track (LAION-CLAP audio 512d, SigLIP2 image 768d, BPR CF 128d, three Qwen3 text 1024d each)
- `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` — pre-computed user embeddings
- `talkpl-ai/TalkPlayData-Challenge-Blind-A` — blind set A (released)
- `talkpl-ai/TalkPlayData-Challenge-Blind-B` — blind set B (released **2026-06-15**)

For blind-set submissions you'll need to re-run inference using the appropriate split; `baselines_v3.py` currently hard-codes the `test` split — change `convo["test"]` to load `Blind-A` / `Blind-B` instead.

---

## License

This repo's code: MIT. Dataset and official baselines belong to the challenge organizers.
