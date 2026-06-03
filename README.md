# Music-CRS — RecSys Challenge 2026

A clean, **CPU-or-GPU** pipeline for the [Music Conversational Recommendation Challenge](https://nlp4musa.github.io/music-crs-challenge/) (RecSys Challenge 2026).

> **Status (CPU-only, devset 1000 sessions × 8 turns):** the pure-numpy BM25 + no-repeat baseline already **beats the official LLaMA-1B+BM25 baseline** on retrieval (`nDCG@20 = 0.100` vs official `0.0815`). v3's sparse-postings BM25 is ~11× faster than v1 (~1 min instead of ~11 min on a laptop). All numbers reproducible from this repo.

---

## Auto CPU/GPU

Every script in `src/` runs on CPU out of the box. If `torch` is installed and a CUDA / MPS device is visible, the heavy linear-algebra steps (BM25 sparse score-add, dense matmul, LM forward/backward) move to GPU automatically. No code changes required:

```bash
python src/_device.py
# torch installed: True
# [device] auto -> cuda (1 device(s); NVIDIA A100-SXM4-80GB)
# selected device: cuda
```

Force a specific device:
```bash
python src/baselines_v3.py --device cuda  ...
python src/baselines_v3.py --device cpu   ...
MUSIC_CRS_DEVICE=cuda python src/baselines_v3.py ...
```

---

## What's in here

```
.
├── src/
│   ├── _device.py                  # CPU/GPU autodetect (zero-config)
│   ├── baselines.py                # v1, original pure-numpy BM25
│   ├── baselines_v2.py             # v2 hybrid (kept for reference)
│   ├── baselines_v3.py             # ⭐ main retrieval; all knobs as CLI flags
│   ├── evaluate.py                 # self-contained evaluator
│   ├── sweep.py                    # run + score 8 retrieval recipes
│   ├── data_prep.py                # build pseudo-labels for the state extractor
│   ├── train_state_extractor.py    # GPU: LoRA fine-tune Qwen3-0.6B/4B
│   └── extractor_inference.py      # GPU: run trained extractor on devset
├── exp/
│   ├── ground_truth/devset.json
│   ├── inference/devset/           # saved predictions for every model
│   └── scores/devset/              # macro nDCG / diversity per model
├── REPORT.md                       # data investigation + per-turn analysis
├── GPU_EXPERIMENTS.md              # full experiment plan with commands
├── requirements.txt                # CPU-only deps
└── requirements-gpu.txt            # add these on the GPU box
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
# 1. install
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. reproduce the best non-LM baseline (~1 min on a laptop with v3)
python src/baselines_v3.py \
    --output exp/inference/devset/bm25_norepeat_repro.json \
    --bm25_only --tag bm25_norepeat

# 3. score it
python src/evaluate.py \
    --inference exp/inference/devset/bm25_norepeat_repro.json \
    --scores exp/scores/devset/bm25_norepeat_repro.json \
    --ground_truth exp/ground_truth/devset.json
```

Expected: `nDCG@20 ≈ 0.100`, beating the official LLaMA-1B+BM25 baseline (0.0815).

### Sweep multiple configs

```bash
python src/sweep.py --runs all
```

Runs 8 retrieval recipes (pure BM25, BM25+no-repeat, hybrid with mean / decay / last / last-k pooling, different embedding modalities, etc.) and prints a leaderboard.

---

## Quick start (GPU server)

```bash
# 1. install GPU extras
pip install -r requirements.txt
pip install -r requirements-gpu.txt

# 2. confirm GPU is detected
python src/_device.py

# 3. fast retrieval sweep — 5 minutes instead of 30
python src/sweep.py --runs all

# 4. train conversation-state extractor (~45 min on A100)
python src/data_prep.py --out data/state_extractor_train.jsonl
python src/data_prep.py --out data/state_extractor_eval.jsonl --split test --max_sessions 200
python src/train_state_extractor.py \
    --train_jsonl data/state_extractor_train.jsonl \
    --eval_jsonl data/state_extractor_eval.jsonl \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/state_extractor_qwen3_0.6b

# 5. run the trained extractor on devset
python src/extractor_inference.py \
    --model_dir out/state_extractor_qwen3_0.6b \
    --split test \
    --out exp/states/test.jsonl
```

Full experiment plan with expected runtimes: [`GPU_EXPERIMENTS.md`](GPU_EXPERIMENTS.md).

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

For Blind A/B inference (server-side eval), point `--split` at the right dataset:

```bash
python src/baselines_v3.py --split Blind-A --output sub_blindA.json --bm25_only
python src/baselines_v3.py --split Blind-B --output sub_blindB.json --bm25_only
```

Submit to [Codabench](https://www.codabench.org/competitions/15786/) for blind-set scoring. Final challenge deadline: **2026-06-30**.

---

## Useful upstream repos

- [`nlp4musa/music-crs-baselines`](https://github.com/nlp4musa/music-crs-baselines) — official BM25 + LLaMA-1B baseline (needs GPU + flash-attn)
- [`nlp4musa/music-crs-evaluator`](https://github.com/nlp4musa/music-crs-evaluator) — official evaluator (this repo's `src/evaluate.py` is compatible)

```bash
git clone --depth 1 https://github.com/nlp4musa/music-crs-baselines
git clone --depth 1 https://github.com/nlp4musa/music-crs-evaluator
```

---

## Datasets

All HuggingFace, no auth required:

- `talkpl-ai/TalkPlayData-Challenge-Dataset` — 15,199 train + 1,000 dev sessions, each 8 turns
- `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` — 47,071 tracks
- `talkpl-ai/TalkPlayData-Challenge-User-Metadata` — 8,772 users
- `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` — 6 modalities per track (LAION-CLAP audio 512d, SigLIP2 image 768d, BPR CF 128d, Qwen3 attributes/lyrics/metadata 1024d each)
- `talkpl-ai/TalkPlayData-Challenge-User-Embeddings`
- `talkpl-ai/TalkPlayData-Challenge-Blind-A` (released)
- `talkpl-ai/TalkPlayData-Challenge-Blind-B` (released **2026-06-15**)

---

## License

This repo's code: MIT. Dataset and official baselines belong to the challenge organizers.
