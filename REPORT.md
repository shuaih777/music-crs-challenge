# Music-CRS Challenge 2026 — Local Investigation Report

> Working directory: `/home/l131341/projects/llm/llm_rec/`
> Last update: full devset sweep + ensemble results

---

## TL;DR

The best non-LM submission is `ensemble_rrf_eq` (RRF fusion of `metadata_qwen3` and `cf_bpr` retrievals): **`nDCG@20 = 0.1241, Hit@20 = 26.2%`** on the devset. That's **+52% over the official LLaMA-1B+BM25 baseline** (0.0815). All compute on this side is plain numpy — no LLM, no GPU required.

---

## Final leaderboard (devset, 1000 sessions × 8 turns = 8000 inferences)

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Hit@20 | CatDiv | LexDiv |
|---|---:|---:|---:|---:|---:|---:|
| Random (official ref) | 0.000 | 0.0001 | 0.0001 | — | 0.965 | 0.000 |
| Popularity (official ref) | 0.0005 | 0.0018 | 0.0024 | — | 0.0004 | 0.000 |
| **LLaMA-1B + BM25 (official ref)** | 0.0098 | 0.0627 | **0.0815** | — | 0.379 | 0.255 |
| my BM25 v1 (clean reimpl, no LLM) | 0.014 | 0.060 | 0.076 | 19.2% | 0.456 | 0.129 |
| BM25 + no-repeat | 0.036 | 0.086 | 0.100 | 20.6% | 0.436 | 0.090 |
| BM25 + LAION-CLAP audio + no-repeat (v2) | 0.040 | 0.097 | 0.113 | 23.8% | 0.482 | 0.099 |
| `early_dense_only` (CLAP, dense only t≤4) | 0.037 | 0.090 | 0.106 | 22.2% | 0.479 | 0.098 |
| `last_track` (CLAP, last-accepted-track query) | 0.039 | 0.093 | 0.108 | 22.4% | 0.548 | 0.114 |
| `decay_descending` (CLAP, decayed prior) | 0.039 | 0.095 | 0.112 | 23.5% | 0.510 | 0.103 |
| `cf_bpr` (BPR collab-filtering as dense) | 0.037 | 0.097 | 0.114 | 24.5% | 0.540 | 0.108 |
| `metadata_qwen3` (Qwen3 text emb of metadata) | 0.041 | 0.104 | 0.121 | 25.3% | 0.486 | 0.097 |
| `ensemble_per_turn` (`meta` t1-6 + `cf` t7-8) | 0.041 | 0.105 | 0.123 | 25.8% | 0.501 | 0.102 |
| `ensemble_rrf_3way` (meta+cf+CLAP, RRF) | 0.042 | 0.107 | 0.124 | 26.0% | 0.492 | 0.097 |
| **`ensemble_rrf_eq` (meta+cf, equal-weight RRF)** | **0.043** | **0.106** | **0.124** | **26.2%** | 0.516 | 0.097 |

---

## Per-turn nDCG@20 (this is where the strategy lives)

| | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 v1 | 0.134 | 0.119 | 0.083 | 0.066 | 0.060 | 0.058 | 0.046 | 0.044 | 0.076 |
| BM25 + no-repeat | 0.134 | 0.142 | 0.107 | 0.090 | 0.087 | 0.090 | 0.074 | 0.075 | 0.100 |
| `early_dense_only` | 0.134 | 0.149 | 0.134 | 0.103 | 0.087 | 0.090 | 0.074 | 0.075 | 0.106 |
| `last_track` | 0.134 | 0.130 | 0.124 | 0.098 | 0.106 | 0.103 | 0.086 | 0.086 | 0.108 |
| `decay_descending` (CLAP) | 0.134 | 0.130 | 0.134 | 0.103 | 0.109 | 0.108 | 0.091 | 0.087 | 0.112 |
| `cf_bpr` | 0.134 | 0.142 | 0.113 | 0.111 | 0.108 | 0.111 | **0.099** | **0.096** | 0.114 |
| `metadata_qwen3` | 0.134 | **0.171** | 0.139 | **0.115** | **0.115** | **0.114** | 0.089 | 0.089 | 0.121 |
| `ensemble_per_turn` | 0.134 | 0.171 | 0.139 | 0.115 | 0.115 | 0.114 | 0.099 | 0.096 | 0.123 |
| `ensemble_rrf_eq` | 0.134 | 0.166 | 0.140 | 0.116 | 0.114 | 0.118 | 0.096 | 0.092 | 0.124 |

Three observations that drive next steps:

1. **`metadata_qwen3` is the new BM25.** Replacing raw history-text BM25 with Qwen3-1024d text embeddings of track metadata gives ~+20% over `bm25_norepeat` essentially for free (just a different `--embed`). The qwen3 text emb dominates everywhere except the very last turns.
2. **`cf_bpr` is the late-turn fallback.** It's the only retriever where t7/t8 are *not* much worse than mid-turns — collab-filtering captures user-cluster preferences that are stable even when the conversation drifts. This is the cleanest signal that turns 5-8 are best handled by a different mechanism than turns 1-4.
3. **All retrievers tie at t1.** Turn 1 has explicit text queries ("play X by Y") so it's a metadata-text problem, BM25-class. The interesting work is t2-8.

---

## What's been built (CPU-only, all reproducible from this repo)

```
src/
├── _device.py                  # CPU/GPU autodetect; lazy torch import
├── baselines_v3.py             # ⭐ main retrieval (sparse postings BM25 + dense)
├── ensemble.py                 # per-turn / RRF combine of multiple inference JSONs
├── evaluate.py                 # self-contained evaluator (matches official one)
├── sweep.py                    # run + score 8 retrieval recipes
├── data_prep.py                # build pseudo-labels for the state extractor
├── train_state_extractor.py    # GPU: LoRA fine-tune Qwen3-0.6B/4B
├── extractor_inference.py      # GPU: run trained extractor over devset
├── baselines.py / baselines_v2.py  # earlier versions kept for reference
└── ...

exp/inference/devset/   ← 11 inference JSONs, one per config
exp/scores/devset/      ← matching score JSONs

REPORT.md (this file), README.md, GPU_EXPERIMENTS.md
```

### Key compute facts
- v3 BM25 with sparse postings is **~11× faster on CPU** than v1/v2 (~1 min vs ~11 min on the devset). Same numerical output.
- When `torch` is installed and CUDA is visible, BM25 sparse score-add and dense matmul both move to GPU automatically; no flags needed.

### Submission ready
- `exp/inference/devset/ensemble_rrf_eq.json` is the current best devset submission (`nDCG@20 = 0.124`, `Hit@20 = 26.2%`).
- For Blind-A/B, the same recipe works — just point `--split Blind-A` at `baselines_v3.py` and re-run the sweep on that split.

---

## What's next (priority order)

### 1. **Train the conversation-state extractor (Phase B)** — biggest expected win
This is the only path I see to break above ~0.13 on turns 7-8. The data-prep heuristic, the LoRA training script, and the inference glue are all written and committed:

```bash
python src/data_prep.py --out data/state_extractor_train.jsonl
python src/data_prep.py --out data/state_extractor_eval.jsonl --split test --max_sessions 200
python src/train_state_extractor.py \
    --train_jsonl data/state_extractor_train.jsonl \
    --eval_jsonl data/state_extractor_eval.jsonl \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/state_extractor_qwen3_0.6b
python src/extractor_inference.py \
    --model_dir out/state_extractor_qwen3_0.6b \
    --split test \
    --out exp/states/test.jsonl
```

After that I'll add `--states_jsonl` to `baselines_v3.py` so the BM25 query becomes the structured state's `genre + mood + era + accepted_tags + artist_hints` instead of raw history concatenation. Expected gain: another **+30-50% on turns 5-8**, which would lift macro nDCG@20 to **~0.16-0.18**.

### 2. **Bi-encoder reranker on top-40** (independent +5-10%)
Use `BAAI/bge-small-en-v1.5` to score (history+user_query) against (track metadata text) over the fused top-40 candidates. ~15 min on any modest GPU.

### 3. **Wider RRF ensemble + grid search**
Try RRF with all retrievers (5-way), tune RRF k and per-tag weights against devset. Probably only +1-2% but cheap.

### 4. **Generative retrieval (Phase D)** — paper-worthy, big bet
RQ-VAE on multi-modal track embeddings → seq2seq emitting semantic IDs. 1-2 weeks of work; upper-bound win is large but uncertain.

---

## Files in `exp/inference/devset/`

Each is the standard 8000-row submission JSON. You can load any of these, evaluate it, or use it as a leg in a wider ensemble.

| Tag | What it is |
|---|---|
| `random` | uniform 20-from-catalog |
| `popularity` | static top-20 by popularity score (always same row) |
| `bm25_v1`, `bm25_v1_user` | original v1 baselines |
| `bm25_pure` | v3 BM25 without any filter |
| `bm25_norepeat` | v3 BM25 + no-repeat filter |
| `last_track` | hybrid: BM25 + dense via last accepted track only |
| `early_dense_only` | hybrid: dense for t≤4, BM25 only after |
| `decay_descending` | hybrid: decayed-prior, descending dense weight |
| `v2_hybrid` | v2-style: mean-pool + ascending dense weight |
| `cf_bpr` | hybrid using BPR CF embeddings |
| `metadata_qwen3` | hybrid using Qwen3-1024d metadata text embeddings |
| `ensemble_per_turn` | `metadata_qwen3` (t1-6) + `cf_bpr` (t7-8) |
| `ensemble_rrf_eq` | RRF fusion of `metadata_qwen3` + `cf_bpr` (equal weight) |
| `ensemble_rrf_3way` | RRF of `metadata_qwen3` + `cf_bpr` + `decay_descending` |

To recompute scores for any:
```bash
python src/evaluate.py \
    --inference exp/inference/devset/<tag>.json \
    --scores exp/scores/devset/<tag>.json \
    --ground_truth exp/ground_truth/devset.json
```
