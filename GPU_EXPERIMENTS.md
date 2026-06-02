# GPU Experiments — what to run on your server

This document lists every experiment that benefits from a GPU, with the exact command, expected runtime, and what to look for in the output. Each one is independent — pick what to run based on time/budget.

> **Setup once:**
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> pip install -r requirements-gpu.txt           # torch, transformers, etc.
> python src/evaluate.py --inference exp/inference/devset/bm25_norepeat.json \
>     --scores /tmp/sanity.json --ground_truth exp/ground_truth/devset.json
> ```
> If that prints `nDCG@20 ≈ 0.10` you're good.

---

## Phase A — fast sweeps (CPU works, GPU is just faster)

You don't strictly need a GPU for these. But on an A100 the BM25 + dense matmul is `~3 min` per config instead of `~25 min`.

### A1. Run the full retrieval-config sweep
**Goal:** find the best non-LM retrieval recipe.
**Expected runtime:** 8 configs × 25 min CPU = ~3.5 h, or ~30 min on GPU
**Cost:** $0 (no LM calls)

```bash
python src/sweep.py --out_dir exp/inference/devset --runs all
```

Outputs:
- `exp/inference/devset/<tag>.json` per config
- `exp/scores/devset/<tag>.json` per config
- a printed leaderboard at the end

**What to look for:** `metadata_qwen3` and `decay_descending` should both beat the `v2_hybrid` and `bm25_norepeat` baselines we already have. If `cf_bpr` does well, that's a hint for downstream personalization work.

### A2. Per-turn breakdown of every sweep run
After A1 finishes, run:
```bash
python -c "
import json, pandas as pd, numpy as np, glob
from datasets import load_dataset
def ndcg(g, p, k):
    p = p[:k]; dcg = sum((1 if x in g else 0)/np.log2(i+2) for i,x in enumerate(p))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(g), k)))
    return dcg/idcg if idcg else 0
gt = json.load(open('exp/ground_truth/devset.json'))
gtdf = pd.DataFrame(gt)
print(f'{'tag':22s}  ' + ' '.join(f't{t}' for t in range(1,9)))
for path in sorted(glob.glob('exp/inference/devset/*.json')):
    tag = path.split('/')[-1].replace('.json','')
    pdf = pd.DataFrame(json.load(open(path)))
    j = gtdf.merge(pdf, on=['session_id','turn_number'])
    j['n20'] = j.apply(lambda r: ndcg([r['ground_truth_track_id']], r['predicted_track_ids'], 20), axis=1)
    pt = j.groupby('turn_number')['n20'].mean()
    print(f'{tag:22s}  ' + ' '.join(f'{pt[t]:.3f}' for t in range(1,9)))
"
```

This is what reveals the **per-turn winners**. Different recipes win different turns; an ensemble that picks the best-per-turn often beats every single config.

### A3. Per-turn ensemble (after A2)
For each turn `t ∈ {1..8}`, find the single config with highest mean `nDCG@20` on devset turn `t`, then build a final submission whose turn-`t` rows come from that config's predictions. ~3 lines of pandas, no extra runs needed.

---

## Phase B — Bi-encoder reranker (small GPU)

**Why:** BM25 + dense find candidates; a learned ranker on the fused top-40 should pick the right one more often. Even a tiny `all-MiniLM-L6-v2` should help.

### B1. Score top-40 with a sentence-encoder
**Hardware:** any CUDA GPU, even 8GB
**Runtime:** ~15 minutes for all 8000 devset turns
**Cost:** $0 (open model)

```bash
# pseudocode — implement as src/rerank_biencoder.py
# 1) Load best inference JSON from Phase A
# 2) For each (history, top-40 candidates) pair:
#       q = encoder(history + user_query)
#       k = encoder(track_meta_text)
#       new_scores = w_dense*old + w_rerank*cosine(q, k)
# 3) Re-rank, write new inference JSON, run evaluate.py
```

I'd implement this here if I had GPU access. Use `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` for fast iteration; `BAAI/bge-large-en-v1.5` for the final attempt.

### B2. Cross-encoder reranker
Same idea but cross-encode `(history, track_meta)` pairs.
**Hardware:** 24GB GPU (4090/A10) for `cross-encoder/ms-marco-MiniLM-L-12-v2`
**Runtime:** ~45 min for top-40 × 8000 = 320,000 pairs.

---

## Phase C — Conversation-state extractor (the real prize)

This is the big lever for **turns 5-8**. The `REPORT.md` per-turn data shows BM25-style retrieval collapses after turn 4 because raw history concatenation drowns out the user's *current* preference. A learned summarizer recovers it.

### C1. Train a 0.6B–4B LM to extract structured preference state
**Hardware:** 1× A100 80GB (or 2× 4090) with LoRA
**Runtime:** 1 epoch ≈ 45-90 min on Qwen3-0.6B / Qwen3-4B
**Cost:** GPU time only

```bash
# Skeleton in src/train_state_extractor.py — finish the data pipeline:
python src/train_state_extractor.py \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/state_extractor_0.6b \
    --epochs 1 \
    --lr 2e-4 \
    --lora
```

The labeling strategy I recommend:
1. Use `conversation_goal.listener_goal` as a soft target for session-level intent.
2. Tags from accepted tracks → `accepted_tags`.
3. Tags from tracks where the user pushed back (sentiment classifier on the next turn) → `rejected_tags`.
4. Train autoregressively on the structured-output format (see `train_state_extractor.py`).

Or, alternatively (cleaner): **distill** by labeling 5k training conversations with Claude/Gemini and fine-tuning Qwen on those. That probably wins the leaderboard.

### C2. Use the extractor at inference
After C1, swap the `tokenize(full_query)` step in `baselines_v3.py` with:
```python
state = extractor.generate(history, user_query)   # returns {genre, mood, era, tags, ...}
qt = tokenize(state.to_query_string())  # e.g. "moody 70s rock soft introspective"
```
Re-run `sweep.py`. Expected gain: **+30-50% on turn 5-8 nDCG@20** if the extractor is decent.

---

## Phase D — Generative retrieval (paper-worthy)

The `tips/` folder of the official baseline mentions semantic IDs.

### D1. Train RQ-VAE on track embeddings
**Hardware:** 1× A100, ~3-4 hours.
Quantize the LAION-CLAP audio (or fused multi-modal) embedding into 4 levels of 256-codeword codebooks. Each track gets a 4-tuple semantic ID.

### D2. Train an encoder-decoder to emit semantic IDs given history
**Hardware:** 1× A100, 4-8 hours.
Trains end-to-end recommendation as text → 4-token sequence. T5-base or similar. Decode with constrained beam search (only valid 4-tuples).

This is a 1-2 week project but the upper-bound win is large; it's also the most paper-friendly.

---

## What I'd prioritize given limited GPU time

1. **A1 + A3** (~ 4 GPU-hours): catches all the easy wins from configurable retrieval. Should land in the **0.11-0.13 nDCG@20** range.
2. **B1 bi-encoder rerank** (~ 1 GPU-hour): independent boost on top of A.
3. **C1 state extractor** (~ 4-12 GPU-hours): the only path I see to get above 0.20 on turns 5-8.
4. (Stretch) **D**: only if you want a paper.

A's results dictate B and C — please share the leaderboard from A1 first and I'll tune the reranker and extractor design accordingly.
