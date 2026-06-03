# GPU Experiments — what to run on your server

This document lists every experiment that benefits from a GPU, with the exact command, expected runtime, and what to look for in the output. **Everything below works on CPU too** — install only `requirements.txt` and the device autodetect will fall back to numpy.

> **One-time setup:**
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> # GPU only:
> pip install -r requirements-gpu.txt
> # sanity check
> python src/_device.py
> python src/evaluate.py \
>     --inference exp/inference/devset/bm25_norepeat.json \
>     --scores /tmp/sanity.json \
>     --ground_truth exp/ground_truth/devset.json
> ```
> The first script prints the chosen device. The second should print `nDCG@20 ≈ 0.10`.

---

## Speed expectations

| Path | CPU | A100 / 4090 |
|---|---|---|
| One BM25-only inference (8000 turns) | ~1 min | ~30 s |
| One BM25 + dense hybrid inference | ~3-5 min | ~20-40 s |
| `sweep.py --runs all` (8 configs) | ~30 min | ~5 min |
| State-extractor data prep (15k sessions × 8 turns) | ~2 min | ~2 min |
| Train Qwen3-0.6B + LoRA, 1 epoch (~120k examples) | impractical | ~45 min |
| Train Qwen3-4B + LoRA, 1 epoch | impractical | ~2-3 h |
| State-extractor inference on devset (8000 turns) | very slow | ~5-15 min |

(BM25 v3 is ~11x faster than v1/v2 thanks to sparse postings; this applies on
both CPU and GPU.)

---

## Phase A — fast retrieval-config sweep (CPU OK; GPU just faster)

### A1. Run the full retrieval sweep
**Goal:** find the best non-LM retrieval recipe.
```bash
python src/sweep.py --runs all
```
Outputs `exp/inference/devset/<tag>.json`, `exp/scores/devset/<tag>.json`, and a leaderboard.

**What to look for:** `decay_descending`, `last_track`, `metadata_qwen3` should compete with `bm25_norepeat`. Push the leaderboard (`exp/scores/devset/*.json`) back to the repo so we can pick best per turn.

### A2. Per-turn breakdown
After A1, run:
```bash
python -c "
import json, glob
import pandas as pd, numpy as np
from datasets import load_dataset
def ndcg(g, p, k):
    p = p[:k]; dcg = sum((1 if x in g else 0)/np.log2(i+2) for i,x in enumerate(p))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(g), k)))
    return dcg/idcg if idcg else 0
gt = json.load(open('exp/ground_truth/devset.json'))
gtdf = pd.DataFrame(gt)
print(f'{\"tag\":22s}  ' + ' '.join(f't{t}' for t in range(1,9)))
for path in sorted(glob.glob('exp/inference/devset/*.json')):
    tag = path.split('/')[-1].replace('.json','')
    pdf = pd.DataFrame(json.load(open(path)))
    j = gtdf.merge(pdf, on=['session_id','turn_number'])
    j['n20'] = j.apply(lambda r: ndcg([r['ground_truth_track_id']], r['predicted_track_ids'], 20), axis=1)
    pt = j.groupby('turn_number')['n20'].mean()
    print(f'{tag:22s}  ' + ' '.join(f'{pt[t]:.3f}' for t in range(1,9)))
"
```
Different recipes win different turns; the per-turn-best ensemble usually beats every single config.

### A3. Per-turn ensemble
For each turn `t`, find the inference JSON whose row for that `(session, turn)` had the highest `nDCG@20` *on the devset*, and stitch together a final submission. ~15 lines of pandas, no extra inference runs.

---

## Phase B — Conversation-state extractor (the real prize)

This is the big lever for **turns 5-8**. The per-turn data shows BM25-style retrieval collapses after turn 4 because raw history concatenation drowns out the user's *current* preference. A learned summarizer recovers it.

### B1. Build pseudo-labeled training data (CPU, ~2 min)

```bash
python src/data_prep.py --out data/state_extractor_train.jsonl
# also build a small held-out eval set (use the official test split)
python src/data_prep.py --out data/state_extractor_eval.jsonl --split test --max_sessions 200
```

Pseudo-label strategy:
- Target attributes derived from the **ground-truth recommended track's metadata** (genres, moods, era, tags).
- `accepted_tags` = tags of tracks recommended in turns < t (excluding ones the user pushed back on).
- `rejected_tags` = tags of tracks where the next user turn shows pushback (heuristic regex on "no", "different", "skip", "instead", etc.).
- `artist_hints` = artists the user accepted earlier.

### B2. Fine-tune Qwen3-0.6B with LoRA

```bash
# Single A100 / 4090 / 24GB card
python src/train_state_extractor.py \
    --train_jsonl data/state_extractor_train.jsonl \
    --eval_jsonl data/state_extractor_eval.jsonl \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/state_extractor_qwen3_0.6b \
    --epochs 1 \
    --lr 2e-4 \
    --batch_size 8 \
    --grad_accum 2

# Multi-GPU
accelerate config       # one-time
accelerate launch src/train_state_extractor.py [...same args...]
```

**What to look for:** training loss should drop from ~2.5 → ~0.4 within 200 steps, eval loss stable. Typical wall-clock: ~30-60 min on a single A100.

### B3. Try the bigger model
```bash
python src/train_state_extractor.py \
    --train_jsonl data/state_extractor_train.jsonl \
    --eval_jsonl data/state_extractor_eval.jsonl \
    --model_id Qwen/Qwen3-4B \
    --output_dir out/state_extractor_qwen3_4b \
    --batch_size 4 --grad_accum 4 --epochs 1
```
~2-3 h on A100. Larger model = more reliable JSON output and richer attribute extraction.

### B4. Run the trained extractor over devset
```bash
python src/extractor_inference.py \
    --model_dir out/state_extractor_qwen3_0.6b \
    --split test \
    --out exp/states/test.jsonl
```
~5-15 min on GPU.

### B5. Wire it into retrieval
*(We'll add a `--states_jsonl` flag to `baselines_v3.py` once you have B4 outputs.
The flag will replace the raw history concatenation with the structured state's
attributes when forming the BM25 query, e.g. `genre + mood + era + accepted_tags`.)*

Expected gain on turns 5-8: **+30-50% nDCG@20** if the extractor is decent.

---

## Phase C — Bi/cross-encoder reranker (optional)

**Why:** BM25 + dense find candidates; a learned ranker on the fused top-40 should pick the right one more often.

### C1. Bi-encoder rerank
**Hardware:** any CUDA GPU (8GB+).
```bash
# (To be implemented as src/rerank_biencoder.py)
# 1) Load best inference JSON from Phase A
# 2) For each (history, top-40 candidates) pair:
#       q = encoder(history + user_query)
#       k = encoder(track_meta_text)
#       new_scores = w_dense*old + w_rerank*cosine(q, k)
# 3) Re-rank, write new inference JSON, run evaluate.py
```
Use `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` for fast iteration; `BAAI/bge-large-en-v1.5` for final.

### C2. Cross-encoder rerank
~24GB GPU. Use `cross-encoder/ms-marco-MiniLM-L-12-v2`. ~45 min on top-40 × 8000 = 320k pairs.

---

## Phase D — Generative retrieval (paper-worthy, big bet)

The official baseline's `tips/` folder mentions semantic IDs.

### D1. Train RQ-VAE on track embeddings (~3-4h on A100)
Quantize the LAION-CLAP audio (or fused multi-modal) embedding into 4 levels of 256-codeword codebooks. Each track gets a 4-tuple semantic ID.

### D2. Train an encoder-decoder to emit semantic IDs given history (~4-8h)
T5-base or similar. Decode with constrained beam search (only valid 4-tuples).

This is a 1-2 week project but the upper-bound win is large; it's also the most paper-friendly angle.

---

## Recommended sequence for limited GPU time

1. **A1 + A2 + A3** (~5 GPU-min): catches all the easy retrieval wins. Should land **0.11–0.13 nDCG@20**.
2. **B1 + B2 + B4** (~1.5 GPU-h): the only path I see to break through 0.20 on turns 5-8.
3. **C1** (~1 GPU-h): independent bonus.
4. (Stretch) **D**: if you want a paper.

After A1 finishes, please push the `exp/scores/devset/*.json` back so I can pick the right embedding modality and dense-pooling for B5 (the integration step).
