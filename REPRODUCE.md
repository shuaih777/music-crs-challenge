# Reproducing the Final Submission

## Quick Summary

Our best submission (nDCG@20 = 0.185 on devset) uses a 13-leg retrieval ensemble reranked by LightGBM. The full pipeline takes ~12-15h on a single H100 (mostly bi-encoder training), or ~3-4h with 4 GPUs in parallel.

**Skip training entirely**: all 6 bi-encoder checkpoints + the LightGBM reranker + PMI matrix are published at [huggingface.co/shuaih777/music-challenge-models](https://huggingface.co/shuaih777/music-challenge-models). See the [Inference Pipeline section of README.md](README.md#inference-pipeline-blind-b--predictionsjson) to go straight from a held-out split to `predictions.json`.

## Hardware Requirements

- **GPU**: 1× H100 80GB (or A100 80GB). Multiple GPUs recommended for parallel bi-encoder training.
- **Disk**: ~50GB (models + embeddings + inference files)
- **RAM**: 32GB+

## Environment Setup

```bash
# Python 3.12+ with CUDA support
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt

# Key versions:
#   torch >= 2.3
#   transformers >= 4.45, < 5.0  (5.x has compat issues with some models)
#   sentence-transformers == 3.3.1
#   lightgbm >= 4.0
```

## One-Command Reproduction

```bash
bash reproduce.sh
```

This runs all 7 steps sequentially. To resume from a specific step:
```bash
bash reproduce.sh --step 3  # resume from bi-encoder training
```

## Step-by-Step Breakdown

### Step 1: Sparse + Dense Retrieval Legs (~5 min, CPU)

Produces 4 legs using BM25 and pre-computed dataset embeddings:
- `metadata_qwen3`: Dense retrieval via metadata-qwen3 (1024d) embeddings
- `cf_bpr`: Dense retrieval via collaborative filtering BPR (128d)
- `decay_descending`: Same as metadata_qwen3 with decay-weighted pooling
- `bm25_norepeat`: BM25 over track metadata text

### Step 2: PMI Co-occurrence Leg (~2 min, CPU)

Builds item-item PPMI matrix from 15K training session co-occurrences:
- `pmi_leg`: Retrieves tracks co-occurring with prior accepted tracks

### Step 3: Bi-Encoder Training (~8-10h sequential, GPU)

Trains 6 bi-encoders with different base models on 120K (conversation, gold_track) pairs using MultipleNegativesRankingLoss:

| Key | Model | Params | Batch | Time |
|-----|-------|--------|-------|------|
| `biencoder` | BAAI/bge-base-en-v1.5 | 110M | 128 | ~30 min |
| `biencoder_large` | BAAI/bge-large-en-v1.5 | 335M | 64 | ~60 min |
| `e5_large` | intfloat/e5-large-v2 | 335M | 32 | ~90 min |
| `biencoder_stella` | dunzhang/stella_en_400M_v5 | 400M | 64 | ~90 min |
| `biencoder_mxbai` | mixedbread-ai/mxbai-embed-large-v1 | 335M | 32 | ~90 min |
| `biencoder_nv_embed` | nvidia/NV-Embed-v2 | 7.8B | 4 | ~4-6h |

**Note**: NV-Embed-v2 requires `--lora` flag (full fine-tune would OOM). Stella may require `transformers < 5.0` due to custom modeling code.

**Parallel execution** (4 GPUs):
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_biencoder.py all --model_id BAAI/bge-base-en-v1.5 ... &
CUDA_VISIBLE_DEVICES=1 python src/train_biencoder.py all --model_id BAAI/bge-large-en-v1.5 ... &
CUDA_VISIBLE_DEVICES=2 python src/train_biencoder.py all --model_id intfloat/e5-large-v2 ... &
CUDA_VISIBLE_DEVICES=3 python src/train_biencoder.py all --model_id nvidia/NV-Embed-v2 --lora ... &
wait
```

### Step 4: Multi-Query Variants (~10 min, GPU)

Uses the trained BGE-large model with alternative query formulations:
- `biencoder_last2turns_top100`: Only last 2 turns of history as query
- `biencoder_current_only_top100`: Only the current user utterance as query

### Step 5: LightGBM LambdaRank (~5 min, CPU)

Trains LightGBM on union(top-100) from all 13 legs with features:
- Per-leg rank and RRF score
- Raw bi-encoder cosine similarity
- Track popularity, turn number, PMI sum
- Number of legs containing each candidate

### Step 6: Response Generation (~5 min, GPU)

Generates natural language responses with Qwen3-0.6B (nucleus sampling).

### Step 7: Evaluation

Reports nDCG@20, Hit@20, Distinct-2 on the devset.

## Expected Results

```
nDCG@20:    ~0.185  (±0.002 due to random seed in LightGBM CV)
Hit@20:     ~39%
Distinct-2: ~0.26
```

## Key Files

| File | Purpose |
|------|---------|
| `src/baselines_v3.py` | Sparse/dense retrieval legs |
| `src/build_item2item.py` | PMI matrix construction |
| `src/retrieval_legs.py` | PMI retrieval inference |
| `src/train_biencoder.py` | Bi-encoder training + inference |
| `src/train_ltr_v2.py` | LightGBM with cosine features |
| `src/generate_responses.py` | LLM response generation |
| `src/evaluate.py` | Evaluation metrics |
| `reproduce.sh` | End-to-end reproduction script |

## Known Issues

- **Stella (dunzhang/stella_en_400M_v5)**: Based on GTE architecture, may have index-out-of-bounds errors with transformers >= 5.0. Use `transformers==4.45` or skip this leg (impact: -0.003 nDCG@20).
- **NV-Embed-v2**: Requires LoRA training (`--lora` flag). Full fine-tune causes OOM even on H100 80GB.
- **Reproducibility variance**: LightGBM 5-fold CV introduces small random variation (~±0.002 nDCG@20).
