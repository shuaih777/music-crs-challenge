# Music-CRS — RecSys Challenge 2026

Our solution for the [Music Conversational Recommendation Challenge](https://nlp4musa.github.io/music-crs-challenge/) (RecSys Challenge 2026). Team: **shuaih**.

> **Final result**: nDCG@20 = **0.185** on devset (+127% over official baseline), **0.304** on Blind-B test set. Composite score 0.252.

All trained weights are public on Hugging Face: **[shuaih777/music-challenge-models](https://huggingface.co/shuaih777/music-challenge-models)** — see [Inference Pipeline](#inference-pipeline-blind-b--predictionsjson) below to run Blind-B → `predictions.json` without retraining.

---

## Approach

Multi-stage pipeline: **diverse retrieval → learned-to-rank fusion → response generation**.

1. **13 retrieval legs** (BM25, pre-computed embeddings, 6 fine-tuned bi-encoders, PMI co-occurrence, multi-query variants) each produce top-100 candidates
2. **LightGBM LambdaRank** reranks the union candidate pool (~200-400/turn) using per-leg rank, cosine scores, and metadata features
3. **Qwen3-0.6B** generates natural language response explanations

Key finding: **model diversity in retrieval** dominates all other improvements. Each bi-encoder with a different architecture (BGE, E5, Stella, NV-Embed-v2) adds +0.003–0.009 nDCG@20 via complementary recall.

---

## Results

### Development set

| System | nDCG@20 | Hit@20 |
|--------|---------|--------|
| Official baseline (LLaMA-1B + BM25) | 0.082 | — |
| Our BM25 + no-repeat | 0.100 | 20.6% |
| + LightGBM (9 legs) | 0.146 | 30.8% |
| + Bi-encoder (BGE-large) | 0.165 | 35.2% |
| **+ NV-Embed-v2 (13 legs)** | **0.185** | **39.1%** |

### Official Blind-B test set

| Metric | Score |
|--------|-------|
| nDCG@20 | 0.304 |
| Catalog Diversity | 0.031 |
| Lexical Diversity | 0.629 |
| LLM-as-Judge | 1.45 |
| **Composite** | **0.252** |

---

## Reproduce

```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt

# Full reproduction (~12-15h on H100, or ~3h with 4 GPUs)
bash reproduce.sh

# Resume from specific step
bash reproduce.sh --step 3
```

See [REPRODUCE.md](REPRODUCE.md) for detailed instructions.

---

## Inference Pipeline: Blind-B → `predictions.json`

All trained weights (6 bi-encoders + LightGBM reranker + PMI matrix) are published publicly, so you can run the full pipeline on a held-out split **without training anything**:

👉 **[huggingface.co/shuaih777/music-challenge-models](https://huggingface.co/shuaih777/music-challenge-models)**

```
Blind-Dataset-B (talkpl-ai/TalkPlayData-Challenge-Blind-B)
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  13 retrieval legs → top-100 candidates each                │
│  • sparse:  BM25 (no-repeat)                                 │
│  • dense (frozen embeddings): metadata-qwen3, cf_bpr,        │
│    decay_descending                                          │
│  • PMI item-item co-occurrence                                │
│  • 6 fine-tuned bi-encoders: bge-base, bge-large, e5-large,   │
│    stella, mxbai, NV-Embed-v2 (LoRA)                          │
│  • 2 multi-query variants of bge-large (last-2-turns,          │
│    current-utterance-only)                                    │
└────────────────────────────────────────────────────────────┘
        │  union candidate pool (~200-400 tracks/turn)
        ▼
┌────────────────────────────────────────────────────────────┐
│  LightGBM LambdaRank reranker (lgbm_reproduce.txt)            │
│  features: per-leg rank + RRF score, bi-encoder cosine,       │
│  track popularity, turn number, PMI sum, #legs agreeing        │
└────────────────────────────────────────────────────────────┘
        │  top-20 tracks/turn
        ▼
┌────────────────────────────────────────────────────────────┐
│  Qwen3-0.6B response generation (nucleus sampling)             │
└────────────────────────────────────────────────────────────┘
        │
        ▼
   predictions.json   (N sessions × 8 turns, submit to Codabench)
```

### Run it

```bash
pip install -r requirements.txt && pip install -r requirements-gpu.txt

# 1. Pull all pretrained weights (~22GB) instead of retraining
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='shuaih777/music-challenge-models', local_dir='.hf_download')
"
mkdir -p out exp/ltr
mv .hf_download/{biencoder,biencoder_large,e5_large,biencoder_mxbai,biencoder_stella,biencoder_nv_embed} out/
mv .hf_download/reranker/lgbm_reproduce.txt exp/ltr/
mv .hf_download/pmi/* exp/
rm -rf .hf_download

# 2. Run the 13-leg pipeline against Blind-B
bash scripts/run_blind_b_13leg.sh
# → exp/inference/blind_b_13leg/predictions.json
```

`run_blind_b_13leg.sh` mirrors `reproduce.sh` step-for-step (same 13 legs, same reranker) but targets `talkpl-ai/TalkPlayData-Challenge-Blind-B` and skips training — every step is inference-only, so no GPU-hours beyond encoding queries and generating responses.

---

## Repository Structure

```
.
├── src/                         # Core source code
│   ├── baselines_v3.py          # BM25 + dense retrieval engine
│   ├── train_biencoder.py       # Bi-encoder training + inference
│   ├── train_ltr_v2.py          # LightGBM LambdaRank reranker
│   ├── build_item2item.py       # PMI co-occurrence matrix
│   ├── generate_responses.py    # LLM response generation
│   ├── evaluate.py              # Evaluation metrics
│   └── ...                      # Additional utilities
├── scripts/                     # Experimental pipeline scripts
│   └── run_blind_b_13leg.sh     # Full 13-leg pipeline on Blind-B → predictions.json
├── exp/                         # Experiment outputs (inference JSONs, scores)
├── research/                    # Research notes and synthesis
├── reproduce.sh                 # End-to-end reproduction script (devset)
├── REPRODUCE.md                 # Detailed reproduction guide
├── requirements.txt             # CPU dependencies
└── requirements-gpu.txt         # GPU dependencies
```

---

## Key Design Decisions

- **Recall > Ranking**: With 47K tracks, finding the correct item is harder than ranking it. Multiple diverse bi-encoders expand recall from 30% → 39%.
- **LightGBM > Neural rerankers**: Pointwise cross-encoders and listwise LLM rerankers both failed. The "voting" signal (how many legs agree) is stronger than semantic re-scoring.
- **NV-Embed-v2 (7.8B)**: Single largest improvement (+0.011). Larger models capture nuances that 335M models miss.

---

## Citation

```bibtex
@inproceedings{huang2026musiccrs,
  title={Multi-Stage Retrieval with Diverse Bi-Encoders for Conversational Music Recommendation},
  author={Huang, Shuai},
  booktitle={RecSys Challenge Workshop at ACM RecSys},
  year={2026}
}
```

## License

Code: MIT. Dataset belongs to the challenge organizers.
