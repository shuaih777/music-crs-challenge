# Reproducibility — Environment (captured 2026-06-30)

Final submission (Blind-B nDCG@20 = 0.25) was produced with this environment.
Full lock file: `requirements-lock.txt` (pip freeze).

## Core
- Python 3.11.10
- CUDA 12.4, GPU: NVIDIA A100 80GB PCIe
- torch==2.4.1+cu124

## Key ML packages (exact)
| package | version |
|---|---|
| transformers | 5.12.1 |
| sentence-transformers | 5.6.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 1.20.1 |
| datasets | 5.0.0 |
| accelerate | 1.14.0 |
| peft | 0.19.1 |
| lightgbm | 4.6.0 |
| xformers | 0.0.28.post1 |
| numpy | 2.2.6 |
| scipy | 1.16.0 |
| pandas | 2.2.3 |
| joblib | 1.5.3 |

## Version-sensitive notes (gotchas hit during this work)
- **xformers must match torch**: torch 2.4.1 → `xformers==0.0.28.post1`. Install with
  `pip install --no-deps xformers==0.0.28.post1` to avoid torch being upgraded.
  Required only by stella (`dunzhang/stella_en_400M_v5`).
- **stella vs transformers 5.x**: stella's custom modeling code crashes on
  transformers 5.12.1 (unpad_inputs / token_type_embeddings index error).
  stella legs in this repo were produced on a separate machine running
  transformers 4.x. Not reproducible under 5.x without patching stella.
- **Trainer API**: transformers 5.x uses `processing_class=` (not `tokenizer=`).
