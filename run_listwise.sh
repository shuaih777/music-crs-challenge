#!/bin/bash
# =============================================================================
# Listwise LLM Reranker + bge-large bi-encoder (parallel on 2 GPUs)
# =============================================================================
# GPU 1: Listwise reranker (Qwen3-8B zero-shot on lgbm_biencoder top-20)
# GPU 2: bge-large-en-v1.5 bi-encoder retrain (335M, 45 min)
#
# Run both in parallel:
#   bash run_listwise.sh        # on GPU 1
#   bash run_biencoder_large.sh # on GPU 2 (optional, separate script)
#
# Or sequential:
#   bash run_listwise.sh
#
# Expected: nDCG@20 0.1557 → 0.16-0.18
# Wall-clock: ~2-4h on H100 (mostly listwise inference)
# =============================================================================

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/listwise.log"
mkdir -p logs exp/inference/devset exp/scores/devset

echo "=== Listwise reranker started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 1: Run listwise reranker on lgbm_biencoder output (current best)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Listwise reranking (Qwen3-8B) ======" | tee -a "$LOG"

INPUT="exp/inference/devset/lgbm_biencoder.json"
if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found. Run run_biencoder.sh first." | tee -a "$LOG"
    exit 1
fi

OUTPUT="exp/inference/devset/listwise_reranked.json"
if [ -f "$OUTPUT" ]; then
    echo "  [skip] $OUTPUT already exists" | tee -a "$LOG"
else
    echo "  Input: $INPUT" | tee -a "$LOG"
    echo "  Model: Qwen/Qwen3-8B (zero-shot listwise)" | tee -a "$LOG"
    echo "  This will take ~2-4 hours..." | tee -a "$LOG"

    PYTHONPATH=src python src/listwise_reranker.py \
        --inference "$INPUT" \
        --out "$OUTPUT" \
        --model Qwen/Qwen3-8B \
        --top_k 20 \
        --temperature 0.0 \
        --batch_size 8 \
        2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Evaluate
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Evaluate ======" | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference "$OUTPUT" \
    --scores exp/scores/devset/listwise_reranked.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 3: Add responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: Responses ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "$OUTPUT" \
    --out exp/inference/devset/listwise_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/listwise_final.json \
    --scores exp/scores/devset/listwise_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_biencoder', 'LightGBM + bi-encoder (input)'),
    ('listwise_reranked', 'After listwise reranking'),
    ('listwise_final', 'After listwise + responses'),
]
print(f'{\"\":40s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s} {\"Dist-2\":>7s}')
print('-'*70)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:40s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f} {s.get(\"lexical_diversity\",0):7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
