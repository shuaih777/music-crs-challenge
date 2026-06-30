#!/bin/bash
# =============================================================================
# GCRS Generative Retrieval (4-level RQ + Constrained Decoding)
# =============================================================================
# Full pipeline: quantization → structured training data → LoRA fine-tune
# (Qwen3-4B) → constrained inference → standalone eval → LightGBM integration.
#
# Based on: "Generative Conversational Recommender System" (Zhang et al., 2026)
#
# Usage:
#   bash run_generative_gcrs.sh
#
# Expected: ~4-5h on H100 (quantize 10min + data 5min + train 2-3h + infer 1h)
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="logs/generative_gcrs.log"
OUT_DIR="out/generative_gcrs"
LEG_PATH="exp/inference/devset/generative_gcrs_top100.json"
SCORES_DIR="exp/scores/devset"

mkdir -p logs "$OUT_DIR" exp/inference/devset "$SCORES_DIR"

echo "=== GCRS Generative Retrieval started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 1-4: Full GCRS pipeline (quantize + data + train + infer)
# =============================================================================

if [ -f "$LEG_PATH" ]; then
    echo "  [skip] $LEG_PATH exists — skipping pipeline" | tee -a "$LOG"
else
    echo "" | tee -a "$LOG"
    echo "--- Running GCRS pipeline (quantize + data + train + infer) ---" | tee -a "$LOG"
    PYTHONPATH=src python src/generative_retrieval.py \
        --version gcrs \
        --model_id Qwen/Qwen3-4B \
        --output_dir "$OUT_DIR" \
        --out_leg "$LEG_PATH" \
        --n_levels 4 \
        --n_clusters 256 \
        --beam_size 20 \
        --epochs 2.0 \
        --batch_size 4 \
        --lr 2e-4 \
        --split test \
        2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 5: Standalone evaluation
# =============================================================================

echo "" | tee -a "$LOG"
echo "--- Step 5: Standalone evaluation ---" | tee -a "$LOG"

python -c "
import json, sys
sys.path.insert(0, 'src')
from evaluate_recall import evaluate_leg

with open('$LEG_PATH') as f:
    preds = json.load(f)
metrics = evaluate_leg(preds, split='test')
print(f'[GCRS standalone] nDCG@20={metrics[\"ndcg@20\"]:.4f}  Recall@20={metrics[\"recall@20\"]:.4f}  Recall@100={metrics[\"recall@100\"]:.4f}')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 6: Add as leg to LightGBM (with existing best legs) and evaluate
# =============================================================================

echo "" | tee -a "$LOG"
echo "--- Step 6: LightGBM integration ---" | tee -a "$LOG"

# Find existing best legs directory
BEST_LEGS_DIR="exp/inference/devset"
EXISTING_LEGS=""
for leg in biencoder_hardneg_top100.json listwise_top100.json bm25_top100.json; do
    if [ -f "$BEST_LEGS_DIR/$leg" ]; then
        if [ -z "$EXISTING_LEGS" ]; then
            EXISTING_LEGS="$BEST_LEGS_DIR/$leg"
        else
            EXISTING_LEGS="$EXISTING_LEGS,$BEST_LEGS_DIR/$leg"
        fi
    fi
done

if [ -n "$EXISTING_LEGS" ]; then
    ALL_LEGS="$EXISTING_LEGS,$LEG_PATH"
    echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

    PYTHONPATH=src python src/train_ltr_v2.py \
        --legs "$ALL_LEGS" \
        --output_dir "$SCORES_DIR/ltr_with_gcrs" \
        2>&1 | tee -a "$LOG" || echo "  [warn] LightGBM integration failed (non-fatal)" | tee -a "$LOG"
else
    echo "  [skip] No existing legs found for LightGBM integration" | tee -a "$LOG"
    echo "  Run standalone: python src/train_ltr_v2.py --legs $LEG_PATH" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== GCRS Generative Retrieval done at $(date) ===" | tee -a "$LOG"
