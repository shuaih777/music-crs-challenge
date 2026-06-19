#!/bin/bash
# =============================================================================
# Cross-encoder LoRA training + reranking pipeline
# =============================================================================
# Total time on A100 80GB: ~45-60 min
#
# Usage:
#   bash run_crossencoder.sh
# =============================================================================

set -euo pipefail

# Environment
if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/crossencoder.log"
mkdir -p logs data out exp/inference/devset exp/scores/devset

echo "=== Cross-encoder pipeline started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build training data (~2 min)
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 1] Building cross-encoder training data..." | tee -a "$LOG"
PYTHONPATH=src python src/train_crossencoder.py build_data \
    --out data/crossencoder_train.jsonl \
    --n_neg 7 \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train Qwen3-0.6B + LoRA (~30 min on A100)
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 2] Training Qwen3-0.6B cross-encoder (LoRA r=16)..." | tee -a "$LOG"
PYTHONPATH=src python src/train_crossencoder.py train \
    --train_jsonl data/crossencoder_train.jsonl \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/crossencoder_qwen3_0.6b \
    --epochs 1 \
    --batch_size 32 \
    --lr 2e-4 \
    --max_seq_len 512 \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Rerank the best LightGBM output (~15 min)
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 3] Reranking with trained cross-encoder..." | tee -a "$LOG"

# Pick the best available LightGBM output
BEST_INPUT=""
for f in exp/inference/devset/lgbm_8leg.json \
         exp/inference/devset/lgbm_6leg.json \
         exp/inference/devset/lgbm_reranked.json \
         exp/inference/devset/ensemble_rrf_pmi3_tuned.json; do
    if [ -f "$f" ]; then
        BEST_INPUT="$f"
        break
    fi
done

if [ -z "$BEST_INPUT" ]; then
    echo "  ERROR: no suitable input found for reranking" | tee -a "$LOG"
    exit 1
fi
echo "  Input: $BEST_INPUT" | tee -a "$LOG"

PYTHONPATH=src python src/train_crossencoder.py rerank \
    --model_dir out/crossencoder_qwen3_0.6b \
    --inference "$BEST_INPUT" \
    --out exp/inference/devset/crossencoder_trained.json \
    --top_k 20 \
    --batch_size 32 \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 4] Evaluating..." | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/crossencoder_trained.json \
    --scores exp/scores/devset/crossencoder_trained.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Add LLM responses
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 5] Generating LLM responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/crossencoder_trained.json \
    --out exp/inference/devset/crossencoder_final.json \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/crossencoder_final.json \
    --scores exp/scores/devset/crossencoder_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== Results ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('ensemble_rrf_pmi3_tuned', 'RRF baseline'),
    ('lgbm_reranked', 'LightGBM 3-leg'),
    ('lgbm_8leg', 'LightGBM 8-leg'),
    ('crossencoder_trained', 'Cross-encoder (no response)'),
    ('crossencoder_final', 'Cross-encoder + LLM response'),
]
print(f'{\"Config\":35s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"Dist-2\":>8s}')
print('-' * 65)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:35s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s.get(\"lexical_diversity\",0):8.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "Submission file: exp/inference/devset/crossencoder_final.json" | tee -a "$LOG"
