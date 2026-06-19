#!/bin/bash
# =============================================================================
# Music-CRS Full Pipeline: LightGBM Reranker + Response Generation
# =============================================================================
# Run on GPU server (or CPU — will be slower for response generation).
# Expects: conda/venv already activated, all deps installed.
#
# Usage:
#   bash run_pipeline.sh
#
# Output:
#   exp/inference/devset/lgbm_reranked.json       (reranked top-20)
#   exp/inference/devset/lgbm_final.json          (+ LLM responses)
#   exp/inference/devset/lgbm_final_template.json (+ template responses, fallback)
#   exp/scores/devset/lgbm_*.json                 (evaluation scores)
#   exp/ltr/lgbm_model.txt                        (saved model)
#   logs/pipeline.log                             (full log)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Step -1: Activate the known working environment
# ---------------------------------------------------------------------------
# This repo has been tested in the foundation_model conda env. Do not load a
# standalone cuda module here; PyTorch gets its CUDA/cuDNN libraries from the
# Python environment/user-site packages on this cluster.
if ! command -v module >/dev/null 2>&1; then
    if [ -f /usr/share/Modules/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/Modules/init/bash
    fi
fi

module load anaconda
conda activate foundation_model
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/pipeline.log"
mkdir -p logs exp/ltr exp/inference/devset exp/scores/devset

echo "=== Music-CRS Pipeline started at $(date) ===" | tee "$LOG"
echo "[env] python=$(which python)" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch
print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[env] gpu_count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
# Step 0: Ensure dependencies
# ---------------------------------------------------------------------------
echo "[step 0] Checking dependencies..." | tee -a "$LOG"
python -c "import lightgbm; print(f'  lightgbm {lightgbm.__version__}')" 2>&1 | tee -a "$LOG" || {
    echo "  Installing lightgbm..." | tee -a "$LOG"
    pip install lightgbm 2>&1 | tail -3 | tee -a "$LOG"
}
python -c "import datasets, pandas, numpy, scipy; print('  core deps OK')" 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 1: LightGBM LambdaMART Reranker
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[step 1] Training LightGBM LambdaMART reranker..." | tee -a "$LOG"
echo "  Legs: metadata_qwen3, cf_bpr, pmi_leg" | tee -a "$LOG"
echo "  Union top-100 candidate pool, 5-fold CV" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs metadata_qwen3,cf_bpr,pmi_leg \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_model.txt \
    --out_inference exp/inference/devset/lgbm_reranked.json \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[step 1b] Evaluating LightGBM reranked output..." | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_reranked.json \
    --scores exp/scores/devset/lgbm_reranked.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 2: Response Generation (template fallback — always runs, instant)
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "[step 2a] Generating template-based responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_reranked.json \
    --out exp/inference/devset/lgbm_final_template.json \
    --mode template \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "[step 2a-eval] Evaluating template responses..." | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_final_template.json \
    --scores exp/scores/devset/lgbm_final_template.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Step 3: Response Generation (LLM — only if GPU available)
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG"
HAS_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$HAS_CUDA" = "True" ]; then
    echo "[step 3] Generating LLM-based responses (Qwen3-0.6B, GPU)..." | tee -a "$LOG"
    PYTHONPATH=src python src/generate_responses.py \
        --inference exp/inference/devset/lgbm_reranked.json \
        --out exp/inference/devset/lgbm_final.json \
        --mode llm \
        --model Qwen/Qwen3-0.6B \
        --batch_size 16 \
        2>&1 | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "[step 3-eval] Evaluating LLM responses..." | tee -a "$LOG"
    PYTHONPATH=src python src/evaluate.py \
        --inference exp/inference/devset/lgbm_final.json \
        --scores exp/scores/devset/lgbm_final.json \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | tee -a "$LOG"
else
    echo "[step 3] SKIPPED — no CUDA detected. Using template responses as final." | tee -a "$LOG"
    cp exp/inference/devset/lgbm_final_template.json exp/inference/devset/lgbm_final.json
    cp exp/scores/devset/lgbm_final_template.json exp/scores/devset/lgbm_final.json 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== Pipeline complete at $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Results summary:" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('ensemble_rrf_pmi3_tuned', 'Previous best (RRF grid)'),
    ('lgbm_reranked', 'LightGBM reranked'),
    ('lgbm_final_template', 'LightGBM + template responses'),
    ('lgbm_final', 'LightGBM + LLM responses'),
]
print(f'{\"Config\":40s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"Distinct-2\":>10s}')
print('-' * 72)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:40s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s.get(\"lexical_diversity\",0):10.4f}')
    else:
        print(f'{desc:40s} {\"(not run)\":>10s}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Key file for submission: exp/inference/devset/lgbm_final.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
