#!/bin/bash
# =============================================================================
# Day 1 v2: Item2Vec + Current-Utterance legs + enhanced LightGBM
# =============================================================================
# Pipeline:
#   1. Train Item2Vec on session sequences + run inference (top-100)
#   2. Run current-utterance BM25 leg (top-100)
#   3. Train LightGBM with ALL legs (existing + item2vec + current_utterance)
#   4. Add LLM responses
#   5. Evaluate and print comparison table
#
# Usage:
#   bash run_day1_v2.sh
#
# Expected wall-clock: ~15-25 min on CPU (no GPU needed for new legs + LightGBM)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi

module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/day1_v2.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

echo "=== Day1 v2 pipeline started at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Item2Vec — train on session sequences + inference
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Item2Vec ======" | tee -a "$LOG"

ITEM2VEC_OUT="exp/inference/devset/item2vec_top100.json"
if [ -f "$ITEM2VEC_OUT" ]; then
    echo "  [skip] $ITEM2VEC_OUT already exists" | tee -a "$LOG"
else
    echo "  [run] Training Item2Vec + inference..." | tee -a "$LOG"
    PYTHONPATH=src python src/build_item2vec.py \
        --out "$ITEM2VEC_OUT" \
        --model_path exp/item2vec_model.bin \
        --n_output 100 \
        --dim 128 \
        --window 5 \
        --epochs 20 \
        2>&1 | tee -a "$LOG"
fi

# (top-100 legs are for LightGBM pool, not direct evaluation)
echo "  [done] Item2Vec leg: $ITEM2VEC_OUT" | tee -a "$LOG"

# =============================================================================
# Step 2: Current-utterance BM25 leg
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Current-utterance BM25 ======" | tee -a "$LOG"

CUR_UTT_OUT="exp/inference/devset/current_utterance_top100.json"
if [ -f "$CUR_UTT_OUT" ]; then
    echo "  [skip] $CUR_UTT_OUT already exists" | tee -a "$LOG"
else
    echo "  [run] Current-utterance BM25 leg..." | tee -a "$LOG"
    PYTHONPATH=src python src/current_utterance_leg.py \
        --out "$CUR_UTT_OUT" \
        --n_output 100 \
        2>&1 | tee -a "$LOG"
fi

# (top-100 legs are for LightGBM pool, not direct evaluation)
echo "  [done] Current-utterance leg: $CUR_UTT_OUT" | tee -a "$LOG"

# =============================================================================
# Step 3: LightGBM with ALL legs (existing top-100 + new legs)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: LightGBM (all legs) ======" | tee -a "$LOG"

# Gather all available top-100 legs
# Existing legs from run_expand_top100.sh:
#   meta_v2_top100, cf_v2_top100, pmi_v2_top100, attr_v2_top100,
#   bm25_v2_top100, last_v2_top100, siglip_v2_top100
# New legs:
#   item2vec_top100, current_utterance_top100

# Build comma-separated leg list from what actually exists
ALL_LEGS=""
for LEG in meta_v2_top100 cf_v2_top100 pmi_v2_top100 attr_v2_top100 \
           bm25_v2_top100 last_v2_top100 siglip_v2_top100 \
           item2vec_top100 current_utterance_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    else
        echo "  [warn] ${LEG}.json not found, skipping" | tee -a "$LOG"
    fi
done

echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_day1_v2.txt \
    --out_inference exp/inference/devset/lgbm_day1_v2.json \
    2>&1 | tee -a "$LOG"

# Evaluate LightGBM
echo "  [eval] LightGBM day1_v2:" | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_day1_v2.json \
    --scores exp/scores/devset/lgbm_day1_v2.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 4: LLM response generation
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 4: LLM response generation ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_day1_v2.json \
    --out exp/inference/devset/day1_v2_final.json \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# Evaluate final
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/day1_v2_final.json \
    --scores exp/scores/devset/day1_v2_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 5: Comparison table
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== RESULTS COMPARISON ======" | tee -a "$LOG"
python -c "
import json, os

configs = [
    ('lgbm_day1_nosiglip', 'Day1 LightGBM (6-leg, no siglip)'),
    ('lgbm_day1_siglip', 'Day1 LightGBM (7-leg, w/ siglip)'),
    ('day1_final', 'Day1 winner + LLM responses'),
    ('item2vec_top100', 'Item2Vec standalone (new)'),
    ('current_utterance_top100', 'Current-utterance BM25 (new)'),
    ('lgbm_day1_v2', 'Day1v2 LightGBM (ALL legs + new feats)'),
    ('day1_v2_final', 'Day1v2 winner + LLM responses'),
]
print(f'{\"\":45s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"nDCG@1\":>8s} {\"Dist-2\":>8s}')
print('-' * 85)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s.get(\"ndcg@1\",0):8.4f} {s.get(\"lexical_diversity\",0):8.4f}')
    else:
        print(f'{desc:45s} {\"(not run)\":>10s}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Submission: exp/inference/devset/day1_v2_final.json" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
