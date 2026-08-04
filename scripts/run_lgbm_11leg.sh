#!/bin/bash
# =============================================================================
# Quick LightGBM: original 9 legs + 2 new legs, NO new features
# =============================================================================
# Uses the PROVEN lgbm_top100 approach (which scored 0.1463) and just adds
# item2vec + current_utterance as extra retrieval legs.
# Does NOT add the cross-features that caused the regression.
#
# Usage:
#   bash run_lgbm_11leg.sh
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

LOG="logs/lgbm_11leg.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

echo "=== LightGBM 11-leg run at $(date) ===" | tee "$LOG"

# Step 1: Ensure item2vec + current_utterance legs exist
ITEM2VEC="exp/inference/devset/item2vec_top100.json"
CUR_UTT="exp/inference/devset/current_utterance_top100.json"

if [ ! -f "$ITEM2VEC" ]; then
    echo "[run] Item2Vec leg..." | tee -a "$LOG"
    PYTHONPATH=src python src/build_item2vec.py \
        --out "$ITEM2VEC" --n_output 100 \
        2>&1 | tail -5 | tee -a "$LOG"
fi

if [ ! -f "$CUR_UTT" ]; then
    echo "[run] Current-utterance leg..." | tee -a "$LOG"
    PYTHONPATH=src python src/current_utterance_leg.py \
        --out "$CUR_UTT" --n_output 100 \
        2>&1 | tail -5 | tee -a "$LOG"
fi

# Step 2: Build leg list from ALL available files
# Original lgbm_top100 used these 9 (some with leg_ prefix):
#   attributes_qwen3, bm25_norepeat, cf_bpr, decay_descending,
#   image_siglip2, lyrics_qwen3, metadata_qwen3, pmi_leg, state_bm25_focused
# Plus new: item2vec_top100, current_utterance_top100

ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           attributes_qwen3 image_siglip2 lyrics_qwen3 state_bm25_focused \
           leg_attributes_qwen3_embedding_0_6b leg_image_siglip2 \
           leg_lyrics_qwen3_embedding_0_6b \
           item2vec_top100 current_utterance_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    fi
done

echo "Legs found: $ALL_LEGS" | tee -a "$LOG"
echo "Leg count: $(echo "$ALL_LEGS" | tr ',' '\n' | wc -l)" | tee -a "$LOG"

# Step 3: Train LightGBM with ORIGINAL params (no new cross-features)
echo "" | tee -a "$LOG"
echo "[train] LightGBM 11-leg (original features + params)..." | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_11leg.txt \
    --out_inference exp/inference/devset/lgbm_11leg.json \
    2>&1 | tee -a "$LOG"

# Step 4: Evaluate
echo "" | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_11leg.json \
    --scores exp/scores/devset/lgbm_11leg.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# Step 5: Responses
echo "" | tee -a "$LOG"
echo "[responses] Adding LLM responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_11leg.json \
    --out exp/inference/devset/lgbm_11leg_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_11leg_final.json \
    --scores exp/scores/devset/lgbm_11leg_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
python -c "
import json, os
for tag, desc in [('lgbm_top100','9-leg (previous best)'), ('lgbm_11leg','11-leg (+ item2vec + utterance)')]:
    p = f'exp/scores/devset/{tag}.json'
    if os.path.exists(p):
        s = json.load(open(p))
        print(f'{desc:35s} nDCG@20={s[\"ndcg@20\"]:.6f} Hit@20={s.get(\"hit@20\",0)*100:.2f}%')
" 2>&1 | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
