#!/bin/bash
# =============================================================================
# LightGBM v2: with bi-encoder cosine scores as continuous feature
# =============================================================================
# The v1 LightGBM only used 5 trees (80% on n_legs_containing) because
# discrete rank features are too coarse. Adding raw cosine similarity
# gives the model a continuous signal to split on → more trees → better.
#
# Usage:
#   bash run_ltr_v2.sh
#
# Expected: ~20-30 min (includes bi-encoder scoring of all candidates)
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

LOG="logs/ltr_v2.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

echo "=== LightGBM v2 (with cosine scores) at $(date) ===" | tee "$LOG"

# Build leg list
ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           attributes_qwen3 image_siglip2 lyrics_qwen3 state_bm25_focused \
           leg_attributes_qwen3_embedding_0_6b leg_image_siglip2 \
           leg_lyrics_qwen3_embedding_0_6b \
           biencoder_top100 biencoder_large_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    fi
done
echo "Legs: $ALL_LEGS" | tee -a "$LOG"

# Find bi-encoder model dir
BIENC_DIR=""
for d in out/biencoder_large out/biencoder; do
    if [ -d "$d" ]; then
        BIENC_DIR="$d"
        break
    fi
done
echo "Bi-encoder: $BIENC_DIR" | tee -a "$LOG"

# Run LightGBM v2
PYTHONPATH=src python src/train_ltr_v2.py \
    --legs "$ALL_LEGS" \
    --biencoder_dir "$BIENC_DIR" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_v2.txt \
    --out_inference exp/inference/devset/lgbm_v2.json \
    --batch_size 256 \
    2>&1 | tee -a "$LOG"

# Evaluate
echo "" | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_v2.json \
    --scores exp/scores/devset/lgbm_v2.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# Responses
echo "" | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_v2.json \
    --out exp/inference/devset/lgbm_v2_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_v2_final.json \
    --scores exp/scores/devset/lgbm_v2_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_biencoder_large_final', 'v1 (rank features only)'),
    ('lgbm_v2', 'v2 (+ cosine score feature)'),
    ('lgbm_v2_final', 'v2 + responses'),
]
print(f'{\"\":35s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s} {\"Dist-2\":>7s}')
print('-'*65)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:35s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f} {s.get(\"lexical_diversity\",0):7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
