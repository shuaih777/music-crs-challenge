#!/bin/bash
# =============================================================================
# Generative Retrieval (TalkPlay-style Semantic IDs)
# =============================================================================
# All-in-one: K-Means quantization → training data → LoRA fine-tune → inference
# Produces a retrieval leg for LightGBM integration.
#
# Usage:
#   bash run_generative.sh
#
# Expected: ~2-3h on H100
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

LOG="logs/generative.log"
mkdir -p logs out/generative exp/inference/devset exp/scores/devset exp/ltr

echo "=== Generative retrieval started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 1-4: All-in-one generative retrieval pipeline
# =============================================================================
echo "" | tee -a "$LOG"
echo "Running full generative retrieval pipeline..." | tee -a "$LOG"

GEN_LEG="exp/inference/devset/generative_top100.json"

if [ -f "$GEN_LEG" ]; then
    echo "  [skip] $GEN_LEG exists" | tee -a "$LOG"
else
    PYTHONPATH=src python src/generative_retrieval.py \
        --model_id Qwen/Qwen3-0.6B \
        --output_dir out/generative \
        --out_leg "$GEN_LEG" \
        --n_clusters 256 \
        --n_levels 2 \
        --epochs 2 \
        --batch_size 8 \
        --n_output 100 \
        2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 5: Recall check
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Recall check ======" | tee -a "$LOG"
python -c "
import json

gt = json.load(open('exp/ground_truth/devset.json'))
preds = json.load(open('$GEN_LEG'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}
pred_by_key = {(p['session_id'], p['turn_number']): p['predicted_track_ids'] for p in preds}

hit1 = hit10 = hit20 = hit100 = 0
total = len(gold_by_key)
for key, gold in gold_by_key.items():
    tracks = pred_by_key.get(key, [])
    if gold in tracks[:1]: hit1 += 1
    if gold in tracks[:10]: hit10 += 1
    if gold in tracks[:20]: hit20 += 1
    if gold in tracks[:100]: hit100 += 1

print(f'Generative retrieval standalone:')
print(f'  Hit@1:   {hit1}/{total} = {hit1/total*100:.1f}%')
print(f'  Hit@10:  {hit10}/{total} = {hit10/total*100:.1f}%')
print(f'  Hit@20:  {hit20}/{total} = {hit20/total*100:.1f}%')
print(f'  Hit@100: {hit100}/{total} = {hit100/total*100:.1f}%')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 6: Add to LightGBM
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== LightGBM with generative leg ======" | tee -a "$LOG"

ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           attributes_qwen3 image_siglip2 lyrics_qwen3 state_bm25_focused \
           leg_attributes_qwen3_embedding_0_6b leg_image_siglip2 \
           leg_lyrics_qwen3_embedding_0_6b \
           biencoder_top100 biencoder_large_top100 \
           generative_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    fi
done
echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr_v2.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_generative.txt \
    --out_inference exp/inference/devset/lgbm_generative.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_generative.json \
    --scores exp/scores/devset/lgbm_generative.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 7: Responses + final
# =============================================================================
echo "" | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_generative.json \
    --out exp/inference/devset/lgbm_generative_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_generative_final.json \
    --scores exp/scores/devset/lgbm_generative_final.json \
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
    ('lgbm_v2_final', 'Previous best (LightGBM v2)'),
    ('lgbm_generative', 'With generative leg'),
    ('lgbm_generative_final', 'Generative + responses'),
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
