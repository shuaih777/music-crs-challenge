#!/bin/bash
# =============================================================================
# Overnight batch: Train multiple bi-encoders with different base models
# =============================================================================
# Each different base model captures different semantic patterns → different
# recall → LightGBM ensemble benefits.
#
# Proven pattern: every new bi-encoder adds +0.003-0.009 nDCG@20
#   bge-base: +0.009, bge-large: +0.009, E5-large: +0.003
#
# This script trains 4 more bi-encoders overnight, then runs ablation.
# Total time: ~6-8h on H100 (can sleep through it)
#
# Usage:
#   nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
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

LOG="logs/overnight.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

# Parse arguments
SELECTED_MODELS="${1:-all}"
echo "Selected models: $SELECTED_MODELS"
echo "=== Overnight batch started at $(date) ===" | tee "$LOG"
echo "Plan: train 4 bi-encoders + final ablation" | tee -a "$LOG"

# =============================================================================
# Models to train (each ~1-1.5h)
# =============================================================================
declare -A MODELS
MODELS[stella]="dunzhang/stella_en_400M_v5"
MODELS[mxbai]="mixedbread-ai/mxbai-embed-large-v1"
MODELS[arctic]="Snowflake/snowflake-arctic-embed-l-v2.0"
MODELS[nv_embed]="nvidia/NV-Embed-v2"

# =============================================================================
# Train each bi-encoder
# =============================================================================
for KEY in stella mxbai arctic nv_embed; do
    # Skip if not selected
    if [ "$SELECTED_MODELS" != "all" ] && ! echo "$SELECTED_MODELS" | grep -qw "$KEY"; then
        echo "[${KEY}] skipped (not selected)" | tee -a "$LOG"
        continue
    fi

    MODEL_ID="${MODELS[$KEY]}"
    OUT_DIR="out/biencoder_${KEY}"
    OUT_LEG="exp/inference/devset/biencoder_${KEY}_top100.json"

    if [ -f "$OUT_LEG" ]; then
        echo "" | tee -a "$LOG"
        echo "[${KEY}] skip: $OUT_LEG exists" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[${KEY}] Training: $MODEL_ID" | tee -a "$LOG"
    echo "  Started at $(date)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    # Determine batch size based on model size
    BATCH=256
    case "$KEY" in
        stella) BATCH=128 ;;
        mxbai) BATCH=128 ;;
        arctic) BATCH=128 ;;
        nv_embed) BATCH=32 ;;
    esac

    PYTHONPATH=src python src/train_biencoder.py all \
        --model_id "$MODEL_ID" \
        --output_dir "$OUT_DIR" \
        --out_leg "$OUT_LEG" \
        --batch_size "$BATCH" \
        --epochs 3 \
        --n_output 100 \
        2>&1 | tee -a "$LOG"

    echo "[${KEY}] Completed at $(date)" | tee -a "$LOG"

    # Quick recall check
    python -c "
import json
gt = json.load(open('exp/ground_truth/devset.json'))
preds = json.load(open('$OUT_LEG'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}
pred_by_key = {(p['session_id'], p['turn_number']): set(p['predicted_track_ids'][:100]) for p in preds}
hit100 = sum(1 for k, g in gold_by_key.items() if g in pred_by_key.get(k, set()))
print(f'  [{\"$KEY\"}] Hit@100: {hit100}/{len(gold_by_key)} = {hit100/len(gold_by_key)*100:.1f}%')
" 2>&1 | tee -a "$LOG"
done

# =============================================================================
# Final ablation: test combinations
# =============================================================================
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "ABLATION: Testing all combinations" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Collect all available bi-encoder legs
ALL_BIENC_LEGS=""
for KEY in biencoder_large_top100 biencoder_top100 e5_large_top100 \
           biencoder_stella_top100 biencoder_mxbai_top100 \
           biencoder_arctic_top100 biencoder_nv_embed_top100; do
    if [ -f "exp/inference/devset/${KEY}.json" ]; then
        ALL_BIENC_LEGS="${ALL_BIENC_LEGS:+$ALL_BIENC_LEGS,}$KEY"
    fi
done

# Base non-biencoder legs
BASE_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        BASE_LEGS="${BASE_LEGS:+$BASE_LEGS,}$LEG"
    fi
done

echo "Base legs: $BASE_LEGS" | tee -a "$LOG"
echo "Bi-encoder legs: $ALL_BIENC_LEGS" | tee -a "$LOG"

# Test: all legs combined
ALL_LEGS="${BASE_LEGS},${ALL_BIENC_LEGS}"
echo "" | tee -a "$LOG"
echo "[ablation] ALL legs combined: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr_v2.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_overnight_all.txt \
    --out_inference exp/inference/devset/lgbm_overnight_all.json \
    2>&1 | grep -E "positive|fold|Feature" | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_overnight_all.json \
    --scores exp/scores/devset/lgbm_overnight_all.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# Add responses to the best result
echo "" | tee -a "$LOG"
echo "[responses] Adding LLM responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_overnight_all.json \
    --out exp/inference/devset/lgbm_overnight_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_overnight_final.json \
    --scores exp/scores/devset/lgbm_overnight_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Final summary
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== FINAL SUMMARY ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_v2', 'Previous best (0.1686)'),
    ('lgbm_abl_plus_e5', '+ E5-large (from final_boost)'),
    ('lgbm_overnight_all', 'ALL bi-encoders combined'),
    ('lgbm_overnight_final', 'ALL + responses'),
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
echo "=== Overnight batch completed at $(date) ===" | tee -a "$LOG"
