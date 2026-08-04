#!/bin/bash
# =============================================================================
# bge-large Bi-Encoder (335M) — run on a second GPU in parallel with listwise
# =============================================================================
# Trains a larger bi-encoder (bge-large-en-v1.5, 335M params, 1024d) on the
# same 120k conversation→track pairs. Larger model = better semantic matching
# = more recall.
#
# Run in parallel with run_listwise.sh on a different GPU:
#   CUDA_VISIBLE_DEVICES=0 bash run_listwise.sh &
#   CUDA_VISIBLE_DEVICES=1 bash run_biencoder_large.sh &
#
# Expected wall-clock: ~60-90 min on A100/H100
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

LOG="logs/biencoder_large.log"
mkdir -p logs out/biencoder_large exp/inference/devset exp/scores/devset exp/ltr

echo "=== bge-large bi-encoder started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 1: Train bge-large bi-encoder
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Train bge-large-en-v1.5 (335M) ======" | tee -a "$LOG"

BIENC_LARGE="exp/inference/devset/biencoder_large_top100.json"

if [ -f "$BIENC_LARGE" ]; then
    echo "  [skip] $BIENC_LARGE already exists" | tee -a "$LOG"
else
    PYTHONPATH=src python src/train_biencoder.py all \
        --model_id BAAI/bge-large-en-v1.5 \
        --output_dir out/biencoder_large \
        --out_leg "$BIENC_LARGE" \
        --batch_size 64 \
        --epochs 3 \
        --n_output 100 \
        2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Recall check
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Recall check ======" | tee -a "$LOG"
python -c "
import json

gt = json.load(open('exp/ground_truth/devset.json'))
preds = json.load(open('$BIENC_LARGE'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}
pred_by_key = {(p['session_id'], p['turn_number']): p['predicted_track_ids'] for p in preds}

hit20 = hit50 = hit100 = 0
total = len(gold_by_key)
for key, gold in gold_by_key.items():
    tracks = pred_by_key.get(key, [])
    if gold in tracks[:20]: hit20 += 1
    if gold in tracks[:50]: hit50 += 1
    if gold in tracks[:100]: hit100 += 1

print(f'bge-large standalone recall:')
print(f'  Hit@20:  {hit20}/{total} = {hit20/total*100:.1f}%')
print(f'  Hit@50:  {hit50}/{total} = {hit50/total*100:.1f}%')
print(f'  Hit@100: {hit100}/{total} = {hit100/total*100:.1f}%')

# Compare vs bge-base
try:
    base_preds = json.load(open('exp/inference/devset/biencoder_top100.json'))
    base_by_key = {(p['session_id'], p['turn_number']): set(p['predicted_track_ids'][:100]) for p in base_preds}
    large_by_key = {(p['session_id'], p['turn_number']): set(p['predicted_track_ids'][:100]) for p in preds}

    # How many golds does large find that base misses?
    large_unique = 0
    for key, gold in gold_by_key.items():
        in_large = gold in large_by_key.get(key, set())
        in_base = gold in base_by_key.get(key, set())
        if in_large and not in_base:
            large_unique += 1
    print(f'  Unique golds (large finds, base misses): {large_unique} ({large_unique/total*100:.1f}%)')
except:
    pass
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: LightGBM with both bi-encoders
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: LightGBM (original legs + both bi-encoders) ======" | tee -a "$LOG"

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
echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_biencoder_large.txt \
    --out_inference exp/inference/devset/lgbm_biencoder_large.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_biencoder_large.json \
    --scores exp/scores/devset/lgbm_biencoder_large.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 4: Responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 4: Responses ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_biencoder_large.json \
    --out exp/inference/devset/lgbm_biencoder_large_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_biencoder_large_final.json \
    --scores exp/scores/devset/lgbm_biencoder_large_final.json \
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
    ('lgbm_biencoder', 'bge-base bi-encoder (previous)'),
    ('lgbm_biencoder_large', 'bge-base + bge-large (both)'),
    ('lgbm_biencoder_large_final', 'Both + responses'),
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
