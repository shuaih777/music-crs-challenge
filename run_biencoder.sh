#!/bin/bash
# =============================================================================
# Bi-Encoder Fine-tuning + LightGBM Integration
# =============================================================================
# Trains a bi-encoder (bge-base-en-v1.5) on 120k conversation→track pairs,
# then uses it as a new retrieval leg in LightGBM.
#
# Usage:
#   bash run_biencoder.sh
#
# Expected wall-clock: ~60-75 min on H100/A100
# Expected result: nDCG@20 0.146 → 0.155-0.18 (if bi-encoder adds unique recall)
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

LOG="logs/biencoder.log"
mkdir -p logs out/biencoder exp/inference/devset exp/scores/devset exp/tracks exp/ltr data

echo "=== Bi-Encoder pipeline started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 1: Train bi-encoder (build data + train + encode tracks + inference)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Bi-Encoder training + inference ======" | tee -a "$LOG"

BIENC_LEG="exp/inference/devset/biencoder_top100.json"

if [ -f "$BIENC_LEG" ]; then
    echo "  [skip] $BIENC_LEG already exists" | tee -a "$LOG"
else
    PYTHONPATH=src python src/train_biencoder.py all \
        --model_id BAAI/bge-base-en-v1.5 \
        --output_dir out/biencoder \
        --out_leg "$BIENC_LEG" \
        --batch_size 256 \
        --epochs 3 \
        --n_output 100 \
        2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Quick standalone recall check (how many golds does bi-encoder find?)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Bi-encoder recall check ======" | tee -a "$LOG"
python -c "
import json

gt = json.load(open('exp/ground_truth/devset.json'))
preds = json.load(open('$BIENC_LEG'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}
pred_by_key = {(p['session_id'], p['turn_number']): p['predicted_track_ids'] for p in preds}

hit20 = hit50 = hit100 = 0
total = len(gold_by_key)
for key, gold in gold_by_key.items():
    tracks = pred_by_key.get(key, [])
    if gold in tracks[:20]: hit20 += 1
    if gold in tracks[:50]: hit50 += 1
    if gold in tracks[:100]: hit100 += 1

print(f'Bi-encoder standalone recall:')
print(f'  Hit@20:  {hit20}/{total} = {hit20/total*100:.1f}%')
print(f'  Hit@50:  {hit50}/{total} = {hit50/total*100:.1f}%')
print(f'  Hit@100: {hit100}/{total} = {hit100/total*100:.1f}%')

# Check how many golds are UNIQUE to bi-encoder (not in existing legs)
existing_legs = []
for leg in ['metadata_qwen3', 'cf_bpr', 'pmi_leg', 'decay_descending', 'bm25_norepeat']:
    try:
        existing_legs.append(json.load(open(f'exp/inference/devset/{leg}.json')))
    except: pass

if existing_legs:
    existing_by_key = {}
    for leg_data in existing_legs:
        for p in leg_data:
            key = (p['session_id'], p['turn_number'])
            if key not in existing_by_key:
                existing_by_key[key] = set()
            existing_by_key[key].update(p['predicted_track_ids'][:100])

    unique_hits = 0
    for key, gold in gold_by_key.items():
        bienc_tracks = set(pred_by_key.get(key, [])[:100])
        existing_tracks = existing_by_key.get(key, set())
        if gold in bienc_tracks and gold not in existing_tracks:
            unique_hits += 1
    print(f'  Unique golds (in bi-encoder but NOT in existing legs): {unique_hits} ({unique_hits/total*100:.1f}%)')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: LightGBM with bi-encoder as additional leg
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: LightGBM (original legs + bi-encoder) ======" | tee -a "$LOG"

# Build leg list: original 9 + biencoder
ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           attributes_qwen3 image_siglip2 lyrics_qwen3 state_bm25_focused \
           leg_attributes_qwen3_embedding_0_6b leg_image_siglip2 \
           leg_lyrics_qwen3_embedding_0_6b \
           biencoder_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    fi
done
echo "  Legs: $ALL_LEGS" | tee -a "$LOG"
echo "  Count: $(echo "$ALL_LEGS" | tr ',' '\n' | wc -l)" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_biencoder.txt \
    --out_inference exp/inference/devset/lgbm_biencoder.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_biencoder.json \
    --scores exp/scores/devset/lgbm_biencoder.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 4: Add LLM responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 4: LLM responses ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_biencoder.json \
    --out exp/inference/devset/lgbm_biencoder_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_biencoder_final.json \
    --scores exp/scores/devset/lgbm_biencoder_final.json \
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
    ('lgbm_top100', 'Previous best (9 legs, no bi-encoder)'),
    ('lgbm_biencoder', 'With bi-encoder (+1 leg)'),
    ('lgbm_biencoder_final', 'With bi-encoder + LLM responses'),
]
print(f'{\"\":45s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s} {\"Dist-2\":>7s}')
print('-'*75)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f} {s.get(\"lexical_diversity\",0):7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "If improved: rerun on Blind-B with --split Blind-B in inference step" | tee -a "$LOG"
