#!/bin/bash
# =============================================================================
# Day 1: Turn-1 fix + top-100 legs + LightGBM (with/without siglip2 ablation)
# =============================================================================
# Does everything in one shot:
#   1. Re-runs legs with fixed tokenizer + n_output=100 (no lyrics)
#   2. Trains two LightGBMs: with and without image_siglip2
#   3. Picks the winner, adds LLM responses
#
# Usage:
#   bash run_expand_top100.sh
#
# Expected wall-clock: ~30-40 min on H100
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

LOG="logs/day1_expand.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

echo "=== Day 1 expand started at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Re-run core legs with n_output=100 (fixed tokenizer auto-applied)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Running legs with n_output=100 + fixed tokenizer..." | tee -a "$LOG"

run_leg() {
    local TAG="$1"
    shift
    local OUT="exp/inference/devset/${TAG}.json"
    if [ -f "$OUT" ]; then
        echo "  [skip] $TAG" | tee -a "$LOG"
        return
    fi
    echo "  [run] $TAG" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --n_output 100 \
        "$@" \
        2>&1 | { grep -E "dense used|state query|no-repeat|Wrote" || true; } | tee -a "$LOG"
}

# Core legs (no lyrics — confirmed harmful by TalkPlay paper)
run_leg "meta_v2_top100" \
    --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag meta_v2

run_leg "cf_v2_top100" \
    --embed cf-bpr --pooling decay --weight_schedule descending --tag cf_v2

run_leg "attr_v2_top100" \
    --embed attributes-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag attr_v2

run_leg "bm25_v2_top100" \
    --bm25_only --tag bm25_v2

run_leg "last_v2_top100" \
    --embed metadata-qwen3_embedding_0.6b --pooling last --weight_schedule descending --tag last_v2

# Image-siglip2 (we'll test with and without)
run_leg "siglip_v2_top100" \
    --embed image-siglip2 --pooling decay --weight_schedule descending --tag siglip_v2

# PMI with top-100
PMI_OUT="exp/inference/devset/pmi_v2_top100.json"
if [ ! -f "$PMI_OUT" ]; then
    echo "  [run] pmi_v2_top100" | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, numpy as np
from scipy import sparse
from datasets import load_dataset
from tqdm import tqdm

pmi = sparse.load_npz('exp/item2item_pmi.npz')
tracks = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_ids = [r['track_id'] for r in tracks]
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')
rows = []
for ex in tqdm(test, desc='pmi_top100'):
    for tn in range(1, 9):
        prior = [c['content'] for c in ex['conversations'] if c['role']=='music' and c['turn_number'] < tn]
        prior_idx = [track_to_idx[t] for t in prior if t in track_to_idx]
        if prior_idx:
            scores = np.asarray(pmi[prior_idx].sum(axis=0)).flatten()
        else:
            scores = np.zeros(len(track_ids), dtype=np.float32)
        for idx in prior_idx:
            scores[idx] = -1e9
        top_indices = np.argsort(-scores)[:100]
        preds = [track_ids[i] for i in top_indices]
        rows.append({'session_id': ex['session_id'], 'user_id': ex['user_id'],
                     'turn_number': int(tn), 'predicted_track_ids': preds,
                     'predicted_response': ''})
with open('${PMI_OUT}', 'w') as f:
    json.dump(rows, f, ensure_ascii=False)
print(f'Wrote {len(rows)} to ${PMI_OUT}')
" 2>&1 | tail -2 | tee -a "$LOG"
else
    echo "  [skip] pmi_v2_top100" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: LightGBM — two variants (with and without siglip2)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Training LightGBM variants..." | tee -a "$LOG"

# Variant A: without siglip2
LEGS_NO_SIGLIP="meta_v2_top100,cf_v2_top100,pmi_v2_top100,attr_v2_top100,bm25_v2_top100,last_v2_top100"
echo "  [2a] LightGBM WITHOUT siglip2: $LEGS_NO_SIGLIP" | tee -a "$LOG"
PYTHONPATH=src python src/train_ltr.py \
    --legs "$LEGS_NO_SIGLIP" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_day1_nosiglip.txt \
    --out_inference exp/inference/devset/lgbm_day1_nosiglip.json \
    2>&1 | { grep -E "positive|fold|Feature|importance" || true; } | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_day1_nosiglip.json \
    --scores exp/scores/devset/lgbm_day1_nosiglip.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | { grep -E "ndcg@20|hit@20" || true; } | tee -a "$LOG"

# Variant B: with siglip2
LEGS_WITH_SIGLIP="meta_v2_top100,cf_v2_top100,pmi_v2_top100,attr_v2_top100,bm25_v2_top100,last_v2_top100,siglip_v2_top100"
echo "" | tee -a "$LOG"
echo "  [2b] LightGBM WITH siglip2: $LEGS_WITH_SIGLIP" | tee -a "$LOG"
PYTHONPATH=src python src/train_ltr.py \
    --legs "$LEGS_WITH_SIGLIP" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_day1_siglip.txt \
    --out_inference exp/inference/devset/lgbm_day1_siglip.json \
    2>&1 | { grep -E "positive|fold|Feature|importance" || true; } | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_day1_siglip.json \
    --scores exp/scores/devset/lgbm_day1_siglip.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | { grep -E "ndcg@20|hit@20" || true; } | tee -a "$LOG"

# =============================================================================
# Step 3: Pick winner + add responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Picking winner + generating responses..." | tee -a "$LOG"

BEST=$(python -c "
import json
a = json.load(open('exp/scores/devset/lgbm_day1_nosiglip.json'))['ndcg@20']
b = json.load(open('exp/scores/devset/lgbm_day1_siglip.json'))['ndcg@20']
print('lgbm_day1_siglip' if b > a else 'lgbm_day1_nosiglip')
")
echo "  Winner: $BEST" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "exp/inference/devset/${BEST}.json" \
    --out exp/inference/devset/day1_final.json \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/day1_final.json \
    --scores exp/scores/devset/day1_final.json \
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
    ('lgbm_top100', 'Previous best (top-100, old tokenizer)'),
    ('lgbm_day1_nosiglip', 'Day1 NO siglip2 (new tokenizer)'),
    ('lgbm_day1_siglip', 'Day1 WITH siglip2 (new tokenizer)'),
    ('day1_final', 'Day1 winner + LLM responses'),
]
print(f'{\"\":45s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"nDCG@1\":>8s} {\"Dist-2\":>8s}')
print('-' * 85)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s[\"ndcg@1\"]:8.4f} {s.get(\"lexical_diversity\",0):8.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Submission: exp/inference/devset/day1_final.json" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
