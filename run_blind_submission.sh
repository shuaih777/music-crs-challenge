#!/bin/bash
# =============================================================================
# Blind-B Submission with LightGBM scoring (best approach)
# =============================================================================
# Uses lgbm_top100_model.txt trained on devset to score Blind-B candidates.
# The model learned meta-rules about which legs to trust — these generalize.
#
# Required legs (matching lgbm_top100 training):
#   attributes_qwen3, bm25_norepeat, cf_bpr, decay_descending,
#   image_siglip2, lyrics_qwen3, metadata_qwen3, pmi_leg, state_bm25_focused
#
# Usage:
#   bash run_blind_submission.sh
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

SPLIT="Blind-B"
OUT_DIR="exp/inference/blind_b"
LOG="logs/blind_b_submission.log"
mkdir -p logs "$OUT_DIR"

echo "=== Blind-B LightGBM submission at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Run all 8 retrieval legs on Blind-B (n_output=100)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Running retrieval legs on $SPLIT (n_output=100)..." | tee -a "$LOG"

run_leg() {
    local TAG="$1"; shift
    local OUT="${OUT_DIR}/${TAG}.json"
    if [ -f "$OUT" ]; then
        echo "  [skip] $TAG" | tee -a "$LOG"; return
    fi
    echo "  [run] $TAG" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" --split "$SPLIT" --n_output 100 "$@" \
        2>&1 | grep -E "Wrote|sessions" | tee -a "$LOG"
}

# Must match the feature names in lgbm_top100_model.txt (sorted order):
run_leg "attributes_qwen3" \
    --embed attributes-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag attr

run_leg "bm25_norepeat" --bm25_only --tag bm25

run_leg "cf_bpr" \
    --embed cf-bpr --pooling decay --weight_schedule descending --tag cf

run_leg "decay_descending" \
    --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag decay

run_leg "image_siglip2" \
    --embed image-siglip2 --pooling decay --weight_schedule descending --tag siglip

run_leg "lyrics_qwen3" \
    --embed lyrics-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag lyrics

run_leg "metadata_qwen3" \
    --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag meta

# PMI leg
PMI_OUT="${OUT_DIR}/pmi_leg.json"
if [ ! -f "$PMI_OUT" ]; then
    echo "  [run] pmi_leg" | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, numpy as np
from scipy import sparse
from datasets import load_dataset
from tqdm import tqdm

pmi = sparse.load_npz('exp/item2item_pmi.npz')
tracks = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_ids = [r['track_id'] for r in tracks]
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-${SPLIT}', split='test')
rows = []
for ex in tqdm(test, desc='pmi'):
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
print(f'Wrote {len(rows)} rows')
" 2>&1 | tail -2 | tee -a "$LOG"
else
    echo "  [skip] pmi_leg" | tee -a "$LOG"
fi

# State BM25 (skip if no states — generates empty predictions as placeholder)
STATE_OUT="${OUT_DIR}/state_bm25_focused.json"
if [ ! -f "$STATE_OUT" ]; then
    echo "  [run] state_bm25_focused (BM25-only fallback, no states)" | tee -a "$LOG"
    # Just use BM25 as a proxy — same as bm25_norepeat for this purpose
    cp "${OUT_DIR}/bm25_norepeat.json" "$STATE_OUT" 2>/dev/null || \
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$STATE_OUT" --split "$SPLIT" --n_output 100 --bm25_only --tag state_bm25 \
        2>&1 | grep "Wrote" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Score with LightGBM model
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Scoring with LightGBM (lgbm_top100_model)..." | tee -a "$LOG"

LEGS="attributes_qwen3,bm25_norepeat,cf_bpr,decay_descending,image_siglip2,lyrics_qwen3,metadata_qwen3,pmi_leg,state_bm25_focused"

PYTHONPATH=src python src/score_blind.py \
    --model exp/ltr/lgbm_top100_model.txt \
    --legs "$LEGS" \
    --inference_dir "$OUT_DIR" \
    --out "${OUT_DIR}/lgbm_scored.json" \
    --pmi_path exp/item2item_pmi.npz \
    --split "$SPLIT" \
    --top_k 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: Add LLM responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Generating responses..." | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/lgbm_scored.json" \
    --out "${OUT_DIR}/submission.json" \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 4: Validate
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Validating..." | tee -a "$LOG"
python -c "
import json
with open('${OUT_DIR}/submission.json') as f:
    rows = json.load(f)
n = len(rows)
sessions = len(set(r['session_id'] for r in rows))
has_resp = sum(1 for r in rows if r.get('predicted_response'))
dupes = sum(1 for r in rows if len(set(r['predicted_track_ids'])) != len(r['predicted_track_ids']))
print(f'✓ {n} rows, {sessions} sessions, {n//sessions} turns/session')
print(f'  Responses: {has_resp}/{n}')
print(f'  Duplicate track_ids: {dupes} rows')
print(f'  Avg tracks/row: {sum(len(r[\"predicted_track_ids\"]) for r in rows)/n:.1f}')
if dupes == 0 and n == sessions * 8:
    print(f'  FORMAT OK — ready for Codabench')
else:
    print(f'  ⚠️ CHECK FORMAT')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "SUBMIT: ${OUT_DIR}/submission.json" | tee -a "$LOG"
echo "  → https://www.codabench.org/competitions/15786/" | tee -a "$LOG"
