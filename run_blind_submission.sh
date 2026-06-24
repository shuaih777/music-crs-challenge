#!/bin/bash
# =============================================================================
# Blind-B Submission Pipeline
# =============================================================================
# Runs the full pipeline on Blind-B data to produce a Codabench submission.
# Uses the same 8-leg LightGBM approach that scored 0.1463 on devset.
#
# NOTE: LightGBM is trained on DEVSET ground truth (since we don't have
# Blind-B labels). The model generalizes because it learns which legs to
# trust, not track-specific patterns.
#
# Usage:
#   bash run_blind_submission.sh
#
# Output:
#   exp/inference/blind_b/submission.json  (for Codabench upload)
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

SPLIT="Blind-B"
OUT_DIR="exp/inference/blind_b"
LOG="logs/blind_b_submission.log"
mkdir -p logs "$OUT_DIR" exp/ltr

echo "=== Blind-B submission pipeline started at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Run retrieval legs on Blind-B (n_output=100)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Running retrieval legs on $SPLIT..." | tee -a "$LOG"

run_leg() {
    local TAG="$1"
    shift
    local OUT="${OUT_DIR}/${TAG}.json"
    if [ -f "$OUT" ]; then
        echo "  [skip] $TAG" | tee -a "$LOG"
        return
    fi
    echo "  [run] $TAG" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --split "$SPLIT" \
        --n_output 100 \
        "$@" \
        2>&1 | grep -E "dense used|Wrote|sessions" | tee -a "$LOG"
}

# Same legs as lgbm_top100 (our best devset config)
run_leg "metadata_qwen3" \
    --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag meta

run_leg "cf_bpr" \
    --embed cf-bpr --pooling decay --weight_schedule descending --tag cf

run_leg "attributes_qwen3" \
    --embed attributes-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag attr

run_leg "image_siglip2" \
    --embed image-siglip2 --pooling decay --weight_schedule descending --tag siglip

run_leg "decay_descending" \
    --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag decay

run_leg "bm25_only" \
    --bm25_only --tag bm25

run_leg "last_track" \
    --embed metadata-qwen3_embedding_0.6b --pooling last --weight_schedule descending --tag last

# State BM25 (needs states for Blind-B — skip if not available)
STATES_PATH="exp/states/blind_b.jsonl"
if [ -f "$STATES_PATH" ]; then
    run_leg "state_bm25" \
        --bm25_only --states_jsonl "$STATES_PATH" --query_mode state --tag state_bm25
else
    echo "  [skip] state_bm25 — no Blind-B states available" | tee -a "$LOG"
fi

# PMI leg for Blind-B
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
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Blind-B', split='test')
rows = []
for ex in tqdm(test, desc='pmi_blind_b'):
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
" 2>&1 | tail -3 | tee -a "$LOG"
else
    echo "  [skip] pmi_leg" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Apply LightGBM (trained on devset) to Blind-B candidates
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Applying LightGBM reranker to Blind-B..." | tee -a "$LOG"

# Build the list of available legs
AVAILABLE_LEGS=""
for leg in metadata_qwen3 cf_bpr pmi_leg attributes_qwen3 image_siglip2 decay_descending bm25_only last_track state_bm25; do
    if [ -f "${OUT_DIR}/${leg}.json" ]; then
        if [ -z "$AVAILABLE_LEGS" ]; then
            AVAILABLE_LEGS="$leg"
        else
            AVAILABLE_LEGS="${AVAILABLE_LEGS},${leg}"
        fi
    fi
done
echo "  Available legs: $AVAILABLE_LEGS" | tee -a "$LOG"

# Use the devset-trained model to score Blind-B candidates
# train_ltr.py needs ground truth for training — we train on devset, apply to blind
# For now, use a simpler approach: RRF ensemble (no LTR training needed for blind set)
echo "  Using RRF ensemble (tuned weights from devset grid search)..." | tee -a "$LOG"

# RRF with best weights from devset: m=1.25, c=1.0, p=0.75, k=20
PYTHONPATH=src python src/ensemble.py rrf \
    --inputs "$AVAILABLE_LEGS" \
    --weights 1.25,1.0,0.75,1.0,1.0,1.0,1.0,0.5 \
    --rrf_k 20 \
    --inference_dir "$OUT_DIR" \
    --out "${OUT_DIR}/rrf_ensemble.json" \
    2>&1 | tail -3 | tee -a "$LOG"

# =============================================================================
# Step 3: Add LLM responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Generating LLM responses..." | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/rrf_ensemble.json" \
    --out "${OUT_DIR}/submission.json" \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 4: Validate submission format
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Validating submission format..." | tee -a "$LOG"
python -c "
import json

with open('${OUT_DIR}/submission.json') as f:
    rows = json.load(f)

print(f'Total rows: {len(rows)}')
# Check all required fields
issues = []
sessions = set()
for i, r in enumerate(rows):
    if 'session_id' not in r: issues.append(f'row {i}: missing session_id')
    if 'user_id' not in r: issues.append(f'row {i}: missing user_id')
    if 'turn_number' not in r: issues.append(f'row {i}: missing turn_number')
    if 'predicted_track_ids' not in r: issues.append(f'row {i}: missing predicted_track_ids')
    if 'predicted_response' not in r: issues.append(f'row {i}: missing predicted_response')
    if len(r.get('predicted_track_ids', [])) > 20:
        issues.append(f'row {i}: >20 track_ids')
    if len(set(r.get('predicted_track_ids', []))) != len(r.get('predicted_track_ids', [])):
        issues.append(f'row {i}: duplicate track_ids')
    sessions.add((r.get('session_id'), r.get('turn_number')))

if issues:
    print(f'ISSUES ({len(issues)}):')
    for iss in issues[:10]:
        print(f'  {iss}')
else:
    print('✓ All rows valid')
    print(f'  Sessions: {len(set(r[\"session_id\"] for r in rows))}')
    print(f'  (session, turn) pairs: {len(sessions)}')
    print(f'  Turns per session: {len(rows) // len(set(r[\"session_id\"] for r in rows))}')
    print(f'  Avg tracks per row: {sum(len(r[\"predicted_track_ids\"]) for r in rows)/len(rows):.1f}')
    print(f'  Has responses: {sum(1 for r in rows if r[\"predicted_response\"])} / {len(rows)}')
print()
print(f'Submission file: ${OUT_DIR}/submission.json')
print(f'Upload to: https://www.codabench.org/competitions/15786/')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "SUBMIT: ${OUT_DIR}/submission.json → Codabench" | tee -a "$LOG"
