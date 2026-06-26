#!/bin/bash
# =============================================================================
# Blind-B Submission with lgbm_biencoder_large (nDCG@20=0.1649 on devset)
# =============================================================================
# Uses the best model (bge-base + bge-large bi-encoders + 5 other legs).
#
# Required: out/biencoder/ and out/biencoder_large/ must exist (trained models)
#
# Usage:
#   bash run_blind_b_best.sh
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
OUT_DIR="exp/inference/blind_b_best"
LOG="logs/blind_b_best.log"
mkdir -p logs "$OUT_DIR"

echo "=== Blind-B best submission at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Run retrieval legs on Blind-B (matching lgbm_biencoder_large features)
# =============================================================================
# Legs needed: biencoder_large_top100, biencoder_top100, bm25_norepeat,
#              cf_bpr, decay_descending, metadata_qwen3, pmi_leg
echo "" | tee -a "$LOG"
echo "[step 1] Running retrieval legs on Blind-B..." | tee -a "$LOG"

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
        2>&1 | grep -E "Wrote|sessions" | tee -a "$LOG"
}

# Standard legs
run_leg "metadata_qwen3" --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag meta
run_leg "cf_bpr" --embed cf-bpr --pooling decay --weight_schedule descending --tag cf
run_leg "decay_descending" --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag decay
run_leg "bm25_norepeat" --bm25_only --tag bm25

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
" 2>&1 | tail -2 | tee -a "$LOG"
else
    echo "  [skip] pmi_leg" | tee -a "$LOG"
fi

# Bi-encoder legs (use trained models)
BIENC_OUT="${OUT_DIR}/biencoder_top100.json"
if [ ! -f "$BIENC_OUT" ]; then
    echo "  [run] biencoder_top100 (bge-base)" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/biencoder \
        --track_emb out/biencoder/track_embeddings.npy \
        --out "$BIENC_OUT" \
        --split Blind-B --n_output 100 --batch_size 256 \
        2>&1 | grep -E "Wrote|queries" | tee -a "$LOG"
else
    echo "  [skip] biencoder_top100" | tee -a "$LOG"
fi

BIENC_LARGE_OUT="${OUT_DIR}/biencoder_large_top100.json"
if [ ! -f "$BIENC_LARGE_OUT" ]; then
    echo "  [run] biencoder_large_top100 (bge-large)" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/biencoder_large \
        --track_emb out/biencoder_large/track_embeddings.npy \
        --out "$BIENC_LARGE_OUT" \
        --split Blind-B --n_output 100 --batch_size 128 \
        2>&1 | grep -E "Wrote|queries" | tee -a "$LOG"
else
    echo "  [skip] biencoder_large_top100" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Score with lgbm_biencoder_large model
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Scoring with lgbm_biencoder_large model..." | tee -a "$LOG"

LEGS="biencoder_large_top100,biencoder_top100,bm25_norepeat,cf_bpr,decay_descending,metadata_qwen3,pmi_leg"

PYTHONPATH=src python src/score_blind.py \
    --model exp/ltr/lgbm_biencoder_large.txt \
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
print(f'Total rows: {len(rows)}')
sessions = set(r['session_id'] for r in rows)
issues = []
for r in rows:
    if len(r.get('predicted_track_ids',[])) > 20: issues.append('too many tracks')
    if len(set(r.get('predicted_track_ids',[]))) != len(r.get('predicted_track_ids',[])): issues.append('dupes')
if issues:
    print(f'ISSUES: {issues[:5]}')
else:
    print(f'✓ Valid: {len(sessions)} sessions, {len(rows)} rows, all ≤20 unique tracks')
    print(f'  Has responses: {sum(1 for r in rows if r.get(\"predicted_response\"))}/{len(rows)}')
print(f'\\nSubmit: ${OUT_DIR}/submission.json → https://www.codabench.org/competitions/15786/')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
