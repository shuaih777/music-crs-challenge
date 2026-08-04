#!/bin/bash
# =============================================================================
# Blind-B Submission with Bi-Encoder (new best: nDCG@20=0.1557 on devset)
# =============================================================================
# Same as run_blind_submission.sh but includes the bi-encoder retrieval leg
# and uses the lgbm_biencoder model.
#
# Usage:
#   bash run_blind_biencoder.sh
#
# Expected wall-clock: ~40-50 min on H100
# Output: exp/inference/blind_b/submission_biencoder.json
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
LOG="logs/blind_b_biencoder.log"
mkdir -p logs "$OUT_DIR" exp/tracks

echo "=== Blind-B (with bi-encoder) started at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Run standard retrieval legs on Blind-B (same as before)
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
        --output "$OUT" --split "$SPLIT" --n_output 100 "$@" \
        2>&1 | grep -E "Wrote|sessions" | tee -a "$LOG"
}

run_leg "metadata_qwen3" --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag meta
run_leg "cf_bpr" --embed cf-bpr --pooling decay --weight_schedule descending --tag cf
run_leg "decay_descending" --embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag decay
run_leg "bm25_norepeat" --bm25_only --tag bm25
run_leg "image_siglip2" --embed image-siglip2 --pooling decay --weight_schedule descending --tag siglip
run_leg "attributes_qwen3" --embed attributes-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag attr
run_leg "lyrics_qwen3" --embed lyrics-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --tag lyrics

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

# State BM25 placeholder
STATE_OUT="${OUT_DIR}/state_bm25_focused.json"
if [ ! -f "$STATE_OUT" ]; then
    cp "${OUT_DIR}/bm25_norepeat.json" "$STATE_OUT" 2>/dev/null || \
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$STATE_OUT" --split "$SPLIT" --n_output 100 --bm25_only --tag state \
        2>&1 | grep "Wrote" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Bi-encoder retrieval leg on Blind-B
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Bi-encoder retrieval on $SPLIT..." | tee -a "$LOG"

BIENC_OUT="${OUT_DIR}/biencoder_top100.json"
if [ -f "$BIENC_OUT" ]; then
    echo "  [skip] $BIENC_OUT exists" | tee -a "$LOG"
else
    # Check if track embeddings exist (encoded during devset run)
    TRACK_EMB="out/biencoder/track_embeddings.npy"
    if [ ! -f "$TRACK_EMB" ]; then
        echo "  [encode] Track embeddings not found, encoding..." | tee -a "$LOG"
        PYTHONPATH=src python src/train_biencoder.py encode_tracks \
            --model_dir out/biencoder \
            --out "$TRACK_EMB" \
            --batch_size 256 \
            2>&1 | tail -3 | tee -a "$LOG"
    fi

    echo "  [run] Bi-encoder inference on $SPLIT..." | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/biencoder \
        --track_emb "$TRACK_EMB" \
        --out "$BIENC_OUT" \
        --n_output 100 \
        --batch_size 256 \
        --split "$SPLIT" \
        2>&1 | tail -5 | tee -a "$LOG"
fi

# =============================================================================
# Step 3: Score with LightGBM (bi-encoder model)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Scoring with lgbm_biencoder model..." | tee -a "$LOG"

LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,attributes_qwen3,image_siglip2,lyrics_qwen3,state_bm25_focused,biencoder_top100"

PYTHONPATH=src python src/score_blind.py \
    --model exp/ltr/lgbm_biencoder.txt \
    --legs "$LEGS" \
    --inference_dir "$OUT_DIR" \
    --out "${OUT_DIR}/lgbm_biencoder_scored.json" \
    --pmi_path exp/item2item_pmi.npz \
    --split "$SPLIT" \
    --top_k 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 4: Add LLM responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Generating responses..." | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/lgbm_biencoder_scored.json" \
    --out "${OUT_DIR}/submission_biencoder.json" \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 5: Validate
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 5] Validating submission..." | tee -a "$LOG"
python -c "
import json
with open('${OUT_DIR}/submission_biencoder.json') as f:
    rows = json.load(f)
print(f'Total rows: {len(rows)}')
sessions = set(r['session_id'] for r in rows)
print(f'Sessions: {len(sessions)}')
print(f'Turns/session: {len(rows)//len(sessions)}')
issues = [i for i,r in enumerate(rows) if len(r.get('predicted_track_ids',[])) > 20 or len(set(r.get('predicted_track_ids',[]))) != len(r.get('predicted_track_ids',[]))]
if issues:
    print(f'ISSUES: {len(issues)} rows with >20 or duplicate track_ids')
else:
    print('✓ All rows valid (≤20 unique track_ids)')
print(f'Has responses: {sum(1 for r in rows if r.get(\"predicted_response\"))}/{len(rows)}')
print(f'\\nSubmit: ${OUT_DIR}/submission_biencoder.json')
print(f'Upload to: https://www.codabench.org/competitions/15786/')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
