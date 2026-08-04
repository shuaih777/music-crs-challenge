#!/bin/bash
# =============================================================================
# Blind-B submission with the full 13-leg reproduce.sh pipeline
# =============================================================================
# Mirrors reproduce.sh exactly (same 13 legs, same LightGBM model), just
# pointed at the Blind-B split instead of devset. Produces a final
# predictions.json with reranked tracks + generated responses.
#
# Requires the 6 bi-encoder checkpoints in out/ (biencoder, biencoder_large,
# e5_large, biencoder_mxbai, biencoder_stella, biencoder_nv_embed) and the
# reranker exp/ltr/lgbm_reproduce.txt. Get them without retraining from:
#   https://huggingface.co/shuaih777/music-challenge-models
#
# Usage:
#   bash scripts/run_blind_b_13leg.sh
# =============================================================================

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

SPLIT="Blind-B"
OUT_DIR="exp/inference/blind_b_13leg"
LOG="logs/blind_b_13leg.log"
mkdir -p logs "$OUT_DIR" exp/tracks

echo "=== Blind-B 13-leg submission at $(date) ===" | tee "$LOG"

# Sanity: all 6 bi-encoder checkpoints + reranker must exist
for CKPT in biencoder biencoder_large e5_large biencoder_mxbai biencoder_stella biencoder_nv_embed; do
    if [ ! -d "out/$CKPT" ]; then
        echo "ERROR: missing out/$CKPT — download from https://huggingface.co/shuaih777/music-challenge-models" | tee -a "$LOG"
        exit 1
    fi
done
if [ ! -f exp/ltr/lgbm_reproduce.txt ]; then
    echo "ERROR: missing exp/ltr/lgbm_reproduce.txt — download the reranker from HF or run reproduce.sh Step 5" | tee -a "$LOG"
    exit 1
fi

# =============================================================================
# Step 1: Sparse + dense legs (BM25, metadata-qwen3, cf_bpr, decay_descending)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Sparse + dense legs on $SPLIT..." | tee -a "$LOG"
for ARGS in \
    "--embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --output ${OUT_DIR}/metadata_qwen3.json --tag meta" \
    "--embed cf-bpr --pooling decay --weight_schedule descending --output ${OUT_DIR}/cf_bpr.json --tag cf" \
    "--embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --output ${OUT_DIR}/decay_descending.json --tag decay" \
    "--bm25_only --output ${OUT_DIR}/bm25_norepeat.json --tag bm25"; do
    OUT=$(echo "$ARGS" | grep -oP '(?<=--output )\S+')
    [ -f "$OUT" ] && { echo "  [skip] $OUT" | tee -a "$LOG"; continue; }
    PYTHONPATH=src python src/baselines_v3.py --split "$SPLIT" --n_output 100 $ARGS 2>&1 | tail -2 | tee -a "$LOG"
done

# =============================================================================
# Step 2: PMI leg
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] PMI leg..." | tee -a "$LOG"
PMI_OUT="${OUT_DIR}/pmi_leg.json"
if [ ! -f "$PMI_OUT" ]; then
    PYTHONPATH=src python -c "
import json, numpy as np
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
        scores = np.asarray(pmi[prior_idx].sum(axis=0)).flatten() if prior_idx else np.zeros(len(track_ids), dtype=np.float32)
        for idx in prior_idx: scores[idx] = -1e9
        top_idx = np.argsort(-scores)[:100]
        rows.append({'session_id': ex['session_id'], 'user_id': ex['user_id'], 'turn_number': int(tn),
                     'predicted_track_ids': [track_ids[i] for i in top_idx], 'predicted_response': ''})
json.dump(rows, open('${PMI_OUT}', 'w'), ensure_ascii=False)
print(f'Wrote {len(rows)} to ${PMI_OUT}')
" 2>&1 | tail -2 | tee -a "$LOG"
else
    echo "  [skip] pmi_leg" | tee -a "$LOG"
fi

# =============================================================================
# Step 3: 6 bi-encoder legs (inference only — checkpoints already trained)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Bi-encoder legs on $SPLIT..." | tee -a "$LOG"
for SPEC in \
    "biencoder|out/biencoder|256" \
    "biencoder_large|out/biencoder_large|128" \
    "e5_large|out/e5_large|128" \
    "biencoder_stella|out/biencoder_stella|64" \
    "biencoder_mxbai|out/biencoder_mxbai|128" \
    "biencoder_nv_embed|out/biencoder_nv_embed|8"; do
    IFS='|' read -r KEY MODEL_DIR BATCH <<< "$SPEC"
    OUT="${OUT_DIR}/${KEY}_top100.json"
    [ -f "$OUT" ] && { echo "  [skip] $KEY" | tee -a "$LOG"; continue; }
    echo "  [run] $KEY" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir "$MODEL_DIR" --track_emb "$MODEL_DIR/track_embeddings.npy" \
        --out "$OUT" --split "$SPLIT" --n_output 100 --batch_size "$BATCH" \
        2>&1 | grep -E "Wrote|queries|Loading" | tee -a "$LOG"
done

# =============================================================================
# Step 4: Multi-query legs (last2turns, current_only) with bge-large
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Multi-query legs on $SPLIT..." | tee -a "$LOG"
MQ_LAST2="${OUT_DIR}/biencoder_last2turns_top100.json"
MQ_CURRENT="${OUT_DIR}/biencoder_current_only_top100.json"
if [ -f "$MQ_LAST2" ] && [ -f "$MQ_CURRENT" ]; then
    echo "  [skip] multi-query legs" | tee -a "$LOG"
else
    PYTHONPATH=src python -c "
import json, os, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)
track_embs = np.load('out/biencoder_large/track_embeddings.npy')
track_ids = json.load(open('out/biencoder_large/track_embeddings_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-${SPLIT}', split='test')
def run_mode(mode, out_path):
    queries, meta_list = [], []
    for ex in test:
        convos = ex['conversations']
        for tn in range(1, 9):
            prior = [c['content'] for c in convos if c['role']=='music' and c['turn_number'] < tn]
            if mode == 'last2':
                lines = []
                for c in convos:
                    if c['turn_number'] > tn: break
                    if c['turn_number'] >= max(1, tn-2):
                        if c['role'] == 'user': lines.append(f'User: {c[\"content\"]}')
                        elif c['role'] == 'assistant': lines.append(f'Assistant: {c[\"content\"][:80]}')
                query = chr(10).join(lines[-6:])
            else:
                query = next((c['content'] for c in convos if c['turn_number']==tn and c['role']=='user'), '')
            queries.append(query or 'music')
            meta_list.append({'session_id': ex['session_id'], 'user_id': ex['user_id'], 'turn_number': tn, 'prior_tracks': prior})
    q_embs = model.encode(queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    rows = []
    for i, m in enumerate(tqdm(meta_list, desc=mode)):
        scores = track_embs @ q_embs[i]
        for tid in m['prior_tracks']:
            if tid in track_to_idx: scores[track_to_idx[tid]] = -1e9
        top_idx = np.argsort(-scores)[:100]
        rows.append({'session_id': m['session_id'], 'user_id': m['user_id'], 'turn_number': m['turn_number'], 'predicted_track_ids': [track_ids[j] for j in top_idx], 'predicted_response': ''})
    json.dump(rows, open(out_path, 'w'), ensure_ascii=False)
    print(f'Wrote {len(rows)} to {out_path}')
if not os.path.exists('$MQ_LAST2'): run_mode('last2', '$MQ_LAST2')
if not os.path.exists('$MQ_CURRENT'): run_mode('current', '$MQ_CURRENT')
" 2>&1 | grep -E "Wrote|Encoding" | tee -a "$LOG"
fi

# =============================================================================
# Step 5: Score with the 13-leg LightGBM reranker
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 5] Scoring with lgbm_reproduce (13 legs)..." | tee -a "$LOG"
LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,biencoder_top100,biencoder_large_top100,e5_large_top100,biencoder_stella_top100,biencoder_mxbai_top100,biencoder_nv_embed_top100,biencoder_last2turns_top100,biencoder_current_only_top100"

PYTHONPATH=src python src/score_blind_v2.py \
    --legs "$LEGS" \
    --biencoder_dir out/biencoder_large \
    --inference_dir "$OUT_DIR" \
    --model exp/ltr/lgbm_reproduce.txt \
    --out "${OUT_DIR}/lgbm_scored.json" \
    --split "$SPLIT" \
    --pmi_path exp/item2item_pmi.npz \
    --top_k 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 6: Generate responses -> final predictions.json
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 6] Generating responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/lgbm_scored.json" \
    --out "${OUT_DIR}/predictions.json" \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 7: Validate submission format
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 7] Validating..." | tee -a "$LOG"
python -c "
import json
rows = json.load(open('${OUT_DIR}/predictions.json'))
sessions = set(r['session_id'] for r in rows)
issues = []
for r in rows:
    ids = r.get('predicted_track_ids', [])
    if len(ids) > 20: issues.append('too many tracks')
    if len(set(ids)) != len(ids): issues.append('dupes')
if issues:
    print(f'ISSUES: {issues[:5]}')
else:
    print(f'OK: {len(sessions)} sessions, {len(rows)} rows, all <=20 unique tracks')
    print(f'  Has responses: {sum(1 for r in rows if r.get(\"predicted_response\"))}/{len(rows)}')
print(f'Submit: ${OUT_DIR}/predictions.json')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
