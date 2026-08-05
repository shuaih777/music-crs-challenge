#!/bin/bash
# =============================================================================
# Blind-B submission with the 14-leg pipeline (13-leg baseline + personalized)
# =============================================================================
# This is the pipeline that produced the ACTUAL locked final submission
# (Blind-B nDCG@20 = 0.25, composite = 0.252 -- the headline numbers in
# README.md). scripts/run_blind_b_13leg.sh reproduces the 0.24 baseline
# that came before personalized was added; this script extends it with the
# 14th leg (biencoder_personalized) and the 14-leg LightGBM reranker.
#
# Reuses the 13 base legs already computed by run_blind_b_13leg.sh (same
# OUT_DIR) -- run that first, or this script will generate them itself via
# the same skip-if-exists logic.
#
# Requires the personalized checkpoint in out/biencoder_personalized and the
# 14-leg reranker exp/ltr/lgbm_reproduce_14leg.txt. Get them without
# retraining from: https://huggingface.co/shuaih777/music-challenge-models
#
# Usage:
#   bash scripts/run_blind_b_14leg.sh
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
OUT_DIR="exp/inference/blind_b_13leg"   # reuse the 13 base legs already there
LOG="logs/blind_b_14leg.log"
mkdir -p logs "$OUT_DIR" exp/tracks

echo "=== Blind-B 14-leg submission at $(date) ===" | tee "$LOG"

# Sanity: 13 base legs + personalized checkpoint + 14-leg reranker
for CKPT in biencoder biencoder_large e5_large biencoder_mxbai biencoder_stella biencoder_nv_embed biencoder_personalized; do
    if [ ! -d "out/$CKPT" ]; then
        echo "ERROR: missing out/$CKPT — download from https://huggingface.co/shuaih777/music-challenge-models" | tee -a "$LOG"
        exit 1
    fi
done
if [ ! -f exp/ltr/lgbm_reproduce_14leg.txt ]; then
    echo "ERROR: missing exp/ltr/lgbm_reproduce_14leg.txt — download the 14-leg reranker from HF" | tee -a "$LOG"
    exit 1
fi
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           biencoder_top100 biencoder_large_top100 e5_large_top100 \
           biencoder_stella_top100 biencoder_mxbai_top100 biencoder_nv_embed_top100 \
           biencoder_last2turns_top100 biencoder_current_only_top100; do
    if [ ! -f "${OUT_DIR}/${LEG}.json" ]; then
        echo "ERROR: missing ${OUT_DIR}/${LEG}.json — run scripts/run_blind_b_13leg.sh first" | tee -a "$LOG"
        exit 1
    fi
done
echo "  13 base legs present, checkpoints present." | tee -a "$LOG"

# =============================================================================
# Step 1: Generate the personalized leg (14th) on Blind-B
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Personalized bi-encoder leg on $SPLIT..." | tee -a "$LOG"
PERS_OUT="${OUT_DIR}/biencoder_personalized_top100.json"
if [ -f "$PERS_OUT" ]; then
    echo "  [skip] biencoder_personalized_top100" | tee -a "$LOG"
else
    PYTHONPATH=src python -c "
import json, os, sys, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('out/biencoder_personalized', trust_remote_code=True)
track_embs = np.load('out/biencoder_personalized/track_embeddings.npy')
track_ids = json.load(open('out/biencoder_personalized/track_embeddings_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

test = load_dataset('talkpl-ai/TalkPlayData-Challenge-${SPLIT}', split='test')

def build_personalized_query(conversations, turn, user_profile):
    age = user_profile.get('age_group', '') or ''
    gender = user_profile.get('gender', '') or ''
    country = user_profile.get('country_name', '') or ''
    prefix = f'[{age} | {gender} | {country}]'
    lines = [prefix]
    for c in conversations:
        if c['turn_number'] > turn: break
        if c['turn_number'] < turn:
            if c['role'] == 'user': lines.append(f'User: {c[\"content\"]}')
            elif c['role'] == 'assistant': lines.append(f'Assistant: {c[\"content\"][:100]}')
        elif c['turn_number'] == turn and c['role'] == 'user':
            lines.append(f'User: {c[\"content\"]}')
    return '\n'.join(lines[-12:])

queries, meta_list = [], []
for ex in test:
    user_profile = ex.get('user_profile', {}) or {}
    for tn in range(1, 9):
        q = build_personalized_query(ex['conversations'], tn, user_profile)
        prior = [c['content'] for c in ex['conversations'] if c['role']=='music' and c['turn_number'] < tn]
        queries.append(q)
        meta_list.append({'session_id': ex['session_id'], 'user_id': ex['user_id'], 'turn_number': tn, 'prior_tracks': prior})

print(f'Encoding {len(queries)} personalized queries...', flush=True)
q_embs = model.encode(queries, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)

rows = []
for i, m in enumerate(tqdm(meta_list, desc='retrieval')):
    scores = track_embs @ q_embs[i]
    for tid in m['prior_tracks']:
        if tid in track_to_idx: scores[track_to_idx[tid]] = -1e9
    top_idx = np.argsort(-scores)[:100]
    rows.append({'session_id': m['session_id'], 'user_id': m['user_id'], 'turn_number': m['turn_number'],
                 'predicted_track_ids': [track_ids[j] for j in top_idx], 'predicted_response': ''})

json.dump(rows, open('$PERS_OUT', 'w'), ensure_ascii=False)
print(f'Wrote {len(rows)} to $PERS_OUT')
" 2>&1 | grep -E "Wrote|Encoding|Error|Traceback|error" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Score with the 14-leg LightGBM reranker
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Scoring with lgbm_reproduce_14leg (14 legs)..." | tee -a "$LOG"
LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,biencoder_top100,biencoder_large_top100,e5_large_top100,biencoder_stella_top100,biencoder_mxbai_top100,biencoder_nv_embed_top100,biencoder_last2turns_top100,biencoder_current_only_top100,biencoder_personalized_top100"

PYTHONPATH=src python src/score_blind_v2.py \
    --legs "$LEGS" \
    --biencoder_dir out/biencoder_large \
    --inference_dir "$OUT_DIR" \
    --model exp/ltr/lgbm_reproduce_14leg.txt \
    --out "${OUT_DIR}/lgbm_scored_14leg.json" \
    --split "$SPLIT" \
    --pmi_path exp/item2item_pmi.npz \
    --top_k 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: Generate responses for all 640 candidates
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Generating responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/lgbm_scored_14leg.json" \
    --out "${OUT_DIR}/lgbm_with_responses_640_14leg.json" \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 4: Filter to one row per session at its LAST turn -> predictions_14leg.json
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Filtering to one prediction per session (last turn)..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json
from datasets import load_dataset

bb = load_dataset('talkpl-ai/TalkPlayData-Challenge-${SPLIT}', split='test')
last_turn = {}
for item in bb:
    sid = item['session_id']
    last = item['conversations'][-1]
    last_turn[sid] = (item['user_id'], int(last['turn_number']))

allpred = json.load(open('${OUT_DIR}/lgbm_with_responses_640_14leg.json'))
idx = {(r['session_id'], int(r['turn_number'])): r for r in allpred}

out, missing = [], []
for sid, (uid, lt) in last_turn.items():
    r = idx.get((sid, lt))
    if r is None:
        missing.append((sid, lt)); continue
    out.append({
        'session_id': sid, 'user_id': uid, 'turn_number': lt,
        'predicted_track_ids': r['predicted_track_ids'][:20],
        'predicted_response': r.get('predicted_response', ''),
    })
if missing:
    print(f'ERROR: {len(missing)} sessions missing their last-turn prediction: {missing[:5]}')
    raise SystemExit(1)
assert all(len(r['predicted_track_ids']) <= 20 and len(set(r['predicted_track_ids'])) == len(r['predicted_track_ids']) for r in out)
json.dump(out, open('${OUT_DIR}/predictions_14leg.json', 'w'), ensure_ascii=False)
print(f'Wrote {len(out)} rows to ${OUT_DIR}/predictions_14leg.json (one per session, last turn)')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 5: Validate submission format
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 5] Validating..." | tee -a "$LOG"
python -c "
import json
rows = json.load(open('${OUT_DIR}/predictions_14leg.json'))
sessions = set(r['session_id'] for r in rows)
issues = []
for r in rows:
    ids = r.get('predicted_track_ids', [])
    if len(ids) > 20: issues.append('too many tracks')
    if len(set(ids)) != len(ids): issues.append('dupes')
if len(rows) != len(sessions):
    issues.append('row count != unique session count')
if issues:
    print(f'ISSUES: {issues[:5]}')
else:
    print(f'OK: {len(sessions)} sessions, {len(rows)} rows (one per session), all <=20 unique tracks')
    print(f'  Has responses: {sum(1 for r in rows if r.get(\"predicted_response\"))}/{len(rows)}')
print(f'Submit: ${OUT_DIR}/predictions_14leg.json')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
