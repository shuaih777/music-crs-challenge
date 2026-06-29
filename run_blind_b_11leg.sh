#!/bin/bash
# =============================================================================
# Blind-B submission with the 11-leg lgbm_v2 model (devset nDCG@20=0.1785)
# =============================================================================
# Model: exp/ltr/lgbm_abl_all_plus_mxbai.txt
# Legs: base7 + e5_large + biencoder_mxbai + multi-query(last2turns, current_only)
#       + bi-encoder cosine feature (out/biencoder_large)
#
# Reuses base legs already in exp/inference/blind_b_best/. Generates the 4 new
# legs on Blind-B, scores, adds responses, validates.
# =============================================================================

set -euo pipefail

LOG="logs/blind_b_11leg.log"
OUT_DIR="exp/inference/blind_b_best"
SPLIT="Blind-B"
mkdir -p logs "$OUT_DIR" exp/tracks

echo "=== Blind-B 11-leg submission at $(date) ===" | tee "$LOG"

# Sanity: base legs must already exist (from run_blind_b_best.sh)
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           biencoder_large_top100 biencoder_top100; do
    if [ ! -f "${OUT_DIR}/${LEG}.json" ]; then
        echo "ERROR: base leg missing: ${OUT_DIR}/${LEG}.json (run run_blind_b_best.sh first)" | tee -a "$LOG"
        exit 1
    fi
done

# =============================================================================
# Step 1: Generate the 4 new legs on Blind-B
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Generating new legs on $SPLIT..." | tee -a "$LOG"

# --- e5_large ---
if [ -f "${OUT_DIR}/e5_large_top100.json" ]; then
    echo "  [skip] e5_large_top100" | tee -a "$LOG"
else
    echo "  [run] e5_large_top100" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/e5_large \
        --track_emb out/e5_large/track_embeddings.npy \
        --out "${OUT_DIR}/e5_large_top100.json" \
        --split "$SPLIT" --n_output 100 --batch_size 256 \
        2>&1 | grep -E "Wrote|queries|Loading" | tee -a "$LOG"
fi

# --- biencoder_mxbai ---
if [ -f "${OUT_DIR}/biencoder_mxbai_top100.json" ]; then
    echo "  [skip] biencoder_mxbai_top100" | tee -a "$LOG"
else
    echo "  [run] biencoder_mxbai_top100" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/biencoder_mxbai \
        --track_emb out/biencoder_mxbai/track_embeddings.npy \
        --out "${OUT_DIR}/biencoder_mxbai_top100.json" \
        --split "$SPLIT" --n_output 100 --batch_size 256 \
        2>&1 | grep -E "Wrote|queries|Loading" | tee -a "$LOG"
fi

# --- multi-query (last2turns + current_only) with bge-large ---
MQ_LAST2="${OUT_DIR}/biencoder_last2turns_top100.json"
MQ_CURRENT="${OUT_DIR}/biencoder_current_only_top100.json"
if [ -f "$MQ_LAST2" ] && [ -f "$MQ_CURRENT" ]; then
    echo "  [skip] multi-query legs" | tee -a "$LOG"
else
    echo "  [run] multi-query legs (last2turns + current_only)" | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, sys, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

print('Loading bge-large model...', flush=True)
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)
track_embs = np.load('exp/tracks/biencoder_large_tracks.npy')
track_ids = json.load(open('exp/tracks/biencoder_large_tracks_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

test = load_dataset('talkpl-ai/TalkPlayData-Challenge-${SPLIT}', split='test')

def run_query_mode(mode, out_path):
    queries, meta_list, rows = [], [], []
    for ex in test:
        convos = ex['conversations']
        for tn in range(1, 9):
            prior_tracks = [c['content'] for c in convos if c['role']=='music' and c['turn_number'] < tn]
            if mode == 'last2':
                lines = []
                for c in convos:
                    if c['turn_number'] > tn: break
                    if c['turn_number'] >= max(1, tn-2):
                        if c['role'] == 'user': lines.append(f\"User: {c['content']}\")
                        elif c['role'] == 'assistant': lines.append(f\"Assistant: {c['content'][:80]}\")
                query = '\\n'.join(lines[-6:])
            else:
                query = ''
                for c in convos:
                    if c['turn_number'] == tn and c['role'] == 'user':
                        query = c['content']; break
            queries.append(query if query else 'music recommendation')
            meta_list.append({'session_id': ex['session_id'], 'user_id': ex['user_id'],
                              'turn_number': tn, 'prior_tracks': prior_tracks})
    print(f'  Encoding {len(queries)} queries ({mode})...', flush=True)
    q_embs = model.encode(queries, batch_size=256, show_progress_bar=True,
                          normalize_embeddings=True, convert_to_numpy=True)
    for i, m in enumerate(tqdm(meta_list, desc=mode)):
        scores = track_embs @ q_embs[i]
        for tid in m['prior_tracks']:
            if tid in track_to_idx: scores[track_to_idx[tid]] = -1e9
        top_idx = np.argsort(-scores)[:100]
        rows.append({'session_id': m['session_id'], 'user_id': m['user_id'],
                     'turn_number': m['turn_number'],
                     'predicted_track_ids': [track_ids[j] for j in top_idx],
                     'predicted_response': ''})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f: json.dump(rows, f, ensure_ascii=False)
    print(f'  Wrote {len(rows)} to {out_path}')

if not os.path.exists('$MQ_LAST2'): run_query_mode('last2', '$MQ_LAST2')
if not os.path.exists('$MQ_CURRENT'): run_query_mode('current', '$MQ_CURRENT')
" 2>&1 | grep -E "Wrote|Encoding|Loading" | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Score with the 11-leg lgbm_v2 model
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Scoring with lgbm_abl_all_plus_mxbai..." | tee -a "$LOG"

LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,biencoder_large_top100,biencoder_top100,e5_large_top100,biencoder_last2turns_top100,biencoder_current_only_top100,biencoder_mxbai_top100"

PYTHONPATH=src python src/score_blind_v2.py \
    --legs "$LEGS" \
    --biencoder_dir out/biencoder_large \
    --inference_dir "$OUT_DIR" \
    --model exp/ltr/lgbm_abl_all_plus_mxbai.txt \
    --out "${OUT_DIR}/lgbm_v2_scored.json" \
    --split "$SPLIT" \
    --pmi_path exp/item2item_pmi.npz \
    --top_k 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: Add LLM responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Generating responses..." | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference "${OUT_DIR}/lgbm_v2_scored.json" \
    --out "${OUT_DIR}/submission_11leg.json" \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# Step 4: Validate
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Validating..." | tee -a "$LOG"
python -c "
import json
rows = json.load(open('${OUT_DIR}/submission_11leg.json'))
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
print(f'Submit: ${OUT_DIR}/submission_11leg.json -> https://www.codabench.org/competitions/15786/')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
