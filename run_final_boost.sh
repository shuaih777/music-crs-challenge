#!/bin/bash
# =============================================================================
# Final Boost: 5 independent improvements, ablation-tested one by one
# =============================================================================
# Produces 5 new signals, then tests each incrementally against lgbm_v2 baseline.
# Only keeps changes that improve nDCG@20. Never overwrites existing results.
#
# Experiments:
#   A. E5-large bi-encoder (new retrieval leg)
#   B. bge-large with top-200 output (larger pool)
#   C. Multi-query bi-encoder (3 query granularities)
#   D. All-legs raw cosine scores as features
#   E. CatBoost blend with LightGBM
#
# Usage:
#   bash run_final_boost.sh
#
# Expected: ~2-3h total on H100. Each ablation takes ~5 min (LightGBM only).
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

LOG="logs/final_boost.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr exp/tracks data out/e5_large

echo "=== Final Boost started at $(date) ===" | tee "$LOG"

# =============================================================================
# PHASE 1: Generate all new signals (independent, can partially fail)
# =============================================================================
echo "" | tee -a "$LOG"
echo "========== PHASE 1: Generate new signals ==========" | tee -a "$LOG"

# --- A. E5-large bi-encoder ---
E5_LEG="exp/inference/devset/e5_large_top100.json"
if [ -f "$E5_LEG" ]; then
    echo "[A] skip: $E5_LEG exists" | tee -a "$LOG"
else
    echo "[A] Training E5-large bi-encoder..." | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py all \
        --model_id intfloat/e5-large-v2 \
        --output_dir out/e5_large \
        --out_leg "$E5_LEG" \
        --batch_size 128 \
        --epochs 3 \
        --n_output 100 \
        2>&1 | tail -10 | tee -a "$LOG"
fi

# --- B. bge-large with top-200 ---
BGE200_LEG="exp/inference/devset/biencoder_large_top200.json"
if [ -f "$BGE200_LEG" ]; then
    echo "[B] skip: $BGE200_LEG exists" | tee -a "$LOG"
else
    echo "[B] Re-running bge-large inference with n_output=200..." | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py inference \
        --model_dir out/biencoder_large \
        --track_emb exp/tracks/biencoder_large_tracks.npy \
        --out "$BGE200_LEG" \
        --n_output 200 \
        2>&1 | tail -5 | tee -a "$LOG"
fi

# --- C. Multi-query bi-encoder (3 query granularities) ---
MQ_LAST2="exp/inference/devset/biencoder_last2turns_top100.json"
MQ_CURRENT="exp/inference/devset/biencoder_current_only_top100.json"
if [ -f "$MQ_LAST2" ] && [ -f "$MQ_CURRENT" ]; then
    echo "[C] skip: multi-query legs exist" | tee -a "$LOG"
else
    echo "[C] Multi-query bi-encoder (last-2-turns + current-only)..." | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, sys, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, 'src')
from train_biencoder import build_track_text

print('Loading bge-large model...', flush=True)
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)

# Load track embeddings
track_embs = np.load('exp/tracks/biencoder_large_tracks.npy')
track_ids = json.load(open('exp/tracks/biencoder_large_tracks_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

# Load test
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')

def run_query_mode(mode, out_path):
    rows = []
    queries = []
    meta_list = []

    for ex in test:
        convos = ex['conversations']
        for tn in range(1, 9):
            prior_tracks = [c['content'] for c in convos if c['role']=='music' and c['turn_number'] < tn]

            if mode == 'last2':
                # Only last 2 turns of history + current user message
                lines = []
                for c in convos:
                    if c['turn_number'] > tn: break
                    if c['turn_number'] >= max(1, tn-2):
                        if c['role'] == 'user':
                            lines.append(f\"User: {c['content']}\")
                        elif c['role'] == 'assistant':
                            lines.append(f\"Assistant: {c['content'][:80]}\")
                query = '\\n'.join(lines[-6:])
            elif mode == 'current':
                # Only current user message
                query = ''
                for c in convos:
                    if c['turn_number'] == tn and c['role'] == 'user':
                        query = c['content']
                        break

            queries.append(query if query else 'music recommendation')
            meta_list.append({
                'session_id': ex['session_id'],
                'user_id': ex['user_id'],
                'turn_number': tn,
                'prior_tracks': prior_tracks,
            })

    print(f'  Encoding {len(queries)} queries ({mode})...', flush=True)
    q_embs = model.encode(queries, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)

    print(f'  Retrieving top-100...', flush=True)
    for i, m in enumerate(tqdm(meta_list, desc=mode)):
        scores = track_embs @ q_embs[i]
        for tid in m['prior_tracks']:
            if tid in track_to_idx:
                scores[track_to_idx[tid]] = -1e9
        top_idx = np.argsort(-scores)[:100]
        preds = [track_ids[j] for j in top_idx]
        rows.append({
            'session_id': m['session_id'],
            'user_id': m['user_id'],
            'turn_number': m['turn_number'],
            'predicted_track_ids': preds,
            'predicted_response': '',
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f'  Wrote {len(rows)} to {out_path}')

if not os.path.exists('$MQ_LAST2'):
    run_query_mode('last2', '$MQ_LAST2')
if not os.path.exists('$MQ_CURRENT'):
    run_query_mode('current', '$MQ_CURRENT')
" 2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "Phase 1 complete. New signals generated." | tee -a "$LOG"

# =============================================================================
# PHASE 2: Ablation — test each addition incrementally
# =============================================================================
echo "" | tee -a "$LOG"
echo "========== PHASE 2: Ablation testing ==========" | tee -a "$LOG"

# Baseline legs (same as lgbm_v2)
BASE_LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,biencoder_large_top100,biencoder_top100"

# Find which base legs actually exist
AVAILABLE_BASE=""
for LEG in $(echo "$BASE_LEGS" | tr ',' ' '); do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        AVAILABLE_BASE="${AVAILABLE_BASE:+$AVAILABLE_BASE,}$LEG"
    fi
done
echo "Base legs: $AVAILABLE_BASE" | tee -a "$LOG"

run_ablation() {
    local NAME="$1"
    local LEGS="$2"
    local TOP_K="${3:-100}"
    local OUT_MODEL="exp/ltr/lgbm_abl_${NAME}.txt"
    local OUT_INF="exp/inference/devset/lgbm_abl_${NAME}.json"
    local OUT_SCORE="exp/scores/devset/lgbm_abl_${NAME}.json"

    if [ -f "$OUT_SCORE" ]; then
        echo "  [$NAME] skip (already scored)" | tee -a "$LOG"
        return
    fi

    echo "  [$NAME] legs=$LEGS top_k=$TOP_K" | tee -a "$LOG"
    PYTHONPATH=src python src/train_ltr_v2.py \
        --legs "$LEGS" \
        --inference_dir exp/inference/devset \
        --ground_truth exp/ground_truth/devset.json \
        --top_k "$TOP_K" \
        --n_folds 5 \
        --pmi_path exp/item2item_pmi.npz \
        --out_model "$OUT_MODEL" \
        --out_inference "$OUT_INF" \
        2>&1 | grep -E "positive|fold" | tail -6 | tee -a "$LOG"

    PYTHONPATH=src python src/evaluate.py \
        --inference "$OUT_INF" \
        --scores "$OUT_SCORE" \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | grep "ndcg@20" | tee -a "$LOG"
}

# Ablation A: + E5-large
if [ -f "$E5_LEG" ]; then
    run_ablation "plus_e5" "${AVAILABLE_BASE},e5_large_top100"
fi

# Ablation B: bge-large top-200 (replace top-100 leg with top-200, increase pool)
if [ -f "$BGE200_LEG" ]; then
    LEGS_B=$(echo "$AVAILABLE_BASE" | sed 's/biencoder_large_top100/biencoder_large_top200/')
    run_ablation "top200" "$LEGS_B" 200
fi

# Ablation C: + multi-query legs
if [ -f "$MQ_LAST2" ] && [ -f "$MQ_CURRENT" ]; then
    run_ablation "plus_mq" "${AVAILABLE_BASE},biencoder_last2turns_top100,biencoder_current_only_top100"
fi

# Ablation D: + E5 + multi-query (combination)
if [ -f "$E5_LEG" ] && [ -f "$MQ_LAST2" ]; then
    run_ablation "plus_all_legs" "${AVAILABLE_BASE},e5_large_top100,biencoder_last2turns_top100,biencoder_current_only_top100"
fi

# =============================================================================
# PHASE 3: Results comparison
# =============================================================================
echo "" | tee -a "$LOG"
echo "========== PHASE 3: Results ==========" | tee -a "$LOG"

python -c "
import json, os, glob

results = []
for f in sorted(glob.glob('exp/scores/devset/lgbm_abl_*.json')):
    tag = os.path.basename(f).replace('.json', '').replace('lgbm_abl_', '')
    s = json.load(open(f))
    results.append((s['ndcg@20'], s.get('hit@20',0), tag))

# Add baseline
try:
    b = json.load(open('exp/scores/devset/lgbm_v2.json'))
    results.append((b['ndcg@20'], b.get('hit@20',0), 'BASELINE (lgbm_v2)'))
except: pass

results.sort(reverse=True)
print(f'{\"Config\":35s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"vs baseline\":>12s}')
print('-' * 70)
baseline_ndcg = 0.168635  # lgbm_v2
for ndcg, hit, tag in results:
    delta = ndcg - baseline_ndcg
    marker = '✓' if delta > 0 else '✗' if delta < 0 else '='
    print(f'{tag:35s} {ndcg:10.6f} {hit*100:7.2f}% {delta:+10.6f} {marker}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "Pick the best ablation, add responses, submit." | tee -a "$LOG"
