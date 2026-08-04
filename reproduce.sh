#!/bin/bash
# See REPRODUCE.md for full documentation
# Usage: bash reproduce.sh [--step N]

set -uo pipefail
if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="logs/reproduce.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr exp/tracks data out
START_STEP="${1:-1}"
[ "$START_STEP" = "--step" ] && START_STEP="${2:-1}"
echo "=== Reproduce started at $(date), step $START_STEP ===" | tee "$LOG"

# Step 1: Sparse + dense legs
if [ "$START_STEP" -le 1 ]; then
echo "[Step 1] Sparse + dense legs" | tee -a "$LOG"
for ARGS in \
    "--embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --output exp/inference/devset/metadata_qwen3.json --tag meta" \
    "--embed cf-bpr --pooling decay --weight_schedule descending --output exp/inference/devset/cf_bpr.json --tag cf" \
    "--embed metadata-qwen3_embedding_0.6b --pooling decay --weight_schedule descending --output exp/inference/devset/decay_descending.json --tag decay" \
    "--bm25_only --output exp/inference/devset/bm25_norepeat.json --tag bm25"; do
    OUT=$(echo "$ARGS" | grep -oP '(?<=--output )\S+')
    [ -f "$OUT" ] && continue
    PYTHONPATH=src python src/baselines_v3.py --n_output 100 $ARGS 2>&1 | tail -2 | tee -a "$LOG"
done
fi

# Step 2: PMI
if [ "$START_STEP" -le 2 ]; then
echo "[Step 2] PMI" | tee -a "$LOG"
[ -f exp/item2item_pmi.npz ] || PYTHONPATH=src python src/build_item2item.py --out exp/item2item_pmi.npz 2>&1 | tail -3 | tee -a "$LOG"
[ -f exp/inference/devset/pmi_leg.json ] || PYTHONPATH=src python src/retrieval_legs.py pmi --pmi_path exp/item2item_pmi.npz --out exp/inference/devset/pmi_leg.json 2>&1 | tail -2 | tee -a "$LOG"
fi

# Step 3: Bi-encoders
if [ "$START_STEP" -le 3 ]; then
echo "[Step 3] Bi-encoders" | tee -a "$LOG"
[ -f data/biencoder_pairs.jsonl ] || PYTHONPATH=src python src/train_biencoder.py build_data --out data/biencoder_pairs.jsonl 2>&1 | tail -2 | tee -a "$LOG"
for SPEC in \
    "biencoder|BAAI/bge-base-en-v1.5|128|" \
    "biencoder_large|BAAI/bge-large-en-v1.5|64|" \
    "e5_large|intfloat/e5-large-v2|32|" \
    "biencoder_stella|dunzhang/stella_en_400M_v5|64|" \
    "biencoder_mxbai|mixedbread-ai/mxbai-embed-large-v1|32|" \
    "biencoder_nv_embed|nvidia/NV-Embed-v2|4|--lora"; do
    IFS='|' read -r KEY MODEL BATCH EXTRA <<< "$SPEC"
    OUT="exp/inference/devset/${KEY}_top100.json"
    [ -f "$OUT" ] && { echo "  [skip] $KEY" | tee -a "$LOG"; continue; }
    echo "  [train] $KEY" | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py all \
        --model_id "$MODEL" --output_dir "out/$KEY" --out_leg "$OUT" \
        --batch_size "$BATCH" --epochs 3 --n_output 100 $EXTRA \
        2>&1 | tail -5 | tee -a "$LOG"
done
fi

# Step 4: Multi-query
if [ "$START_STEP" -le 4 ]; then
echo "[Step 4] Multi-query" | tee -a "$LOG"
if [ ! -f exp/inference/devset/biencoder_last2turns_top100.json ] || [ ! -f exp/inference/devset/biencoder_current_only_top100.json ]; then
    PYTHONPATH=src python -c "
import json, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)
track_embs = np.load('exp/tracks/biencoder_large_tracks.npy')
track_ids = json.load(open('exp/tracks/biencoder_large_tracks_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')
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
    with open(out_path, 'w') as f: json.dump(rows, f, ensure_ascii=False)
    print(f'Wrote {len(rows)} to {out_path}')
run_mode('last2', 'exp/inference/devset/biencoder_last2turns_top100.json')
run_mode('current', 'exp/inference/devset/biencoder_current_only_top100.json')
" 2>&1 | tee -a "$LOG"
fi
fi

# Step 5: LightGBM
if [ "$START_STEP" -le 5 ]; then
echo "[Step 5] LightGBM" | tee -a "$LOG"
ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat biencoder_top100 biencoder_large_top100 e5_large_top100 biencoder_stella_top100 biencoder_mxbai_top100 biencoder_nv_embed_top100 biencoder_last2turns_top100 biencoder_current_only_top100; do
    [ -f "exp/inference/devset/${LEG}.json" ] && ALL_LEGS="${ALL_LEGS:+$ALL_LEGS,}$LEG"
done
echo "  Legs: $(echo $ALL_LEGS | tr ',' '\n' | wc -l)" | tee -a "$LOG"
PYTHONPATH=src python src/train_ltr_v2.py --legs "$ALL_LEGS" --inference_dir exp/inference/devset --ground_truth exp/ground_truth/devset.json --top_k 100 --n_folds 5 --pmi_path exp/item2item_pmi.npz --out_model exp/ltr/lgbm_reproduce.txt --out_inference exp/inference/devset/lgbm_reproduce.json 2>&1 | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py --inference exp/inference/devset/lgbm_reproduce.json --scores exp/scores/devset/lgbm_reproduce.json --ground_truth exp/ground_truth/devset.json 2>&1 | tail -15 | tee -a "$LOG"
fi

# Step 6: Responses
if [ "$START_STEP" -le 6 ]; then
echo "[Step 6] Responses" | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py --inference exp/inference/devset/lgbm_reproduce.json --out exp/inference/devset/lgbm_final_13leg.json --mode auto --model Qwen/Qwen3-0.6B 2>&1 | tail -3 | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py --inference exp/inference/devset/lgbm_final_13leg.json --scores exp/scores/devset/lgbm_final_13leg.json --ground_truth exp/ground_truth/devset.json 2>&1 | tail -15 | tee -a "$LOG"
fi

echo "=== Done at $(date) ===" | tee -a "$LOG"
