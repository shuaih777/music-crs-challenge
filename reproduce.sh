#!/bin/bash
# =============================================================================
# REPRODUCE.sh — Full pipeline from scratch to final submission
# =============================================================================
# Reproduces the 13-leg + NV-Embed system (devset nDCG@20 = 0.1848)
#
# Prerequisites:
#   - GPU: 1x H100 80GB (or A100 80GB)
#   - Python 3.10+ with: pip install -r requirements.txt -r requirements-gpu.txt
#   - Additional: pip install lightgbm gensim
#   - Internet access to HuggingFace (for datasets + model downloads)
#
# Total runtime: ~12-16h on single H100 (mostly bi-encoder training)
#   - With 4 GPUs running in parallel: ~4-5h
#
# Usage:
#   bash reproduce.sh           # full pipeline
#   bash reproduce.sh --skip-training  # use pre-trained models (if available)
#
# Output:
#   exp/inference/devset/lgbm_13leg_final.json  (devset evaluation)
#   exp/inference/blind_b/submission.json       (blind-B submission)
# =============================================================================

set -uo pipefail

# Environment setup (adjust to your cluster)
if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/reproduce.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr exp/tracks data

echo "=== Reproduce started at $(date) ===" | tee "$LOG"
echo "System: $(python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")' 2>/dev/null)" | tee -a "$LOG"

SKIP_TRAINING="${1:-}"

# =============================================================================
# STAGE 1: Basic retrieval legs (CPU, ~5 min total)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== STAGE 1: Basic retrieval legs ======" | tee -a "$LOG"

# 1a. BM25
OUT="exp/inference/devset/bm25_norepeat.json"
if [ ! -f "$OUT" ]; then
    echo "  [1/13] BM25 no-repeat..." | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" --bm25_only --n_output 100 --tag bm25 \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# 1b. metadata-qwen3 dense
OUT="exp/inference/devset/metadata_qwen3.json"
if [ ! -f "$OUT" ]; then
    echo "  [2/13] metadata-qwen3 dense..." | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" --embed metadata-qwen3_embedding_0.6b \
        --pooling decay --weight_schedule descending --n_output 100 --tag meta \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# 1c. cf-bpr dense
OUT="exp/inference/devset/cf_bpr.json"
if [ ! -f "$OUT" ]; then
    echo "  [3/13] cf-bpr dense..." | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" --embed cf-bpr \
        --pooling decay --weight_schedule descending --n_output 100 --tag cf \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# 1d. decay_descending (metadata-qwen3 with decay pooling)
OUT="exp/inference/devset/decay_descending.json"
if [ ! -f "$OUT" ]; then
    echo "  [4/13] decay_descending..." | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" --embed metadata-qwen3_embedding_0.6b \
        --pooling decay --weight_schedule descending --n_output 100 --tag decay \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# 1e. PMI co-occurrence
OUT="exp/inference/devset/pmi_leg.json"
if [ ! -f "$OUT" ]; then
    echo "  [5/13] PMI leg (build + inference)..." | tee -a "$LOG"
    PYTHONPATH=src python src/build_item2item.py --out exp/item2item_pmi.npz \
        2>&1 | tail -3 | tee -a "$LOG"
    PYTHONPATH=src python src/retrieval_legs.py pmi \
        --pmi_path exp/item2item_pmi.npz --out "$OUT" \
        2>&1 | tail -3 | tee -a "$LOG"
fi

echo "  Stage 1 complete: 5 basic legs" | tee -a "$LOG"

# =============================================================================
# STAGE 2: Bi-encoder training + inference (~10-14h sequential, ~3h parallel)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== STAGE 2: Bi-encoder legs (8 models) ======" | tee -a "$LOG"

declare -A BIENC_MODELS
BIENC_MODELS[biencoder_top100]="BAAI/bge-base-en-v1.5"
BIENC_MODELS[biencoder_large_top100]="BAAI/bge-large-en-v1.5"
BIENC_MODELS[e5_large_top100]="intfloat/e5-large-v2"
BIENC_MODELS[biencoder_stella_top100]="dunzhang/stella_en_400M_v5"
BIENC_MODELS[biencoder_mxbai_top100]="mixedbread-ai/mxbai-embed-large-v1"
BIENC_MODELS[biencoder_nv_embed_top100]="nvidia/NV-Embed-v2"

COUNTER=6
for KEY in biencoder_top100 biencoder_large_top100 e5_large_top100 \
           biencoder_stella_top100 biencoder_mxbai_top100 biencoder_nv_embed_top100; do
    MODEL_ID="${BIENC_MODELS[$KEY]}"
    OUT="exp/inference/devset/${KEY}.json"
    OUT_DIR="out/${KEY//_top100/}"

    if [ -f "$OUT" ]; then
        echo "  [${COUNTER}/13] skip: $KEY" | tee -a "$LOG"
    elif [ "$SKIP_TRAINING" = "--skip-training" ] && [ -d "$OUT_DIR" ]; then
        echo "  [${COUNTER}/13] $KEY (inference only, model exists)..." | tee -a "$LOG"
        PYTHONPATH=src python src/train_biencoder.py encode_tracks \
            --model_dir "$OUT_DIR" --out "exp/tracks/${KEY//_top100/}_tracks.npy" --batch_size 256 \
            2>&1 | tail -3 | tee -a "$LOG"
        PYTHONPATH=src python src/train_biencoder.py inference \
            --model_dir "$OUT_DIR" --track_emb "exp/tracks/${KEY//_top100/}_tracks.npy" \
            --out "$OUT" --n_output 100 \
            2>&1 | tail -3 | tee -a "$LOG"
    else
        echo "  [${COUNTER}/13] $KEY (train + inference): $MODEL_ID" | tee -a "$LOG"
        LORA_FLAG=""
        BATCH=64
        case "$KEY" in
            biencoder_nv_embed_top100) LORA_FLAG="--lora"; BATCH=4 ;;
            biencoder_stella_top100) BATCH=64 ;;
            *) BATCH=128 ;;
        esac
        PYTHONPATH=src python src/train_biencoder.py all \
            --model_id "$MODEL_ID" \
            --output_dir "$OUT_DIR" \
            --out_leg "$OUT" \
            --batch_size $BATCH \
            --epochs 3 \
            --n_output 100 \
            $LORA_FLAG \
            2>&1 | tail -10 | tee -a "$LOG"
    fi
    COUNTER=$((COUNTER + 1))
done

# Multi-query legs (using bge-large model)
for MQ_KEY in biencoder_last2turns_top100 biencoder_current_only_top100; do
    OUT="exp/inference/devset/${MQ_KEY}.json"
    if [ -f "$OUT" ]; then
        echo "  [${COUNTER}/13] skip: $MQ_KEY" | tee -a "$LOG"
    else
        echo "  [${COUNTER}/13] $MQ_KEY (multi-query with bge-large)..." | tee -a "$LOG"
        MODE="last2"
        [ "$MQ_KEY" = "biencoder_current_only_top100" ] && MODE="current"

        PYTHONPATH=src python -c "
import json, os, sys, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)
track_embs = np.load('exp/tracks/biencoder_large_tracks.npy')
track_ids = json.load(open('exp/tracks/biencoder_large_tracks_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')

queries, meta_list = [], []
for ex in test:
    convos = ex['conversations']
    for tn in range(1, 9):
        prior = [c['content'] for c in convos if c['role']=='music' and c['turn_number'] < tn]
        if '$MODE' == 'last2':
            lines = []
            for c in convos:
                if c['turn_number'] > tn: break
                if c['turn_number'] >= max(1, tn-2):
                    if c['role'] == 'user': lines.append(f\"User: {c['content']}\")
                    elif c['role'] == 'assistant': lines.append(f\"Assistant: {c['content'][:80]}\")
            query = '\n'.join(lines[-6:])
        else:
            query = next((c['content'] for c in convos if c['turn_number']==tn and c['role']=='user'), '')
        queries.append(query or 'music')
        meta_list.append({'session_id': ex['session_id'], 'user_id': ex['user_id'],
                          'turn_number': tn, 'prior_tracks': prior})

q_embs = model.encode(queries, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
rows = []
for i, m in enumerate(tqdm(meta_list)):
    scores = track_embs @ q_embs[i]
    for tid in m['prior_tracks']:
        if tid in track_to_idx: scores[track_to_idx[tid]] = -1e9
    top_idx = np.argsort(-scores)[:100]
    rows.append({'session_id': m['session_id'], 'user_id': m['user_id'],
                 'turn_number': m['turn_number'],
                 'predicted_track_ids': [track_ids[j] for j in top_idx],
                 'predicted_response': ''})
with open('$OUT', 'w') as f: json.dump(rows, f, ensure_ascii=False)
print(f'Wrote {len(rows)} to $OUT')
" 2>&1 | tail -5 | tee -a "$LOG"
    fi
    COUNTER=$((COUNTER + 1))
done

echo "  Stage 2 complete: 8 bi-encoder legs (total 13 legs)" | tee -a "$LOG"

# =============================================================================
# STAGE 3: LightGBM LambdaRank reranking
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== STAGE 3: LightGBM reranking ======" | tee -a "$LOG"

ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           biencoder_top100 biencoder_large_top100 e5_large_top100 \
           biencoder_stella_top100 biencoder_mxbai_top100 biencoder_nv_embed_top100 \
           biencoder_last2turns_top100 biencoder_current_only_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        ALL_LEGS="${ALL_LEGS:+$ALL_LEGS,}$LEG"
    else
        echo "  WARNING: missing leg $LEG" | tee -a "$LOG"
    fi
done
echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr_v2.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_13leg_final.txt \
    --out_inference exp/inference/devset/lgbm_13leg_final.json \
    2>&1 | tee -a "$LOG"

# =============================================================================
# STAGE 4: Response generation
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== STAGE 4: Response generation ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_13leg_final.json \
    --out exp/inference/devset/lgbm_13leg_with_responses.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -5 | tee -a "$LOG"

# =============================================================================
# STAGE 5: Evaluate
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== STAGE 5: Evaluation ======" | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_13leg_with_responses.json \
    --scores exp/scores/devset/lgbm_13leg_with_responses.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# =============================================================================
# DONE
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== Reproduce complete at $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Final devset results:" | tee -a "$LOG"
python -c "
import json
s = json.load(open('exp/scores/devset/lgbm_13leg_with_responses.json'))
print(f'  nDCG@20 = {s[\"ndcg@20\"]:.4f}')
print(f'  Hit@20  = {s.get(\"hit@20\",0)*100:.1f}%')
print(f'  Dist-2  = {s.get(\"lexical_diversity\",0):.4f}')
" 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Submission file: exp/inference/devset/lgbm_13leg_with_responses.json" | tee -a "$LOG"
echo "For Blind-B: re-run all legs with --split Blind-B, then score with the saved LightGBM model" | tee -a "$LOG"
