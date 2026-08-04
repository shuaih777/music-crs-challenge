#!/bin/bash
# =============================================================================
# Music-CRS Extended Experiments — H100 full utilization
# =============================================================================
# Prerequisite: run_pipeline.sh already completed (lgbm_reranked.json exists)
#
# This script:
#   1. Generates 3 new dense retrieval legs (attributes/lyrics/siglip2)
#   2. Runs 6-leg and 8-leg LightGBM (wider recall pool)
#   3. Trains a cross-encoder reranker (Qwen3-0.6B LoRA)
#   4. Generates high-quality responses (Qwen3-4B if available, else 0.6B)
#   5. Prints leaderboard
#
# Usage:
#   bash run_experiments.sh
#
# Expected wall-clock on H100: ~60-90 min total
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup (same as run_pipeline.sh)
# ---------------------------------------------------------------------------
if ! command -v module >/dev/null 2>&1; then
    if [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi

module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/experiments.log"
mkdir -p logs exp/inference/devset exp/scores/devset exp/ltr

echo "=== Extended experiments started at $(date) ===" | tee "$LOG"
python -c "import torch; print(f'[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>&1 | tee -a "$LOG"

# =============================================================================
# PHASE 1: Generate new dense retrieval legs (expand recall diversity)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== PHASE 1: New dense retrieval legs ======" | tee -a "$LOG"

# Each uses prior-track-pool dense (same approach as metadata_qwen3/cf_bpr)
# but with different track-embedding modalities → different tracks get retrieved
for EMBED in attributes-qwen3_embedding_0.6b lyrics-qwen3_embedding_0.6b image-siglip2; do
    TAG="${EMBED//-/_}"
    TAG="${TAG//./_}"
    TAG="leg_${TAG}"
    OUT="exp/inference/devset/${TAG}.json"

    if [ -f "$OUT" ]; then
        echo "  [skip] $OUT already exists" | tee -a "$LOG"
        continue
    fi

    echo "  [run] Dense leg: $EMBED → $TAG" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --embed "$EMBED" \
        --pooling decay \
        --weight_schedule descending \
        --tag "$TAG" \
        2>&1 | tail -5 | tee -a "$LOG"

    PYTHONPATH=src python src/evaluate.py \
        --inference "$OUT" \
        --scores "exp/scores/devset/${TAG}.json" \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | grep -E "ndcg@20|hit@20" | tee -a "$LOG"
done

# Also generate a BM25-only with state query (focused retrieval, no dense)
STATE_BM25="exp/inference/devset/state_bm25_focused.json"
if [ ! -f "$STATE_BM25" ]; then
    echo "  [run] State-focused BM25 leg" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$STATE_BM25" \
        --bm25_only \
        --states_jsonl exp/states/test.jsonl \
        --query_mode state \
        --state_subtract_rejected --neg_weight 0.3 \
        --tag state_bm25_focused \
        2>&1 | tail -5 | tee -a "$LOG"

    PYTHONPATH=src python src/evaluate.py \
        --inference "$STATE_BM25" \
        --scores exp/scores/devset/state_bm25_focused.json \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | grep -E "ndcg@20|hit@20" | tee -a "$LOG"
fi

# =============================================================================
# PHASE 2: Multi-leg LightGBM (wider recall pool)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== PHASE 2: Multi-leg LightGBM ======" | tee -a "$LOG"

# 6-leg: original 3 + 3 new dense
echo "  [2a] 6-leg LightGBM (meta + cf + pmi + attributes + lyrics + siglip)" | tee -a "$LOG"
PYTHONPATH=src python src/train_ltr.py \
    --legs metadata_qwen3,cf_bpr,pmi_leg,leg_attributes_qwen3_embedding_0_6b,leg_lyrics_qwen3_embedding_0_6b,leg_image_siglip2 \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_6leg_model.txt \
    --out_inference exp/inference/devset/lgbm_6leg.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_6leg.json \
    --scores exp/scores/devset/lgbm_6leg.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# 8-leg: 6 + decay_descending + last_track + state_bm25
echo "" | tee -a "$LOG"
echo "  [2b] 8-leg LightGBM (all available independent legs)" | tee -a "$LOG"
PYTHONPATH=src python src/train_ltr.py \
    --legs metadata_qwen3,cf_bpr,pmi_leg,leg_attributes_qwen3_embedding_0_6b,leg_lyrics_qwen3_embedding_0_6b,leg_image_siglip2,decay_descending,state_bm25_focused \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_8leg_model.txt \
    --out_inference exp/inference/devset/lgbm_8leg.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_8leg.json \
    --scores exp/scores/devset/lgbm_8leg.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# PHASE 3: Cross-encoder reranker (Qwen3-0.6B pointwise scoring on top-50)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== PHASE 3: Cross-encoder reranker ======" | tee -a "$LOG"

# Use the best LightGBM output as the candidate source
BEST_LGBM="exp/inference/devset/lgbm_8leg.json"
if [ ! -f "$BEST_LGBM" ]; then
    BEST_LGBM="exp/inference/devset/lgbm_6leg.json"
fi
if [ ! -f "$BEST_LGBM" ]; then
    BEST_LGBM="exp/inference/devset/lgbm_reranked.json"
fi

# Cross-encoder: score (conversation, track_metadata) pairs
# This requires src/rerank_crossencoder.py which we'll create
if [ -f "src/rerank_crossencoder.py" ]; then
    echo "  [run] Cross-encoder reranking on $BEST_LGBM" | tee -a "$LOG"
    PYTHONPATH=src python src/rerank_crossencoder.py \
        --inference "$BEST_LGBM" \
        --out exp/inference/devset/crossencoder_reranked.json \
        --model Qwen/Qwen3-0.6B \
        --batch_size 32 \
        --top_k 20 \
        2>&1 | tee -a "$LOG"

    PYTHONPATH=src python src/evaluate.py \
        --inference exp/inference/devset/crossencoder_reranked.json \
        --scores exp/scores/devset/crossencoder_reranked.json \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | tail -15 | tee -a "$LOG"
else
    echo "  [skip] src/rerank_crossencoder.py not found — skipping" | tee -a "$LOG"
fi

# =============================================================================
# PHASE 4: High-quality response generation
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== PHASE 4: Response generation ======" | tee -a "$LOG"

# Pick the best reranked output for final response generation
BEST_RANKING="exp/inference/devset/lgbm_8leg.json"
if [ ! -f "$BEST_RANKING" ]; then
    BEST_RANKING="exp/inference/devset/lgbm_6leg.json"
fi
if [ ! -f "$BEST_RANKING" ]; then
    BEST_RANKING="exp/inference/devset/lgbm_reranked.json"
fi

# Try Qwen3-4B first (better responses), fall back to 0.6B
echo "  [run] LLM response generation on $BEST_RANKING" | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference "$BEST_RANKING" \
    --out exp/inference/devset/final_submission.json \
    --mode llm \
    --model Qwen/Qwen3-0.6B \
    --batch_size 16 \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/final_submission.json \
    --scores exp/scores/devset/final_submission.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# PHASE 5: Final leaderboard
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== FINAL LEADERBOARD ======" | tee -a "$LOG"
python -c "
import json, os

configs = [
    'ensemble_rrf_pmi3_tuned',
    'lgbm_reranked',
    'lgbm_6leg',
    'lgbm_8leg',
    'crossencoder_reranked',
    'lgbm_final',
    'final_submission',
]
print(f'{\"Config\":35s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"Dist-2\":>8s} {\"CatDiv\":>8s}')
print('-' * 75)
for tag in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{tag:35s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s.get(\"lexical_diversity\",0):8.4f} {s.get(\"catalog_diversity\",0):8.4f}')
    else:
        print(f'{tag:35s} {\"(not run)\":>10s}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Experiments complete at $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Best submission candidate: exp/inference/devset/final_submission.json" | tee -a "$LOG"
echo "Full log: $LOG" | tee -a "$LOG"
