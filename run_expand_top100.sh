#!/bin/bash
# =============================================================================
# Expand all retrieval legs to top-100 output and retrain LightGBM
# =============================================================================
# This is the #1 highest-ROI experiment: current legs only output top-20,
# limiting the candidate pool to ~96 tracks/turn. Expanding to top-100
# should raise gold-in-pool from ~40% to 55-60%, giving LightGBM much
# more room to find gold tracks.
#
# Expected: nDCG@20 from 0.141 → 0.17-0.19
# Wall-clock: ~30 min on H100 (legs: ~3 min each × 8 + LTR: ~10 min)
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

LOG="logs/expand_top100.log"
mkdir -p logs exp/inference/devset/top100 exp/scores/devset exp/ltr

echo "=== Expand legs to top-100 — started $(date) ===" | tee "$LOG"

N_OUTPUT=100
OUTDIR="exp/inference/devset/top100"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Re-run baselines_v3 legs with --n_output 100
# ─────────────────────────────────────────────────────────────────────────────
echo "[step 1] Running dense legs with n_output=$N_OUTPUT..." | tee -a "$LOG"

declare -A LEGS=(
    ["metadata_qwen3"]="metadata-qwen3_embedding_0.6b"
    ["cf_bpr"]="cf-bpr"
    ["attributes_qwen3"]="attributes-qwen3_embedding_0.6b"
    ["lyrics_qwen3"]="lyrics-qwen3_embedding_0.6b"
    ["image_siglip2"]="image-siglip2"
)

for LEG in "${!LEGS[@]}"; do
    EMBED="${LEGS[$LEG]}"
    OUT="$OUTDIR/${LEG}.json"
    if [ -f "$OUT" ]; then
        echo "  [skip] $OUT exists" | tee -a "$LOG"
        continue
    fi
    echo "  [run] $LEG (embed=$EMBED, n_output=$N_OUTPUT)" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --embed "$EMBED" \
        --pooling decay \
        --weight_schedule descending \
        --n_output $N_OUTPUT \
        --tag "${LEG}_top100" \
        2>&1 | tail -3 | tee -a "$LOG"
done

# decay_descending (uses audio-laion_clap with specific settings)
OUT="$OUTDIR/decay_descending.json"
if [ ! -f "$OUT" ]; then
    echo "  [run] decay_descending (CLAP, n_output=$N_OUTPUT)" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --embed audio-laion_clap \
        --pooling decay \
        --weight_schedule descending \
        --n_output $N_OUTPUT \
        --tag decay_desc_top100 \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# BM25-only (no-repeat filter)
OUT="$OUTDIR/bm25_norepeat.json"
if [ ! -f "$OUT" ]; then
    echo "  [run] bm25_norepeat (n_output=$N_OUTPUT)" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --bm25_only \
        --n_output $N_OUTPUT \
        --tag bm25_top100 \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# State-focused BM25
OUT="$OUTDIR/state_bm25_focused.json"
if [ ! -f "$OUT" ]; then
    echo "  [run] state_bm25_focused (n_output=$N_OUTPUT)" | tee -a "$LOG"
    PYTHONPATH=src python src/baselines_v3.py \
        --output "$OUT" \
        --bm25_only \
        --states_jsonl exp/states/test.jsonl \
        --query_mode state \
        --state_subtract_rejected --neg_weight 0.3 \
        --n_output $N_OUTPUT \
        --tag state_bm25_top100 \
        2>&1 | tail -3 | tee -a "$LOG"
fi

# PMI leg
OUT="$OUTDIR/pmi_leg.json"
if [ ! -f "$OUT" ]; then
    echo "  [run] pmi_leg (n_output=$N_OUTPUT)" | tee -a "$LOG"
    PYTHONPATH=src python src/retrieval_legs.py pmi \
        --pmi_path exp/item2item_pmi.npz \
        --out "$OUT" \
        --n_output $N_OUTPUT \
        2>&1 | tail -3 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "[step 1 done] All legs generated in $OUTDIR/" | tee -a "$LOG"
ls "$OUTDIR"/*.json | wc -l | xargs echo "  total leg files:" | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Measure new gold-in-pool ceiling
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 2] Measuring gold-in-pool ceiling..." | tee -a "$LOG"
python -c "
import json, os, glob
gt = json.load(open('exp/ground_truth/devset.json'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}

# Union of all top-100 legs
pool = {}
for f in glob.glob('$OUTDIR/*.json'):
    rows = json.load(open(f))
    for r in rows:
        key = (r['session_id'], r['turn_number'])
        if key not in pool:
            pool[key] = set()
        pool[key].update(r['predicted_track_ids'])

in_pool = 0
for key, gold_id in gold_by_key.items():
    if gold_id in pool.get(key, set()):
        in_pool += 1
total = len(gold_by_key)
print(f'  Gold-in-pool: {in_pool}/{total} ({in_pool/total*100:.1f}%)')
print(f'  Avg pool size per turn: {sum(len(v) for v in pool.values())/len(pool):.0f}')
" 2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Retrain LightGBM on expanded pool
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 3] Training LightGBM on expanded top-100 pool..." | tee -a "$LOG"

# List all available legs in the top100 dir
LEGS_LIST=$(ls "$OUTDIR"/*.json | xargs -I{} basename {} .json | paste -sd, -)
echo "  Legs: $LEGS_LIST" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$LEGS_LIST" \
    --inference_dir "$OUTDIR" \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 \
    --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_top100_model.txt \
    --out_inference exp/inference/devset/lgbm_top100.json \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 4] Evaluating..." | tee -a "$LOG"
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_top100.json \
    --scores exp/scores/devset/lgbm_top100.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Add LLM responses
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[step 5] Generating responses..." | tee -a "$LOG"
PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_top100.json \
    --out exp/inference/devset/lgbm_top100_final.json \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_top100_final.json \
    --scores exp/scores/devset/lgbm_top100_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_8leg', 'Previous best (top-20 legs)'),
    ('lgbm_top100', 'LightGBM on top-100 legs'),
    ('lgbm_top100_final', 'Top-100 + LLM responses'),
]
print(f'{\"Config\":40s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"Dist-2\":>8s}')
print('-' * 70)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:40s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s.get(\"lexical_diversity\",0):8.4f}')
    else:
        print(f'{desc:40s} {\"(missing)\":>10s}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
