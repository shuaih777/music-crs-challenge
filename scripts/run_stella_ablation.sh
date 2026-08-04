#!/bin/bash
# =============================================================================
# Stella ablation: add biencoder_stella as a 12th leg on top of the best 11-leg
# =============================================================================
# Baseline (11-leg, all+mxbai): nDCG@20 = 0.17854
# Fair test: does stella add incremental value on top of the saturated best?
#
# Uses train_ltr_v2.py (parallel folds) + the biencoder_cosine feature from
# out/biencoder_large, identical to how 0.1785 was produced.
# =============================================================================

set -euo pipefail

LOG="logs/stella_ablation.log"
mkdir -p logs exp/ltr exp/inference/devset exp/scores/devset

echo "=== Stella ablation (12-leg) at $(date) ===" | tee "$LOG"

# 11-leg best + stella
LEGS="metadata_qwen3,cf_bpr,pmi_leg,decay_descending,bm25_norepeat,biencoder_large_top100,biencoder_top100,e5_large_top100,biencoder_last2turns_top100,biencoder_current_only_top100,biencoder_mxbai_top100,biencoder_stella_top100"

# Sanity: all leg files present
for LEG in $(echo "$LEGS" | tr ',' ' '); do
    if [ ! -f "exp/inference/devset/${LEG}.json" ]; then
        echo "ERROR: missing leg exp/inference/devset/${LEG}.json" | tee -a "$LOG"
        exit 1
    fi
done
echo "Legs (12): $LEGS" | tee -a "$LOG"

# Train + CV predict (parallel folds)
PYTHONPATH=src python src/train_ltr_v2.py \
    --legs "$LEGS" \
    --biencoder_dir out/biencoder_large \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_abl_plus_stella.txt \
    --out_inference exp/inference/devset/lgbm_abl_plus_stella.json \
    2>&1 | grep -E "Loading|positives|fold|Feature matrix|importance|:" | tail -25 | tee -a "$LOG"

# Evaluate
PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_abl_plus_stella.json \
    --scores exp/scores/devset/lgbm_abl_plus_stella.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | grep -E "ndcg@20|hit@20|ndcg@1\b" | tee -a "$LOG"

# Comparison
echo "" | tee -a "$LOG"
echo "=== COMPARISON ===" | tee -a "$LOG"
python3 -c "
import json, os
rows = [
    ('11-leg best (all+mxbai)', 'exp/scores/devset/lgbm_abl_all_plus_mxbai.json'),
    ('12-leg (+ stella)',       'exp/scores/devset/lgbm_abl_plus_stella.json'),
]
print(f'{\"\":28s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s}')
print('-'*56)
base = None
for name, path in rows:
    if not os.path.exists(path): continue
    s = json.load(open(path))
    nd = s['ndcg@20']
    if base is None: base = nd
    delta = f'  ({nd-base:+.5f})' if nd != base else ''
    print(f'{name:28s} {nd:9.5f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f}{delta}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
