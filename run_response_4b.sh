#!/bin/bash
# =============================================================================
# Qwen3-4B Response Generation (higher quality than 0.6B)
# =============================================================================
# Independent from ranking — only improves Distinct-2 and LLM-as-judge score.
# Can run in parallel with run_biencoder_hardneg.sh on a separate GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_response_4b.sh
#
# Expected: ~30-45 min on H100. Distinct-2 from 0.26 → 0.30+
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

LOG="logs/response_4b.log"
mkdir -p logs exp/inference/devset exp/scores/devset

echo "=== Qwen3-4B response generation at $(date) ===" | tee "$LOG"

# Pick the best ranking available
BEST_RANKING=""
for f in exp/inference/devset/lgbm_hardneg.json \
         exp/inference/devset/lgbm_biencoder_large.json \
         exp/inference/devset/lgbm_biencoder.json \
         exp/inference/devset/lgbm_top100.json; do
    if [ -f "$f" ]; then
        BEST_RANKING="$f"
        break
    fi
done

if [ -z "$BEST_RANKING" ]; then
    echo "ERROR: No ranking file found" | tee -a "$LOG"
    exit 1
fi
echo "Input ranking: $BEST_RANKING" | tee -a "$LOG"

# =============================================================================
# Step 1: Generate responses with Qwen3-4B
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Qwen3-4B response generation ======" | tee -a "$LOG"

OUTPUT="exp/inference/devset/best_4b_response.json"

PYTHONPATH=src python src/generate_responses.py \
    --inference "$BEST_RANKING" \
    --out "$OUTPUT" \
    --mode llm \
    --model Qwen/Qwen3-4B \
    --batch_size 8 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 2: Evaluate
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Evaluate ======" | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference "$OUTPUT" \
    --scores exp/scores/devset/best_4b_response.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 3: Compare response quality
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: Response quality comparison ======" | tee -a "$LOG"

python -c "
import json, os

# Compare Distinct-2 across different response generators
configs = [
    ('lgbm_biencoder_large_final', '0.6B response (current)'),
    ('best_4b_response', '4B response (new)'),
]
print(f'{\"\":35s} {\"nDCG@20\":>9s} {\"Dist-2\":>8s} {\"Dist-2 delta\":>12s}')
print('-'*70)
baseline_d2 = None
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        d2 = s.get('lexical_diversity', 0)
        if baseline_d2 is None:
            baseline_d2 = d2
            delta = ''
        else:
            delta = f'{d2 - baseline_d2:+.4f}'
        print(f'{desc:35s} {s[\"ndcg@20\"]:9.6f} {d2:8.4f} {delta:>12s}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
