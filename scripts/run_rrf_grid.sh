#!/usr/bin/env bash
set -euo pipefail

# Grid-search RRF weights for the three strongest retrieval legs:
# metadata_qwen3 + cf_bpr + pmi_leg.
#
# Run from the repo root:
#   bash run_rrf_grid.sh

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/Modules/init/bash ]]; then
    # Make "module load ..." available in non-interactive bash.
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
fi

module load anaconda
conda activate foundation_model
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

mkdir -p exp/inference/devset/rrf_grid
mkdir -p exp/scores/devset/rrf_grid
mkdir -p logs

K_VALUES=(20 40 60 80 100)
META_WEIGHTS=(0.75 1.0 1.25 1.5)
CF_WEIGHTS=(0.5 0.75 1.0 1.25)
PMI_WEIGHTS=(0.5 0.75 1.0 1.25 1.5 2.0)

total=$(( ${#K_VALUES[@]} * ${#META_WEIGHTS[@]} * ${#CF_WEIGHTS[@]} * ${#PMI_WEIGHTS[@]} ))
i=0

echo "[rrf_grid] total configs: $total"
echo "[rrf_grid] current target: ensemble_rrf_pmi3 nDCG@20 = 0.127588595"

for k in "${K_VALUES[@]}"; do
  for wm in "${META_WEIGHTS[@]}"; do
    for wc in "${CF_WEIGHTS[@]}"; do
      for wp in "${PMI_WEIGHTS[@]}"; do
        i=$((i + 1))
        tag="rrf_m${wm}_c${wc}_p${wp}_k${k}"
        out_json="exp/inference/devset/rrf_grid/${tag}.json"
        score_json="exp/scores/devset/rrf_grid/${tag}.json"

        echo "[$i/$total] $tag"
        python src/ensemble.py rrf \
          --inputs metadata_qwen3,cf_bpr,pmi_leg \
          --weights "${wm},${wc},${wp}" \
          --rrf_k "$k" \
          --inference_dir exp/inference/devset \
          --out "$out_json" >/dev/null

        python src/evaluate.py \
          --inference "$out_json" \
          --scores "$score_json" \
          --ground_truth exp/ground_truth/devset.json >/dev/null
      done
    done
  done
done

python - <<'PY' | tee logs/rrf_grid_top20.txt
import glob
import json
import os

rows = []
for path in glob.glob("exp/scores/devset/rrf_grid/*.json"):
    with open(path) as f:
        score = json.load(f)
    rows.append((
        score["ndcg@20"],
        score["ndcg@10"],
        score["ndcg@1"],
        score["hit@20"],
        os.path.basename(path),
    ))

print("\nTop 20 RRF grid configs:")
for n20, n10, n1, hit20, name in sorted(rows, reverse=True)[:20]:
    print(f"{name:45s} n20={n20:.9f} n10={n10:.9f} n1={n1:.6f} hit20={hit20:.6f}")
PY

echo
echo "[rrf_grid] wrote summary to logs/rrf_grid_top20.txt"
