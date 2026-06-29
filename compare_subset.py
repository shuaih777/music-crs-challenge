"""Compare two rankings on the SAME subset of devset turns.

Evaluates nDCG@{1,10,20} on exactly the (session_id, turn_number) pairs
present in the listwise output, for both the baseline 13-leg ranking and
the listwise-reranked ranking. Fair apples-to-apples on the subset.

Usage:
    PYTHONPATH=src python compare_subset.py \
        --baseline exp/inference/devset/lgbm_abl_plus_nvembed.json \
        --listwise exp/inference/devset/listwise_finetuned.json
"""
import argparse, json, sys
sys.path.insert(0, "src")
from metrics import compute_recsys_metrics

p = argparse.ArgumentParser()
p.add_argument("--baseline", default="exp/inference/devset/lgbm_abl_plus_nvembed.json")
p.add_argument("--listwise", required=True)
p.add_argument("--ground_truth", default="exp/ground_truth/devset.json")
a = p.parse_args()

gt = {(r["session_id"], int(r["turn_number"])): r["ground_truth_track_id"]
      for r in json.load(open(a.ground_truth))}
base = {(r["session_id"], int(r["turn_number"])): r["predicted_track_ids"]
        for r in json.load(open(a.baseline))}
lw = {(r["session_id"], int(r["turn_number"])): r["predicted_track_ids"]
      for r in json.load(open(a.listwise))}

# subset = turns present in listwise output (the 1000 reranked)
keys = [k for k in lw if k in gt and k in base]
print(f"Comparing on {len(keys)} shared turns")

def avg_ndcg(pred_map):
    import numpy as np
    accum = {}
    for k in keys:
        m = compute_recsys_metrics(pred_map[k], [gt[k]], [1, 10, 20])
        for kk, vv in m.items():
            accum.setdefault(kk, []).append(vv)
    return {kk: float(np.mean(vv)) for kk, vv in accum.items()}

b = avg_ndcg(base)
l = avg_ndcg(lw)
print(f"{'metric':12s} {'13-leg':>10s} {'+listwise':>10s} {'delta':>10s}")
for kk in b:
    print(f"{kk:12s} {b[kk]:10.5f} {l[kk]:10.5f} {l[kk]-b[kk]:+10.5f}")
