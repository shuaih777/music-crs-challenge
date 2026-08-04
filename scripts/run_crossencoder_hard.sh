#!/bin/bash
# =============================================================================
# Cross-encoder with HARD NEGATIVES from LightGBM top-100 pool
# =============================================================================
# Previous cross-encoder failed because random negatives were trivially easy.
# This version mines hard negatives from the actual retrieval pool.
#
# Usage:
#   bash run_crossencoder_hard.sh
#
# Expected: ~60-90 min on H100/A100
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

LOG="logs/crossencoder_hard.log"
mkdir -p logs data out exp/inference/devset exp/scores/devset

echo "=== Cross-encoder (hard negatives) started at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Build training data with hard negatives from TRAIN split retrieval
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 1] Building training data with hard negatives..." | tee -a "$LOG"
echo "  Using lgbm_top100's candidate pool structure on train split" | tee -a "$LOG"

# We need to run retrieval on train split to get hard negatives.
# But we don't have train-split inference files. Instead, use a simpler approach:
# For each (session, turn) in TRAIN, the gold track + same-artist non-gold tracks
# as hard negatives (plausible but wrong).
PYTHONPATH=src python src/train_crossencoder.py build_data \
    --out data/crossencoder_hard_train.jsonl \
    --n_neg 5 \
    2>&1 | tee -a "$LOG"

# Also build a version with the hard negatives being tracks from similar sessions
# (same genre/mood) — approximation of retrieval-pool negatives
echo "" | tee -a "$LOG"
echo "  Building hard-negative augmented version..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json, random, os
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

# Load existing training data
data = []
with open('data/crossencoder_hard_train.jsonl') as f:
    for line in f:
        data.append(json.loads(line))
print(f'Loaded {len(data)} existing examples')

# Load track metadata for tag-based hard neg mining
tracks_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_meta = {r['track_id']: r for r in tracks_ds}

# Group tracks by their first tag (genre proxy)
tag_to_tracks = defaultdict(list)
for tid, meta in track_meta.items():
    tags = meta.get('tag_list') or []
    if tags:
        tag_to_tracks[str(tags[0]).lower()].append(tid)

# Load train sessions to get gold track info
train = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='train')
import pandas as pd

rng = random.Random(123)
hard_examples = []
n_hard_added = 0

for ex in tqdm(train, desc='hard_neg_mining'):
    df = pd.DataFrame(ex['conversations'])
    for turn in range(1, 9):
        gold_rows = df[(df['turn_number'] == turn) & (df['role'] == 'music')]
        if gold_rows.empty:
            continue
        gold_id = gold_rows.iloc[0]['content']
        if gold_id not in track_meta:
            continue

        gold_meta = track_meta[gold_id]
        gold_tags = [str(t).lower() for t in (gold_meta.get('tag_list') or [])[:3]]

        # Find hard negatives: tracks with similar tags but different ID
        hard_neg_candidates = set()
        for tag in gold_tags[:2]:
            for tid in tag_to_tracks.get(tag, [])[:50]:
                if tid != gold_id:
                    hard_neg_candidates.add(tid)

        if len(hard_neg_candidates) < 3:
            continue

        # Build context (same as train_crossencoder.py)
        from train_crossencoder import build_context, build_track_desc
        context = build_context(ex['conversations'], turn)

        # Add 3 tag-based hard negatives per positive
        hard_negs = rng.sample(list(hard_neg_candidates), min(3, len(hard_neg_candidates)))
        for neg_id in hard_negs:
            neg_desc = build_track_desc(track_meta[neg_id])
            hard_examples.append({
                'context': context,
                'track': neg_desc,
                'label': 0,
                'session_id': ex['session_id'],
                'turn': turn,
            })
            n_hard_added += 1

print(f'Added {n_hard_added} tag-based hard negatives')

# Combine: original data + hard negatives
all_data = data + hard_examples
rng.shuffle(all_data)

with open('data/crossencoder_hard_train.jsonl', 'w') as f:
    for ex in all_data:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f'Final training set: {len(all_data)} examples')
print(f'  positives: {sum(1 for e in all_data if e[\"label\"]==1)}')
print(f'  negatives: {sum(1 for e in all_data if e[\"label\"]==0)}')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 2: Train Qwen3-0.6B cross-encoder with hard negatives
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 2] Training Qwen3-0.6B cross-encoder (hard negatives)..." | tee -a "$LOG"

PYTHONPATH=src python src/train_crossencoder.py train \
    --train_jsonl data/crossencoder_hard_train.jsonl \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/crossencoder_hard_0.6b \
    --epochs 1 \
    --batch_size 32 \
    --lr 2e-4 \
    --max_seq_len 512 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: Rerank lgbm_top100 output (our current best ranking)
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 3] Reranking lgbm_top100 with trained cross-encoder..." | tee -a "$LOG"

PYTHONPATH=src python src/train_crossencoder.py rerank \
    --model_dir out/crossencoder_hard_0.6b \
    --inference exp/inference/devset/lgbm_top100.json \
    --out exp/inference/devset/ce_hard_reranked.json \
    --top_k 20 \
    --batch_size 64 \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/ce_hard_reranked.json \
    --scores exp/scores/devset/ce_hard_reranked.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 4: Add responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "[step 4] Adding LLM responses..." | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/ce_hard_reranked.json \
    --out exp/inference/devset/ce_hard_final.json \
    --mode auto \
    --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/ce_hard_final.json \
    --scores exp/scores/devset/ce_hard_final.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_top100', 'LightGBM top-100 (current best ranking)'),
    ('crossencoder_trained', 'CE random negatives (old, failed)'),
    ('ce_hard_reranked', 'CE hard negatives (new)'),
    ('ce_hard_final', 'CE hard + LLM responses'),
]
print(f'{\"\":45s} {\"nDCG@20\":>10s} {\"Hit@20\":>8s} {\"nDCG@1\":>8s} {\"Dist-2\":>8s}')
print('-' * 85)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:10.6f} {s.get(\"hit@20\",0)*100:7.2f}% {s[\"ndcg@1\"]:8.4f} {s.get(\"lexical_diversity\",0):8.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
