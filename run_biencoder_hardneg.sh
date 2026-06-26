#!/bin/bash
# =============================================================================
# Hard-Negative Bi-Encoder Retrain (bge-large, 2nd round)
# =============================================================================
# Takes the trained bge-large bi-encoder, mines hard negatives from its own
# top-50 retrievals, and retrains 1 epoch to sharpen the embeddings.
#
# This is the standard "mine → retrain" loop from DPR/ANCE papers.
# Expected: embeddings become better at distinguishing "close but wrong" tracks.
#
# Usage:
#   bash run_biencoder_hardneg.sh
#
# Expected wall-clock: ~45-60 min on H100
# Expected lift: nDCG@20 0.1649 → 0.17-0.175
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

LOG="logs/biencoder_hardneg.log"
mkdir -p logs data out/biencoder_large_hardneg exp/inference/devset exp/scores/devset exp/ltr

echo "=== Hard-neg bi-encoder retrain at $(date) ===" | tee "$LOG"

# =============================================================================
# Step 1: Mine hard negatives from bge-large's own top-50 on TRAIN split
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 1: Mine hard negatives ======" | tee -a "$LOG"

HARD_NEG_DATA="data/biencoder_hardneg_pairs.jsonl"
if [ -f "$HARD_NEG_DATA" ]; then
    echo "  [skip] $HARD_NEG_DATA exists" | tee -a "$LOG"
else
    PYTHONPATH=src python -c "
import json, os, sys, random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd

sys.path.insert(0, 'src')
from train_biencoder import build_conversation_text, build_track_text

print('Loading trained bge-large model...', flush=True)
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)

print('Loading track metadata...', flush=True)
tracks_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_meta = {r['track_id']: r for r in tracks_ds}
track_ids = list(track_meta.keys())
track_texts = [build_track_text(r) for r in tqdm(tracks_ds, desc='track texts')]

print('Encoding all tracks...', flush=True)
track_embs = model.encode(track_texts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)

print('Loading train sessions...', flush=True)
train = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='train')

rng = random.Random(42)
pairs = []
n_hard = 0

print('Mining hard negatives from top-50...', flush=True)
# Process in chunks to manage memory
for ex in tqdm(train, desc='mining'):
    convos = ex['conversations']
    df = pd.DataFrame(convos)
    for turn in range(1, 9):
        gold_rows = df[(df['turn_number'] == turn) & (df['role'] == 'music')]
        if gold_rows.empty:
            continue
        gold_id = gold_rows.iloc[0]['content']
        if gold_id not in track_meta:
            continue

        conv_text = build_conversation_text(convos, turn)
        gold_text = build_track_text(track_meta[gold_id])

        # Encode conversation
        q_emb = model.encode(conv_text, normalize_embeddings=True)

        # Find top-50 nearest tracks
        scores = track_embs @ q_emb
        top50_idx = np.argsort(-scores)[:50]
        top50_ids = [track_ids[i] for i in top50_idx]

        # Hard negatives: top-50 tracks that are NOT gold (ranks 5-50)
        hard_negs = [tid for tid in top50_ids[5:] if tid != gold_id][:3]

        # Also keep 2 random negatives for diversity
        rand_negs = rng.sample(track_ids, 2)
        rand_negs = [t for t in rand_negs if t != gold_id][:2]

        # Positive pair
        pairs.append({'query': conv_text, 'positive': gold_text,
                      'negative': build_track_text(track_meta[hard_negs[0]]) if hard_negs else None})
        n_hard += 1

        # Additional hard neg pairs (for TripletLoss style)
        for neg_id in hard_negs[:2] + rand_negs[:1]:
            if neg_id in track_meta:
                pairs.append({
                    'query': conv_text,
                    'positive': gold_text,
                    'negative': build_track_text(track_meta[neg_id]),
                })

# Write JSONL
os.makedirs('data', exist_ok=True)
with open('$HARD_NEG_DATA', 'w') as f:
    for p in pairs:
        if p.get('negative'):
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
print(f'Wrote {len([p for p in pairs if p.get(\"negative\")])} triplets to $HARD_NEG_DATA')
print(f'Hard negatives mined: {n_hard}')
" 2>&1 | tee -a "$LOG"
fi

# =============================================================================
# Step 2: Retrain bge-large with hard negatives (1 epoch, lower LR)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 2: Retrain with hard negatives (1 epoch) ======" | tee -a "$LOG"

PYTHONPATH=src python -c "
import json, sys
try:
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)

print('Loading trained bge-large model...', flush=True)
model = SentenceTransformer('out/biencoder_large', trust_remote_code=True)

print('Loading hard-negative triplets...', flush=True)
examples = []
with open('$HARD_NEG_DATA') as f:
    for line in f:
        d = json.loads(line)
        if d.get('negative'):
            examples.append(InputExample(texts=[d['query'], d['positive'], d['negative']]))
print(f'  {len(examples)} triplets', flush=True)

# Use TripletLoss with hard negatives
train_dataloader = DataLoader(examples, shuffle=True, batch_size=128)
train_loss = losses.TripletLoss(model, distance_metric=losses.TripletDistanceMetric.COSINE,
                                 triplet_margin=0.2)

warmup_steps = int(len(train_dataloader) * 0.1)
print(f'Training: 1 epoch, batch=128, lr=5e-6, warmup={warmup_steps}', flush=True)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=warmup_steps,
    output_path='out/biencoder_large_hardneg',
    optimizer_params={'lr': 5e-6},  # Very low LR for refinement
    show_progress_bar=True,
    use_amp=True,
)
print('Model saved to out/biencoder_large_hardneg')
" 2>&1 | tee -a "$LOG"

# =============================================================================
# Step 3: Re-encode tracks + inference
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 3: Re-encode tracks + inference ======" | tee -a "$LOG"

PYTHONPATH=src python src/train_biencoder.py encode_tracks \
    --model_dir out/biencoder_large_hardneg \
    --out exp/tracks/biencoder_large_hardneg.npy \
    --batch_size 256 \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/train_biencoder.py inference \
    --model_dir out/biencoder_large_hardneg \
    --track_emb exp/tracks/biencoder_large_hardneg.npy \
    --out exp/inference/devset/biencoder_hardneg_top100.json \
    --n_output 100 \
    2>&1 | tee -a "$LOG"

# =============================================================================
# Step 4: LightGBM with hard-neg bi-encoder (replacing old bi-encoder leg)
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 4: LightGBM with hard-neg bi-encoder ======" | tee -a "$LOG"

ALL_LEGS=""
for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
           attributes_qwen3 image_siglip2 lyrics_qwen3 state_bm25_focused \
           leg_attributes_qwen3_embedding_0_6b leg_image_siglip2 \
           leg_lyrics_qwen3_embedding_0_6b \
           biencoder_large_top100 biencoder_hardneg_top100; do
    if [ -f "exp/inference/devset/${LEG}.json" ]; then
        if [ -z "$ALL_LEGS" ]; then
            ALL_LEGS="$LEG"
        else
            ALL_LEGS="${ALL_LEGS},${LEG}"
        fi
    fi
done
echo "  Legs: $ALL_LEGS" | tee -a "$LOG"

PYTHONPATH=src python src/train_ltr.py \
    --legs "$ALL_LEGS" \
    --inference_dir exp/inference/devset \
    --ground_truth exp/ground_truth/devset.json \
    --top_k 100 --n_folds 5 \
    --pmi_path exp/item2item_pmi.npz \
    --out_model exp/ltr/lgbm_hardneg.txt \
    --out_inference exp/inference/devset/lgbm_hardneg.json \
    2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_hardneg.json \
    --scores exp/scores/devset/lgbm_hardneg.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tail -15 | tee -a "$LOG"

# =============================================================================
# Step 5: Responses
# =============================================================================
echo "" | tee -a "$LOG"
echo "====== Step 5: Responses ======" | tee -a "$LOG"

PYTHONPATH=src python src/generate_responses.py \
    --inference exp/inference/devset/lgbm_hardneg.json \
    --out exp/inference/devset/lgbm_hardneg_final.json \
    --mode auto --model Qwen/Qwen3-0.6B \
    2>&1 | tail -3 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/lgbm_hardneg_final.json \
    --scores exp/scores/devset/lgbm_hardneg_final.json \
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
    ('lgbm_biencoder_large_final', 'bge-large (previous best)'),
    ('lgbm_hardneg', 'bge-large + hard-neg retrain'),
    ('lgbm_hardneg_final', 'Hard-neg + responses'),
]
print(f'{\"\":40s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s} {\"Dist-2\":>7s}')
print('-'*70)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:40s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f} {s.get(\"lexical_diversity\",0):7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
