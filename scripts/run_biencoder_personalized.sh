#!/bin/bash
# =============================================================================
# Personalized Bi-Encoder (Experiment A from run_personalized_listwise.sh,
# extracted standalone). Trains bge-large with user demographic prefix
# ([age_group | gender | country]) prepended to queries. This is the 14th
# leg that took the final Blind-B submission from 0.24 -> 0.25.
# =============================================================================
set -euo pipefail

LOG="logs/biencoder_personalized.log"
mkdir -p logs data out/biencoder_personalized exp/inference/devset exp/tracks
echo "=== Personalized bi-encoder started at $(date) ===" | tee "$LOG"

PERS_LEG="exp/inference/devset/biencoder_personalized_top100.json"

echo "[1/4] Building personalized training pairs..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json, os, sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, 'src')
from train_biencoder import build_track_text

def build_personalized_query(conversations, turn, user_profile):
    age = user_profile.get('age_group', '')
    gender = user_profile.get('gender', '')
    country = user_profile.get('country_name', '')
    prefix = f'[{age} | {gender} | {country}]'
    lines = [prefix]
    for c in conversations:
        if c['turn_number'] > turn:
            break
        if c['turn_number'] < turn:
            if c['role'] == 'user':
                lines.append(f'User: {c[\"content\"]}')
            elif c['role'] == 'assistant':
                lines.append(f'Assistant: {c[\"content\"][:100]}')
        elif c['turn_number'] == turn and c['role'] == 'user':
            lines.append(f'User: {c[\"content\"]}')
    return '\n'.join(lines[-12:])

print('Loading data...', flush=True)
train = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='train')
tracks_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_meta = {r['track_id']: r for r in tracks_ds}

pairs = []
for ex in tqdm(train, desc='building personalized pairs'):
    convos = ex['conversations']
    user_profile = ex.get('user_profile', {})
    df = pd.DataFrame(convos)
    for turn in range(1, 9):
        gold_rows = df[(df['turn_number'] == turn) & (df['role'] == 'music')]
        if gold_rows.empty: continue
        gold_id = gold_rows.iloc[0]['content']
        if gold_id not in track_meta: continue

        query = build_personalized_query(convos, turn, user_profile)
        track_text = build_track_text(track_meta[gold_id])
        if query and track_text:
            pairs.append({'query': query, 'positive': track_text,
                          'session_id': ex['session_id'], 'turn': turn, 'gold_id': gold_id})

os.makedirs('data', exist_ok=True)
with open('data/biencoder_personalized_pairs.jsonl', 'w') as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
print(f'Wrote {len(pairs)} personalized pairs')
" 2>&1 | tee -a "$LOG"

echo "[2/4] Training bge-large with personalized queries..." | tee -a "$LOG"
PYTHONPATH=src python src/train_biencoder.py train \
    --train_jsonl data/biencoder_personalized_pairs.jsonl \
    --model_id BAAI/bge-large-en-v1.5 \
    --output_dir out/biencoder_personalized \
    --batch_size 64 \
    --epochs 3 \
    2>&1 | tee -a "$LOG"

echo "[3/4] Encoding tracks..." | tee -a "$LOG"
PYTHONPATH=src python src/train_biencoder.py encode_tracks \
    --model_dir out/biencoder_personalized \
    --out exp/tracks/biencoder_personalized_tracks.npy \
    --batch_size 256 \
    2>&1 | tee -a "$LOG"

echo "[4/4] Inference with personalized queries (devset)..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json, os, sys, numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, 'src')

print('Loading model...', flush=True)
model = SentenceTransformer('out/biencoder_personalized', trust_remote_code=True)

track_embs = np.load('exp/tracks/biencoder_personalized_tracks.npy')
track_ids = json.load(open('exp/tracks/biencoder_personalized_tracks_ids.json'))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')

def build_personalized_query(conversations, turn, user_profile):
    age = user_profile.get('age_group', '')
    gender = user_profile.get('gender', '')
    country = user_profile.get('country_name', '')
    prefix = f'[{age} | {gender} | {country}]'
    lines = [prefix]
    for c in conversations:
        if c['turn_number'] > turn: break
        if c['turn_number'] < turn:
            if c['role'] == 'user': lines.append(f'User: {c[\"content\"]}')
            elif c['role'] == 'assistant': lines.append(f'Assistant: {c[\"content\"][:100]}')
        elif c['turn_number'] == turn and c['role'] == 'user':
            lines.append(f'User: {c[\"content\"]}')
    return '\n'.join(lines[-12:])

queries, meta_list = [], []
for ex in test:
    user_profile = ex.get('user_profile', {})
    for tn in range(1, 9):
        q = build_personalized_query(ex['conversations'], tn, user_profile)
        prior = [c['content'] for c in ex['conversations'] if c['role']=='music' and c['turn_number'] < tn]
        queries.append(q)
        meta_list.append({'session_id': ex['session_id'], 'user_id': ex['user_id'],
                          'turn_number': tn, 'prior_tracks': prior})

print(f'Encoding {len(queries)} queries...', flush=True)
q_embs = model.encode(queries, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)

rows = []
for i, m in enumerate(tqdm(meta_list, desc='retrieval')):
    scores = track_embs @ q_embs[i]
    for tid in m['prior_tracks']:
        if tid in track_to_idx: scores[track_to_idx[tid]] = -1e9
    top_idx = np.argsort(-scores)[:100]
    preds = [track_ids[j] for j in top_idx]
    rows.append({'session_id': m['session_id'], 'user_id': m['user_id'],
                 'turn_number': m['turn_number'], 'predicted_track_ids': preds,
                 'predicted_response': ''})

with open('$PERS_LEG', 'w') as f:
    json.dump(rows, f, ensure_ascii=False)
print(f'Wrote {len(rows)} to $PERS_LEG')
" 2>&1 | tee -a "$LOG"

echo "=== Done at $(date) ===" | tee -a "$LOG"
echo "Note: $PERS_LEG holds top-100 candidates (for the LightGBM pool), not a" | tee -a "$LOG"
echo "submission-format top-20, so it isn't scored standalone via evaluate.py" | tee -a "$LOG"
echo "(same reason no other *_top100.json leg has a score file — see the" | tee -a "$LOG"
echo "archived 'More than 20 predictions' bug/fix in session_archive/)." | tee -a "$LOG"
