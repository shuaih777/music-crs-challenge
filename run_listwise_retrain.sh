#!/bin/bash
# Retrain listwise LoRA with shuffled candidates, then run full 8000-row inference
set -eo pipefail

LOG="logs/listwise_retrain.log"
mkdir -p logs data out/listwise_reranker_v2 exp/inference/devset exp/scores/devset

echo "=== Listwise retrain started at $(date) ===" | tee "$LOG"

# ─── Step 1: Rebuild training data with shuffled candidates ───────────────────
echo "[1/4] Building shuffled listwise training data..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json, os, sys, random
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

gt = json.load(open('exp/ground_truth/devset.json'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}

best_inf = None
for f in ['exp/inference/devset/lgbm_abl_plus_nvembed.json',
          'exp/inference/devset/lgbm_overnight_all.json',
          'exp/inference/devset/lgbm_biencoder_large.json']:
    if os.path.exists(f):
        best_inf = f
        break
if not best_inf:
    print('ERROR: no LightGBM inference found'); sys.exit(1)
print(f'Using candidates from: {best_inf}', flush=True)
preds = json.load(open(best_inf))

tracks_ds = load_dataset('parquet', data_files={'train': 'hf://datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet'}, split='train')
track_meta = {r['track_id']: r for r in tracks_ds}

test = load_dataset('talkpl-ai/TalkPlayData-Challenge-Dataset', split='test')
conversations = {ex['session_id']: ex['conversations'] for ex in test}

def build_context(convos, turn):
    lines = []
    for c in convos:
        if c['turn_number'] > turn: break
        if c['turn_number'] < turn:
            if c['role'] == 'user': lines.append(f'User: {c[\"content\"]}')
            elif c['role'] == 'assistant': lines.append(f'Assistant: {c[\"content\"][:80]}')
        elif c['turn_number'] == turn and c['role'] == 'user':
            lines.append(f'User: {c[\"content\"]}')
    return '\n'.join(lines[-8:])

def track_desc(tid):
    m = track_meta.get(tid, {})
    name = ', '.join(m.get('track_name', []))[:40]
    artist = ', '.join(m.get('artist_name', []))[:30]
    tags = ', '.join(str(t) for t in (m.get('tag_list') or [])[:6])
    year = (m.get('release_date') or '')[:4]
    return f'\"{name}\" by {artist} ({tags}) [{year}]'

examples = []
n_with_gold = 0
for row in tqdm(preds, desc='building shuffled listwise data'):
    key = (row['session_id'], row['turn_number'])
    gold = gold_by_key.get(key)
    orig_candidates = row['predicted_track_ids'][:20]
    if not gold or gold not in orig_candidates:
        continue

    # Shuffle candidates randomly
    candidates = list(orig_candidates)
    random.shuffle(candidates)

    n_with_gold += 1
    convos = conversations.get(row['session_id'], [])
    context = build_context(convos, row['turn_number'])
    track_lines = [f'[{i+1}] {track_desc(tid)}' for i, tid in enumerate(candidates)]

    gold_idx = candidates.index(gold)
    target_order = [gold_idx + 1]
    for i in range(len(candidates)):
        if i != gold_idx:
            target_order.append(i + 1)
    target_str = ', '.join(str(x) for x in target_order)

    user_msg = f'Conversation:\n{context}\n\nCandidate tracks:\n' + '\n'.join(track_lines) + \
               '\n\nRank ALL tracks from most to least relevant. Output only track numbers as comma-separated list:'
    examples.append({
        'messages': [
            {'role': 'system', 'content': 'You are a music recommendation expert. Rank tracks by relevance to the conversation.'},
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': target_str},
        ],
        'session_id': row['session_id'],
        'turn_number': row['turn_number'],
    })

os.makedirs('data', exist_ok=True)
with open('data/listwise_train_shuffled.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f'Wrote {len(examples)} examples ({n_with_gold} turns had gold in top-20)')
print(f'Gold position distribution (first 5 positions):')
from collections import Counter
positions = []
for ex in examples:
    target = ex['messages'][-1]['content']
    nums = [int(x.strip()) for x in target.split(',') if x.strip().isdigit()]
    if nums: positions.append(nums[0])
c = Counter(positions)
for pos in sorted(c)[:5]:
    print(f'  pos {pos}: {c[pos]} ({c[pos]/len(examples)*100:.1f}%)')
" 2>&1 | tee -a "$LOG"

# ─── Step 2: Retrain LoRA (single GPU to avoid DDP issues) ────────────────────
echo "" | tee -a "$LOG"
echo "[2/4] Retraining LoRA on GPU 0 (~10 min)..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python src/train_state_extractor.py \
    --train_jsonl data/listwise_train_shuffled.jsonl \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir out/listwise_reranker_v2 \
    --epochs 2 \
    --lr 2e-4 \
    --batch_size 2 \
    --grad_accum 16 \
    --max_seq_len 2048 \
    --lora_r 16 \
    --lora_alpha 32 \
    2>&1 | tee -a "$LOG"

# ─── Step 3: Inference on full 8000 devset rows (3 shards, GPUs 0,1,2) ───────
echo "" | tee -a "$LOG"
echo "[3/4] Running inference on 8000 devset rows (3 GPUs)..." | tee -a "$LOG"

for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src python src/listwise_shard_infer.py \
        --shard $i --total_shards 3 \
        --model_dir out/listwise_reranker_v2 \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --inference exp/inference/devset/lgbm_abl_plus_nvembed.json \
        --out "exp/inference/devset/listwise_v2_shard_${i}.json" \
        2>&1 | tail -3 | tee -a "$LOG" &
done
wait
echo "All shards complete." | tee -a "$LOG"

# ─── Step 4: Merge and evaluate ───────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "[4/4] Merging and evaluating..." | tee -a "$LOG"
python -c "
import json
rows = []
for i in range(3):
    data = json.load(open(f'exp/inference/devset/listwise_v2_shard_{i}.json'))
    rows.extend(data)
    print(f'shard {i}: {len(data)} rows')
json.dump(rows, open('exp/inference/devset/listwise_v2.json','w'), ensure_ascii=False)
print(f'Merged {len(rows)} rows')
" 2>&1 | tee -a "$LOG"

PYTHONPATH=src python src/evaluate.py \
    --inference exp/inference/devset/listwise_v2.json \
    --scores exp/scores/devset/listwise_v2.json \
    --ground_truth exp/ground_truth/devset.json \
    2>&1 | tee -a "$LOG"

# ─── Summary ──────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_abl_plus_nvembed', 'Baseline (NV-Embed ensemble)'),
    ('listwise_v2',           'Listwise LoRA v2 (shuffled training)'),
]
print(f'{\"\":45s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s}')
print('-'*72)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
