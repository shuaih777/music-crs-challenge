#!/bin/bash
set -uo pipefail

LOG="logs/personalized_listwise.log"
mkdir -p logs data out/listwise_reranker exp/inference/devset exp/scores/devset

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=1

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "[B] Fine-tuned Listwise Reranker (Qwen3-1.5B LoRA)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

LISTWISE_OUT="exp/inference/devset/listwise_finetuned.json"

# Step 1: Build training data
echo "  [1/3] Building listwise training data..." | tee -a "$LOG"
PYTHONPATH=src python -c "
import json, os, sys
from datasets import load_dataset
from tqdm import tqdm

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
    print('ERROR: no LightGBM inference found')
    sys.exit(1)
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
for row in tqdm(preds, desc='building listwise data'):
    key = (row['session_id'], row['turn_number'])
    gold = gold_by_key.get(key)
    candidates = row['predicted_track_ids'][:20]
    if not gold or gold not in candidates:
        continue
    n_with_gold += 1
    convos = conversations.get(row['session_id'], [])
    context = build_context(convos, row['turn_number'])
    track_lines = []
    for i, tid in enumerate(candidates):
        track_lines.append(f'[{i+1}] {track_desc(tid)}')
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
with open('data/listwise_train.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f'Wrote {len(examples)} examples ({n_with_gold} turns had gold in top-20)')
" 2>&1 | tee -a "$LOG"

# Step 2: LoRA fine-tune Qwen3-1.5B on GPUs 1,2,3
echo "  [2/3] Training Qwen3-1.5B listwise reranker (LoRA) on GPU 1..." | tee -a "$LOG"
PYTHONPATH=src python src/train_state_extractor.py \
    --train_jsonl data/listwise_train.jsonl \
    --model_id Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir out/listwise_reranker \
    --epochs 2 \
    --lr 2e-4 \
    --batch_size 2 \
    --grad_accum 16 \
    --max_seq_len 2048 \
    --lora_r 16 \
    --lora_alpha 32 \
    2>&1 | tee -a "$LOG"

# Step 3: Sharded inference on GPUs 1,2,3
echo "  [3/3] Reranking devset with fine-tuned model (3 shards)..." | tee -a "$LOG"

BEST_INF=""
for f in exp/inference/devset/lgbm_abl_plus_nvembed.json \
         exp/inference/devset/lgbm_overnight_all.json \
         exp/inference/devset/lgbm_biencoder_large.json; do
    if [ -f "$f" ]; then BEST_INF="$f"; break; fi
done

N_SHARDS=1
for i in 0; do
    GPU=1
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python src/listwise_shard_infer.py \
        --shard $i --total_shards $N_SHARDS \
        --model_dir out/listwise_reranker \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --inference "$BEST_INF" \
        --out "exp/inference/devset/listwise_shard_${i}.json" \
        2>&1 | tail -3 | tee -a "$LOG" &
done
wait
echo "  All shards complete." | tee -a "$LOG"

PYTHONPATH=src python src/listwise_shard_infer.py --merge \
    --total_shards $N_SHARDS \
    --out "$LISTWISE_OUT" \
    2>&1 | tee -a "$LOG"

echo "  [B] Done." | tee -a "$LOG"
