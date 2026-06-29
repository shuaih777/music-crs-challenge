#!/bin/bash
# =============================================================================
# Personalized Bi-Encoder + Fine-tuned Listwise Reranker
# =============================================================================
# Two independent experiments that can run in parallel on separate GPUs:
#   GPU 0: Personalized bi-encoder (~1h)
#   GPU 1-3: Fine-tuned listwise reranker (~1h with 4 GPU, ~3h with 1 GPU)
#
# Usage:
#   bash run_personalized_listwise.sh          # sequential on 1 GPU
#   bash run_personalized_listwise.sh parallel  # parallel on multiple GPUs
#
# =============================================================================

set -uo pipefail

if ! command -v module >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda 2>/dev/null || true
conda activate foundation_model 2>/dev/null || true
export PATH="${CONDA_PREFIX:-$HOME/.conda/envs/foundation_model}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
module unload cuda 2>/dev/null || true

LOG="logs/personalized_listwise.log"
mkdir -p logs data out/biencoder_personalized out/listwise_reranker \
         exp/inference/devset exp/scores/devset exp/ltr

echo "=== Personalized + Listwise started at $(date) ===" | tee "$LOG"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "GPUs available: $NUM_GPUS" | tee -a "$LOG"

MODE="${1:-sequential}"
echo "Mode: $MODE" | tee -a "$LOG"

# =============================================================================
# EXPERIMENT A: Personalized Bi-Encoder
# =============================================================================
run_personalized() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[A] Personalized Bi-Encoder (user metadata in query)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    PERS_LEG="exp/inference/devset/biencoder_personalized_top100.json"
    if [ -f "$PERS_LEG" ]; then
        echo "  [skip] $PERS_LEG exists" | tee -a "$LOG"
        return
    fi

    # Step 1: Build personalized training data
    echo "  [1/4] Building personalized training pairs..." | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, 'src')
from train_biencoder import build_track_text

def build_personalized_query(conversations, turn, user_profile):
    '''Query with user demographic prefix.'''
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

    # Step 2: Train
    echo "  [2/4] Training bge-large with personalized queries..." | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py train \
        --train_jsonl data/biencoder_personalized_pairs.jsonl \
        --model_id BAAI/bge-large-en-v1.5 \
        --output_dir out/biencoder_personalized \
        --batch_size 64 \
        --epochs 3 \
        2>&1 | tail -10 | tee -a "$LOG"

    # Step 3: Encode tracks
    echo "  [3/4] Encoding tracks..." | tee -a "$LOG"
    PYTHONPATH=src python src/train_biencoder.py encode_tracks \
        --model_dir out/biencoder_personalized \
        --out exp/tracks/biencoder_personalized_tracks.npy \
        --batch_size 256 \
        2>&1 | tail -5 | tee -a "$LOG"

    # Step 4: Inference with personalized queries
    echo "  [4/4] Inference with personalized queries..." | tee -a "$LOG"
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

    echo "  [A] Done." | tee -a "$LOG"
}

# =============================================================================
# EXPERIMENT B: Fine-tuned Listwise Reranker
# =============================================================================
run_listwise_finetune() {
    echo "" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"
    echo "[B] Fine-tuned Listwise Reranker (Qwen3-1.5B LoRA)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    LISTWISE_OUT="exp/inference/devset/listwise_finetuned.json"
    if [ -f "$LISTWISE_OUT" ]; then
        echo "  [skip] $LISTWISE_OUT exists" | tee -a "$LOG"
        return
    fi

    # Step 1: Build training data (permutation targets from LightGBM candidates)
    echo "  [1/3] Building listwise training data..." | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, sys
from datasets import load_dataset
from tqdm import tqdm

# Load ground truth
gt = json.load(open('exp/ground_truth/devset.json'))
gold_by_key = {(g['session_id'], g['turn_number']): g['ground_truth_track_id'] for g in gt}

# Load best LightGBM candidates (lgbm_abl_plus_nvembed or lgbm_overnight_all)
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

# Load track metadata for descriptions
tracks_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
track_meta = {r['track_id']: r for r in tracks_ds}

# Load conversations
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

# Build training examples
examples = []
n_with_gold = 0
for row in tqdm(preds, desc='building listwise data'):
    key = (row['session_id'], row['turn_number'])
    gold = gold_by_key.get(key)
    candidates = row['predicted_track_ids'][:20]
    if not gold or gold not in candidates:
        continue  # Skip if gold not in top-20 (can't learn from this)

    n_with_gold += 1
    convos = conversations.get(row['session_id'], [])
    context = build_context(convos, row['turn_number'])

    # Build track list
    track_lines = []
    for i, tid in enumerate(candidates):
        track_lines.append(f'[{i+1}] {track_desc(tid)}')

    # Target: gold track first, then rest in original order
    gold_idx = candidates.index(gold)
    target_order = [gold_idx + 1]  # 1-indexed
    for i in range(len(candidates)):
        if i != gold_idx:
            target_order.append(i + 1)
    target_str = ', '.join(str(x) for x in target_order)

    # Format as chat messages
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

    # Step 2: LoRA fine-tune Qwen3-1.5B
    echo "  [2/3] Training Qwen3-1.5B listwise reranker (LoRA)..." | tee -a "$LOG"

    # Use accelerate for multi-GPU if available
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Using accelerate with $NUM_GPUS GPUs" | tee -a "$LOG"
        TRAIN_CMD="accelerate launch --num_processes $NUM_GPUS --mixed_precision bf16"
    else
        TRAIN_CMD="python"
    fi

    PYTHONPATH=src $TRAIN_CMD src/train_state_extractor.py \
        --train_jsonl data/listwise_train.jsonl \
        --model_id Qwen/Qwen3-1.5B \
        --output_dir out/listwise_reranker \
        --epochs 2 \
        --lr 2e-4 \
        --batch_size 2 \
        --grad_accum 16 \
        --max_seq_len 2048 \
        --lora_r 16 \
        --lora_alpha 32 \
        2>&1 | tee -a "$LOG"

    # Step 3: Inference — rerank devset
    echo "  [3/3] Reranking devset with fine-tuned model..." | tee -a "$LOG"
    PYTHONPATH=src python -c "
import json, os, sys, re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

print('Loading model...', flush=True)
base_id = 'Qwen/Qwen3-1.5B'
tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, padding_side='left')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(model, 'out/listwise_reranker')
model.eval()
device = next(model.parameters()).device

# Load candidates + metadata + conversations
best_inf = None
for f in ['exp/inference/devset/lgbm_abl_plus_nvembed.json',
          'exp/inference/devset/lgbm_overnight_all.json']:
    if os.path.exists(f): best_inf = f; break
preds = json.load(open(best_inf))
tracks_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata', split='all_tracks')
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

def parse_ranking(output, n):
    numbers = [int(x) for x in re.findall(r'\d+', output)]
    valid, seen = [], set()
    for x in numbers:
        if 1 <= x <= n and x not in seen:
            valid.append(x - 1); seen.add(x)
    for i in range(n):
        if i not in seen: valid.append(i)
    return valid[:n]

print(f'Reranking {len(preds)} turns...', flush=True)
output_rows = []
for row in tqdm(preds, desc='listwise_ft'):
    candidates = row['predicted_track_ids'][:20]
    convos = conversations.get(row['session_id'], [])
    context = build_context(convos, row['turn_number'])
    track_lines = [f'[{i+1}] {track_desc(tid)}' for i, tid in enumerate(candidates)]

    user_msg = f'Conversation:\n{context}\n\nCandidate tracks:\n' + '\n'.join(track_lines) + \
               '\n\nRank ALL tracks from most to least relevant. Output only track numbers as comma-separated list:'
    messages = [
        {'role': 'system', 'content': 'You are a music recommendation expert. Rank tracks by relevance to the conversation.'},
        {'role': 'user', 'content': user_msg},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    ranking = parse_ranking(gen, len(candidates))
    reranked = [candidates[i] for i in ranking]
    output_rows.append({
        'session_id': row['session_id'], 'user_id': row.get('user_id', ''),
        'turn_number': row['turn_number'], 'predicted_track_ids': reranked,
        'predicted_response': row.get('predicted_response', ''),
    })

with open('$LISTWISE_OUT', 'w') as f:
    json.dump(output_rows, f, ensure_ascii=False)
print(f'Wrote {len(output_rows)} to $LISTWISE_OUT')
" 2>&1 | tee -a "$LOG"

    echo "  [B] Done." | tee -a "$LOG"
}

# =============================================================================
# Run experiments
# =============================================================================
if [ "$MODE" = "parallel" ] && [ "$NUM_GPUS" -ge 2 ]; then
    echo "Running A and B in parallel..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 run_personalized &
    PID_A=$!
    # Give B the remaining GPUs
    REMAINING_GPUS=$(seq 1 $((NUM_GPUS-1)) | tr '\n' ',')
    REMAINING_GPUS=${REMAINING_GPUS%,}
    CUDA_VISIBLE_DEVICES=$REMAINING_GPUS run_listwise_finetune &
    PID_B=$!
    wait $PID_A $PID_B
else
    run_personalized
    run_listwise_finetune
fi

# =============================================================================
# Evaluate both
# =============================================================================
echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "EVALUATION" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

for TAG in biencoder_personalized_top100 listwise_finetuned; do
    INF="exp/inference/devset/${TAG}.json"
    if [ -f "$INF" ]; then
        echo "  Evaluating $TAG..." | tee -a "$LOG"
        PYTHONPATH=src python src/evaluate.py \
            --inference "$INF" \
            --scores "exp/scores/devset/${TAG}.json" \
            --ground_truth exp/ground_truth/devset.json \
            2>&1 | grep "ndcg@20\|hit@20" | tee -a "$LOG"
    fi
done

# Also test personalized as an additional leg in LightGBM
PERS_LEG="exp/inference/devset/biencoder_personalized_top100.json"
if [ -f "$PERS_LEG" ]; then
    echo "" | tee -a "$LOG"
    echo "  Testing personalized leg in LightGBM ensemble..." | tee -a "$LOG"

    ALL_LEGS=""
    for LEG in metadata_qwen3 cf_bpr pmi_leg decay_descending bm25_norepeat \
               biencoder_large_top100 biencoder_top100 e5_large_top100 \
               biencoder_stella_top100 biencoder_nv_embed_top100 \
               generative_top100 biencoder_personalized_top100; do
        if [ -f "exp/inference/devset/${LEG}.json" ]; then
            ALL_LEGS="${ALL_LEGS:+$ALL_LEGS,}$LEG"
        fi
    done

    PYTHONPATH=src python src/train_ltr_v2.py \
        --legs "$ALL_LEGS" \
        --inference_dir exp/inference/devset \
        --ground_truth exp/ground_truth/devset.json \
        --top_k 100 --n_folds 5 \
        --pmi_path exp/item2item_pmi.npz \
        --out_model exp/ltr/lgbm_personalized.txt \
        --out_inference exp/inference/devset/lgbm_personalized.json \
        2>&1 | grep -E "positive|fold" | tail -6 | tee -a "$LOG"

    PYTHONPATH=src python src/evaluate.py \
        --inference exp/inference/devset/lgbm_personalized.json \
        --scores exp/scores/devset/lgbm_personalized.json \
        --ground_truth exp/ground_truth/devset.json \
        2>&1 | grep "ndcg@20\|hit@20" | tee -a "$LOG"
fi

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a "$LOG"
echo "=== RESULTS ===" | tee -a "$LOG"
python -c "
import json, os
configs = [
    ('lgbm_abl_plus_nvembed', 'Current best (NV-Embed ensemble)'),
    ('biencoder_personalized_top100', 'Personalized bi-encoder (standalone)'),
    ('lgbm_personalized', 'Current best + personalized leg'),
    ('listwise_finetuned', 'Listwise fine-tuned reranker'),
]
print(f'{\"\":45s} {\"nDCG@20\":>9s} {\"Hit@20\":>7s} {\"nDCG@1\":>7s}')
print('-'*68)
for tag, desc in configs:
    path = f'exp/scores/devset/{tag}.json'
    if os.path.exists(path):
        s = json.load(open(path))
        print(f'{desc:45s} {s[\"ndcg@20\"]:9.6f} {s.get(\"hit@20\",0)*100:6.2f}% {s[\"ndcg@1\"]:7.4f}')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Done at $(date) ===" | tee -a "$LOG"
