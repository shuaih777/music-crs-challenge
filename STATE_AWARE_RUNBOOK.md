# State-aware retrieval — how to run

You've already got `exp/states/test.jsonl` from the trained Qwen3-0.6B extractor (99.9% parse rate after the thinking-mode fix). This doc shows how to plug those states into retrieval through the new flags in `src/baselines_v3.py`.

There are **three independent levers** the new flags expose:
1. `--query_mode {state, state_only}` — replace raw history concat with weighted state attributes for BM25
2. `--state_subtract_rejected --neg_weight W` — subtract `BM25(rejected_tags) × W` from the BM25 scores
3. `--state_emb_path PATH` — use a pre-encoded state vector as the dense-retrieval query (instead of pooling prior accepted tracks)

The first two need only the existing `exp/states/test.jsonl`. The third needs you to encode state texts once with a sentence-encoder; that's what `src/encode_states.py` does.

---

## Step 0 — pull and verify

```bash
git pull
python src/_device.py            # confirm CUDA visible
ls exp/states/test.jsonl         # 8000 rows, 99.9% parsed
```

---

## Step 1 — sparse-only state experiments (fast, ~1 min each)

These are pure CPU/numpy and tell us how much the **BM25 query construction** matters before we touch dense retrieval. Run each, then `evaluate.py`:

```bash
# 1.a state-as-query (no negative)
python src/baselines_v3.py \
    --output exp/inference/devset/state_bm25.json \
    --bm25_only \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --tag state_bm25
python src/evaluate.py \
    --inference exp/inference/devset/state_bm25.json \
    --scores exp/scores/devset/state_bm25.json \
    --ground_truth exp/ground_truth/devset.json

# 1.b state + subtract rejected (default neg_weight=0.5)
python src/baselines_v3.py \
    --output exp/inference/devset/state_bm25_neg.json \
    --bm25_only \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --state_subtract_rejected \
    --neg_weight 0.5 \
    --tag state_bm25_neg
python src/evaluate.py \
    --inference exp/inference/devset/state_bm25_neg.json \
    --scores exp/scores/devset/state_bm25_neg.json \
    --ground_truth exp/ground_truth/devset.json

# 1.c state-only (skip user_query duplication)
python src/baselines_v3.py \
    --output exp/inference/devset/state_only_bm25.json \
    --bm25_only \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state_only \
    --tag state_only_bm25
python src/evaluate.py \
    --inference exp/inference/devset/state_only_bm25.json \
    --scores exp/scores/devset/state_only_bm25.json \
    --ground_truth exp/ground_truth/devset.json
```

Local CPU smoke shows 1.a/1.b ≈ 0.10 nDCG@20 (similar to your `bm25_norepeat_states.json` at 0.1023). The interesting result is the **per-turn breakdown** — `state` query lifts t1 nDCG@1 from 0.014 → 0.06, the largest t1 jump we've seen.

---

## Step 2 — encode state texts to dense vectors (one-time, ~3-5 min on GPU)

```bash
python src/encode_states.py \
    --states_jsonl exp/states/test.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --out exp/states/test_qwen3_emb.npz \
    --batch_size 32
```

Output: `exp/states/test_qwen3_emb.npz` — a (~7995, 1024) float32 matrix keyed by `(session_id, turn_number)`. Same dimensionality as the `*-qwen3_embedding_0.6b` track-embedding fields, so dense retrieval works directly.

---

## Step 3 — dense retrieval with state embeddings (the real prize)

Now use the pre-encoded state vectors as the **dense query** against each Qwen3 track-embedding modality. This bypasses the prior-pool drift problem completely:

```bash
# 3.a state dense over metadata-qwen3 tracks (mirrors the current best leg)
python src/baselines_v3.py \
    --output exp/inference/devset/state_dense_metadata.json \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --state_subtract_rejected --neg_weight 0.5 \
    --embed metadata-qwen3_embedding_0.6b \
    --state_emb_path exp/states/test_qwen3_emb.npz \
    --tag state_dense_metadata
python src/evaluate.py \
    --inference exp/inference/devset/state_dense_metadata.json \
    --scores exp/scores/devset/state_dense_metadata.json \
    --ground_truth exp/ground_truth/devset.json

# 3.b state dense over attributes-qwen3 (often the best for tag/mood queries)
python src/baselines_v3.py \
    --output exp/inference/devset/state_dense_attributes.json \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --state_subtract_rejected --neg_weight 0.5 \
    --embed attributes-qwen3_embedding_0.6b \
    --state_emb_path exp/states/test_qwen3_emb.npz \
    --tag state_dense_attributes
python src/evaluate.py \
    --inference exp/inference/devset/state_dense_attributes.json \
    --scores exp/scores/devset/state_dense_attributes.json \
    --ground_truth exp/ground_truth/devset.json

# 3.c state dense over lyrics-qwen3 (catches mood/theme)
python src/baselines_v3.py \
    --output exp/inference/devset/state_dense_lyrics.json \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --state_subtract_rejected --neg_weight 0.5 \
    --embed lyrics-qwen3_embedding_0.6b \
    --state_emb_path exp/states/test_qwen3_emb.npz \
    --tag state_dense_lyrics
python src/evaluate.py \
    --inference exp/inference/devset/state_dense_lyrics.json \
    --scores exp/scores/devset/state_dense_lyrics.json \
    --ground_truth exp/ground_truth/devset.json
```

Each is ~30 sec on GPU, ~3 min on CPU.

---

## Step 4 — RRF ensemble with the new state legs

Once 3.a/3.b/3.c are scored, RRF-fuse them with the existing legs:

```bash
# add the 3 new state-dense legs to the existing best ensemble
python src/ensemble.py rrf \
    --inputs metadata_qwen3,cf_bpr,state_dense_metadata,state_dense_attributes,state_dense_lyrics \
    --weights 1.0,1.0,1.0,1.0,0.5 \
    --inference_dir exp/inference/devset \
    --out exp/inference/devset/ensemble_rrf_state5.json
python src/evaluate.py \
    --inference exp/inference/devset/ensemble_rrf_state5.json \
    --scores exp/scores/devset/ensemble_rrf_state5.json \
    --ground_truth exp/ground_truth/devset.json
```

If `state_dense_metadata` alone outscores `metadata_qwen3` (which is plausible — same track embeddings but a cleaner query), you can drop the latter.

---

## What we expect to see

| config | nDCG@20 expected | Hit@20 expected |
|---|---:|---:|
| state_bm25 | ≈ 0.100 | ≈ 21% |
| state_bm25_neg | ≈ 0.100-0.105 | ≈ 21-22% |
| state_dense_metadata | **0.13-0.16** | 27-32% |
| state_dense_attributes | **0.13-0.16** | 27-32% |
| state_dense_lyrics | 0.10-0.13 | 22-27% |
| ensemble_rrf_state5 | **0.14-0.18** | 30-35% |

Big claims — the state-as-dense-query is the whole reason we trained the extractor. If 3.a doesn't beat 0.124, something's wrong (state quality, encoder, dim mismatch). Most likely failure mode: the encoder produces vectors that aren't in the same semantic space as the track-side `*-qwen3_embedding_0.6b` (those were almost certainly produced by an earlier Qwen3 emb model). In that case, swap to whichever Qwen3-Embedding the dataset's track embeddings actually used — check the dataset card on HF.

---

## Failure-mode debugging

If **state_dense_* scores look random** (≤ 0.01 nDCG@20):
- The encoder model is wrong (different from what made the track embeddings). Try `Alibaba-NLP/gte-Qwen2-1.5B-instruct` or whatever HF dataset card mentions.
- Check `q_vec` dimensions match track_emb — `--state_emb_path` will fall back to prior-pool if dims differ, and you'd see "dense used in 0 calls" in the log.

If **nothing changes versus prior-pool dense**:
- Double-check the .npz is actually being used: log line should say `state-emb dense queries: <path>`.
- Confirm `len(state_emb_by_turn) == 7995` in the run log; if zero, the keys aren't matching `(session_id, turn_number)`.
