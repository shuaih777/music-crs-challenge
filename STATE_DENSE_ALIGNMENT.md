# State-dense alignment — diagnosis + recipe

## What we observed

Pulled your `state_dense_*` results. They underperform the existing legs:

| Config | nDCG@20 | Hit@20 |
|---|---:|---:|
| `metadata_qwen3` (raw history → mean of prior tracks) | **0.1208** | **25.3%** |
| `state_dense_metadata` (state vec → metadata-qwen3 tracks) | 0.0957 | 20.9% |
| `state_dense_attributes` | 0.0853 | 18.6% |
| `state_dense_lyrics` | 0.0649 | 14.8% |

After diagnostic in `src/diagnose_alignment.py`:

```
state_emb (no instruction) vs attributes-qwen3_embedding_0.6b tracks:
  Hit@20:   3.8%   (random = 0.1%)
  Hit@100:  9.7%   (random = 0.5%)
  Hit@500: 26.5%   (random = 1.7%)
```

## Root cause

State embeddings and track embeddings **are not in the same Qwen3-Embedding subspace**. Track-side embeddings were pre-computed by the dataset authors with some prompt template; we encoded query side **without any instruction**, so `state_text` lands in a different region of the same model's output space.

Qwen3-Embedding is an instruction-tuned retrieval model — when you encode without an instruction prefix, you get an "uncalibrated" vector. Track-track cos-sim is still well-structured (verified: nearest neighbors of "Emancipator" are correctly other relaxing/electronic acts), but query-track cos-sim is broken.

## Fix: scan instruction templates

`encode_states.py` now has `--instruction {none,qwen3_default,qwen3_music,qwen3_attributes,qwen3_metadata,custom}`. We don't know the exact prompt the dataset used, so try several and pick by `diagnose_alignment.py`'s Hit@100 metric — no need to run full inference.

### Step 1 — re-encode with each candidate instruction

```bash
git pull

# Five candidates ~3-5 min each on A100
for INST in none qwen3_default qwen3_music qwen3_attributes qwen3_metadata; do
    python src/encode_states.py \
        --states_jsonl exp/states/test.jsonl \
        --model Qwen/Qwen3-Embedding-0.6B \
        --instruction $INST \
        --out exp/states/test_qwen3_emb_$INST.npz \
        --batch_size 32
done
```

### Step 2 — diagnose alignment on each (no full inference)

```bash
for INST in none qwen3_default qwen3_music qwen3_attributes qwen3_metadata; do
    echo "=== $INST ==="
    python src/diagnose_alignment.py \
        --state_emb exp/states/test_qwen3_emb_$INST.npz \
        --track_field attributes-qwen3_embedding_0.6b \
        --sample 1000
done | tee logs/alignment_attributes.txt
```

This prints Hit@K for each — we want **Hit@100 > 30%** to make dense retrieval worthwhile. If none of the templates clear that bar, the dataset likely used a different base model entirely (we'd need to read the dataset card or ask the organizers).

### Step 3 — repeat for the other two text modalities

Track fields `metadata-qwen3_embedding_0.6b` and `lyrics-qwen3_embedding_0.6b` may have different prompts than `attributes-qwen3_embedding_0.6b`. Run diagnose against each. The winning instruction can be different per modality.

### Step 4 — pick winners and re-run inference

For each `(instruction, track_field)` pair where Hit@100 cleared 30%, re-run:

```bash
python src/baselines_v3.py \
    --output exp/inference/devset/state_dense_<field>_<inst>.json \
    --states_jsonl exp/states/test.jsonl \
    --query_mode state \
    --state_subtract_rejected --neg_weight 0.5 \
    --embed <field> \
    --state_emb_path exp/states/test_qwen3_emb_<inst>.npz \
    --tag state_dense_<field>_<inst>
python src/evaluate.py \
    --inference exp/inference/devset/state_dense_<field>_<inst>.json \
    --scores exp/scores/devset/state_dense_<field>_<inst>.json \
    --ground_truth exp/ground_truth/devset.json
```

Then re-fuse with `ensemble.py`.

---

## Custom instruction (if dataset card surfaces a clue)

If you find the actual prompt the organizers used (e.g. through a HF discussion, dataset card update, or the original paper), pass it directly:

```bash
python src/encode_states.py \
    --states_jsonl exp/states/test.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B \
    --instruction custom \
    --custom_instruction "Instruct: Retrieve a music track that matches the listener's preferences." \
    --out exp/states/test_qwen3_emb_custom.npz
```

---

## What to expect

If you find a working template, `state_dense_<field>` should beat its corresponding raw-history dense leg (`metadata_qwen3`, etc.) **especially on turns 5-8** where the state cleanly encodes the user's current intent and raw-history pooling has drifted. That should push the 5-way RRF ensemble well past 0.124 nDCG@20.

If no template clears Hit@100=30%, the next move is to **encode track text ourselves** with the same model+instruction, throwing away the dataset's pre-computed `*-qwen3_embedding_0.6b` and using only `track_card_text → emb` from our own pipeline. That guarantees alignment but takes ~10 min on A100.
