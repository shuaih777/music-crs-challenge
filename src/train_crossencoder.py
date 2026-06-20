"""Train a cross-encoder reranker (Qwen3-0.6B + LoRA) for Music-CRS.

The cross-encoder learns to score (conversation_context, track_metadata) pairs
for relevance. Unlike the zero-shot logit approach in rerank_crossencoder.py,
this fine-tunes the model on actual train-session data.

Training data:
  - From 15k train sessions, for each (session, turn):
    - Positive: (context, gold_track_metadata) → label=1
    - Negatives: (context, random_non_gold_tracks) → label=0
    - Hard negatives: tracks from the retrieval legs' top-100 that are NOT gold

The model is trained with binary cross-entropy on the "relevant"/"irrelevant"
classification task.

Usage:
    # 1. Build training data (~2 min CPU)
    python src/train_crossencoder.py build_data \
        --out data/crossencoder_train.jsonl \
        --n_neg 7 \
        --hard_neg_legs metadata_qwen3,cf_bpr,pmi_leg

    # 2. Train (~30 min on A100 80GB)
    python src/train_crossencoder.py train \
        --train_jsonl data/crossencoder_train.jsonl \
        --model_id Qwen/Qwen3-0.6B \
        --output_dir out/crossencoder_qwen3_0.6b \
        --epochs 1 --batch_size 32 --lr 2e-4

    # 3. Rerank devset with the trained model (~15 min)
    python src/train_crossencoder.py rerank \
        --model_dir out/crossencoder_qwen3_0.6b \
        --inference exp/inference/devset/lgbm_8leg.json \
        --out exp/inference/devset/crossencoder_trained.json

    # Or all-in-one:
    bash run_crossencoder.sh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Data building
# ============================================================================

def _as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v[:5] if x)
    return str(v) if v else ""


def build_context(conversations: list, turn: int) -> str:
    """Build conversation context up to and including the user message at `turn`."""
    lines = []
    for c in conversations:
        if c["turn_number"] > turn:
            break
        if c["turn_number"] < turn:
            if c["role"] == "user":
                lines.append(f"User: {c['content']}")
            elif c["role"] == "assistant":
                lines.append(f"Assistant: {c['content'][:120]}")
        elif c["turn_number"] == turn and c["role"] == "user":
            lines.append(f"User: {c['content']}")
    return "\n".join(lines[-10:])


def build_track_desc(meta: dict) -> str:
    name = _as_text(meta.get("track_name", [""]))
    artist = _as_text(meta.get("artist_name", [""]))
    album = _as_text(meta.get("album_name", [""]))
    tags = (meta.get("tag_list") or [])[:12]
    tags_str = ", ".join(str(t) for t in tags)
    release = meta.get("release_date") or ""
    parts = [f"{name} by {artist}"]
    if album:
        parts.append(f"album: {album}")
    if tags_str:
        parts.append(f"tags: {tags_str}")
    if release:
        parts.append(f"year: {release[:4]}")
    return " | ".join(parts)


def cmd_build_data(args) -> None:
    """Build training JSONL for the cross-encoder."""
    print("Loading train sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    print(f"  {len(train)} sessions", flush=True)

    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    all_track_ids = list(track_meta.keys())

    # Optionally load hard negatives from retrieval legs (run on TRAIN if available)
    hard_neg_pool: Dict[Tuple[str, int], List[str]] = {}
    # For now we just use random negatives (hard negs from devset legs would leak)

    rng = random.Random(args.seed)
    examples: List[dict] = []

    print(f"Building pairs (n_neg={args.n_neg})...", flush=True)
    for ex in tqdm(train, desc="building"):
        session_id = ex["session_id"]
        convos = ex["conversations"]
        df = pd.DataFrame(convos)

        for turn in range(1, 9):
            # Get gold track
            gold_rows = df[(df["turn_number"] == turn) & (df["role"] == "music")]
            if gold_rows.empty:
                continue
            gold_id = gold_rows.iloc[0]["content"]
            if gold_id not in track_meta:
                continue

            context = build_context(convos, turn)
            gold_desc = build_track_desc(track_meta[gold_id])

            # Positive example
            examples.append({
                "context": context,
                "track": gold_desc,
                "label": 1,
                "session_id": session_id,
                "turn": turn,
            })

            # Negative examples (random tracks from catalog)
            neg_ids = rng.sample(all_track_ids, min(args.n_neg, len(all_track_ids)))
            for neg_id in neg_ids:
                if neg_id == gold_id:
                    continue
                neg_desc = build_track_desc(track_meta[neg_id])
                examples.append({
                    "context": context,
                    "track": neg_desc,
                    "label": 0,
                    "session_id": session_id,
                    "turn": turn,
                })

    # Shuffle
    rng.shuffle(examples)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {args.out}")
    print(f"  positives: {sum(1 for e in examples if e['label']==1)}")
    print(f"  negatives: {sum(1 for e in examples if e['label']==0)}")


# ============================================================================
# Training
# ============================================================================

def cmd_train(args) -> None:
    """Fine-tune Qwen3-0.6B with LoRA for pointwise relevance scoring."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer)
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: {e}\npip install -r requirements-gpu.txt peft", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model {args.model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,  # regression (relevance score)
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.pad_token_id = tok.pad_token_id

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Load data
    print(f"Loading {args.train_jsonl}...", flush=True)
    data = []
    with open(args.train_jsonl) as f:
        for line in f:
            data.append(json.loads(line))
    if args.max_examples:
        data = data[:args.max_examples]
    print(f"  {len(data)} examples", flush=True)

    ds = Dataset.from_list(data)

    def tokenize_fn(batch):
        texts = [f"{ctx}\n\nTrack: {trk}" for ctx, trk in
                 zip(batch["context"], batch["track"])]
        encoded = tok(texts, max_length=args.max_seq_len, truncation=True, padding=False)
        encoded["labels"] = [[float(l)] for l in batch["label"]]
        return encoded

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names,
                desc="tokenize")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        bf16=(device == "cuda"),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
    )

    print("Training...", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


# ============================================================================
# Reranking with trained model
# ============================================================================

def cmd_rerank(args) -> None:
    """Rerank an inference JSON using the trained cross-encoder."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rerank] device={device} model_dir={args.model_dir}", flush=True)

    # Detect LoRA adapter
    is_lora = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))
    if is_lora:
        with open(os.path.join(args.model_dir, "adapter_config.json")) as f:
            cfg = json.load(f)
        base_id = args.base_model or cfg.get("base_model_name_or_path")
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=1, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        base.config.pad_token_id = tok.pad_token_id
        model = PeftModel.from_pretrained(base, args.model_dir)
    else:
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, num_labels=1, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.config.pad_token_id = tok.pad_token_id
    model.eval()

    # Load inference + metadata + conversations
    with open(args.inference) as f:
        rows = json.load(f)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    convo_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex for ex in convo_ds}

    output_rows = []
    for row in tqdm(rows, desc="reranking"):
        session_id = row["session_id"]
        turn = int(row["turn_number"])
        candidates = row["predicted_track_ids"][:args.top_k]

        conv = conversations.get(session_id, {})
        context = build_context(conv.get("conversations", []), turn)

        if not candidates:
            output_rows.append(row)
            continue

        # Build texts
        texts = []
        for tid in candidates:
            meta = track_meta.get(tid, {})
            desc = build_track_desc(meta)
            texts.append(f"{context}\n\nTrack: {desc}")

        # Score in batches
        scores = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            inputs = tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**inputs)
                batch_scores = out.logits.squeeze(-1).float().cpu().numpy()
                if batch_scores.ndim == 0:
                    batch_scores = np.array([float(batch_scores)])
                scores.extend(batch_scores.tolist())

        # Rerank
        order = np.argsort(-np.array(scores))
        reranked = [candidates[i] for i in order[:20]]

        output_rows.append({
            "session_id": session_id,
            "user_id": row.get("user_id", ""),
            "turn_number": turn,
            "predicted_track_ids": reranked,
            "predicted_response": row.get("predicted_response", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"Wrote {len(output_rows)} predictions to {args.out}")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # build_data
    bd = sub.add_parser("build_data", help="Build training JSONL")
    bd.add_argument("--out", default="data/crossencoder_train.jsonl")
    bd.add_argument("--n_neg", type=int, default=7,
                    help="Negative samples per positive")
    bd.add_argument("--hard_neg_legs", default="",
                    help="Comma-separated leg names for hard negatives (not yet implemented)")
    bd.add_argument("--seed", type=int, default=42)

    # train
    tr = sub.add_parser("train", help="Train cross-encoder with LoRA")
    tr.add_argument("--train_jsonl", required=True)
    tr.add_argument("--model_id", default="Qwen/Qwen3-0.6B")
    tr.add_argument("--output_dir", default="out/crossencoder_qwen3_0.6b")
    tr.add_argument("--epochs", type=float, default=1.0)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--batch_size", type=int, default=32)
    tr.add_argument("--grad_accum", type=int, default=1)
    tr.add_argument("--max_seq_len", type=int, default=512)
    tr.add_argument("--max_examples", type=int, default=None)

    # rerank
    rr = sub.add_parser("rerank", help="Rerank with trained model")
    rr.add_argument("--model_dir", required=True)
    rr.add_argument("--base_model", default=None)
    rr.add_argument("--inference", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--top_k", type=int, default=20)
    rr.add_argument("--batch_size", type=int, default=32)

    args = p.parse_args()
    if args.cmd == "build_data":
        cmd_build_data(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "rerank":
        cmd_rerank(args)


if __name__ == "__main__":
    main()
