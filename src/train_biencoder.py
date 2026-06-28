"""Bi-Encoder fine-tuning: train a conversation→track retrieval model.

Trains a bi-encoder (BAAI/bge-base-en-v1.5) on 120k (conversation, gold_track)
pairs using MultipleNegativesRankingLoss. Produces a new retrieval leg that
directly encodes conversation context and finds semantically matching tracks.

This is DIFFERENT from our failed state-dense approach:
  - State-dense: used a FROZEN encoder and tried to align with dataset's
    pre-computed track embeddings (failed due to space mismatch)
  - Bi-encoder: trains BOTH sides on our data, so alignment is guaranteed

Pipeline:
  1. Build training pairs from 15k train sessions
  2. Fine-tune bge-base-en-v1.5 with MNR loss (~20-30 min on A100)
  3. Encode all 47k tracks
  4. Inference: encode each conversation, cosine top-100
  5. Output as a new retrieval leg for LightGBM

Usage:
    # All-in-one (recommended):
    python src/train_biencoder.py all \
        --model_id BAAI/bge-base-en-v1.5 \
        --output_dir out/biencoder \
        --out_leg exp/inference/devset/biencoder_top100.json

    # Or step by step:
    python src/train_biencoder.py build_data --out data/biencoder_pairs.jsonl
    python src/train_biencoder.py train --train_jsonl data/biencoder_pairs.jsonl --output_dir out/biencoder
    python src/train_biencoder.py encode_tracks --model_dir out/biencoder --out exp/tracks/biencoder_tracks.npy
    python src/train_biencoder.py inference --model_dir out/biencoder --track_emb exp/tracks/biencoder_tracks.npy --out exp/inference/devset/biencoder_top100.json

GPU: 1x A100/H100. Total time: ~45-60 min.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def _apply_transformers_compat_patches():
    """Runtime patches for models written against older transformers (<5.x)."""
    # 1. DynamicCache.from_legacy_cache was removed in transformers 5.x
    try:
        from transformers import DynamicCache
        if not hasattr(DynamicCache, 'from_legacy_cache'):
            @classmethod
            def from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                if past_key_values is not None:
                    for layer_idx, (k, v) in enumerate(past_key_values):
                        cache.update(k, v, layer_idx)
                return cache
            DynamicCache.from_legacy_cache = from_legacy_cache
    except Exception:
        pass

    # 2. DynamicCache.get_usable_length renamed to get_seq_length in transformers 5.x
    try:
        from transformers import DynamicCache
        if not hasattr(DynamicCache, 'get_usable_length') and hasattr(DynamicCache, 'get_seq_length'):
            DynamicCache.get_usable_length = DynamicCache.get_seq_length
    except Exception:
        pass

    # 3. all_tied_weights_keys: accessed as instance attr in transformers 5.x
    #    but custom models (e.g. NVEmbedModel) may not initialize it
    try:
        import transformers.modeling_utils as _mu
        _orig = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
        def _patched(self, *args, **kwargs):
            if not hasattr(self, 'all_tied_weights_keys'):
                self.all_tied_weights_keys = {}
            return _orig(self, *args, **kwargs)
        _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched
    except Exception:
        pass


_apply_transformers_compat_patches()


# ============================================================================
# Data preparation
# ============================================================================

def _as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v[:5] if x)
    return str(v) if v else ""


def build_conversation_text(conversations: list, turn: int) -> str:
    """Build conversation context up to current turn's user message."""
    lines = []
    for c in conversations:
        if c["turn_number"] > turn:
            break
        if c["turn_number"] < turn:
            if c["role"] == "user":
                lines.append(f"User: {c['content']}")
            elif c["role"] == "assistant":
                lines.append(f"Assistant: {c['content'][:100]}")
        elif c["turn_number"] == turn and c["role"] == "user":
            lines.append(f"User: {c['content']}")
    # Keep last 6 exchanges to fit in context
    return "\n".join(lines[-12:])


def build_track_text(meta: dict) -> str:
    """Build track text representation. NO lyrics (TalkPlay: lyrics hurt)."""
    name = _as_text(meta.get("track_name", [""]))
    artist = _as_text(meta.get("artist_name", [""]))
    album = _as_text(meta.get("album_name", [""]))
    tags = (meta.get("tag_list") or [])[:15]
    tags_str = ", ".join(str(t) for t in tags if t)
    release = meta.get("release_date") or ""
    year = release[:4] if release else ""

    parts = [f"{name} by {artist}"]
    if album:
        parts.append(f"from {album}")
    if tags_str:
        parts.append(f"[{tags_str}]")
    if year:
        parts.append(f"({year})")
    return " ".join(parts)


def cmd_build_data(args) -> None:
    """Build (conversation, track_text) training pairs."""
    print("Loading train sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    print(f"  {len(train)} sessions", flush=True)

    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}

    pairs: List[dict] = []
    for ex in tqdm(train, desc="building pairs"):
        convos = ex["conversations"]
        df = pd.DataFrame(convos)
        for turn in range(1, 9):
            gold_rows = df[(df["turn_number"] == turn) & (df["role"] == "music")]
            if gold_rows.empty:
                continue
            gold_id = gold_rows.iloc[0]["content"]
            if gold_id not in track_meta:
                continue

            conv_text = build_conversation_text(convos, turn)
            track_text = build_track_text(track_meta[gold_id])

            if conv_text and track_text:
                pairs.append({
                    "query": conv_text,
                    "positive": track_text,
                    "session_id": ex["session_id"],
                    "turn": turn,
                    "gold_id": gold_id,
                })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.out}")


# ============================================================================
# Training
# ============================================================================

def cmd_train(args) -> None:
    """Fine-tune bi-encoder with MultipleNegativesRankingLoss."""
    try:
        import torch
        from sentence_transformers import (
            SentenceTransformer, InputExample, losses, evaluation
        )
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"ERROR: {e}\npip install sentence-transformers torch", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model_id}", flush=True)
    model = SentenceTransformer(args.model_id, trust_remote_code=True)

    # Gradient checkpointing: trades compute for memory (~50-70% less activation memory)
    try:
        model[0].auto_model.gradient_checkpointing_enable()
        print("  gradient checkpointing enabled", flush=True)
    except Exception as e:
        print(f"  gradient checkpointing not available: {e}", flush=True)

    # LoRA: for large models (e.g. NV-Embed-v2 7B) where full fine-tune exceeds GPU memory
    if getattr(args, 'lora', False):
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # NV-Embed style: outer NVEmbedModel wraps an inner embedding_model (Mistral LLM)
        # Apply LoRA to the inner model, not the outer wrapper
        inner = model[0].auto_model
        if hasattr(inner, 'embedding_model') and inner.embedding_model is not None:
            inner.embedding_model = get_peft_model(inner.embedding_model, lora_config)
            inner.embedding_model.print_trainable_parameters()
        else:
            model[0].auto_model = get_peft_model(inner, lora_config)
            model[0].auto_model.print_trainable_parameters()

    # Load training pairs
    print(f"Loading pairs from {args.train_jsonl}...", flush=True)
    examples = []
    with open(args.train_jsonl) as f:
        for line in f:
            d = json.loads(line)
            examples.append(InputExample(texts=[d["query"], d["positive"]]))
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"  {len(examples)} training pairs", flush=True)

    # DataLoader
    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    # Loss: MultipleNegativesRankingLoss (in-batch negatives)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"warmup={warmup_steps} steps", flush=True)

    # use_amp=True uses FP16 scaler which breaks with accelerate>=1.0 + PyTorch>=2.4.
    # Cast model to BF16 manually instead — H100 native, fast, numerically stable.
    import torch
    model = model.to(torch.bfloat16)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
        use_amp=False,
    )

    # Merge LoRA weights back into the base model so encode/inference work normally
    if getattr(args, 'lora', False):
        print("Merging LoRA weights into base model...", flush=True)
        inner = model[0].auto_model
        if hasattr(inner, 'embedding_model') and hasattr(inner.embedding_model, 'merge_and_unload'):
            inner.embedding_model = inner.embedding_model.merge_and_unload()
        elif hasattr(model[0].auto_model, 'merge_and_unload'):
            model[0].auto_model = model[0].auto_model.merge_and_unload()
        model.save(args.output_dir)
        print(f"Merged model saved to {args.output_dir}", flush=True)
    print(f"Model saved to {args.output_dir}")


# ============================================================================
# Track encoding
# ============================================================================

def cmd_encode_tracks(args) -> None:
    """Encode all 47k tracks with the trained model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_dir}...", flush=True)
    model = SentenceTransformer(args.model_dir, trust_remote_code=True)

    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_ids = [r["track_id"] for r in tracks_ds]
    track_texts = [build_track_text(r) for r in tqdm(tracks_ds, desc="building texts")]

    print(f"Encoding {len(track_texts)} tracks...", flush=True)
    track_embs = model.encode(
        track_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"  shape: {track_embs.shape}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, track_embs)
    # Save track_ids order
    meta_path = args.out.replace(".npy", "_ids.json")
    with open(meta_path, "w") as f:
        json.dump(track_ids, f)
    print(f"Saved {args.out} + {meta_path}")


# ============================================================================
# Inference
# ============================================================================

def cmd_inference(args) -> None:
    """Run bi-encoder retrieval on devset."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_dir}...", flush=True)
    model = SentenceTransformer(args.model_dir, trust_remote_code=True)

    # Load track embeddings
    print(f"Loading track embeddings from {args.track_emb}...", flush=True)
    track_embs = np.load(args.track_emb).astype(np.float32)
    ids_path = args.track_emb.replace(".npy", "_ids.json")
    track_ids = json.load(open(ids_path))
    assert len(track_ids) == track_embs.shape[0]
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"  {track_embs.shape[0]} tracks, dim={track_embs.shape[1]}", flush=True)

    # Load test conversations
    split = args.split
    if split == "test":
        test = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        test = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{split}", split="test")
    print(f"  {len(test)} test sessions", flush=True)

    # Build queries
    queries: List[dict] = []
    for ex in test:
        for tn in range(1, 9):
            conv_text = build_conversation_text(ex["conversations"], tn)
            prior_tracks = [c["content"] for c in ex["conversations"]
                           if c["role"] == "music" and c["turn_number"] < tn]
            queries.append({
                "session_id": ex["session_id"],
                "user_id": ex["user_id"],
                "turn_number": tn,
                "query_text": conv_text,
                "prior_tracks": prior_tracks,
            })
    print(f"  {len(queries)} queries", flush=True)

    # Encode queries in batches
    print("Encoding queries...", flush=True)
    query_texts = [q["query_text"] for q in queries]
    query_embs = model.encode(
        query_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Retrieve top-K for each query
    print(f"Retrieving top-{args.n_output} per query...", flush=True)
    rows: List[dict] = []
    for i, q in enumerate(tqdm(queries, desc="retrieval")):
        scores = track_embs @ query_embs[i]

        # No-repeat: mask prior tracks
        for tid in q["prior_tracks"]:
            if tid in track_to_idx:
                scores[track_to_idx[tid]] = -1e9

        top_indices = np.argsort(-scores)[:args.n_output]
        preds = [track_ids[idx] for idx in top_indices]

        rows.append({
            "session_id": q["session_id"],
            "user_id": q["user_id"],
            "turn_number": q["turn_number"],
            "predicted_track_ids": preds,
            "predicted_response": "",
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} predictions to {args.out}")


# ============================================================================
# All-in-one
# ============================================================================

def cmd_all(args) -> None:
    """Run the full pipeline: build_data → train → encode_tracks → inference."""
    data_path = os.path.join(args.output_dir, "train_pairs.jsonl")
    track_emb_path = os.path.join(args.output_dir, "track_embeddings.npy")

    # Step 1: Build data
    print("=" * 60)
    print("STEP 1: Building training pairs")
    print("=" * 60)
    args.out = data_path
    cmd_build_data(args)

    # Step 2: Train
    print("\n" + "=" * 60)
    print("STEP 2: Training bi-encoder")
    print("=" * 60)
    args.train_jsonl = data_path
    cmd_train(args)

    # Step 3: Encode tracks
    print("\n" + "=" * 60)
    print("STEP 3: Encoding tracks")
    print("=" * 60)
    args.model_dir = args.output_dir
    args.out = track_emb_path
    cmd_encode_tracks(args)

    # Step 4: Inference
    print("\n" + "=" * 60)
    print("STEP 4: Running inference on devset")
    print("=" * 60)
    args.track_emb = track_emb_path
    args.out = args.out_leg
    cmd_inference(args)

    print("\n" + "=" * 60)
    print(f"DONE! Retrieval leg saved to: {args.out_leg}")
    print("Add to LightGBM: --legs ...,biencoder_top100")
    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # build_data
    bd = sub.add_parser("build_data")
    bd.add_argument("--out", default="data/biencoder_pairs.jsonl")

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--train_jsonl", required=True)
    tr.add_argument("--model_id", default="BAAI/bge-base-en-v1.5")
    tr.add_argument("--output_dir", default="out/biencoder")
    tr.add_argument("--batch_size", type=int, default=256)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--max_examples", type=int, default=None)
    tr.add_argument("--lora", action="store_true", help="Use LoRA for large models")
    tr.add_argument("--lora_r", type=int, default=16)
    tr.add_argument("--lora_alpha", type=int, default=32)
    tr.add_argument("--lora_dropout", type=float, default=0.05)

    # encode_tracks
    et = sub.add_parser("encode_tracks")
    et.add_argument("--model_dir", required=True)
    et.add_argument("--out", default="exp/tracks/biencoder_tracks.npy")
    et.add_argument("--batch_size", type=int, default=256)

    # inference
    inf = sub.add_parser("inference")
    inf.add_argument("--model_dir", required=True)
    inf.add_argument("--track_emb", required=True)
    inf.add_argument("--out", required=True)
    inf.add_argument("--n_output", type=int, default=100)
    inf.add_argument("--batch_size", type=int, default=256)
    inf.add_argument("--split", default="test", choices=["test", "Blind-A", "Blind-B"])

    # all-in-one
    al = sub.add_parser("all")
    al.add_argument("--model_id", default="BAAI/bge-base-en-v1.5")
    al.add_argument("--output_dir", default="out/biencoder")
    al.add_argument("--out_leg", default="exp/inference/devset/biencoder_top100.json")
    al.add_argument("--batch_size", type=int, default=256)
    al.add_argument("--epochs", type=int, default=3)
    al.add_argument("--max_examples", type=int, default=None)
    al.add_argument("--n_output", type=int, default=100)
    al.add_argument("--split", default="test", choices=["test", "Blind-A", "Blind-B"])
    al.add_argument("--lora", action="store_true", help="Use LoRA for large models")
    al.add_argument("--lora_r", type=int, default=16)
    al.add_argument("--lora_alpha", type=int, default=32)
    al.add_argument("--lora_dropout", type=float, default=0.05)

    args = p.parse_args()
    if args.cmd == "build_data":
        cmd_build_data(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "encode_tracks":
        cmd_encode_tracks(args)
    elif args.cmd == "inference":
        cmd_inference(args)
    elif args.cmd == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
