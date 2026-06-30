"""Generative Retrieval via Semantic IDs (TalkPlay/GCRS).

End-to-end: K-Means quantization → training data → LoRA fine-tune → inference.
Produces a retrieval leg (top-100) for LightGBM integration.

Two versions:
  --version simple (default): 2-level K-Means, <music><m_X><m_Y></music> format
  --version gcrs: 4-level RQ-VAE style with collision resolution, structured
      generation (<MODE=REC><BOI><s1_N><s2_N><s3_N><s4_N><EOI><RESP>...),
      constrained trie-based decoding, beam search. Based on:
      "Generative Conversational Recommender System" (Zhang et al., 2026)

Usage:
    # Simple (original)
    python src/generative_retrieval.py --version simple \
        --model_id Qwen/Qwen3-0.6B \
        --output_dir out/generative \
        --out_leg exp/inference/devset/generative_top100.json

    # GCRS
    python src/generative_retrieval.py --version gcrs \
        --model_id Qwen/Qwen3-4B \
        --output_dir out/generative_gcrs \
        --out_leg exp/inference/devset/generative_gcrs_top100.json \
        --n_levels 4 --n_clusters 256 --beam_size 20

GPU: 1x H100. Total: ~2-3h (simple) / ~4-5h (gcrs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Step 1: Semantic ID construction (K-Means Residual Quantization)
# ============================================================================

def build_semantic_ids(
    n_clusters: int = 256,
    n_levels: int = 2,
) -> Tuple[Dict[str, List[int]], np.ndarray, List[str]]:
    """Assign each track a multi-level semantic ID via residual quantization.

    Returns:
        track_to_sid: {track_id: [level1_cluster, level2_cluster, ...]}
        track_embs: (N, D) normalized embeddings
        track_ids: ordered list
    """
    from sklearn.cluster import MiniBatchKMeans

    print("Loading track embeddings (metadata-qwen3)...", flush=True)
    emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")

    track_ids = []
    vecs = []
    for r in tqdm(emb_ds, desc="loading"):
        e = r["metadata-qwen3_embedding_0.6b"]
        if isinstance(e, list) and len(e) == 1024:
            track_ids.append(r["track_id"])
            vecs.append(e)
        else:
            track_ids.append(r["track_id"])
            vecs.append([0.0] * 1024)

    X = np.array(vecs, dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-9, None)

    print(f"Quantizing {len(track_ids)} tracks with {n_levels} levels × {n_clusters} clusters...", flush=True)
    track_to_sid: Dict[str, List[int]] = {tid: [] for tid in track_ids}
    residual = X.copy()

    for level in range(n_levels):
        print(f"  Level {level+1}: fitting K-Means...", flush=True)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096,
            n_init=3, max_iter=100, random_state=42 + level,
        )
        labels = kmeans.fit_predict(residual)

        for i, tid in enumerate(track_ids):
            track_to_sid[tid].append(int(labels[i]))

        # Compute residuals for next level
        centroids = kmeans.cluster_centers_
        residual = residual - centroids[labels]

    # Check collision rate
    sid_strings = [tuple(v) for v in track_to_sid.values()]
    n_unique = len(set(sid_strings))
    print(f"  Unique SIDs: {n_unique}/{len(track_ids)} ({n_unique/len(track_ids)*100:.1f}%)")
    print(f"  Collisions: {len(track_ids) - n_unique}")

    return track_to_sid, X, track_ids


# ============================================================================
# Step 1 GCRS: 4-level RQ with collision resolution
# ============================================================================

def build_semantic_ids_gcrs(
    n_clusters: int = 256,
    n_levels: int = 4,
) -> Tuple[Dict[str, List[int]], np.ndarray, List[str]]:
    """Assign each track a unique multi-level semantic ID via RQ-VAE style quantization.

    Collision resolution: if two tracks have the same 4-digit SID, append a
    disambiguating index (stored as an extra level). The trie and token format
    handle this transparently.

    Returns:
        track_to_sid: {track_id: [s1, s2, s3, s4]} — unique per track
        track_embs: (N, D) normalized embeddings
        track_ids: ordered list
    """
    from sklearn.cluster import MiniBatchKMeans

    print("Loading track embeddings (metadata-qwen3)...", flush=True)
    emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")

    track_ids: List[str] = []
    vecs: List[list] = []
    for r in tqdm(emb_ds, desc="loading"):
        e = r["metadata-qwen3_embedding_0.6b"]
        if isinstance(e, list) and len(e) == 1024:
            track_ids.append(r["track_id"])
            vecs.append(e)
        else:
            track_ids.append(r["track_id"])
            vecs.append([0.0] * 1024)

    X = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-9, None)

    print(f"[GCRS] Quantizing {len(track_ids)} tracks: {n_levels} levels × {n_clusters} clusters...", flush=True)
    raw_sids: Dict[str, List[int]] = {tid: [] for tid in track_ids}
    residual = X.copy()

    for level in range(n_levels):
        print(f"  Level {level+1}: fitting K-Means...", flush=True)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=4096,
            n_init=3, max_iter=100, random_state=42 + level,
        )
        labels = kmeans.fit_predict(residual)
        for i, tid in enumerate(track_ids):
            raw_sids[tid].append(int(labels[i]))
        centroids = kmeans.cluster_centers_
        residual = residual - centroids[labels]

    # Collision resolution: append disambiguating index for duplicates
    sid_groups: Dict[tuple, List[str]] = defaultdict(list)
    for tid, sid in raw_sids.items():
        sid_groups[tuple(sid)].append(tid)

    track_to_sid: Dict[str, List[int]] = {}
    n_collisions = 0
    for sid_tuple, tids in sid_groups.items():
        if len(tids) == 1:
            track_to_sid[tids[0]] = list(sid_tuple)
        else:
            n_collisions += len(tids) - 1
            for idx, tid in enumerate(tids):
                # Append disambiguating index as extra element
                track_to_sid[tid] = list(sid_tuple) + [idx]

    n_unique = len(set(tuple(v) for v in track_to_sid.values()))
    print(f"  Unique SIDs after collision resolution: {n_unique}/{len(track_ids)}")
    print(f"  Collisions resolved: {n_collisions} tracks got extra disambig token")

    return track_to_sid, X, track_ids


# ============================================================================
# Step 2: Build training data (simple version)
# ============================================================================

def build_training_data(
    track_to_sid: Dict[str, List[int]],
    output_path: str,
    n_levels: int = 2,
) -> None:
    """Build (conversation, semantic_id_sequence) training pairs."""
    print("Loading train sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")

    def sid_to_tokens(sid: List[int]) -> str:
        """Convert [123, 45] to '<m_123><m_45>'"""
        return "".join(f"<m_{c}>" for c in sid)

    examples = []
    for ex in tqdm(train, desc="building data"):
        convos = ex["conversations"]
        df = pd.DataFrame(convos)
        for turn in range(1, 9):
            gold_rows = df[(df["turn_number"] == turn) & (df["role"] == "music")]
            if gold_rows.empty:
                continue
            gold_id = gold_rows.iloc[0]["content"]
            if gold_id not in track_to_sid:
                continue

            # Build conversation context
            lines = []
            for c in convos:
                if c["turn_number"] > turn:
                    break
                if c["turn_number"] < turn:
                    if c["role"] == "user":
                        lines.append(f"User: {c['content']}")
                    elif c["role"] == "assistant":
                        lines.append(f"Assistant: {c['content'][:80]}")
                    elif c["role"] == "music":
                        # Show prior recommendations as their SIDs
                        prior_sid = track_to_sid.get(c["content"])
                        if prior_sid:
                            lines.append(f"[Recommended: {sid_to_tokens(prior_sid)}]")
                elif c["turn_number"] == turn and c["role"] == "user":
                    lines.append(f"User: {c['content']}")

            context = "\n".join(lines[-12:])
            target_sid = sid_to_tokens(track_to_sid[gold_id])

            examples.append({
                "messages": [
                    {"role": "system", "content": "You are a music recommendation system. Given a conversation, output the semantic ID of the best track to recommend. Format: <music><m_X><m_Y></music>"},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": f"<music>{target_sid}</music>"},
                ],
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {output_path}")


# ============================================================================
# Step 2 GCRS: Structured generation training data
# ============================================================================

def gcrs_sid_to_tokens(sid: List[int]) -> str:
    """Convert [45, 123, 67, 89] to '<s1_45><s2_123><s3_67><s4_89>'.

    For collision-resolved SIDs with 5+ elements, the extra element is appended
    as <s5_N>.
    """
    return "".join(f"<s{i+1}_{c}>" for i, c in enumerate(sid))


def build_training_data_gcrs(
    track_to_sid: Dict[str, List[int]],
    output_path: str,
    n_levels: int = 4,
) -> None:
    """Build GCRS structured training data.

    Format for recommendation turns:
        Assistant: <MODE=REC><BOI><s1_N><s2_N><s3_N><s4_N><EOI><RESP>{text}
    For chat-only turns:
        Assistant: <MODE=CHAT><RESP>{text}
    """
    print("[GCRS] Loading train sessions...", flush=True)
    train = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")

    SYSTEM_PROMPT = (
        "You are a conversational music recommender. When recommending, "
        "output the track's semantic ID."
    )

    examples = []
    for ex in tqdm(train, desc="building GCRS data"):
        convos = ex["conversations"]
        df = pd.DataFrame(convos)

        for turn in range(1, 9):
            # Determine if this turn has a music recommendation
            gold_rows = df[(df["turn_number"] == turn) & (df["role"] == "music")]
            assistant_rows = df[(df["turn_number"] == turn) & (df["role"] == "assistant")]
            response_text = ""
            if not assistant_rows.empty:
                response_text = str(assistant_rows.iloc[0]["content"])[:200]

            # Build conversation history up to this turn as user message
            history_lines = []
            for c in convos:
                if c["turn_number"] > turn:
                    break
                if c["turn_number"] < turn:
                    if c["role"] == "user":
                        history_lines.append(f"User: {c['content']}")
                    elif c["role"] == "assistant":
                        history_lines.append(f"Assistant: {c['content'][:100]}")
                    elif c["role"] == "music":
                        sid = track_to_sid.get(c["content"])
                        if sid:
                            history_lines.append(
                                f"[Recommended: {gcrs_sid_to_tokens(sid)}]"
                            )
                elif c["turn_number"] == turn and c["role"] == "user":
                    history_lines.append(f"User: {c['content']}")

            user_msg = "\n".join(history_lines[-14:])

            if not gold_rows.empty:
                gold_id = gold_rows.iloc[0]["content"]
                if gold_id not in track_to_sid:
                    continue
                sid_tokens = gcrs_sid_to_tokens(track_to_sid[gold_id])
                # MODE=REC format
                assistant_output = (
                    f"<MODE=REC><BOI>{sid_tokens}<EOI><RESP>{response_text}"
                )
            else:
                if not response_text:
                    continue  # Skip empty turns
                # MODE=CHAT format
                assistant_output = f"<MODE=CHAT><RESP>{response_text}"

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_output},
                ],
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[GCRS] Wrote {len(examples)} examples to {output_path}")


# ============================================================================
# Step 3: Fine-tune LLM (simple version)
# ============================================================================

def train_model(
    train_path: str,
    model_id: str,
    output_dir: str,
    n_clusters: int = 256,
    n_levels: int = 2,
    epochs: float = 2.0,
    batch_size: int = 8,
    lr: float = 2e-4,
    max_seq_len: int = 1024,
    max_examples: int = None,
) -> None:
    """LoRA fine-tune LLM to generate semantic IDs."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,
                                  TrainingArguments, Trainer)
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Add special tokens for semantic IDs
    special_tokens = ["<music>", "</music>"]
    for level in range(n_levels):
        for c in range(n_clusters):
            special_tokens.append(f"<m_{c}>")
    tok.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"  Added {len(special_tokens)} special tokens", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.resize_token_embeddings(len(tok))

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Load data
    print(f"Loading {train_path}...", flush=True)
    data = []
    with open(train_path) as f:
        for line in f:
            data.append(json.loads(line))
    if max_examples:
        data = data[:max_examples]
    ds = Dataset.from_list(data)
    print(f"  {len(ds)} examples", flush=True)

    def tokenize_fn(example):
        msgs = example["messages"]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        out = tok(text, max_length=max_seq_len, truncation=True, padding=False)
        # Mask prompt tokens from loss
        try:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
        prompt_ids = tok(prompt, truncation=True, max_length=max_seq_len)["input_ids"]
        input_ids = list(out["input_ids"])
        attention_mask = list(out["attention_mask"])
        labels = list(input_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, desc="tokenize")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        bf16=(device == "cuda"),
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=training_args, train_dataset=ds, processing_class=tok, data_collator=data_collator)
    print("Training...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


# ============================================================================
# Step 3 GCRS: Fine-tune with frozen original embeddings
# ============================================================================

def train_model_gcrs(
    train_path: str,
    model_id: str,
    output_dir: str,
    n_clusters: int = 256,
    n_levels: int = 4,
    epochs: float = 2.0,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_seq_len: int = 2048,
    max_examples: Optional[int] = None,
) -> None:
    """GCRS LoRA fine-tune: add SID tokens, freeze original embeddings, train."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  DataCollatorForSeq2Seq,
                                  TrainingArguments, Trainer)
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[GCRS] Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build the full set of special tokens
    # Control tokens
    gcrs_special = ["<MODE=REC>", "<MODE=CHAT>", "<BOI>", "<EOI>", "<RESP>"]
    # SID tokens: <s1_0>..<s1_255>, <s2_0>..<s2_255>, ..., <sN_0>..<sN_255>
    for level in range(1, n_levels + 1):
        for c in range(n_clusters):
            gcrs_special.append(f"<s{level}_{c}>")
    # Disambiguating tokens for collisions (level n_levels+1)
    # Allow up to 64 disambiguation indices
    for c in range(64):
        gcrs_special.append(f"<s{n_levels+1}_{c}>")

    original_vocab_size = len(tok)
    tok.add_special_tokens({"additional_special_tokens": gcrs_special})
    new_vocab_size = len(tok)
    n_new_tokens = new_vocab_size - original_vocab_size
    print(f"  Added {n_new_tokens} special tokens (vocab: {original_vocab_size} -> {new_vocab_size})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Resize embeddings and freeze original ones
    model.resize_token_embeddings(new_vocab_size)

    # Freeze original embedding weights, only new tokens are trainable
    embed_in = model.get_input_embeddings()
    embed_out = model.get_output_embeddings()

    # Freeze all embedding weight initially
    if embed_in is not None:
        embed_in.weight.requires_grad = False
    if embed_out is not None:
        embed_out.weight.requires_grad = False

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Now make new token embeddings trainable
    # Access the base model embeddings through peft
    base_model = model.get_base_model()
    base_embed_in = base_model.get_input_embeddings()
    base_embed_out = base_model.get_output_embeddings()
    if base_embed_in is not None:
        base_embed_in.weight.requires_grad = True
    if base_embed_out is not None:
        base_embed_out.weight.requires_grad = True

    # Register a hook to zero out gradients for original tokens
    def _freeze_original_embed_hook(grad):
        """Zero gradients for original vocabulary tokens."""
        grad[:original_vocab_size] = 0
        return grad

    if base_embed_in is not None and base_embed_in.weight.requires_grad:
        base_embed_in.weight.register_hook(_freeze_original_embed_hook)
    if base_embed_out is not None and base_embed_out.weight.requires_grad:
        base_embed_out.weight.register_hook(_freeze_original_embed_hook)

    model.print_trainable_parameters()

    # Load data
    print(f"[GCRS] Loading {train_path}...", flush=True)
    data = []
    with open(train_path) as f:
        for line in f:
            data.append(json.loads(line))
    if max_examples:
        data = data[:max_examples]
    ds = Dataset.from_list(data)
    print(f"  {len(ds)} examples", flush=True)

    def tokenize_fn(example):
        msgs = example["messages"]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        out = tok(text, max_length=max_seq_len, truncation=True, padding=False)
        # Mask prompt tokens from loss
        try:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True)
        prompt_ids = tok(prompt, truncation=True, max_length=max_seq_len)["input_ids"]
        input_ids = list(out["input_ids"])
        attention_mask = list(out["attention_mask"])
        labels = list(input_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, desc="tokenize")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        bf16=(device == "cuda"),
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tok, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=ds,
        processing_class=tok, data_collator=data_collator,
    )
    print("[GCRS] Training...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[GCRS] Saved to {output_dir}")


# ============================================================================
# Step 4: Inference — generate SIDs and map back to tracks (simple)
# ============================================================================

def run_inference(
    model_dir: str,
    track_to_sid: Dict[str, List[int]],
    track_embs: np.ndarray,
    track_ids: List[str],
    n_clusters: int = 256,
    n_levels: int = 2,
    n_output: int = 100,
    split: str = "test",
    batch_size: int = 8,
) -> List[dict]:
    """Generate SIDs for each turn, map to nearest tracks."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Build reverse mapping: SID tuple -> list of track_ids
    sid_to_tracks: Dict[tuple, List[str]] = {}
    for tid, sid in track_to_sid.items():
        key = tuple(sid)
        if key not in sid_to_tracks:
            sid_to_tracks[key] = []
        sid_to_tracks[key].append(tid)

    # We'll use the track embeddings directly for ranking
    from sklearn.cluster import MiniBatchKMeans  # noqa: F401

    # Load model
    print(f"Loading model from {model_dir}...", flush=True)
    base_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path", "Qwen/Qwen3-0.6B")
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        base.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    device = next(model.parameters()).device

    # Load test data
    if split == "test":
        test = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        test = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{split}", split="test")

    # Build SID token IDs for parsing
    m_token_ids = {}
    for c in range(n_clusters):
        token_str = f"<m_{c}>"
        ids = tok.encode(token_str, add_special_tokens=False)
        if ids:
            m_token_ids[ids[0]] = c

    rows = []
    print(f"Running inference on {len(test)} sessions...", flush=True)

    for ex in tqdm(test, desc="inference"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        convos = ex["conversations"]
        df = pd.DataFrame(convos)

        for turn in range(1, 9):
            # Build context
            lines = []
            prior_tracks = []
            for c in convos:
                if c["turn_number"] > turn:
                    break
                if c["turn_number"] < turn:
                    if c["role"] == "user":
                        lines.append(f"User: {c['content']}")
                    elif c["role"] == "assistant":
                        lines.append(f"Assistant: {c['content'][:80]}")
                    elif c["role"] == "music":
                        prior_tracks.append(c["content"])
                        sid = track_to_sid.get(c["content"])
                        if sid:
                            sid_str = "".join(f"<m_{x}>" for x in sid)
                            lines.append(f"[Recommended: {sid_str}]")
                elif c["turn_number"] == turn and c["role"] == "user":
                    lines.append(f"User: {c['content']}")
            context = "\n".join(lines[-12:])

            # Generate
            messages = [
                {"role": "system", "content": "You are a music recommendation system. Given a conversation, output the semantic ID of the best track to recommend. Format: <music><m_X><m_Y></music>"},
                {"role": "user", "content": context},
            ]
            try:
                prompt = tok.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)

            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=1024).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=20,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1]:].tolist()

            # Parse generated SID
            generated_clusters = []
            for gid in gen_ids:
                if gid in m_token_ids:
                    generated_clusters.append(m_token_ids[gid])
                    if len(generated_clusters) >= n_levels:
                        break

            # Map SID to tracks
            preds = []
            if len(generated_clusters) == n_levels:
                sid_key = tuple(generated_clusters)
                # Exact match
                exact = sid_to_tracks.get(sid_key, [])
                preds.extend([t for t in exact if t not in prior_tracks])

            # Fallback: find tracks with matching first cluster, rank by embedding similarity
            if len(preds) < n_output and generated_clusters:
                # Get all tracks in the same coarse cluster
                target_cluster = generated_clusters[0]
                candidate_idx = [i for i, tid in enumerate(track_ids)
                                 if track_to_sid[tid][0] == target_cluster
                                 and tid not in prior_tracks
                                 and tid not in preds]
                if candidate_idx:
                    remaining = n_output - len(preds)
                    preds.extend([track_ids[i] for i in candidate_idx[:remaining]])

            # If still short, fill with global popular tracks
            if len(preds) < n_output:
                for tid in track_ids:
                    if tid not in preds and tid not in prior_tracks:
                        preds.append(tid)
                        if len(preds) >= n_output:
                            break

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(turn),
                "predicted_track_ids": preds[:n_output],
                "predicted_response": "",
            })

    return rows


# ============================================================================
# Step 4 GCRS: Constrained decoding with trie + beam search
# ============================================================================

def build_sid_trie(
    track_to_sid: Dict[str, List[int]],
) -> Dict[int, Any]:
    """Build a nested trie: {s1_id: {s2_id: {s3_id: {s4_id: [track_ids]}}}}.

    Leaf values are lists of track_ids (usually length 1 after collision resolution).
    """
    trie: Dict[int, Any] = {}
    for tid, sid in track_to_sid.items():
        node = trie
        for i, level_id in enumerate(sid[:-1]):
            if level_id not in node:
                node[level_id] = {}
            node = node[level_id]
        # Last level: store track_id
        last = sid[-1]
        if last not in node:
            node[last] = []
        if isinstance(node[last], list):
            node[last].append(tid)
        else:
            # Shouldn't happen if SIDs are unique after collision resolution
            node[last] = [tid]
    return trie


class GCRSConstrainedLogitsProcessor:
    """LogitsProcessor that constrains SID generation to valid trie paths.

    After <BOI> is generated, this processor restricts the vocabulary at each
    position to only valid continuations in the SID trie.
    """

    def __init__(
        self,
        trie: Dict[int, Any],
        token_to_id: Dict[str, int],
        boi_token_id: int,
        eoi_token_id: int,
        n_levels: int,
        n_clusters: int,
    ):
        self.trie = trie
        self.token_to_id = token_to_id  # e.g. "<s1_45>" -> token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.n_levels = n_levels
        self.n_clusters = n_clusters

        # Build level-specific token ID mappings
        # level_token_ids[level] = {cluster_idx: token_id}
        self.level_token_ids: List[Dict[int, int]] = []
        for level in range(1, n_levels + 1):
            mapping = {}
            for c in range(n_clusters):
                tok_str = f"<s{level}_{c}>"
                if tok_str in token_to_id:
                    mapping[c] = token_to_id[tok_str]
            self.level_token_ids.append(mapping)
        # Disambiguation level
        disambig_mapping = {}
        for c in range(64):
            tok_str = f"<s{n_levels+1}_{c}>"
            if tok_str in token_to_id:
                disambig_mapping[c] = token_to_id[tok_str]
        self.level_token_ids.append(disambig_mapping)

    def get_valid_next_tokens(
        self, generated_sid_clusters: List[int]
    ) -> List[int]:
        """Given the SID clusters generated so far, return valid next token IDs."""
        # Navigate the trie
        node = self.trie
        for cluster_id in generated_sid_clusters:
            if cluster_id in node:
                node = node[cluster_id]
            else:
                return []  # Invalid path

        # If node is a list (leaf), we should emit EOI
        if isinstance(node, list):
            return [self.eoi_token_id]

        # Otherwise, node is a dict — valid next are its keys
        level_idx = len(generated_sid_clusters)
        if level_idx < len(self.level_token_ids):
            level_map = self.level_token_ids[level_idx]
            valid_token_ids = []
            for cluster_id in node.keys():
                if cluster_id in level_map:
                    valid_token_ids.append(level_map[cluster_id])
            return valid_token_ids
        else:
            return [self.eoi_token_id]


def run_inference_gcrs(
    model_dir: str,
    track_to_sid: Dict[str, List[int]],
    track_embs: np.ndarray,
    track_ids: List[str],
    n_clusters: int = 256,
    n_levels: int = 4,
    n_output: int = 100,
    beam_size: int = 20,
    split: str = "test",
) -> List[dict]:
    """GCRS inference with constrained decoding and beam search."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Build trie for constrained decoding
    print("[GCRS] Building SID trie...", flush=True)
    trie = build_sid_trie(track_to_sid)

    # Build reverse mapping: SID tuple -> track_id
    sid_to_track: Dict[tuple, str] = {}
    for tid, sid in track_to_sid.items():
        sid_to_track[tuple(sid)] = tid

    # Load model
    print(f"[GCRS] Loading model from {model_dir}...", flush=True)
    base_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path", "Qwen/Qwen3-4B")
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        base.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    device = next(model.parameters()).device

    # Build token string -> token_id map for SID tokens
    token_to_id: Dict[str, int] = {}
    for level in range(1, n_levels + 2):  # +1 for disambig level
        limit = n_clusters if level <= n_levels else 64
        for c in range(limit):
            tok_str = f"<s{level}_{c}>"
            ids = tok.encode(tok_str, add_special_tokens=False)
            if len(ids) == 1:
                token_to_id[tok_str] = ids[0]

    # Get control token IDs
    boi_ids = tok.encode("<BOI>", add_special_tokens=False)
    eoi_ids = tok.encode("<EOI>", add_special_tokens=False)
    mode_rec_ids = tok.encode("<MODE=REC>", add_special_tokens=False)
    resp_ids = tok.encode("<RESP>", add_special_tokens=False)

    boi_token_id = boi_ids[0] if boi_ids else None
    eoi_token_id = eoi_ids[0] if eoi_ids else None

    # Build constrained processor
    processor = GCRSConstrainedLogitsProcessor(
        trie=trie,
        token_to_id=token_to_id,
        boi_token_id=boi_token_id,
        eoi_token_id=eoi_token_id,
        n_levels=n_levels,
        n_clusters=n_clusters,
    )

    # Reverse token_id -> (level, cluster)
    id_to_level_cluster: Dict[int, Tuple[int, int]] = {}
    for level in range(1, n_levels + 2):
        limit = n_clusters if level <= n_levels else 64
        for c in range(limit):
            tok_str = f"<s{level}_{c}>"
            if tok_str in token_to_id:
                id_to_level_cluster[token_to_id[tok_str]] = (level, c)

    # Load test data
    if split == "test":
        test = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        test = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{split}", split="test")

    SYSTEM_PROMPT = (
        "You are a conversational music recommender. When recommending, "
        "output the track's semantic ID."
    )

    rows = []
    print(f"[GCRS] Running inference on {len(test)} sessions (beam_size={beam_size})...", flush=True)

    for ex in tqdm(test, desc="gcrs inference"):
        session_id = ex["session_id"]
        user_id = ex["user_id"]
        convos = ex["conversations"]

        for turn in range(1, 9):
            # Build context
            history_lines = []
            prior_tracks = []
            for c in convos:
                if c["turn_number"] > turn:
                    break
                if c["turn_number"] < turn:
                    if c["role"] == "user":
                        history_lines.append(f"User: {c['content']}")
                    elif c["role"] == "assistant":
                        history_lines.append(f"Assistant: {c['content'][:100]}")
                    elif c["role"] == "music":
                        prior_tracks.append(c["content"])
                        sid = track_to_sid.get(c["content"])
                        if sid:
                            history_lines.append(
                                f"[Recommended: {gcrs_sid_to_tokens(sid)}]"
                            )
                elif c["turn_number"] == turn and c["role"] == "user":
                    history_lines.append(f"User: {c['content']}")

            user_msg = "\n".join(history_lines[-14:])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            try:
                prompt_text = tok.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tok.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = tok(prompt_text, return_tensors="pt", truncation=True,
                         max_length=2048).to(device)
            input_len = inputs["input_ids"].shape[1]

            # Beam search with constrained decoding
            # Step 1: Generate up to <BOI> (or <MODE=REC>)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=3,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            prefix_ids = out[0][input_len:].tolist()

            # Check if mode is REC (has BOI)
            is_rec_mode = boi_token_id is not None and boi_token_id in prefix_ids

            if not is_rec_mode:
                # Chat mode — no track recommendation for this turn
                # Fill with fallback tracks
                preds = [tid for tid in track_ids
                         if tid not in prior_tracks][:n_output]
                rows.append({
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_number": int(turn),
                    "predicted_track_ids": preds[:n_output],
                    "predicted_response": "",
                })
                continue

            # REC mode: constrained beam search over SID tokens
            # Prepare prefix: input + <MODE=REC><BOI>
            prefix_tokens = []
            if mode_rec_ids:
                prefix_tokens.extend(mode_rec_ids)
            if boi_ids:
                prefix_tokens.extend(boi_ids)

            prefix_input_ids = inputs["input_ids"][0].tolist() + prefix_tokens
            prefix_tensor = torch.tensor(
                [prefix_input_ids], dtype=torch.long, device=device
            )

            # Get logits for first SID position
            with torch.no_grad():
                outputs = model(input_ids=prefix_tensor)
                logits = outputs.logits[0, -1, :]  # (vocab_size,)
                log_probs = torch.log_softmax(logits, dim=-1)

            # Initialize beams
            valid_first = processor.get_valid_next_tokens([])
            if not valid_first:
                # Fallback
                preds = [tid for tid in track_ids
                         if tid not in prior_tracks][:n_output]
                rows.append({
                    "session_id": session_id,
                    "user_id": user_id,
                    "turn_number": int(turn),
                    "predicted_track_ids": preds[:n_output],
                    "predicted_response": "",
                })
                continue

            # beams: list of (cumulative_log_prob, cluster_ids_so_far, token_ids_so_far)
            beams: List[Tuple[float, List[int], List[int]]] = []
            for token_id in valid_first:
                if token_id in id_to_level_cluster:
                    _, cluster = id_to_level_cluster[token_id]
                    score = log_probs[token_id].item()
                    beams.append((score, [cluster], prefix_tokens + [token_id]))

            # Sort and keep top beam_size
            beams.sort(key=lambda x: x[0], reverse=True)
            beams = beams[:beam_size]

            # Expand beams level by level
            max_sid_len = max(len(sid) for sid in track_to_sid.values())
            for step in range(1, max_sid_len):
                if not beams:
                    break

                new_beams: List[Tuple[float, List[int], List[int]]] = []
                for cum_score, clusters, tok_ids in beams:
                    valid_next = processor.get_valid_next_tokens(clusters)
                    if not valid_next:
                        continue

                    if valid_next == [eoi_token_id]:
                        # This beam is complete
                        new_beams.append((cum_score, clusters, tok_ids))
                        continue

                    # Get logits for next position
                    full_ids = inputs["input_ids"][0].tolist() + tok_ids
                    full_tensor = torch.tensor(
                        [full_ids], dtype=torch.long, device=device
                    )
                    with torch.no_grad():
                        out_step = model(input_ids=full_tensor)
                        step_logits = out_step.logits[0, -1, :]
                        step_log_probs = torch.log_softmax(step_logits, dim=-1)

                    for next_token_id in valid_next:
                        if next_token_id in id_to_level_cluster:
                            _, next_cluster = id_to_level_cluster[next_token_id]
                            next_score = cum_score + step_log_probs[next_token_id].item()
                            new_beams.append((
                                next_score,
                                clusters + [next_cluster],
                                tok_ids + [next_token_id],
                            ))

                new_beams.sort(key=lambda x: x[0], reverse=True)
                # Keep completed beams + top-K incomplete
                completed = [b for b in new_beams
                             if processor.get_valid_next_tokens(b[1]) == [eoi_token_id]
                             or not processor.get_valid_next_tokens(b[1])]
                incomplete = [b for b in new_beams
                              if b not in completed]
                beams = completed + incomplete[:beam_size]
                beams.sort(key=lambda x: x[0], reverse=True)
                beams = beams[:beam_size]

            # Map beam results to track_ids
            preds = []
            seen_tracks = set(prior_tracks)
            for cum_score, clusters, tok_ids in beams:
                sid_key = tuple(clusters)
                tid = sid_to_track.get(sid_key)
                if tid and tid not in seen_tracks:
                    preds.append(tid)
                    seen_tracks.add(tid)

            # Fallback: nearest-neighbor in embedding space for incomplete
            if len(preds) < n_output and track_embs is not None:
                # Use the centroid of found tracks as query
                if preds:
                    found_idx = [track_ids.index(t) for t in preds
                                 if t in track_ids]
                    if found_idx:
                        query_vec = track_embs[found_idx].mean(axis=0)
                        sims = track_embs @ query_vec
                        ranked = np.argsort(-sims)
                        for idx in ranked:
                            tid = track_ids[idx]
                            if tid not in seen_tracks:
                                preds.append(tid)
                                seen_tracks.add(tid)
                                if len(preds) >= n_output:
                                    break

            # If still short, fill with remaining tracks
            if len(preds) < n_output:
                for tid in track_ids:
                    if tid not in seen_tracks:
                        preds.append(tid)
                        seen_tracks.add(tid)
                        if len(preds) >= n_output:
                            break

            rows.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": int(turn),
                "predicted_track_ids": preds[:n_output],
                "predicted_response": "",
            })

    return rows


# ============================================================================
# Main: all-in-one pipeline
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generative Retrieval via Semantic IDs (simple or GCRS)"
    )
    p.add_argument("--version", choices=["simple", "gcrs"], default="simple",
                   help="simple=2-level KMeans (default), gcrs=4-level RQ + constrained decoding")
    p.add_argument("--model_id", default=None,
                   help="Base model (default: Qwen/Qwen3-0.6B for simple, Qwen/Qwen3-4B for gcrs)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: out/generative or out/generative_gcrs)")
    p.add_argument("--out_leg", default=None,
                   help="Output leg JSON path")
    p.add_argument("--n_clusters", type=int, default=256,
                   help="Clusters per RQ level (default: 256)")
    p.add_argument("--n_levels", type=int, default=None,
                   help="Number of RQ levels (default: 2 for simple, 4 for gcrs)")
    p.add_argument("--beam_size", type=int, default=20,
                   help="Beam search width for GCRS inference (default: 20)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Training batch size (default: 8 for simple, 4 for gcrs)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n_output", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--max_examples", type=int, default=None)
    args = p.parse_args()

    # Set version-specific defaults
    if args.version == "gcrs":
        if args.model_id is None:
            args.model_id = "Qwen/Qwen3-4B"
        if args.output_dir is None:
            args.output_dir = "out/generative_gcrs"
        if args.out_leg is None:
            args.out_leg = "exp/inference/devset/generative_gcrs_top100.json"
        if args.n_levels is None:
            args.n_levels = 4
        if args.batch_size is None:
            args.batch_size = 4
    else:
        if args.model_id is None:
            args.model_id = "Qwen/Qwen3-0.6B"
        if args.output_dir is None:
            args.output_dir = "out/generative"
        if args.out_leg is None:
            args.out_leg = "exp/inference/devset/generative_top100.json"
        if args.n_levels is None:
            args.n_levels = 2
        if args.batch_size is None:
            args.batch_size = 8

    os.makedirs(args.output_dir, exist_ok=True)

    if args.version == "gcrs":
        _run_gcrs_pipeline(args)
    else:
        _run_simple_pipeline(args)


def _run_simple_pipeline(args: argparse.Namespace) -> None:
    """Original simple generative retrieval pipeline."""
    # Step 1: Build semantic IDs
    print("=" * 60)
    print("STEP 1: Building Semantic IDs (K-Means RQ)")
    print("=" * 60)
    sid_path = os.path.join(args.output_dir, "track_sids.json")
    if os.path.exists(sid_path):
        print(f"  Loading cached SIDs from {sid_path}")
        with open(sid_path) as f:
            track_to_sid = json.load(f)
        # Also need embeddings for inference fallback
        emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        track_ids = [r["track_id"] for r in emb_ds]
        track_embs = None  # Will reload if needed
    else:
        track_to_sid, track_embs, track_ids = build_semantic_ids(
            args.n_clusters, args.n_levels)
        with open(sid_path, "w") as f:
            json.dump(track_to_sid, f)
        print(f"  Saved SIDs to {sid_path}")

    # Step 2: Build training data
    print("\n" + "=" * 60)
    print("STEP 2: Building Training Data")
    print("=" * 60)
    train_path = os.path.join(args.output_dir, "train_data.jsonl")
    if os.path.exists(train_path):
        print(f"  [skip] {train_path} exists")
    else:
        build_training_data(track_to_sid, train_path, args.n_levels)

    # Step 3: Train
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning LLM")
    print("=" * 60)
    model_out = os.path.join(args.output_dir, "model")
    if os.path.exists(os.path.join(model_out, "adapter_config.json")):
        print(f"  [skip] {model_out} exists")
    else:
        train_model(
            train_path, args.model_id, model_out,
            args.n_clusters, args.n_levels,
            args.epochs, args.batch_size, args.lr,
            max_examples=args.max_examples,
        )

    # Step 4: Inference
    print("\n" + "=" * 60)
    print("STEP 4: Inference")
    print("=" * 60)
    if track_embs is None:
        # Reload embeddings
        emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        track_ids_reload = []
        vecs = []
        for r in tqdm(emb_ds, desc="reload embs"):
            e = r["metadata-qwen3_embedding_0.6b"]
            track_ids_reload.append(r["track_id"])
            vecs.append(e if isinstance(e, list) and len(e) == 1024 else [0.0] * 1024)
        track_embs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(track_embs, axis=1, keepdims=True)
        track_embs = track_embs / np.clip(norms, 1e-9, None)
        track_ids = track_ids_reload

    rows = run_inference(
        model_out, track_to_sid, track_embs, track_ids,
        args.n_clusters, args.n_levels, args.n_output, args.split,
    )

    os.makedirs(os.path.dirname(args.out_leg) or ".", exist_ok=True)
    with open(args.out_leg, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\nWrote {len(rows)} predictions to {args.out_leg}")
    print("Add to LightGBM: --legs ...,generative_top100")


def _run_gcrs_pipeline(args: argparse.Namespace) -> None:
    """GCRS generative retrieval pipeline with constrained decoding."""
    # Step 1: Build semantic IDs (4-level with collision resolution)
    print("=" * 60)
    print("STEP 1: Building Semantic IDs (GCRS 4-level RQ)")
    print("=" * 60)
    sid_path = os.path.join(args.output_dir, "track_sids.json")
    if os.path.exists(sid_path):
        print(f"  Loading cached SIDs from {sid_path}")
        with open(sid_path) as f:
            track_to_sid = json.load(f)
        emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        track_ids = [r["track_id"] for r in emb_ds]
        track_embs = None
    else:
        track_to_sid, track_embs, track_ids = build_semantic_ids_gcrs(
            args.n_clusters, args.n_levels)
        with open(sid_path, "w") as f:
            json.dump(track_to_sid, f)
        print(f"  Saved SIDs to {sid_path}")

    # Step 2: Build structured training data
    print("\n" + "=" * 60)
    print("STEP 2: Building GCRS Training Data")
    print("=" * 60)
    train_path = os.path.join(args.output_dir, "train_data.jsonl")
    if os.path.exists(train_path):
        print(f"  [skip] {train_path} exists")
    else:
        build_training_data_gcrs(track_to_sid, train_path, args.n_levels)

    # Step 3: Train with frozen original embeddings
    print("\n" + "=" * 60)
    print("STEP 3: Fine-tuning LLM (GCRS)")
    print("=" * 60)
    model_out = os.path.join(args.output_dir, "model")
    if os.path.exists(os.path.join(model_out, "adapter_config.json")):
        print(f"  [skip] {model_out} exists")
    else:
        train_model_gcrs(
            train_path, args.model_id, model_out,
            args.n_clusters, args.n_levels,
            args.epochs, args.batch_size, args.lr,
            max_examples=args.max_examples,
        )

    # Step 4: Constrained inference with beam search
    print("\n" + "=" * 60)
    print("STEP 4: GCRS Constrained Inference")
    print("=" * 60)
    if track_embs is None:
        emb_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", split="all_tracks")
        track_ids_reload = []
        vecs = []
        for r in tqdm(emb_ds, desc="reload embs"):
            e = r["metadata-qwen3_embedding_0.6b"]
            track_ids_reload.append(r["track_id"])
            vecs.append(e if isinstance(e, list) and len(e) == 1024 else [0.0] * 1024)
        track_embs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(track_embs, axis=1, keepdims=True)
        track_embs = track_embs / np.clip(norms, 1e-9, None)
        track_ids = track_ids_reload

    rows = run_inference_gcrs(
        model_out, track_to_sid, track_embs, track_ids,
        args.n_clusters, args.n_levels, args.n_output,
        args.beam_size, args.split,
    )

    os.makedirs(os.path.dirname(args.out_leg) or ".", exist_ok=True)
    with open(args.out_leg, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\n[GCRS] Wrote {len(rows)} predictions to {args.out_leg}")
    print("Add to LightGBM: --legs ...,generative_gcrs_top100")


if __name__ == "__main__":
    main()
