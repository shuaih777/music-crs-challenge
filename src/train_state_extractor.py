"""Train a small LM as a music conversation-state extractor.

Usage:
    # 1. build training data (CPU only, ~1-2 min)
    python src/data_prep.py --out data/state_extractor_train.jsonl

    # 2. train (GPU; ~45-90 min on A100, depending on model size)
    python src/train_state_extractor.py \
        --train_jsonl data/state_extractor_train.jsonl \
        --model_id Qwen/Qwen3-0.6B \
        --output_dir out/state_extractor_qwen3_0.6b

The trainer uses LoRA (r=16, alpha=32) by default — a single A100/24GB card
fits Qwen3-0.6B at batch=16, Qwen3-4B at batch=4. For multi-GPU just run
with `accelerate launch` instead of `python`.

CPU mode also works (for smoke tests / Colab T4) — it will be slow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# These imports are heavy and only needed for training. We import lazily so
# this file can still be parsed without torch installed.


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--eval_jsonl", default=None,
                   help="Optional held-out JSONL (use data_prep.py with --max_sessions on test split)")
    p.add_argument("--model_id", default="Qwen/Qwen3-0.6B",
                   help="HF model id; tested with Qwen/Qwen3-0.6B and Qwen/Qwen3-4B")
    p.add_argument("--output_dir", default="out/state_extractor")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Per-device train batch size; lower for larger models")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true",
                   help="Full fine-tune (needs much more VRAM)")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--fp16", action="store_true",
                   help="Use fp16 instead of bf16 (older GPUs)")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_examples", type=int, default=None,
                   help="Truncate training data — useful for quick smoke tests")
    args = p.parse_args()

    # Lazy imports so `--help` works without torch installed
    try:
        import torch
        from datasets import Dataset
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  TrainingArguments, Trainer,
                                  DataCollatorForLanguageModeling)
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}\n"
              "Install GPU deps: pip install -r requirements-gpu.txt", file=sys.stderr)
        sys.exit(1)

    use_lora = not args.no_lora
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            print("ERROR: peft not installed; pip install peft  (or pass --no_lora)",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Loading {args.model_id} ...")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else (
        torch.float16 if args.fp16 else torch.float32
    )
    if not torch.cuda.is_available():
        print("[warn] no CUDA visible — falling back to fp32 on CPU. "
              "Training will be very slow.")
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_lora:
        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, cfg)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    # Load JSONL into HF Dataset
    def load_jsonl(path: str) -> Dataset:
        data: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
        if args.max_examples:
            data = data[: args.max_examples]
        return Dataset.from_list(data)

    print(f"Loading {args.train_jsonl} ...")
    train_ds = load_jsonl(args.train_jsonl)
    print(f"  train: {len(train_ds)}")
    eval_ds = None
    if args.eval_jsonl:
        eval_ds = load_jsonl(args.eval_jsonl)
        print(f"  eval:  {len(eval_ds)}")

    # Tokenize messages -> tokens; mask user/system tokens out of the loss so
    # the model only learns to *generate* the assistant's structured output.
    def tokenize(example: dict) -> dict:
        msgs = example["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        out = tok(text, max_length=args.max_seq_len, truncation=True, padding=False)
        # Build labels: copy input_ids, mask everything before the assistant turn
        # by finding the assistant header in the rendered text
        prompt_only = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
        )
        prompt_tok = tok(prompt_only, truncation=True, max_length=args.max_seq_len)
        n_prompt = len(prompt_tok["input_ids"])
        labels = list(out["input_ids"])
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100
        out["labels"] = labels
        return out

    cols_to_remove = train_ds.column_names
    train_tok = train_ds.map(tokenize, remove_columns=cols_to_remove,
                             desc="tokenize:train", num_proc=1)
    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(tokenize, remove_columns=eval_ds.column_names,
                               desc="tokenize:eval", num_proc=1)

    def collator(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tok.pad(inputs, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for lab in labels:
            pad_len = max_len - len(lab)
            if tok.padding_side == "left":
                padded = [-100] * pad_len + lab
            else:
                padded = lab + [-100] * pad_len
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_tok is not None else "no",
        eval_steps=args.save_steps if eval_tok is not None else None,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving final model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
