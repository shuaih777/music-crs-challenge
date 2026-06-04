"""Run the trained conversation-state extractor over devset / blind sets.

Produces a JSONL of structured states per (session, turn) that
`baselines_v3.py --states <jsonl>` can read in to drive BM25 + dense.

Usage:
    python src/extractor_inference.py \
        --model_dir out/state_extractor_qwen3_0.6b \
        --split test \
        --out exp/states/test.jsonl

Output schema (one line per row):
    {"session_id": str, "turn_number": int, "state": {...}}

CPU works (slow); GPU is auto-selected when available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd


SYSTEM_PROMPT_FALLBACK = (
    "You are a music preference extractor. Read the conversation between a "
    "user and a music recommendation assistant. Output a JSON object that "
    "summarizes what the user wants RIGHT NOW (after the latest user message). "
    "Be specific and use only what's grounded in the conversation. "
    "Schema:\n"
    "{\"genre\": [str], \"mood\": [str], \"era\": str, \"energy\": str, "
    "\"accepted_tags\": [str], \"rejected_tags\": [str], "
    "\"artist_hints\": [str]}\n"
    "Use [] for empty lists, \"\" for unknown strings."
)


def build_user_block(conversations: list, turn: int, tracks_by_id: Dict[str, dict]) -> str:
    df = pd.DataFrame(conversations)
    hist = df[df["turn_number"] < turn]
    cur_user = df[(df["turn_number"] == turn) & (df["role"] == "user")]
    user_query = cur_user.iloc[0]["content"] if len(cur_user) else ""
    lines = []
    for _, row in hist.iterrows():
        role, content = row["role"], row["content"]
        if role == "music":
            meta = tracks_by_id.get(content, {})
            name = meta.get("track_name") or [""]
            artist = meta.get("artist_name") or [""]
            name_s = ", ".join(name) if isinstance(name, list) else str(name)
            artist_s = ", ".join(artist) if isinstance(artist, list) else str(artist)
            lines.append(f"system_recommended: {name_s} by {artist_s}")
        else:
            lines.append(f"{role}: {content}")
    history_block = "\n".join(lines)
    return f"<history>\n{history_block}\n</history>\n<user>\n{user_query}\n</user>"


def safe_parse_json(s: str) -> Dict[str, Any]:
    """Try to parse the assistant output as JSON; fall back to a permissive
    extractor that grabs the first balanced { ... } block.

    Robust to:
      - leading/trailing whitespace
      - <think>...</think> blocks before the JSON (Qwen3 reasoning mode)
      - text after the JSON block ("Here's the answer: {...} Let me explain...")
      - braces inside string literals (handled by full json.loads on the slice)
    """
    s = (s or "").strip()
    if not s:
        return {}

    # Strip <think>...</think> block(s) at the start
    while True:
        m = re.match(r"^\s*<think>.*?</think>\s*", s, flags=re.DOTALL)
        if not m:
            break
        s = s[m.end():]

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Find first balanced { ... } block, respecting strings + escapes.
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        return {}
    return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Directory saved by train_state_extractor.py "
                        "(may be a LoRA adapter; the base model id is read from adapter_config.json)")
    p.add_argument("--base_model", default=None,
                   help="If --model_dir is a LoRA adapter, fall back to this base model id")
    p.add_argument("--split", default="test", choices=["test", "Blind-A", "Blind-B"])
    p.add_argument("--out", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Generation budget per turn. With enable_thinking=False "
                        "the JSON target is rarely > 200 tokens; keeping headroom.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_sessions", type=int, default=None)
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}\n"
              "Install GPU deps: pip install -r requirements-gpu.txt", file=sys.stderr)
        sys.exit(1)

    # Detect: pure save vs LoRA adapter
    is_lora = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))

    if is_lora:
        from peft import PeftModel
        with open(os.path.join(args.model_dir, "adapter_config.json")) as f:
            cfg = json.load(f)
        base_id = args.base_model or cfg.get("base_model_name_or_path")
        if not base_id:
            print("ERROR: cannot determine base model; pass --base_model", file=sys.stderr)
            sys.exit(1)
        print(f"Loading base {base_id} + LoRA adapter from {args.model_dir} ...")
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True,
                                            padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_id, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, args.model_dir)
    else:
        print(f"Loading model from {args.model_dir} ...")
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True,
                                            padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"Inference device: {device}")

    if args.split == "test":
        convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    else:
        convo = load_dataset(f"talkpl-ai/TalkPlayData-Challenge-{args.split}", split="test")
    if args.max_sessions:
        convo = convo.select(range(min(args.max_sessions, len(convo))))
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    tracks_by_id = {t["track_id"]: t for t in tracks}
    print(f"Sessions: {len(convo)}, tracks: {len(tracks_by_id)}")

    # Build prompts
    # IMPORTANT: Qwen3 enables <think> reasoning by default which causes the
    # model to spend its budget on free-form reasoning instead of producing
    # the JSON object. enable_thinking=False forces direct emission.
    def render(messages):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older / non-Qwen tokenizers don't accept enable_thinking
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    prompts: List[dict] = []
    for ex in convo:
        for tn in range(1, 9):
            user_block = build_user_block(ex["conversations"], tn, tracks_by_id)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_FALLBACK},
                {"role": "user", "content": user_block},
            ]
            prompt = render(messages)
            prompts.append({
                "session_id": ex["session_id"],
                "turn_number": tn,
                "prompt": prompt,
            })
    print(f"Built {len(prompts)} prompts")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_parsed = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="generate"):
            batch = prompts[i:i + args.batch_size]
            inputs = tok([b["prompt"] for b in batch], return_tensors="pt",
                         padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            # With padding_side='left', every row's generated tokens start at
            # the same column = the padded prompt length.
            prompt_len = inputs["input_ids"].shape[1]
            for j, b in enumerate(batch):
                gen = out_ids[j][prompt_len:]
                text = tok.decode(gen, skip_special_tokens=True)
                state = safe_parse_json(text)
                if state:
                    n_parsed += 1
                fout.write(json.dumps({
                    "session_id": b["session_id"],
                    "turn_number": b["turn_number"],
                    "state": state,
                    "raw": text,
                }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} states to {args.out}; {n_parsed} parsed cleanly")


if __name__ == "__main__":
    main()
