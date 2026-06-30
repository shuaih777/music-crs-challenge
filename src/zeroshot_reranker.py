"""
Zero-shot listwise reranker using a large instruction-tuned model.
Run as a shard: python src/zeroshot_reranker.py --shard 0 --total_shards 3 ...
"""
import argparse, json, os, re, sys
from typing import List

def track_desc(track_meta: dict, tid: str) -> str:
    m = track_meta.get(tid, {})
    name = ", ".join(m.get("track_name", []))[:50]
    artist = ", ".join(m.get("artist_name", []))[:40]
    tags = ", ".join(str(t) for t in (m.get("tag_list") or [])[:6])
    year = (m.get("release_date") or "")[:4]
    return f'"{name}" by {artist} ({tags}) [{year}]'


def build_context(convos: list, turn: int) -> str:
    lines = []
    for c in convos:
        if c["turn_number"] > turn:
            break
        if c["turn_number"] < turn:
            if c["role"] == "user":
                lines.append(f"User: {c['content']}")
            elif c["role"] == "assistant":
                lines.append(f"Assistant: {c['content'][:100]}")
        elif c["turn_number"] == turn and c["role"] == "user":
            lines.append(f"User: {c['content']}")
    return "\n".join(lines[-8:])


def parse_ranking(output: str, n: int) -> List[int]:
    numbers = [int(x) for x in re.findall(r"\d+", output)]
    valid, seen = [], set()
    for x in numbers:
        if 1 <= x <= n and x not in seen:
            valid.append(x - 1)
            seen.add(x)
    for i in range(n):
        if i not in seen:
            valid.append(i)
    return valid[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--total_shards", type=int, required=True)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--inference", default="exp/inference/devset/lgbm_abl_plus_nvembed.json")
    p.add_argument("--out", required=True)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"[shard {args.shard}/{args.total_shards}] Loading {args.model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device

    print(f"[shard {args.shard}] Loading data...", flush=True)
    preds = json.load(open(args.inference))
    tracks_ds = load_dataset(
        "parquet",
        data_files={"train": "hf://datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"},
        split="train")
    track_meta = {r["track_id"]: r for r in tracks_ds}
    test_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    conversations = {ex["session_id"]: ex["conversations"] for ex in test_ds}

    total = len(preds)
    shard_size = (total + args.total_shards - 1) // args.total_shards
    start = args.shard * shard_size
    end = min(start + shard_size, total)
    my_preds = preds[start:end]
    print(f"[shard {args.shard}] Processing rows {start}-{end} ({len(my_preds)} turns)", flush=True)

    output_rows = []
    for row in tqdm(my_preds, desc=f"shard-{args.shard}"):
        candidates = row["predicted_track_ids"][:args.top_k]
        convos = conversations.get(row["session_id"], [])
        context = build_context(convos, row["turn_number"])
        track_lines = [f"[{i+1}] {track_desc(track_meta, tid)}" for i, tid in enumerate(candidates)]

        user_msg = (
            f"Conversation:\n{context}\n\n"
            f"Candidate tracks:\n" + "\n".join(track_lines) +
            f"\n\nRank ALL {len(candidates)} tracks from most to least relevant to the conversation. "
            f"Output only the track numbers as a comma-separated list, e.g.: 3, 1, 5, 2, ..."
        )
        messages = [
            {"role": "system", "content": (
                "You are a music recommendation expert. "
                "Given a conversation, rank the candidate tracks by relevance. "
                "Consider the mood, genre, artist style, and context of the conversation."
            )},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=120, do_sample=False,
                pad_token_id=tok.pad_token_id,
                temperature=1.0)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ranking = parse_ranking(gen, len(candidates))
        reranked = [candidates[i] for i in ranking]
        output_rows.append({
            "session_id": row["session_id"],
            "user_id": row.get("user_id", ""),
            "turn_number": row["turn_number"],
            "predicted_track_ids": reranked,
            "predicted_response": row.get("predicted_response", ""),
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False)
    print(f"[shard {args.shard}] Wrote {len(output_rows)} rows → {args.out}", flush=True)


if __name__ == "__main__":
    main()
