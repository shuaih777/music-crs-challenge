"""Generate natural-language recommendation responses using Qwen3-0.6B.

Replaces the templated "How about X by Y?" responses with grounded,
diverse explanations that should score higher on Distinct-2 and LLM-as-judge.

Usage:
    python src/generate_responses.py \
        --inference exp/inference/devset/lgbm_reranked.json \
        --out exp/inference/devset/lgbm_reranked_with_responses.json \
        --model Qwen/Qwen3-0.6B \
        --batch_size 16

CPU works but slow (~30 min). GPU: ~2-3 min on A100.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm


RESPONSE_TEMPLATES = [
    "Based on your love of {genre} music, I think you'll really enjoy \"{track}\" by {artist}. "
    "It's got {mood_desc} that matches what you've been looking for{era_note}.",

    "Here's a great pick for you: \"{track}\" by {artist}. "
    "This {genre} track{era_note} brings {mood_desc} that fits perfectly with your taste.",

    "I'd recommend \"{track}\" by {artist} — it's a {genre} gem{era_note}. "
    "{mood_desc_cap} and {energy} energy make it an ideal match for your current mood.",

    "You might love \"{track}\" by {artist}! "
    "It's {genre}{era_note}, with {mood_desc}. {follow_up}",

    "Given what you've shared, \"{track}\" by {artist} seems like a perfect fit. "
    "This {energy}-energy {genre} track{era_note} delivers {mood_desc}.",

    "Let me suggest \"{track}\" by {artist}. "
    "As a {genre} track{era_note}, it offers {mood_desc} — {follow_up}",

    "How about \"{track}\" by {artist}? "
    "It's a {mood_adj} {genre} piece{era_note} that I think captures exactly what you're after.",

    "I think \"{track}\" by {artist} would resonate with you. "
    "It's {genre}{era_note}, bringing {mood_desc}. {personalization}",
]

FOLLOW_UPS = [
    "Would you like more in this direction?",
    "Let me know if you'd like something similar or different!",
    "I can find more along these lines if you're interested.",
    "Want me to explore more of this vibe?",
    "Shall I look for other tracks with a similar feel?",
    "I have more where that came from if you like it!",
    "Feel free to tell me if this hits the mark or not.",
    "Happy to adjust if you want something different!",
]

PERSONALIZATIONS = [
    "I chose this based on your preference for {genre} sounds.",
    "This fits well with the artists you've been enjoying.",
    "It matches the mood you described perfectly.",
    "Your taste pointed me right toward this one.",
]


def _as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v[:3] if x)
    return str(v) if v else ""


def generate_template_response(
    track_meta: dict,
    turn: int,
    state: dict | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a diverse templated response grounded in track metadata."""
    rng = rng or random.Random()

    name = _as_text(track_meta.get("track_name", [""]))
    artist = _as_text(track_meta.get("artist_name", [""]))
    tags = track_meta.get("tag_list") or []
    popularity = track_meta.get("popularity", 0) or 0
    release_date = track_meta.get("release_date") or ""

    # Extract genre/mood from tags
    mood_tags = [t for t in tags[:15] if t.lower() in
                 {"chill", "relaxing", "energetic", "upbeat", "melancholic",
                  "happy", "sad", "aggressive", "dreamy", "atmospheric",
                  "smooth", "intense", "epic", "dark", "groovy", "funky",
                  "romantic", "nostalgic", "euphoric", "mellow"}]
    genre_tags = [t for t in tags[:10] if t.lower() in
                  {"rock", "pop", "electronic", "hip-hop", "jazz", "classical",
                   "r&b", "folk", "indie", "alternative", "metal", "punk",
                   "soul", "blues", "country", "reggae", "ambient", "dance",
                   "lo-fi", "experimental", "downtempo", "trip-hop"}]

    genre = genre_tags[0] if genre_tags else (tags[0] if tags else "music")
    mood_adj = mood_tags[0] if mood_tags else "captivating"
    mood_desc = f"a {mood_adj} atmosphere" if mood_tags else "a unique musical character"
    mood_desc_cap = mood_desc[0].upper() + mood_desc[1:]
    energy = "high" if popularity > 60 else ("medium" if popularity > 30 else "low")
    era_note = f" from {release_date[:4]}" if release_date and len(release_date) >= 4 else ""

    template = rng.choice(RESPONSE_TEMPLATES)
    follow_up = rng.choice(FOLLOW_UPS)
    personalization = rng.choice(PERSONALIZATIONS).format(genre=genre)

    try:
        response = template.format(
            track=name, artist=artist, genre=genre,
            mood_desc=mood_desc, mood_desc_cap=mood_desc_cap,
            mood_adj=mood_adj, energy=energy, era_note=era_note,
            follow_up=follow_up, personalization=personalization,
        )
    except (KeyError, IndexError):
        response = f"I'd recommend \"{name}\" by {artist}. It's a great {genre} track{era_note}."

    return response


def generate_llm_responses(
    inference_rows: List[dict],
    track_meta: Dict[str, dict],
    model_id: str,
    batch_size: int = 16,
    max_new_tokens: int = 80,
) -> List[str]:
    """Use a small LLM to generate responses (GPU path)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("torch/transformers not available, falling back to templates",
              file=sys.stderr)
        return None

    print(f"Loading {model_id} for response generation...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    print(f"  device={device}", flush=True)

    responses = []
    prompts = []
    for r in inference_rows:
        top_track = r["predicted_track_ids"][0] if r["predicted_track_ids"] else ""
        meta = track_meta.get(top_track, {})
        name = _as_text(meta.get("track_name", [""]))
        artist = _as_text(meta.get("artist_name", [""]))
        tags = (meta.get("tag_list") or [])[:8]
        tags_str = ", ".join(str(t) for t in tags)

        prompt = (
            f"You are a music recommendation assistant. "
            f"Briefly explain why you recommend \"{name}\" by {artist} "
            f"(tags: {tags_str}). "
            f"Be enthusiastic, mention the track's qualities, and ask if they want more. "
            f"Keep it under 50 words."
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(rendered)

    print(f"Generating {len(prompts)} responses...", flush=True)
    for i in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[i:i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.85,
                repetition_penalty=1.15,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for j in range(len(batch)):
            gen = out_ids[j][prompt_len:]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            responses.append(text)

    return responses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inference", required=True,
                   help="Input inference JSON (with predicted_track_ids)")
    p.add_argument("--out", required=True,
                   help="Output inference JSON (same but with better responses)")
    p.add_argument("--mode", default="auto", choices=["template", "llm", "auto"],
                   help="'template' = fast diverse templates (CPU). "
                        "'llm' = Qwen3-0.6B generation (GPU). "
                        "'auto' = llm if torch+cuda available, else template.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading inference from {args.inference}...", flush=True)
    with open(args.inference) as f:
        rows = json.load(f)
    print(f"  {len(rows)} rows", flush=True)

    print("Loading track metadata...", flush=True)
    tracks_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    track_meta = {r["track_id"]: r for r in tracks_ds}

    # Determine mode
    use_llm = False
    if args.mode == "llm":
        use_llm = True
    elif args.mode == "auto":
        try:
            import torch
            use_llm = torch.cuda.is_available()
        except ImportError:
            use_llm = False

    if use_llm:
        print("Using LLM-based response generation...", flush=True)
        responses = generate_llm_responses(rows, track_meta, args.model, args.batch_size)
        if responses is None:
            use_llm = False

    if not use_llm:
        print("Using template-based diverse response generation...", flush=True)
        rng = random.Random(args.seed)
        responses = []
        for r in tqdm(rows, desc="templates"):
            top_track = r["predicted_track_ids"][0] if r["predicted_track_ids"] else ""
            meta = track_meta.get(top_track, {})
            resp = generate_template_response(meta, r["turn_number"], rng=rng)
            responses.append(resp)

    # Patch responses into rows
    for r, resp in zip(rows, responses):
        r["predicted_response"] = resp

    # Write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Wrote {len(rows)} rows to {args.out}")

    # Quick diversity check
    from baselines_v3 import tokenize  # noqa
    bigrams = set()
    total_bigrams = 0
    for r in rows:
        tokens = r["predicted_response"].lower().split()
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i + 1]))
            total_bigrams += 1
    distinct2 = len(bigrams) / total_bigrams if total_bigrams else 0
    print(f"\nDistinct-2 (self-measured): {distinct2:.4f}")


if __name__ == "__main__":
    main()
