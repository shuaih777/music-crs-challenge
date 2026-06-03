"""Build training data for the conversation-state extractor.

Strategy: for every (session, turn) in the *train* split, the gold target is
the metadata of the track that the system actually recommended at that turn.
We extract attributes that downstream retrieval can use:

    {
      "genre":       comma-separated tag-list values that look like genres,
      "mood":        same, but mood-flavored,
      "era":         "1990s" / "2000s" / "2010s" derived from release_date,
      "energy":      "low" | "medium" | "high" inferred from mood/duration,
      "accepted_tags": tags of all music tracks recommended in turns < t,
      "rejected_tags": tags of tracks where the user pushed back next turn,
      "artist_hints": artist names previously accepted (helps continuity),
    }

The model is trained autoregressively to emit this JSON given:
    <history>   role:content for all messages with turn < t (music rows
                replaced with "system: recommended <name> by <artist>"),
    <user>      the current turn's user utterance.

Output: a JSONL file in HF "messages" format ready for SFTTrainer or
Trainer. Each line:
    {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user",   "content": "<history>...<user>..."},
                  {"role": "assistant", "content": "<extracted_state JSON>"}]}

Run on CPU; takes ~1-2 minutes for 15,199 sessions x 8 turns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
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


# Heuristic taxonomies (small; we let the model extend naturally during training)
GENRE_KEYWORDS = {
    "rock", "pop", "metal", "electronic", "edm", "hip-hop", "rap", "jazz",
    "blues", "classical", "country", "folk", "indie", "alternative",
    "punk", "r&b", "soul", "disco", "funk", "reggae", "house", "techno",
    "ambient", "instrumental", "lo-fi", "trap", "k-pop", "j-pop",
    "latin", "afrobeat", "grunge", "dubstep", "synthwave", "post-punk",
}
MOOD_KEYWORDS = {
    "happy", "sad", "energetic", "chill", "relaxing", "melancholic",
    "atmospheric", "uplifting", "dark", "moody", "introspective",
    "romantic", "aggressive", "calm", "peaceful", "groovy", "smooth",
    "dreamy", "anthemic", "haunting", "epic", "playful", "nostalgic",
}
HIGH_ENERGY_TAGS = {"energetic", "aggressive", "anthemic", "epic", "groovy"}
LOW_ENERGY_TAGS = {"calm", "chill", "relaxing", "peaceful", "atmospheric",
                   "ambient", "smooth", "dreamy", "introspective"}


def classify_tags(tags: List[str]) -> Dict[str, list]:
    g, m = [], []
    for t in tags:
        tl = t.lower()
        for kw in GENRE_KEYWORDS:
            if kw in tl:
                g.append(t)
                break
        else:
            for kw in MOOD_KEYWORDS:
                if kw in tl:
                    m.append(t)
                    break
    return {"genre": g[:5], "mood": m[:5]}


def era_from_date(release_date: str) -> str:
    if not release_date or len(release_date) < 4:
        return ""
    try:
        year = int(release_date[:4])
    except ValueError:
        return ""
    decade = (year // 10) * 10
    return f"{decade}s"


def energy_from_tags(tags: List[str]) -> str:
    high = sum(1 for t in tags if any(kw in t.lower() for kw in HIGH_ENERGY_TAGS))
    low = sum(1 for t in tags if any(kw in t.lower() for kw in LOW_ENERGY_TAGS))
    if high > low:
        return "high"
    if low > high:
        return "low"
    return "medium"


# Sentiment / pushback detection in a user message — simple keyword model.
PUSHBACK_RE = re.compile(
    r"\b(no|not really|don'?t|isn'?t|wasn'?t|something (?:else|different)|"
    r"too\s+\w+|skip|next|other|instead|different)\b",
    re.IGNORECASE,
)


def detected_pushback(text: str) -> bool:
    return bool(PUSHBACK_RE.search(text or ""))


def make_target(prior_track_metas: List[dict],
                rejected_track_metas: List[dict],
                gold_track_meta: dict) -> Dict:
    """Build the JSON target the model should emit at this turn."""
    gold_tags = list(gold_track_meta.get("tag_list") or [])
    gold_split = classify_tags(gold_tags)

    accepted_tags: List[str] = []
    artist_hints: List[str] = []
    for m in prior_track_metas[-5:]:
        accepted_tags.extend((m.get("tag_list") or [])[:5])
        an = m.get("artist_name")
        if isinstance(an, list):
            artist_hints.extend(an)
        elif an:
            artist_hints.append(an)
    rejected_tags: List[str] = []
    for m in rejected_track_metas:
        rejected_tags.extend((m.get("tag_list") or [])[:5])

    # dedupe
    def _u(seq: list) -> list:
        seen = set(); out = []
        for x in seq:
            k = (x or "").strip().lower()
            if not k or k in seen:
                continue
            seen.add(k); out.append(x)
        return out

    return {
        "genre": gold_split["genre"],
        "mood": gold_split["mood"],
        "era": era_from_date(gold_track_meta.get("release_date") or ""),
        "energy": energy_from_tags(gold_tags),
        "accepted_tags": _u(accepted_tags)[:8],
        "rejected_tags": _u(rejected_tags)[:8],
        "artist_hints": _u(artist_hints)[:5],
    }


def build_history_string(conversations: List[dict], turn: int,
                         tracks_by_id: Dict[str, dict]) -> tuple[str, str]:
    """Return (history_block, user_query) for the given turn cutoff.

    `history_block` contains all messages with turn_number < `turn`.
    """
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
    return "\n".join(lines), user_query


def build_dataset(out_path: str, max_sessions: int | None = None,
                  split: str = "train") -> None:
    print(f"Loading conversations split={split}...", flush=True)
    convo = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split=split)
    if max_sessions is not None:
        convo = convo.select(range(min(max_sessions, len(convo))))
    print(f"  {len(convo)} sessions", flush=True)

    print("Loading track metadata...", flush=True)
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    tracks_by_id = {t["track_id"]: t for t in tracks}
    print(f"  {len(tracks_by_id)} tracks", flush=True)

    n_written = 0
    n_pushback = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(convo, desc="building"):
            df = pd.DataFrame(ex["conversations"])
            # Pre-extract per-turn music track + per-turn user message
            music_by_turn = {int(r["turn_number"]): r["content"]
                             for _, r in df[df["role"] == "music"].iterrows()}
            user_by_turn = {int(r["turn_number"]): r["content"]
                            for _, r in df[df["role"] == "user"].iterrows()}

            for tn in range(1, 9):
                gold_id = music_by_turn.get(tn)
                if gold_id is None:
                    continue
                gold_meta = tracks_by_id.get(gold_id)
                if not gold_meta:
                    continue

                # Build prior + rejected track metas
                prior_metas = []
                rejected_metas = []
                for prev in range(1, tn):
                    pid = music_by_turn.get(prev)
                    if not pid:
                        continue
                    pmeta = tracks_by_id.get(pid)
                    if not pmeta:
                        continue
                    next_user = user_by_turn.get(prev + 1, "")
                    if detected_pushback(next_user):
                        rejected_metas.append(pmeta)
                        n_pushback += 1
                    else:
                        prior_metas.append(pmeta)

                target = make_target(prior_metas, rejected_metas, gold_meta)
                history, user_query = build_history_string(ex["conversations"], tn, tracks_by_id)

                user_block = f"<history>\n{history}\n</history>\n<user>\n{user_query}\n</user>"
                assistant_block = json.dumps(target, ensure_ascii=False)

                rec = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_block},
                        {"role": "assistant", "content": assistant_block},
                    ],
                    "session_id": ex["session_id"],
                    "turn_number": tn,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\nWrote {n_written} training examples to {out_path}")
    print(f"  Detected pushback in {n_pushback} prior turns")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/state_extractor_train.jsonl")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--max_sessions", type=int, default=None)
    args = p.parse_args()
    build_dataset(args.out, args.max_sessions, args.split)


if __name__ == "__main__":
    main()
