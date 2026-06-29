"""Generate the personalized bi-encoder leg on Blind-B (run on the machine
that has out/biencoder_personalized/). Mirrors the devset personalized
inference exactly (demographic prefix in query), only changing the split
and output path.

Usage (on the machine with the trained personalized model):
    PYTHONPATH=src python infer_personalized_blindb.py
Output: exp/inference/blind_b/biencoder_personalized_top100.json (640 rows)
"""
import json, os, sys
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, "src")

MODEL_DIR = "out/biencoder_personalized"
TRACK_EMB = "exp/tracks/biencoder_personalized_tracks.npy"
TRACK_IDS = "exp/tracks/biencoder_personalized_tracks_ids.json"
OUT = "exp/inference/blind_b/biencoder_personalized_top100.json"

print("Loading model...", flush=True)
model = SentenceTransformer(MODEL_DIR, trust_remote_code=True)
track_embs = np.load(TRACK_EMB)
track_ids = json.load(open(TRACK_IDS))
track_to_idx = {tid: i for i, tid in enumerate(track_ids)}

test = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-B", split="test")


def build_personalized_query(conversations, turn, user_profile):
    age = user_profile.get("age_group", "")
    gender = user_profile.get("gender", "")
    country = user_profile.get("country_name", "")
    prefix = f"[{age} | {gender} | {country}]"
    lines = [prefix]
    for c in conversations:
        if c["turn_number"] > turn:
            break
        if c["turn_number"] < turn:
            if c["role"] == "user":
                lines.append(f'User: {c["content"]}')
            elif c["role"] == "assistant":
                lines.append(f'Assistant: {c["content"][:100]}')
        elif c["turn_number"] == turn and c["role"] == "user":
            lines.append(f'User: {c["content"]}')
    return "\n".join(lines[-12:])


queries, meta_list = [], []
for ex in test:
    user_profile = ex.get("user_profile", {}) or {}
    for tn in range(1, 9):
        q = build_personalized_query(ex["conversations"], tn, user_profile)
        prior = [c["content"] for c in ex["conversations"]
                 if c["role"] == "music" and c["turn_number"] < tn]
        queries.append(q)
        meta_list.append({"session_id": ex["session_id"], "user_id": ex["user_id"],
                          "turn_number": tn, "prior_tracks": prior})

print(f"Encoding {len(queries)} queries...", flush=True)
q_embs = model.encode(queries, batch_size=256, show_progress_bar=True,
                      normalize_embeddings=True, convert_to_numpy=True)

rows = []
for i, m in enumerate(tqdm(meta_list, desc="retrieval")):
    scores = track_embs @ q_embs[i]
    for tid in m["prior_tracks"]:
        if tid in track_to_idx:
            scores[track_to_idx[tid]] = -1e9
    top_idx = np.argsort(-scores)[:100]
    rows.append({"session_id": m["session_id"], "user_id": m["user_id"],
                 "turn_number": m["turn_number"],
                 "predicted_track_ids": [track_ids[j] for j in top_idx],
                 "predicted_response": ""})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(rows, f, ensure_ascii=False)
print(f"Wrote {len(rows)} rows to {OUT}")
