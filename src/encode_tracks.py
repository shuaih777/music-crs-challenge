"""Encode track metadata text into dense vectors.

This is the fallback path for state-dense retrieval when the dataset's
precomputed Qwen3 track embeddings are not aligned with our state embeddings.
Encode both state text and track text with the same model/instruction so cosine
similarity is meaningful.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from encode_states import apply_instruction


def _as_text(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return str(v) if v else ""


def build_track_text(row: dict, mode: str) -> str:
    parts: List[str] = []

    name = _as_text(row.get("track_name"))
    artists = _as_text(row.get("artist_name"))
    album = _as_text(row.get("album_name"))
    if name:
        parts.append(f"title: {name}")
    if artists:
        parts.append(f"artist: {artists}")
    if album and mode in ("metadata", "full"):
        parts.append(f"album: {album}")

    tags = row.get("tag_list") or []
    if isinstance(tags, list) and tags:
        parts.append("tags: " + ", ".join(str(t) for t in tags[:40] if t))

    if mode in ("metadata", "full"):
        rd = row.get("release_date") or ""
        if rd:
            parts.append(f"release_date: {rd}")

    if mode == "full":
        for field in ("genre", "mood", "era", "energy"):
            v = _as_text(row.get(field))
            if v:
                parts.append(f"{field}: {v}")

    return "\n".join(parts) or name or artists or " "


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,
                   help="Output .npz with keys=track_id bytes and embeddings.")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--instruction", default="qwen3_music",
                   choices=["none", "qwen3_default", "qwen3_music",
                            "qwen3_attributes", "qwen3_metadata", "custom"])
    p.add_argument("--custom_instruction", default="")
    p.add_argument("--text_mode", default="metadata",
                   choices=["metadata", "tags", "full"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--max_examples", type=int, default=None)
    args = p.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}\n"
              "Install GPU deps: pip install -r requirements-gpu.txt\n"
              "(this script needs torch + sentence-transformers)",
              file=sys.stderr)
        sys.exit(1)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[encode_tracks] device={device} model={args.model}", flush=True)

    print("  loading track metadata...", flush=True)
    tracks = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    rows = list(tracks)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print(f"  tracks: {len(rows)}", flush=True)

    print(f"  text_mode={args.text_mode} instruction={args.instruction}", flush=True)
    texts = [
        apply_instruction(build_track_text(r, args.text_mode),
                          args.instruction,
                          args.custom_instruction)
        for r in rows
    ]
    if texts:
        print(f"  example formatted document: {texts[0][:240]}", flush=True)

    print("  loading encoder...", flush=True)
    encoder = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    encoder.max_seq_length = args.max_length

    print(f"  encoding {len(texts)} tracks ...", flush=True)
    embs = encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"  embeddings shape={embs.shape}, norm[0]={np.linalg.norm(embs[0]):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = np.array([str(r["track_id"]).encode("utf-8") for r in rows])
    np.savez(args.out, keys=keys, embeddings=embs)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
