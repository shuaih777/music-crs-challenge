"""Encode extractor states as dense vectors for use as the dense query.

Given a state JSONL (output of extractor_inference.py) and a sentence-encoder
model, produce a (session, turn) -> dense vector mapping aligned to one of
the track-embedding modalities.

Why: the current dense query is a pooled mean of *previously accepted track*
embeddings, which:
  - drifts when the conversation pivots (turn 5+ mean of t1-4 tracks doesn't
    reflect what the user wants now)
  - is identically zero at turn 1 (no prior tracks)

A state-text encoding ("rock alternative 1990s energetic accepted_tags=...")
is a clean snapshot of *current* intent and works at every turn.

The output dim must match the track-embedding modality you'll fuse against.
For TalkPlayData-Challenge, two natural pairings are:
  - track field `metadata-qwen3_embedding_0.6b`  ↔ Qwen3-Embedding-0.6B (1024d)
  - track field `attributes-qwen3_embedding_0.6b` ↔ Qwen3-Embedding-0.6B (1024d)
  - track field `lyrics-qwen3_embedding_0.6b`     ↔ Qwen3-Embedding-0.6B (1024d)
The default model below is Qwen/Qwen3-Embedding-0.6B which produces 1024d
vectors out of the box.

Usage:

  # run on GPU server, output a .npz aligned to qwen3 track embeddings
  python src/encode_states.py \\
      --states_jsonl exp/states/test.jsonl \\
      --model Qwen/Qwen3-Embedding-0.6B \\
      --out exp/states/test_qwen3_emb.npz \\
      --batch_size 32

CPU works (slow). The output is also useful as a feature for any reranker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm


# Local imports — same project tree
from baselines_v3 import state_positive_text


def build_state_text(state: dict) -> str:
    """Verbose-but-compact textual rendering of a state, suitable for an
    embedding model. Mirrors the BM25 query weights so both retrievers see
    the same rank-ordered information.
    """
    return state_positive_text(state, user_query="")


# Qwen3-Embedding retrieval instruction templates. The HF model card's
# canonical retrieval format is: "Instruct: <task>\nQuery: <query>". Track-side
# embeddings may have used different (or empty) instructions during their
# pre-computation -- we don't have the dataset card -- so try a few.
INSTRUCTION_TEMPLATES = {
    "none":
        "{query}",
    "qwen3_default":
        "Instruct: Given a music preference description, retrieve relevant tracks.\n"
        "Query: {query}",
    "qwen3_music":
        "Instruct: Given a description of musical taste (genre, mood, era, "
        "energy, accepted tags, artists), retrieve a music track that matches.\n"
        "Query: {query}",
    "qwen3_attributes":
        "Instruct: Given a description of music attributes, retrieve tracks "
        "with similar attributes.\n"
        "Query: {query}",
    "qwen3_metadata":
        "Instruct: Given a music preference profile, retrieve tracks whose "
        "metadata matches the profile.\n"
        "Query: {query}",
}


def apply_instruction(text: str, mode: str, custom: str = "") -> str:
    if mode == "custom":
        return f"{custom}\nQuery: {text}" if custom else text
    return INSTRUCTION_TEMPLATES[mode].format(query=text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--states_jsonl", required=True,
                   help="Output of src/extractor_inference.py")
    p.add_argument("--out", required=True,
                   help="Path ending in .npz (recommended) or .npy. "
                        ".npy will write a parallel .meta.jsonl listing "
                        "(session_id, turn_number) per row.")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="HF Sentence-Transformers compatible encoder. "
                        "Default Qwen3-Embedding-0.6B yields 1024d vectors "
                        "matching the *-qwen3_embedding_0.6b track fields.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512,
                   help="Truncate state text to this many model tokens.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--include_user_query", action="store_true",
                   help="Append the current user_query to each state text. "
                        "Off by default to keep the embedding focused on the "
                        "extracted state attributes.")
    p.add_argument("--instruction", default="qwen3_default",
                   choices=["none", "qwen3_default", "qwen3_music",
                            "qwen3_attributes", "qwen3_metadata", "custom"],
                   help="Instruction prefix for Qwen3-Embedding-style models. "
                        "Qwen3-Embedding requires retrieval task instructions; "
                        "without them, query and document vectors land in "
                        "different subspaces. Try a few prefixes and pick by "
                        "downstream nDCG@20.")
    p.add_argument("--custom_instruction", default="",
                   help="Used when --instruction=custom.")
    p.add_argument("--max_examples", type=int, default=None,
                   help="Smoke-test cap; if set, only encode the first N states.")
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

    # device autodetect
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[encode_states] device={device} model={args.model}", flush=True)

    # Load states
    rows: List[dict] = []
    with open(args.states_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("state"):
                rows.append(r)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print(f"  states with non-empty parse: {len(rows)}", flush=True)

    # Build texts
    print(f"  applying instruction template: {args.instruction}", flush=True)
    texts: List[str] = []
    for r in rows:
        t = build_state_text(r["state"])
        if args.include_user_query and r.get("raw"):
            # raw contains the full extractor output incl. the user_query echo
            # We'd ideally pass user_query separately; settle for state alone
            pass
        if not t:
            t = " "
        t = apply_instruction(t, args.instruction, args.custom_instruction)
        texts.append(t)
    if texts:
        print(f"  example formatted query: {texts[0][:200]}", flush=True)

    # Load encoder
    print("  loading encoder...", flush=True)
    encoder = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    encoder.max_seq_length = args.max_length

    # Encode
    print(f"  encoding {len(texts)} state texts ...", flush=True)
    embs = encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"  embeddings shape={embs.shape}, norm[0]={np.linalg.norm(embs[0]):.4f}", flush=True)

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = np.array(
        [f"{r['session_id']}|{r['turn_number']}".encode("utf-8") for r in rows]
    )
    if args.out.endswith(".npz"):
        np.savez(args.out, keys=keys, embeddings=embs)
        print(f"Wrote {args.out}", flush=True)
    elif args.out.endswith(".npy"):
        np.save(args.out, embs)
        meta_path = args.out[:-4] + ".meta.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "session_id": r["session_id"],
                    "turn_number": int(r["turn_number"]),
                }) + "\n")
        print(f"Wrote {args.out} + {meta_path}", flush=True)
    else:
        raise ValueError(f"--out must end in .npz or .npy: {args.out}")


if __name__ == "__main__":
    main()
