"""Diagnose extractor_inference.py output quality.

Categorizes each line into:
  ok        - state is a valid JSON object with required keys
  empty     - state is {} (parser gave up)
  truncated - raw text doesn't end with '}' (likely max_new_tokens hit)
  bad_json  - raw has '{' but parser couldn't find a balanced block
  no_brace  - raw has no '{' at all
  short_raw - raw text < 20 chars (model produced almost nothing)

Usage:
    python src/diagnose_states.py --states exp/states/test.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter, defaultdict


REQUIRED_KEYS = {"genre", "mood", "era", "energy", "accepted_tags",
                 "rejected_tags", "artist_hints"}


def categorize(rec: dict) -> tuple[str, dict]:
    state = rec.get("state", {}) or {}
    raw = (rec.get("raw") or "").strip()

    if state and isinstance(state, dict):
        missing = REQUIRED_KEYS - set(state.keys())
        if not missing:
            return "ok", {"raw_len": len(raw)}
        return "ok_partial", {"raw_len": len(raw), "missing": list(missing)}

    if not raw or len(raw) < 20:
        return "short_raw", {"raw_len": len(raw), "raw_sample": raw[:60]}

    if "{" not in raw:
        return "no_brace", {"raw_len": len(raw), "raw_sample": raw[:120]}

    if not raw.rstrip().endswith("}"):
        return "truncated", {"raw_len": len(raw), "raw_tail": raw[-200:]}

    return "bad_json", {"raw_len": len(raw), "raw_sample": raw[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--show", type=int, default=2,
                    help="Sample lines to print per failure category")
    args = ap.parse_args()

    cats = Counter()
    samples: dict[str, list] = defaultdict(list)
    raw_lens: list[int] = []

    with open(args.states) as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except Exception:
                cats["malformed_jsonl"] += 1
                continue
            cat, info = categorize(rec)
            cats[cat] += 1
            raw_lens.append(info.get("raw_len", 0))
            if len(samples[cat]) < args.show:
                samples[cat].append({
                    "session_id": rec.get("session_id"),
                    "turn_number": rec.get("turn_number"),
                    **info,
                })

    total = sum(cats.values())
    print(f"Total lines: {total}")
    print()
    print(f"{'category':<20} {'count':>6} {'pct':>6}")
    for c, n in cats.most_common():
        pct = 100.0 * n / total
        print(f"{c:<20} {n:>6} {pct:>5.1f}%")

    if raw_lens:
        rl = sorted(raw_lens)
        n = len(rl)
        print(f"\nraw text length percentiles:")
        for p in (10, 25, 50, 75, 90, 99):
            idx = int(n * p / 100)
            print(f"  p{p:>2}: {rl[min(idx, n-1)]}")

    print("\nSamples per category:")
    for cat in ["truncated", "bad_json", "short_raw", "no_brace",
                "empty", "ok_partial", "malformed_jsonl"]:
        if cat not in samples:
            continue
        print(f"\n--- {cat} ({len(samples[cat])} sample(s)) ---")
        for s in samples[cat]:
            print(f"  {s.get('session_id')}/{s.get('turn_number')}  "
                  f"raw_len={s.get('raw_len')}")
            if "raw_tail" in s:
                print(f"    last 200 chars: {s['raw_tail']!r}")
            if "raw_sample" in s:
                print(f"    first chars:    {s['raw_sample']!r}")


if __name__ == "__main__":
    main()
