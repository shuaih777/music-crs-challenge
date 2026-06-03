# Research synthesis — how to push nDCG@20 above 0.124

> Output of a multi-agent literature/repo survey (run 2026-06-03).
> 11 agents, 75 method findings, 52 repo findings, raw blob in `research/raw_workflow_result.json`.

This is a strategy doc — actual implementation will go into `src/` as we work through it.

---

## TL;DR — what the data points to

1. The dominant problem is **recall, not ranking.** Hit@20 = 26.2% means 73.8% of (session, turn) pairs miss the gold track entirely; **no reranker can save us** until we widen the candidate net.
2. Five of the six pre-computed embedding modalities (`attributes-qwen3`, `lyrics-qwen3`, `image-siglip2`, `audio-laion_clap`, plus user-side embeddings) are not in the current best ensemble. Their misses are largely disjoint, so their union should push Hit@100 to 40-50%+.
3. **Turn 1 nDCG = 0.134 across every retriever.** That's a smoking gun for a tokenization bug (Beyoncé / AC/DC / "The Beatles" all break on `[A-Za-z0-9]+` + stopword strip). A 1-hour fix could deliver more turn-1 lift than weeks of neural reranker work.
4. The 3-way RRF (`meta+cf+CLAP`) actually *lost* nDCG vs 2-way — that's a tuning failure (equal weight + default k=60, no sweep), not a signal failure.
5. Generative retrieval (TIGER / OneRec / HSTU / Text2Tracks) looks great on paper but **needs a new tokenizer + decoder trained from scratch**. Under a 27-day deadline (2026-06-30) classic IR plumbing dominates it on ROI.

---

## Top 8 ranked recommendations

### #1 — Wide-net 6-modality RRF with per-leg / per-turn weight sweep ⭐ blocker fix

> **Effort: days · GPU: small · Lift: +0.020-0.040 nDCG@20**

Wire all 5 unused track embeddings (`attributes-qwen3`, `lyrics-qwen3`, `image-siglip2`, `audio-laion_clap`, `user-emb × track`) into RRF, push fuse depth from 20 → 100/200, and grid-search per-leg + per-turn weights on devset.

**Why for us:** Hit@20 = 26.2% is the hard ceiling. Each modality solo has 22-25% Hit@20 on largely disjoint misses, so their union should push Hit@100 well past 40-50%. The current 3-way ensemble lost nDCG only because CLAP got equal RRF weight — that's a **tuning** failure, not a signal failure.

**How to start:**
1. Add 5 new retriever legs in `src/baselines_v3.py` mirroring the existing dense path but swapping the modality field.
2. In `src/ensemble.py` raise `topk_fuse=200`, emit top-100 per leg into RRF.
3. Add `src/sweep_rrf.py` looping weights `{0, 0.25, 0.5, 1, 2}` per leg, per-turn buckets `{1, 2-4, 5-6, 7-8}`. Optimize Hit@20 on devset.

**Sources:** [arXiv 2210.11934](https://arxiv.org/abs/2210.11934), [arXiv 2502.13713](https://arxiv.org/abs/2502.13713)

---

### #2 — Current-utterance-only leg + pushback regex (cheap state extractor)

> **Effort: hours · GPU: small · Lift: +0.010-0.020, concentrated on t5-8**

Add a parallel retrieval leg encoding **only the turn-t user utterance** (with `attributes-qwen3` and `lyrics-qwen3`); plus a regex pushback detector that drops/negates the last system-recommended track from the prior pool.

**Why for us:** Direct fix for the turn 5-8 collapse. Every retriever degrades monotonically because full-history concatenation drowns the current intent. `attributes-qwen3` is the closest text-side proxy to the genre/mood/era language users actually use at turn 5+. Costs zero GPU training and lands before the LoRA Phase B.

**How to start:**
- `build_query_current_only(turn)` takes only the latest `user_query` and encodes with the Qwen3-Embedding model already in the pipeline.
- Regex set: `('not that', 'different', 'no', 'something else', 'too \\w+', 'less \\w+')`. If matched at turn t, subtract `0.5 × emb(last_recommended_track)` from the dense query and remove from `prior_idx`.

**Sources:** [arXiv 2308.06212](https://arxiv.org/abs/2308.06212), [arXiv 2510.04812](https://arxiv.org/abs/2510.04812)

---

### #3 — LightGBM reranker over union(top-100) with cross-modal + user features

> **Effort: days · GPU: none · Lift: +0.015-0.030 (after #1 lands)**

Train a small LambdaRank LightGBM on the 15,199 train sessions (~121k turns). Features per (track, turn):
- rank-in-each-leg, score-in-each-leg (5-6 legs from #1)
- `cosine(user_emb, track_emb_*)` per modality
- `turn_number`, `prior_track_count`
- tag-overlap with regex-extracted query tags
- `log_popularity`, era difference

Label = 1 for gold track, 0 for sampled negatives from union(top-100).

**Why for us:** Once #1 lifts Hit@100 to ~50-60%, the conditional-on-hit ceiling rises and a learned blender that *knows which leg to trust per turn* captures most of that headroom. Replaces the hand-coded "1-6:meta, 7-8:cf" table with a learned function. 30 min CPU training, ~50ms inference per turn.

**Sources:** [recsys-challenge-2024 1st place repo](https://github.com/k-fujikawa/recsys-challenge-2024-1st-place), [arXiv 2502.07673](https://arxiv.org/abs/2502.07673)

---

### #4 — Turn-1 literal-match leg with unicode-aware normalization 🎯 free win

> **Effort: hours · GPU: none · Lift: +0.005-0.010 overall (turn-1: 0.134 → 0.20+)**

Add normalized exact-match-then-BM25 leg on `track_name + artist_name` with `unidecode` + parenthetical-suffix stripping. Fires when current utterance matches the "play X by Y" pattern.

**Why for us:** Turn-1 nDCG = 0.134 is *identical across BM25, metadata-qwen3, cf-bpr* — that's not a retrieval-algorithm issue. Current tokenizer (`[A-Za-z0-9]+` + stopwords) **guarantees failure** on:
- `Beyoncé`, `Sigur Rós`, `AC/DC`, `$uicideboy$`
- `The Beatles` (drops "The")
- `Heart-Shaped Box (Remastered 2011)` (parenthetical suffix)

**How to start:**
```python
def normalize(s):
    s = unidecode(s).lower()
    s = re.sub(r'\([^)]*(?:remaster|version|edit|live)[^)]*\)', '', s)
    return s

# Build (norm_artist, norm_track) -> idx dict
# Match user query against r'(?:play|put on|listen to) (.+?) (?:by|from) (.+)'
```

**Sources:** [arXiv 2510.01698](https://arxiv.org/abs/2510.01698)

---

### #5 — Qwen3-Reranker-0.6B cross-encoder on top-100 union

> **Effort: days · GPU: small · Lift: +0.010-0.025 (weighted to t5-8)**

Stage-2 cross-encoder rerank conditioned on running dialogue state (`"user has accepted X, rejected Y, current request: Z"`). Qwen3-Reranker-0.6B in fp16 on A100.

**Why for us:** Listwise/cross-encoder rerank is the documented highest-ROI second stage when recall is healthy. Qwen3-Reranker is purpose-built for instruction-aware scoring — the prompt "rerank these tracks for a user who already accepted A,B,C and just said *more upbeat*" is its sweet spot.

**How to start:**
```bash
pip install sentence-transformers>=5.0
# Download Qwen/Qwen3-Reranker-0.6B
```
Add `src/rerank_qwen3.py` with `rerank(query_str, candidate_track_texts, top_k=20)`. Build query string from running state (#2's output) + current `user_query`. Run after #1 ensemble; either replace #3 LightGBM or feed its score as an extra LightGBM feature.

**Sources:** [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B), [castorini/rank_llm](https://github.com/castorini/rank_llm), [arXiv 2312.02724](https://arxiv.org/abs/2312.02724)

---

### #6 — Session-level item2item co-occurrence PMI retriever

> **Effort: hours · GPU: none · Lift: +0.005-0.015, concentrated on t4-8**

Build track-track co-occurrence PMI matrix from the 15,199 train sessions (each session ≈ 8 accepted tracks by the same user). Retrieve via mean-pool of accepted-track PMI rows.

**Why for us:** BPR-CF is global pairwise; **session co-occurrence is a much stronger session-level CF**, especially at turn 5+ when 4-7 prior accepted tracks form a strong "playlist fingerprint". Explains why `cf_bpr` only catches up at turn 7 — there's untapped session-CF structure orthogonal to BPR. Pure CPU, <10 min build.

**How to start:**
```python
# src/build_item2item.py
# For each train session, increment scipy.sparse.coo_matrix on every accepted-track pair
# Compute PPMI; save .npz
# Retrieval leg: given prior_idx, return top-N tracks by sum of PPMI rows
```

**Sources:** [antklen/recsys_challenge_2025](https://github.com/antklen/recsys_challenge_2025)

---

### #7 — MuQ-MuLan replacement for LAION-CLAP audio modality

> **Effort: days · GPU: big · Lift: +0.005-0.015**

Re-encode catalog with **Tencent MuQ-MuLan** (700M, music-specific contrastive joint music/text). LAION-CLAP repo is **dormant since April 2023** — community successor for music is MuQ-MuLan.

**Why for us:** Audio in the current 3-way RRF actively *hurt* because LAION-CLAP-512d is weaker than the text modalities. MuQ-MuLan is purpose-built for music tag/mood retrieval (the exact "more upbeat / darker" refinement language at turn 5+) and shares a text/audio space, so query-side text encoding still works.

**How to start:**
```bash
pip install muq
# src/encode_muq_mulan.py: iterate track audio previews -> audio-muq_mulan.npy (768d)
```
Add as new leg in #1 RRF. A/B against keeping LAION-CLAP. Use primarily on turns ≥3 where mood/tag refinements appear.

**Sources:** [arXiv 2501.01108](https://arxiv.org/abs/2501.01108), [tencent-ailab/MuQ](https://github.com/tencent-ailab/MuQ)

---

### #8 — BGE-M3 unified dense+sparse+ColBERT leg

> **Effort: days · GPU: small · Lift: +0.005-0.015**

Replace the BM25-only sparse leg with **BGE-M3**, which produces dense + sparse + ColBERT-style multi-vec from one forward pass over the same conversational text.

**Why for us:** Music-CRS turns mix entity tokens (`AC/DC`, `Drake`) with mood adjectives (`chill`, `upbeat`) — the regime where naive BM25+dense fusion is most unstable. BGE-M3's three signals share a tokenizer/training distribution, so hybrid fusion behaves better than independently-trained BM25+dense.

**How to start:**
```bash
pip install FlagEmbedding
```
Add `BGEM3FlagModel` index over `track_card_text`. Replace BM25 with BGE-M3 sparse, add BGE-M3 dense as another leg, optionally ColBERT multi-vec only on top-200 from dense as a cheap reranker.

**Sources:** [arXiv 2402.03216](https://arxiv.org/abs/2402.03216), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)

---

## Five surprises from the survey

1. **LAION-CLAP is effectively abandoned** (no checkpoints since April 2023). Community successor for music is **MuQ-MuLan** — your unused audio modality is on a dead-end stack.
2. **Turn-1 nDCG = 0.134 being identical across all retrievers is a smoking gun for a tokenizer bug**, not a retrieval-algorithm issue. 1-hour unicode normalization could pay more than weeks of neural reranker work.
3. **The 3-way RRF lost nDCG (0.1241 → 0.1237) versus 2-way** — widely interpreted as "audio doesn't help" but the real cause is RRF with equal weight + k=60 default + no sweep. Multi-day weight sweep is competitive with multi-week neural reranker work.
4. **User embeddings are loaded but completely unused** — passed through to JSON only. The single most stable preference signal in the pipeline is being thrown away. Adding it as a retrieval leg attacks the t1-t2 cold-conversation regime AND t7-t8 drift regime simultaneously.
5. **TIGER / OneRec / HSTU look attractive on paper** but ALL require training a new tokenizer + decoder under a 27-day deadline. Surprise: how much classic IR plumbing (RRF tuning, literal-match normalization, item2item PMI, LightGBM reranker, cross-encoder) is still on the table and probably dominates them on ROI.

---

## Suggested execution order (matched to deadline 2026-06-30)

| Day | Item | Cumulative est. nDCG@20 |
|---|---|---:|
| 1 (today) | #4 turn-1 literal-match (free win) | 0.130 |
| 1-2 | #2 current-utterance leg + pushback regex (overnight on CPU) | 0.140 |
| 1-2 | #6 item2item PMI (afternoon on CPU) | 0.145 |
| 3-5 | **#1 wide-net 6-modality RRF + weight sweep** ← biggest single lever | 0.165 |
| 6-8 | #3 LightGBM reranker over union(top-100) | 0.185 |
| 9-12 | #5 Qwen3-Reranker-0.6B cross-encoder | 0.200+ |
| 13-15 | #7 MuQ-MuLan re-encode (only if GPU budget allows) | 0.205 |
| 16+ | #8 BGE-M3 swap, plus our existing Phase B state extractor (Qwen3-0.6B LoRA) | 0.210-0.220 |

Time-permitting stretch: keep the **TIGER/OneRec generative-retrieval** experiment (Phase D from `GPU_EXPERIMENTS.md`) on the back-burner — research-novel but a 1-2 week project on its own.

---

## Where each rec attacks the bottleneck table

| Bottleneck (severity) | Recs that attack it |
|---|---|
| **Recall ceiling (Hit@20=26%, blocker)** | #1, #6, #7, #8 |
| **Per-turn collapse (major)** | #2, #3, #5 |
| **Modalities not fused (major)** | #1, #6 (PMI is a 7th modality) |
| **Turn-1 plateau (major)** | #4 |
| **User embeddings unused (major)** | #1 (user × track leg), #3 (user-emb features in LightGBM) |
| **No reranker (minor)** | #3, #5 |
| **No session state (minor)** | #2 (running state), #3 (state features), Phase B LoRA |

---

## Files in `research/`
- `raw_workflow_result.json` — full unedited output from the workflow (75 method findings + 52 repo findings + diagnosis + 8 ranked recommendations)
- `synthesis.md` — this document
