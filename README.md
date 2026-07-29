# Semantic Photo Search

Search 25,000 Unsplash photos by **meaning, not tags**: type *"golden hour by the sea"* and get ranked, relevant photos — powered by CLIP embeddings and vector search.

> 🚧 **Work in progress.** Built session-by-session following [`PLAN.md`](PLAN.md). This README grows into the full portfolio write-up in Session 12; for now it's a signpost.

## Live demo

> **Deploying 27 Aug 2026.** The app is built and verified locally; Hugging Face
> requires an account to be 30+ days old before it will host a free Space, and this
> one was created 28 Jul 2026. The URL below goes live on that date — it is *not*
> live yet, and this note comes down when it is.

**▶ <!-- SPACE_URL -->https://huggingface.co/spaces/greisikertuka/latent-photo-search<!-- /SPACE_URL -->**

A Gradio app on Hugging Face's free tier, searching the full 25k corpus with the
aperture/ISO/focal filters working. *The Space sleeps after ~48 h idle — the first
visit after that takes a minute to wake.*

Its whole payload is **29.3 MB**: float16 embeddings, the id vector, and a slim
display parquet. There's no vector database in the container — the deployed demo
filters with a NumPy boolean mask, because [`FilterSpec`](src/photosearch/models.py)
was designed as a store-agnostic interface back in Session 4. The only model that
runs at query time is the CLIP *text* encoder, on CPU; the photographs themselves
are hotlinked from Unsplash's CDN and never touch the server.

## What this is

A web app that turns a natural-language query into a vector, compares it against precomputed image vectors, and returns the closest photos. The CLIP model is **frozen** — nothing is trained. Indexing runs every image through the encoder once; search runs the query text through the encoder and finds the nearest saved vectors.

The differentiator: **EXIF-aware search** — combine a semantic query with real photographic filters (*"night street, shot at f/1.8 or wider, ISO ≤ 800"*), fusing photography domain knowledge with the AI retrieval.

## Stack

| Layer | Choice |
|---|---|
| Package manager | uv |
| Language | Python 3.12 |
| Embeddings | CLIP `clip-ViT-B-32` via sentence-transformers |
| Vector store | NumPy brute force → Chroma (for filtering) |
| API | FastAPI + uvicorn |
| UI | Static HTML/JS (+ Gradio for the deployed demo) |
| Tests & CI | pytest + ruff + GitHub Actions |

See [`DECISIONS.md`](DECISIONS.md) for the *why* behind each choice.

## Evaluation

*"How do you know it works?"* — 24 hand-labeled queries across four buckets, 679
pooled relevance judgments, measured against a **BM25-over-captions baseline**
(the keyword ranker at the heart of Elasticsearch). One command:

```bash
uv run python eval/run_eval.py
```

### Retrieval quality (K=10, 23 queries)

| system | P@10 | R@10 | MRR |
|---|---|---|---|
| **CLIP** (ViT-B/32) | **0.691** | **0.517** | **0.873** |
| BM25 over captions | 0.283 | 0.241 | 0.623 |

CLIP puts **69%** of the first page on target versus BM25's **28%** — a **+0.41
P@10** gap. MRR 0.87 means the first relevant photo is usually the *first* photo.

### Where the gap comes from (P@10 by bucket)

| bucket | CLIP | BM25 | |
|---|---|---|---|
| easy — one concrete subject | 0.733 | 0.467 | keywords cope when the caption says the thing |
| **compositional** — several clauses | **0.460** | 0.100 | CLIP's weakest bucket; see below |
| **abstract** — mood, no object | **0.767** | 0.267 | *"loneliness"*, *"nostalgia"* — nothing to keyword-match |
| **jargon** — photographic technique | **0.767** | 0.267 | *"long exposure light trails"*, *"leading lines"* |

The headline number hides the real story: on *easy* queries a keyword search is a
respectable half as good, and BM25 even **beats** CLIP on two compositional queries
where the caption literally spells the query out. The gap opens on queries with no
lexical foothold — mood and technique — which is exactly the case semantic search
was supposed to make.

### Latency (median over the eval queries)

| store | vectors | encode ms | search ms | total |
|---|---|---|---|---|
| NumPy (exact brute force) | 24,994 | 34.1 | **5.10** | 39.5 |
| Chroma (HNSW, cosine) | 24,994 | 34.0 | **5.33** | 39.4 |

**Encoding dominates search by ~7×**, and the approximate index is not faster than
brute force at this size — 25k × 512 is 25M multiply-adds, which a laptop CPU eats
for breakfast. Chroma earns its place here for *metadata filtering and incremental
adds*, not for speed; an ANN index at 25k vectors would have been premature
optimisation, and now that's a measured claim rather than a guess.

### Three failure cases worth discussing

1. **Attribute binding** — *"a man in a red jacket on a mountain"* (P@10 = 0.20).
   CLIP returns people on mountains in yellow and brown jackets. The colour
   detaches from the object: it retrieves *red-ish, jacket-ish, mountain-ish* rather
   than the conjunction. BM25 scores **higher** here (0.30) purely because some
   captions say "man in red jacket" verbatim.
2. **Clause dropping** — *"shallow depth of field portrait with creamy bokeh"*
   (P@10 = 0.10, CLIP's worst). It nails the *texture* — blurred flowers, bokeh
   lights, defocused foliage — and forgets the *subject*. Nine of ten results are
   beautifully out-of-focus photographs of no one.
3. **Multi-clause scenes** — *"two people sitting on a bench by the water"*
   (P@10 = 0.00). Benches, water, and pairs of people all come back; all three
   together, never. This is the classic "bag of concepts" limitation of a
   contrastive image-text model with a single global embedding.

A fourth result is about the *corpus*, not the model: **"a black cat sitting on a
windowsill" has no relevant photo anywhere in the pool.** Every system scores 0, so
the query is excluded from the averages (standard TREC practice) and reported
separately — an empty result set is sometimes the honest answer.

Negation (*"a street without any people"*) is the textbook CLIP weakness that did
**not** show up: it scored 0.90, because empty streets are what "street" retrieves
in this corpus anyway. Worth naming — the failure modes you predict aren't always
the ones you measure.

### Honest limitations

- **Pooling makes Recall@10 optimistic.** Candidates were pooled from CLIP, two
  hand-written rephrasings, and BM25; a relevant photo none of them surfaced is
  invisible to the metric. Comparable *between* these systems, not an absolute.
- **One labeller, 24 queries.** No inter-annotator agreement, and no confidence
  intervals tight enough to argue about a two-point difference.
- The relevance rules — and the close calls they settled — are written down in
  [`eval/POLICY.md`](eval/POLICY.md), before labeling started.

<details>
<summary>Rebuilding the gold set from scratch</summary>

```bash
uv run python eval/label.py pool                       # candidates per query
uv run python eval/label.py sheets                     # numbered contact sheets
uv run python eval/label.py record --query e1 --relevant 1,4,7-9
uv run python eval/label.py status                     # coverage
```
</details>

## Search your own photos

The same search core runs over a local folder — real EXIF read straight from the
files, thumbnails generated at index time, originals never leaving the machine:

```bash
uv add pillow-heif
uv run python scripts/05_index_folder.py --path "D:\Photos" --dry-run
uv run python scripts/05_index_folder.py --path "D:\Photos"
```

Re-running is incremental: only new or modified files are embedded, and files you
deleted are removed from the index. The web UI then shows an **Unsplash / My
library** toggle that swaps corpora without a restart.

Measured on a real 5,932-photo archive: the first pass embedded 5,919 files in
57 min (1.7 img/s on CPU), **81% carrying filterable EXIF** across five camera
makes; the follow-up run reported *2 new, 0 modified, 5,917 unchanged* and
embedded only the two. That second line is the whole point of the session.

## Deploying the demo

```bash
uv run python scripts/06_build_space_artifacts.py     # fp16 + slim parquet -> data/space/
uv run --group space python space/app.py              # ALWAYS test locally first
git clone https://huggingface.co/spaces/greisikertuka/latent-photo-search ../latent-space
uv run python scripts/sync_space.py --space ../latent-space --check   # see the payload
uv run python scripts/sync_space.py --space ../latent-space
```

`sync_space.py` flattens `src/photosearch/` next to `app.py` (a Space is its own
git repo and expects that layout), sets up Git LFS for the binaries, and then
stops — **you** review the file list and push. Everything it copies becomes
public, so the review is the point; see the licensing reasoning in
[`DECISIONS.md`](DECISIONS.md#session-11--deployment-hugging-face-space).

## Development

```bash
# One-time: install uv (https://docs.astral.sh/uv/)
uv sync                 # create .venv and install everything from uv.lock
uv run python -c "import photosearch; print('ok')"
uv run pytest           # (tests arrive from Session 2)
```

`uv run <cmd>` is the universal prefix — it auto-creates/syncs the environment first, so there's no venv to activate.

## Status

- [x] **Session 0** — toolchain, project skeleton, CPU-torch pin
- [ ] Session 1 — CLIP hello-world experiment *(needs your own photos)*
- [x] **Sessions 2–3** — dataset prep, EXIF parsers, 24,994 photos indexed
- [x] **Sessions 4–5** — NumPy search core + FilterSpec seam, FastAPI, CI
- [x] **Session 6** — web UI (`v1.0` MVP)
- [x] **Session 7** — Chroma store, EXIF filters fused with vector search
- [x] **Session 8** — filter panel, "more like this", search-by-image
- [x] **Session 9** — your own photo library: incremental indexing + source toggle
- [x] **Session 10** — evaluation harness, BM25 baseline, failure analysis
- [x] **Session 11** — HF Space: fp16 artifacts, NumPy-only filtered search, 29.3 MB payload
- [ ] Session 11b — FastAPI on Render with a quantized ONNX text encoder
- [ ] Session 12 — README, reproducibility, portfolio polish

## License

MIT (see `LICENSE`, added in Session 12). Dataset & photo licensing documented there — the Unsplash Lite data is **not** redistributed in this repo.
