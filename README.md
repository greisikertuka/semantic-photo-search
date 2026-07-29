# Semantic Photo Search

[![CI](https://github.com/greisikertuka/semantic-photo-search/actions/workflows/ci.yml/badge.svg)](https://github.com/greisikertuka/semantic-photo-search/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-black.svg)](.python-version)

Search 25,000 photographs by **meaning, not tags**. Type *"golden hour by the sea"* and
get ranked, relevant frames — then narrow them the way a photographer thinks: *shot at
f/1.8 or wider, ISO ≤ 800, on a 35mm lens.*

No model is trained here. CLIP maps images and text into one shared 512-dimensional
space; the whole search is a dot product against 25,000 precomputed vectors, and it
takes **5 milliseconds**.

<!-- DEMO_GIF: drop docs/demo.gif here (search -> results -> aperture filter changes them, ~15s) -->

---

## Live demo

| | link | note |
|---|---|---|
| **Gradio demo** — 25k corpus, filters, no server state | <!-- SPACE_URL -->*deploying 27 Aug 2026*<!-- /SPACE_URL --> | Hugging Face needs an account 30+ days old before it will host a free Space; this one was created 28 Jul 2026. |
| **The full app** — the real FastAPI backend + this UI | <!-- RENDER_URL -->*deploying*<!-- /RENDER_URL --> | Render's free instance sleeps after 15 min idle and takes ~a minute to wake. The UI says so, then re-runs your query. |

Both are built and verified locally. See [Deployment](#deployment) for what each one
had to fit inside.

---

## How it works

```
                       ┌─────────────────────────────────────────────┐
  OFFLINE (runs once)  │  INDEXING PIPELINE                          │
                       │                                             │
  Unsplash Lite TSV ──▶│  fetch each photo (?w=336, via the CDN)     │
  (25k rows, metadata) │        │                                    │
                       │        ▼                                    │
                       │  CLIP image encoder ──▶ 24,994 × 512 floats │
                       │                         (embeddings + ids)  │
                       │  parse EXIF strings ──▶ photos.parquet      │
                       └─────────────────────────────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────────────┐
  ONLINE (per search)  │  SEARCH CORE  (src/photosearch/)            │
                       │                                             │
  "foggy empty street" │  CLIP text encoder ──▶ 1 × 512 floats       │
        ──────────────▶│  cosine similarity vs all 25k image vectors │
                       │  + EXIF FilterSpec (NumPy mask / Chroma     │
                       │    where-clause — same interface, 2 stores) │
                       │  top-K results, best first                  │
                       └─────────────────────────────────────────────┘
                                │                      │
                                ▼                      ▼
                       FastAPI + HTML/JS UI      Gradio app (NumPy store
                       (local; Render deploy)     + FilterSpec) — HF Space
```

**The five sentences.** CLIP was trained on ~400M image–caption pairs with two separate
encoders — one for pixels, one for text — pulled together so that a matching pair lands
in nearly the same direction and a mismatched pair doesn't. That leaves one vector space
holding both media types, which is the entire trick: a sentence and a photograph become
comparable numbers. Because every vector is normalized to length 1, cosine similarity
collapses into a plain dot product, so scoring the whole corpus is **one matrix–vector
product** — 25,000 × 512 ≈ 25M multiply-adds over a 51 MB array. The model is **frozen**:
indexing runs each image through it once and saves 512 floats; searching runs the query
through it once and finds the nearest saved rows. Nothing is trained, nothing is
fine-tuned, and "the index" is a `.npy` file.

One consequence worth internalizing: **CLIP scores are relative rankings, not
probabilities.** A strong match here lands around 0.28–0.34, not 0.9 — which is why this
UI shows the raw number and refuses to dress it up as a confidence percentage.

---

## Features

| | what it does | where |
|---|---|---|
| **Semantic search** | natural-language query → ranked photos, 39 ms end to end | [`search.py`](src/photosearch/search.py), [`store.py`](src/photosearch/store.py) |
| **EXIF filters** ★ | fuse the query with aperture / ISO / focal range / camera make | [`models.py`](src/photosearch/models.py) — `FilterSpec` |
| **Two stores, one interface** | NumPy brute force or Chroma HNSW, swapped by env var, proven equal by a parity check | `PHOTOSEARCH_STORE=numpy\|chroma` |
| **"More like this"** | nearest neighbours of any photo — costs *zero* model calls | `GET /api/similar/{id}` |
| **Search by image** | upload a photo, get stylistically similar frames | `POST /api/search/by-image` |
| **Your own library** | index a local folder, real EXIF from the files, incremental re-runs | [`library.py`](src/photosearch/library.py) |
| **Evaluation harness** | P@10 / R@10 / MRR vs a BM25 baseline, on hand-labeled judgments | [`eval/`](eval/) |
| **Two deployments** | 29 MB Gradio Space; the real API in 512 MB with no PyTorch | [`space/`](space/), [`render.yaml`](render.yaml) |

★ is the differentiator. Every "CLIP search in 50 lines" tutorial stops at the first row.

---

## Quickstart

Rebuilding the index means downloading 25,000 photos and running them all through CLIP —
an overnight job. So the index ships as a release asset instead:

```bash
git clone https://github.com/greisikertuka/semantic-photo-search
cd semantic-photo-search
uv sync
uv run python scripts/download_artifacts.py
uv run fastapi dev src/photosearch/api.py
```

Open <http://127.0.0.1:8000>. That's ~26 MB of index and about ten minutes, most of it
`uv sync`. The **first query** additionally pulls the CLIP model (~600 MB, cached in
`~/.cache/huggingface`); after that, encoding is ~34 ms and search ~5 ms.

<details>
<summary>Building the index yourself instead (the overnight path)</summary>

```bash
uv run python scripts/01_prepare_dataset.py    # download + parse the Unsplash Lite TSV
uv run python scripts/02_build_index.py --sample 100   # measure throughput, get an ETA
uv run python scripts/02_build_index.py        # the real run: ~25k photos, hours
uv run python scripts/04_ingest_chroma.py      # optional: the Chroma store
uv run python scripts/04_ingest_chroma.py --verify   # parity check vs NumPy
```

The pipeline is chunked and resumable — it checkpoints every 500 rows, so a failure at
photo 24,000 costs one chunk, not the run. Corpus snapshot: **25 Jun 2026**
(the `/latest` download is a moving target; every downstream artifact refers to that date).
</details>

<details>
<summary>Useful environment variables</summary>

| var | values | what it selects |
|---|---|---|
| `PHOTOSEARCH_STORE` | `numpy` (default) · `chroma` · `library` | the back-end that answers |
| `PHOTOSEARCH_ENCODER` | `clip` (default) · `onnx` | sentence-transformers, or the torch-free ONNX text tower |
| `PHOTOSEARCH_DATA_DIR` | path | where the artifacts live (a deploy points this at its unpacked release) |
| `PHOTOSEARCH_ONNX_DIR` | path | the ONNX graph + tokenizer |

The response body carries `store`, `source` and `encoder` back, so which combination
answered is visible in the JSON rather than a matter of faith.
</details>

---

## The API

`uv run fastapi dev src/photosearch/api.py` serves both the JSON API and the UI, with
Swagger at [`/docs`](http://127.0.0.1:8000/docs).

| endpoint | what |
|---|---|
| `GET /api/search` | `q`, `k`, and the FilterSpec params |
| `GET /api/similar/{photo_id}` | "more like this" from the photo's stored vector |
| `POST /api/search/by-image` | multipart upload → encode → same search path |
| `GET /api/health` | corpus size, EXIF-bearing count, active store/source/encoder |
| `GET /api/photo/{id}/thumb` · `/full` | local-library files, resolved by id — never by path |

Filter params on every search route: `aperture_max`, `iso_max`, `focal_min`, `focal_max`,
`camera_make`. One example, which is the whole project in a URL:

```
/api/search?q=night+street&aperture_max=2.0&iso_max=800
```

Every response reports `corpus` and `exif_count` — about **12–15% of the photos carry no
EXIF at all** and can never match a numeric filter, so an active filter searches a smaller
universe than the corpus. The UI says *"searching 21,852 of 24,994 frames with EXIF"*
rather than leaving the shrunken candidate set mysterious.

---

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

---

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

Local photos land in the same `Result` shape as Unsplash ones — nothing downstream
learns that a second corpus exists. File-serving endpoints take an **id, never a
path**, so traversal isn't defended against, it's unrepresentable.

---

## Deployment

### The Gradio Space — 29.3 MB, and no vector database

A Gradio app on Hugging Face's free tier searching the full 25k corpus with the
aperture/ISO/focal filters working. There is **no Chroma in the container**: the
deployed demo filters with a NumPy boolean mask, because `FilterSpec` was designed as
a store-agnostic interface back in Session 4 — a decision made three sessions before
the constraint that would collect on it existed. A smoke test asserts `"chromadb" not
in sys.modules` after importing the app, so the boundary can't rot silently.

The payload is the fp16 embeddings, the id vector, and a slim display parquet.
float16 **on disk only** — it's a storage format, not a compute format, so it's widened
to fp32 at load; NumPy has no fast half-precision matmul and would emulate it in
software, turning a 5 ms search into a few hundred.

```bash
uv run python scripts/06_build_space_artifacts.py     # fp16 + slim parquet -> data/space/
uv run --group space python space/app.py              # ALWAYS test locally first
git clone https://huggingface.co/spaces/greisikertuka/latent-photo-search ../latent-space
uv run python scripts/sync_space.py --space ../latent-space --check   # see the payload
uv run python scripts/sync_space.py --space ../latent-space
```

`sync_space.py` flattens `src/photosearch/` next to `app.py` (a Space is its own git
repo and expects that layout), sets up Git LFS for the binaries, and then **stops** — it
never commits and never pushes. Everything it copies becomes public, including the
licensing decision below, so the human review step is the point.

### The full app on Render — 512 MB, no PyTorch

The hand-built UI above — filters, "more like this", the lot — running the real
FastAPI backend on a **free 512 MB instance**.

The interesting part is what it took to fit. The CLIP text encoder is 254 MB and the
obvious move was to quantize it to the ready-made 64 MB int8 export. Measuring that
export against the full model killed the idea: it agrees at **cosine 0.88** and returns
the same top result **8% of the time**. Profiling then showed the model's size was
never the binding constraint — ONNX stores weights *inside* the graph by default, so
loading it parsed 254 MB and materialized every weight again, peaking at **598 MB**.

Moving the weights to a sidecar file so ONNX Runtime memory-maps them:

| encoder | app peak RSS | encode | vs. the full model |
|---|---|---|---|
| fp32, weights inline *(the naive load)* | **598 MB** — doesn't fit | 26 ms | — |
| ready-made int8, 64 MB *(the obvious pick)* | 366 MB | 5 ms | cosine 0.88, top-1 **8%** |
| ready-made 4-bit, 72 MB | 366 MB | 45 ms | cosine 0.988, **−0.087 P@10** |
| **fp32, weights external** *(shipped)* | **400 MB** | **25 ms** | **identical** |

Fastest, most accurate, and it fits — *without quantizing anything*. Reproduce the
whole table with `scripts/07_build_encoder.py --sweep`.

That third row is the one worth staring at. Cosine 0.988 reads like a rounding error;
running the same 23 labeled queries through it costs **P@10 0.691 → 0.604**. Cosine
between query vectors measures the embedding, not the product. Because Session 10 left
a gold set lying around, checking cost one command:

```bash
uv run python eval/run_eval.py --system deploy    # deployed encoder vs. the real one
```

The shipped encoder scores **0.691 / 0.517 / 0.873 — identical in every bucket.**

Three absences make it fit, and each is load-bearing: no PyTorch (`import torch` alone
costs hundreds of MB before a weight loads), no vector DB (the mask again), and no
vision tower (image vectors were computed offline in Session 3). The last one is
user-visible and handled honestly: `POST /api/search/by-image` returns **501 Not
Implemented**, and `/api/health` advertises `supports_images: false` so the frontend
hides the upload affordance rather than letting someone drag in a photo and get an error.

[`render.yaml`](render.yaml) is a Blueprint — the whole service definition lives in the
repo, so the deploy is reviewable in a diff instead of clicked into a dashboard. Point
Render at the repo (**New → Blueprint Instance**) and it builds itself; the ~210 MB
payload comes from a [GitHub Release][release] at build time.

```bash
uv run --group render python scripts/07_build_encoder.py   # data/encoder/, ~256 MB
python scripts/fetch_deploy_artifacts.py --into .          # or just download them
```

To run the deploy configuration locally — same encoder, same store, same env:

```bash
PHOTOSEARCH_ENCODER=onnx PHOTOSEARCH_DATA_DIR=data/space uv run uvicorn photosearch.api:app --port 8012
```

*The instance sleeps after 15 minutes idle and takes ~a minute to wake; the UI shows a
"waking up" banner only after 1.5 s of real waiting (so it never lies on a warm load),
backs off to 5 s, and re-fires the query you already asked for.*

[release]: https://github.com/greisikertuka/semantic-photo-search/releases/tag/deploy-artifacts-v1

---

## Dataset & licensing

The corpus is the **Unsplash Lite** dataset (25,000 photos, snapshot **25 Jun 2026**).
Its terms permit use but prohibit **redistributing the Licensed Data**, so this repo
takes an explicit position rather than a shrug:

- **`data/` is gitignored.** The TSV and the photographs are never committed.
- **What gets published** is (a) *model outputs* — the embeddings are derived data, not
  the dataset — and (b) the minimum needed to render a result and **credit its
  photographer**, which Unsplash's attribution requirements make mandatory: photo id,
  image URL, photo page URL, photographer name, dimensions, blur hash, numeric EXIF.
- **What does not get published:** the `photo_description` and `ai_description` caption
  columns are dropped from the slim parquet, and no TSV is ever exported. Captions are
  plainly dataset content rather than display necessity, and nothing renders them — so
  dropping them makes *"the corpus is not reconstructible from what we publish"* an
  accurate statement instead of a hopeful one.
- **Photographs are hotlinked** from Unsplash's imgix CDN with the required UTM
  attribution params, which is how Unsplash asks to be used. Images never touch the
  server.

Source code is [MIT](LICENSE). CLIP ViT-B/32 weights are MIT (OpenAI). If you are
Unsplash or a photographer in this set and want something removed, open an issue and
it comes down.

---

## Tech

| Layer | Choice |
|---|---|
| Package manager | uv (`pyproject.toml` ≈ `package.json`, `uv.lock` ≈ lockfile, `uv run` ≈ `npx`) |
| Language | Python 3.12 — pinned to match the HF Space runtime |
| Embeddings | CLIP `clip-ViT-B-32` (512-dim) via sentence-transformers |
| Vector store | NumPy brute force → Chroma, behind one interface |
| API | FastAPI + uvicorn, Pydantic response models, Swagger at `/docs` |
| Deploy encoder | ONNX Runtime + `tokenizers` — CLIP text tower, no PyTorch |
| UI | Hand-written HTML/JS/CSS, no build step (+ Gradio for the Space) |
| Tests & CI | pytest + ruff + GitHub Actions — 181 tests, model-free and artifact-free |

CI is green in seconds because the whole suite runs against a stubbed encoder and
synthetic fixtures — no 600 MB model, no 51 MB index. That was a design choice
(the encoder is an injectable interface), not a happy accident.

**[`DECISIONS.md`](DECISIONS.md) is the other half of this repo** — every non-obvious
choice and why it beat the alternatives, written as the decisions were made. Start
there if you're evaluating the engineering rather than the demo. Full technical
documentation lives in [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md).

---

## Build log

Built session by session against [`PLAN.md`](PLAN.md):

- [x] **0** — toolchain, project skeleton, CPU-torch pin
- [x] **1** — CLIP hello-world: one matrix multiply, all pairwise similarities
  ([`scripts/00_hello_clip.py`](scripts/00_hello_clip.py) — point it at a folder of your own photos)
- [x] **2–3** — dataset prep, EXIF parsers, 24,994 photos indexed
- [x] **4–5** — NumPy search core + the FilterSpec seam, FastAPI, CI
- [x] **6** — web UI, score calibration measured over 16 queries (`v1.0`)
- [x] **7** — Chroma store, EXIF filters fused with vector search, parity check
- [x] **8** — filter panel, "more like this", search-by-image
- [x] **9** — own photo library: incremental indexing + source toggle
- [x] **10** — evaluation harness, BM25 baseline, failure analysis
- [x] **11** — HF Space: fp16 artifacts, NumPy-only filtered search, 29.3 MB
- [x] **11b** — FastAPI on Render: torch-free ONNX encoder, mmapped weights
- [x] **12** — README, reproducibility, licensing, documentation

### Where this goes next

- **Better model, same architecture** — re-index with an open_clip LAION checkpoint and
  let the eval harness *prove* the improvement. The cleanest demonstration of why the
  gold set was worth building.
- **Hybrid ranking** — blend cosine with BM25-over-captions; both halves already exist.
- **Natural-language filters** — parse *"at f/1.8 on my 35mm"* into a `FilterSpec`.
- **Scale story** — load-test, then migrate to Qdrant and write the before/after. The
  switch is worth making around ~1M vectors, multi-instance, or payload-indexed
  filtering; at 25k it would have been cosplay.
