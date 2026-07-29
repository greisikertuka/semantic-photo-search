# Semantic Photo Search — Technical Documentation

**Project:** `semantic-photo-search`
**Author:** Greisi Kertuka
**Version:** 1.0 (Sessions 0–12 complete)
**Document date:** 29 July 2026
**Repository:** <https://github.com/greisikertuka/semantic-photo-search>
**Corpus snapshot:** Unsplash Lite, 25 June 2026 — 24,994 indexed photos

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Core concepts](#2-core-concepts)
3. [Architecture](#3-architecture)
4. [Repository structure](#4-repository-structure)
5. [Module reference](#5-module-reference)
6. [Data artifacts](#6-data-artifacts)
7. [HTTP API reference](#7-http-api-reference)
8. [The web UI](#8-the-web-ui)
9. [Scripts reference](#9-scripts-reference)
10. [Configuration](#10-configuration)
11. [How to run it](#11-how-to-run-it)
12. [Testing](#12-testing)
13. [Evaluation](#13-evaluation)
14. [Deployment](#14-deployment)
15. [Performance characteristics](#15-performance-characteristics)
16. [Troubleshooting](#16-troubleshooting)
17. [Licensing](#17-licensing)
18. [Glossary](#18-glossary)

---

## 1. What this project is

A web application that searches 25,000 photographs **by meaning rather than by
keyword**, and then narrows the results using real photographic metadata.

Typing *"golden hour by the sea"* returns golden-hour seascapes even when no caption
contains those words. Adding *aperture ≤ f/2.0* and *ISO ≤ 800* restricts that same
semantic result set to frames actually shot wide open at low ISO. That combination —
semantic retrieval fused with EXIF filtering — is the project's differentiator; most
CLIP-search demos stop at the first half.

### What it is not

- **Nothing is trained.** CLIP is used frozen, inference-only. There is no fine-tuning,
  no training loop, no labels used as supervision.
- **It is not a large-scale system.** 25k vectors is small on purpose, and the
  documentation is explicit about where the design would have to change (see
  §15 and the Qdrant note in §18).
- **It does not host photographs.** Unsplash images are hotlinked from their CDN.
  Local-library photos never leave the user's machine.

### Design goals, in priority order

1. **Understandability** — brute-force search is written by hand before any vector DB
   is adopted, so the mechanism is legible rather than delegated.
2. **Honest measurement** — every performance and quality claim in this document
   corresponds to a command that reproduces it.
3. **Seams over features** — the store, the encoder, and the corpus are each swappable
   behind a small interface. Both deployments exist only because those seams do.

---

## 2. Core concepts

### 2.1 Embedding

An embedding is a learned list of numbers — here, **512 floats** — representing a piece
of content. Think of it as coordinates in a 512-dimensional space in which *position
encodes meaning*: content with similar meaning lands nearby. Individual numbers are
meaningless; only the relationships between vectors matter.

### 2.2 Cosine similarity, and why it becomes a dot product

Similarity between two vectors is the cosine of the angle between them: `1.0` = same
direction, `0` = unrelated, negative = opposed.

If both vectors are **normalized to length 1**, the cosine reduces to the plain dot
product — one multiply-add per dimension, no division, no square roots. Every vector in
this system is stored normalized (`normalize_embeddings=True` at index time), so
scoring the entire corpus is a single matrix–vector product:

```
scores = embeddings @ query_vec        # (24994, 512) @ (512,) -> (24994,)
```

That is ~25 million multiply-adds over a 51 MB array: **about 5 ms** on a laptop CPU.

### 2.3 CLIP

OpenAI's 2021 model, trained on roughly 400 million (image, caption) pairs with a
*contrastive* objective. Two separate encoders — one for images, one for text — are
trained jointly so that matching pairs get high cosine similarity and mismatched pairs
get low similarity.

The result is **one shared vector space for two different media types**. That is the
entire foundation of this application: a text query can be compared directly against
image vectors, because both are points in the same space.

The checkpoint used is **ViT-B/32** (512-dimensional output, ~600 MB). ViT-L/14 scores
better but is ~10–20× slower and 1.7 GB — the wrong trade for CPU inference on free
hosting.

### 2.4 Zero-shot inference

CLIP is applied to this corpus with no training, no fine-tuning and no labels. Only
forward passes ever run. "Indexing" means: run each image through the frozen encoder
once and save the 512 numbers.

### 2.5 Score calibration — an important, measured caveat

CLIP similarity scores are **relative rankings, not probabilities**. Measured on this
corpus over 16 queries (10 sensible, 6 deliberately absurd):

| query type | top-1 score range |
|---|---|
| sensible (*"a bowl of ramen"*) | 0.264 – 0.340 |
| absurd (*"purple elephant playing chess"*) | 0.246 – 0.292 |

**The ranges overlap.** *"purple elephant playing chess"* scores 0.292 — higher than the
legitimate *"a bowl of ramen"* at 0.264 — because CLIP genuinely finds purple imagery.
There is therefore **no threshold that cleanly separates sense from nonsense**, and any
feature claiming otherwise would be lying.

The UI's honest design follows from that measurement: always display the raw score,
color-coded by band (≥ 0.28 strong, ≥ 0.24 decent, else weak), and show a *"no strong
matches — showing the closest anyway"* banner only when the **best** hit falls below
**0.26** (which catches pure gibberish such as *"xqzptn wobble"* at 0.246 while leaving
real-but-weak queries alone).

### 2.6 Pre-filtering vs post-filtering

**Post-filtering** — rank the top-K, then discard non-matching results — breaks under
selective filters: request the top 50, filter to f/1.8, and you may be left with 2.

**Pre-filtering** — restrict the candidate set first, then rank the survivors — honours
K. Both stores pre-filter: `NumpyStore` compiles a boolean mask, `ChromaStore` compiles
a `where=` clause evaluated before ranking.

A useful consequence falls out for free: photos with no EXIF have `NaN` in the filter
columns, and **every comparison against NaN is False**, so "no EXIF ⇒ excluded when a
filter is active" needs no special-casing. It is also why the API reports `exif_count`
and the UI says *"searching 21,852 of 24,994 frames with EXIF"*.

### 2.7 Approximate nearest neighbour (ANN)

At millions of vectors, brute force becomes slow, and structures such as **HNSW** (a
layered navigable graph walked greedily) find *approximately* nearest neighbours in
roughly logarithmic time, trading a little recall for a lot of speed. Chroma, Qdrant and
FAISS all do this internally.

At 25k vectors it is **not** faster — measured at 5.33 ms (Chroma/HNSW) versus 5.10 ms
(NumPy exact). Chroma is adopted here for *metadata filtering and incremental
add/delete*, not for speed.

---

## 3. Architecture

### 3.1 The two phases

```
                       ┌─────────────────────────────────────────────┐
  OFFLINE (runs once)  │  INDEXING PIPELINE                          │
                       │                                             │
  Unsplash Lite TSV ──▶│  fetch each photo (?w=336&q=80, via CDN)    │
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

### 3.2 The three seams

The architecture is defined by three interfaces, each introduced *before* a second
implementation existed. Every deployment constraint later in the project was absorbed by
one of them.

| seam | interface | implementations | what it bought |
|---|---|---|---|
| **Store** | `search(query_vec, k, filters) -> list[Result]` | `NumpyStore`, `ChromaStore`, `LibraryStore` | The Space filters with NumPy alone — no vector DB in the container. |
| **Encoder** | `encode_text(str)`, `encode_image(PIL.Image)` | `Encoder` (sentence-transformers), `OnnxTextEncoder` | Model-free tests *and* a torch-free 512 MB deploy. |
| **Filters** | `FilterSpec` dataclass | compiled to a NumPy mask *or* a Chroma `where=` | One filter language; the frontend treats filters as one orthogonal axis across all search modes. |

**`FilterSpec` is the spine.** Had EXIF filters been written as Chroma `where=` clauses
when they were introduced, the free-tier Space would now need chromadb plus a 35 MB
SQLite file to get the project's headline feature onto the internet. Instead the whole
Space payload is 29.3 MB.

### 3.3 Request flow (text search)

```
Browser
  └─ GET /api/search?q=...&k=12&aperture_max=2.0
       └─ FastAPI route (plain `def` → runs in the thread pool, CPU-bound work)
            ├─ Depends(get_service)   → the SearchService for the active source
            ├─ Depends(filter_params) → FilterSpec
            └─ SearchService.search_timed()
                 ├─ encoder.encode_text(q)      ~34 ms  → (512,) unit vector
                 └─ store.search(vec, k, spec)  ~5 ms
                      ├─ compile FilterSpec → boolean mask (pre-filter)
                      ├─ scores = embeddings @ vec
                      ├─ np.argpartition → top-k, best first
                      └─ join to the display parquet → list[Result]
       └─ SearchResponse (+ corpus, exif_count, store, source, encode_ms, search_ms)
```

---

## 4. Repository structure

```
semantic-photo-search/
├── .github/workflows/ci.yml      GitHub Actions: ruff + pytest on every push
├── .claude/launch.json           dev-server profiles (api / chroma / render / space)
├── .python-version               3.12 — matches the HF Space runtime
├── pyproject.toml                deps, dependency groups, the CPU-torch index pin
├── uv.lock                       exact resolved versions (committed, never hand-edited)
├── render.yaml                   Render Blueprint — the deploy as a reviewable file
├── LICENSE                       MIT (code only; dataset terms are separate)
├── README.md                     the portfolio front door
├── PLAN.md                       the original 13-session build plan
├── DECISIONS.md                  engineering log — every non-obvious choice and why
├── docs/DOCUMENTATION.md         this document
│
├── src/photosearch/              the importable core library
│   ├── models.py                 FilterSpec, Result
│   ├── exif.py                   EXIF string → typed number parsers
│   ├── encoder.py                CLIP wrapper (sentence-transformers)
│   ├── onnx_encoder.py           torch-free CLIP text tower (onnxruntime)
│   ├── store.py                  NumpyStore, ChromaStore, load_store()
│   ├── library.py                local-folder ingestion + LibraryStore
│   ├── search.py                 SearchService — encoder + store composed
│   ├── baseline.py               BM25-over-captions, same search() shape
│   ├── evaluation.py             P@K, R@K, MRR, pooling, unanswerable handling
│   └── api.py                    the FastAPI application
│
├── web/                          hand-written frontend, no build step
│   ├── index.html                search box, filter panel, result grid
│   ├── app.js                    search modes, filters, BlurHash, cold-start UX
│   └── style.css                 the "editorial darkroom" palette
│
├── scripts/                      one-off pipeline + ops scripts (00 → 07)
├── eval/                         queries, judgments, POLICY.md, harness
├── space/                        the Gradio app deployed to Hugging Face
├── deploy/requirements.txt       the Render runtime's entire dependency list
├── tests/                        pytest — 181 tests, model-free and artifact-free
└── data/                         gitignored: TSV, parquet, embeddings, Chroma, thumbs
```

---

## 5. Module reference

### 5.1 `models.py` — the store-agnostic types

```python
@dataclass(frozen=True)
class FilterSpec:
    aperture_max: float | None = None   # keep f/<= this (inclusive)
    iso_max: int | None = None          # keep ISO <= this
    focal_min: float | None = None      # keep focal >= this (mm)
    focal_max: float | None = None      # keep focal <= this (mm)
    camera_make: str | None = None      # exact, case-insensitive
    def is_active(self) -> bool: ...
```

`Result` carries everything the UI needs to render a card and a detail view — id, score,
image URL, photo page URL, photographer, dimensions, blur hash, both caption fields, and
the five EXIF values — so the API layer never reaches back into the parquet.

### 5.2 `exif.py` — parsers

Turns the dataset's EXIF *strings* into typed numbers at ingest: `"f/1.8"` → `1.8`,
`"1/250"` → `0.004` seconds, ISO → nullable `Int64`, camera make title-cased.
Unparseable values become `NaN`.

This matters more than it looks: **Chroma's `$lt`/`$gte` operators only work on values
stored as numbers.** A stringy `"1.8"` silently matches nothing — no error, no log line,
just zero results. Typing at ingest is what makes the entire EXIF feature function.

Measured coverage over 25,000 photos: aperture 85.2%, focal length 85.5%, exposure
86.8%, ISO 86.8%, camera make/model ~88%.

### 5.3 `encoder.py` / `onnx_encoder.py` — the model seam

`Encoder` wraps one `SentenceTransformer("clip-ViT-B-32")` instance and exposes
`encode_text(str)` and `encode_image(PIL.Image)`, both returning L2-normalized float32
vectors.

`OnnxTextEncoder` implements the same `encode_text` on `onnxruntime` + `tokenizers`,
with **no PyTorch and no vision tower**. It sets `supports_images = False`, which
propagates to `/api/health` so the frontend hides the upload affordance rather than
offering a button that errors.

`load_encoder()` selects between them via `PHOTOSEARCH_ENCODER`.

### 5.4 `store.py` — the store seam

**`NumpyStore`** — brute-force exact cosine search.

- Loads `embeddings.npy` + `photo_ids.npy` + `photos.parquet`, falling back to the
  deploy artifacts (`embeddings.f16.npy` + `photos.slim.parquet`) when the
  full-precision pair is absent — so the same class serves local dev and both deploys.
- **Asserts row alignment element-wise** at construction:
  `photo_ids == photos.photo_id` — not merely equal lengths. A scrambled order returns
  *plausible-looking but wrong* photos with no error anywhere, which is the nastiest bug
  class in the system. A shape check cannot catch it; an ID-equality check can.
- Pre-extracts filterable columns as float arrays once, so mask compilation is pure
  vectorized NumPy.
- `get_embedding(photo_id)` powers "more like this" with **zero model calls**.

**`ChromaStore`** — the same interface over an embedded Chroma collection.

- The collection is created with **cosine** space (`hnsw:space`), not Chroma's default
  L2, and `query()` distances are converted with `similarity = 1 - distance`. On
  normalized vectors L2 and cosine rank *identically* but produce different numbers — an
  L2 collection would therefore rank correctly while silently poisoning every score
  badge and the weak-match threshold.
- Only *filterable* fields are stored as metadata; display fields stay in the parquet
  and are joined back by `photo_id`, keeping one source of truth for rendering.

**`load_store(kind)`** returns `numpy` (default), `chroma`, or `library`.

### 5.5 `library.py` — the local-photo source

Scans a folder recursively, reads EXIF straight from JPEG/HEIC via Pillow, generates
thumbnails (640 px long edge), and incrementally upserts into a `library` Chroma
collection. `LibraryStore` subclasses `ChromaStore`.

Three details that are load-bearing:

- **Path-derived ids.** `photo_id = sha1(absolute path)[:16]`, case-folded on Windows. A
  *content* hash would give a re-edited photo a new id, orphaning its old vector. A file
  counts as modified when mtime **or** size differs from the manifest.
- **Deletions are scoped to the folder being scanned** — without that, indexing
  `D:\Photos` would silently wipe everything indexed from `E:\Archive`.
- **Real EXIF is rationals, tuples and bytes.** Pillow returns `ExposureTime` as a bare
  `(1, 500)` tuple on some files; naively taking element `[0]` records a one-second
  exposure for every 1/500 s frame, with no error. `str(IFDRational(9, 5))` is `"9/5"`,
  which a string parser reads as `9.0` — an f/1.8 lens recorded as f/9. Values are
  coerced to plain numbers *before* the parsers see them, and the tests build their own
  JPEGs rather than mocking Pillow.

`load_photo` raises Pillow's decompression-bomb ceiling to 300 MP for the duration of
one open, in a `try/finally` that always restores the previous value — because
`MAX_IMAGE_PIXELS` is **global to the process**, and forgetting to restore it would
silently disarm the upload endpoint's bomb guard for the life of the server.

### 5.6 `search.py` — `SearchService`

Composes an encoder and a store. `search()`, `similar()` and `search_by_image()` plus
`*_timed()` variants returning encode-ms and search-ms separately.

"More like this" asks the store for `k+1` results and drops the seed, because a photo
always scores 1.0 against itself.

### 5.7 `baseline.py` / `evaluation.py` — the eval machinery

`Bm25Baseline` exposes the same `search(query, k)` shape as `SearchService`, so both run
through one harness. It is deliberately **un-crippled** — no stopword list, no stemming,
both caption columns, zero-score documents dropped rather than padded — because a
baseline you tuned down proves nothing.

`evaluation.py` implements Precision@K, Recall@K, MRR, the pooling logic, and
`drop_unanswerable()`, which splits off queries whose pool contains nothing relevant and
reports them separately instead of dragging every system's average down equally.

### 5.8 `api.py` — the FastAPI application

- **The model loads once, at startup**, in the `lifespan` context manager, and is warmed
  with a throwaway query so visitor #1 doesn't pay the first-call cost.
- **Search endpoints are plain `def`, not `async def`.** Inference is CPU-bound, so
  FastAPI runs them in its thread pool and the event loop stays free. Marking them
  `async` would block the loop and help nothing.
- `filter_params` is declared once as a dependency and reused by all three search
  routes, so their signatures stay short and identical.
- Static files are mounted at `/` **after** the `/api/*` routes so the API wins, and the
  mount is guarded by `WEB_DIR.is_dir()` so the app still imports headless.

---

## 6. Data artifacts

Everything below lives under `data/`, which is **gitignored in its entirety**.

| file | size | what it is |
|---|---|---|
| `raw/photos.tsv000` | ~320 MB | the Unsplash Lite metadata TSV (never committed, never republished) |
| `photos.parquet` | ~6 MB | typed display + EXIF table, pruned to successfully indexed photos |
| `embeddings.npy` | 51 MB | float32, `(24994, 512)`, L2-normalized — the working format |
| `photo_ids.npy` | 1.1 MB | row-aligned join key; the alignment assert's second witness |
| `chroma/` | ~35 MB | the embedded Chroma collection (SQLite + HNSW index) |
| `space/embeddings.f16.npy` | 25.6 MB | float16 — **storage only**, widened to fp32 at load |
| `space/photos.slim.parquet` | 2.6 MB | 13 display columns; caption columns deliberately dropped |
| `encoder/text_model.onnx` + `.onnx.data` | 254 MB | the CLIP text tower with external (mmappable) weights |
| `library/` | varies | local-photo manifest, thumbnails, and its own Chroma collection |

**Why `photo_ids.npy` ships even though the parquet has the same column** (a 1.1 MB
cost): it is what keeps `NumpyStore`'s element-wise alignment assert a *real* check at
startup. A half-synced deploy — new embeddings pushed against a stale parquet — then
dies loudly on boot instead of serving confidently mismatched photos.

**Why fp16 on disk but fp32 in RAM.** Halving the file halves the LFS push, the
container's cold-start download and the release asset, for a quantization error measured
at **max |Δscore| = 4.7e-05** over 40 random unit-vector probes — two orders of magnitude
below the score gap between adjacent results. But fp16 is a *storage* format, not a
*compute* format: NumPy has no fast half-precision matmul and would emulate it in
software, turning a 5 ms search into hundreds of ms. The conversion costs 51 MB of RAM,
which every target tier has.

Measured honestly: **36 of 40 probes had a bit-identical top-10 order**, not 40/40.
Random probes produce near-ties in the tail of the top-10, so fp16 occasionally swaps
ranks 9 and 10. Real queries have far wider gaps — but "the ranking never changes" would
have been a claim the measurement does not support.

---

## 7. HTTP API reference

Base URL locally: `http://127.0.0.1:8000`. Interactive Swagger docs: `/docs`.

### Shared filter parameters

Accepted by `/api/search`, `/api/similar/{id}` and `/api/search/by-image`:

| param | type | constraint | meaning |
|---|---|---|---|
| `aperture_max` | float | `> 0` | keep photos at f/≤ this (wider or equal) |
| `iso_max` | int | `> 0` | keep photos at ISO ≤ this |
| `focal_min` | float | `> 0` | keep focal length ≥ this (mm) |
| `focal_max` | float | `> 0` | keep focal length ≤ this (mm) |
| `camera_make` | str | — | exact make, case-insensitive |

Photos with no value for a filtered field are **excluded** while that filter is active.

### `GET /api/search`

| param | type | default | constraint |
|---|---|---|---|
| `q` | str | required | `min_length=1` |
| `k` | int | 12 | `1 ≤ k ≤ 50` |

```bash
curl "http://127.0.0.1:8000/api/search?q=night+street&aperture_max=2.0&iso_max=800&k=12"
```

**Response** (`SearchResponse`):

```jsonc
{
  "query": "night street",
  "k": 12,
  "count": 12,
  "filtered": true,
  "corpus": 24994,        // total photos indexed
  "exif_count": 21852,    // of those, how many carry filterable EXIF
  "store": "numpy",       // which back-end answered — the seam, made visible
  "source": "unsplash",   // which corpus answered
  "encode_ms": 34.1,
  "search_ms": 5.1,
  "results": [
    {
      "photo_id": "…", "score": 0.3162,
      "photo_image_url": "…", "photo_url": "…", "photographer": "…",
      "width": 4000, "height": 6000, "blur_hash": "…",
      "description": null, "ai_description": "…",
      "aperture": 1.8, "focal_length": 35.0, "exposure_s": 0.004,
      "iso": 400, "camera_make": "Fujifilm", "camera_model": "X-T3"
    }
  ]
}
```

Invalid parameters (`k=0`, `k=999`, `aperture_max=-1`, missing `q`) return **422** with
FastAPI's standard validation body.

### `GET /api/similar/{photo_id}`

Nearest neighbours of an already-indexed photo, using **its own stored embedding** as the
query vector. No encoder call happens at all. Filters still apply — *"like this one, but
shot wide open"*. Returns **404** if the id is not in the index.

### `POST /api/search/by-image`

`multipart/form-data` upload → PIL decode → `encoder.encode_image()` → the same search
path, filters included.

```bash
curl -F "file=@my-photo.jpg" "http://127.0.0.1:8000/api/search/by-image?k=12"
```

- **400** if the file cannot be decoded (a client's bad file is a client error).
- **501 Not Implemented** on the Render deploy, which ships no vision tower.

### `GET /api/health`

```json
{
  "status": "ok", "indexed": 24994, "exif_count": 21852,
  "store": "numpy", "source": "unsplash", "sources": ["unsplash", "library"],
  "encoder": "encoder", "supports_images": true
}
```

`supports_images` is what the frontend reads to decide whether to render the upload
zone. `sources` lists `library` only if a local folder has actually been indexed.

### `GET /api/photo/{photo_id}/thumb` · `/full`

Serves local-library files. The endpoints take an **id, never a path**: the id is
resolved through the manifest server-side, so path traversal is not defended against —
it is *unrepresentable*, because the only reachable paths are ones the user's own
indexing run recorded.

---

## 8. The web UI

Hand-written HTML/JS/CSS, **no build step**, served by FastAPI's `StaticFiles`.

| feature | how |
|---|---|
| **Result grid** | masonry layout; per-card `padding-bottom` reserves the true aspect ratio from the photo's dimensions, so nothing reflows when images load |
| **BlurHash placeholders** | the dataset's ~30-char `blur_hash` decoded client-side into a `<canvas>`, crossfaded out when the CDN image arrives |
| **Score badges** | raw cosine score, banded ≥ 0.28 / ≥ 0.24 / below; a *"no strong matches"* banner only when the best hit is < 0.26 (see §2.5) |
| **Filter panel** | aperture / ISO / focal range / camera make, with an active-filter count, a clear button, and the *"N of M frames carry EXIF"* note |
| **One filter axis, three modes** | a single `activeMode` (`text` / `similar` / `image`); changing a filter re-runs whatever is currently on screen through the same params |
| **Request-id guard** | superseded responses are dropped, so fast filter toggling never renders stale results |
| **Attribution** | photographer credit links to the Unsplash photo page with the required UTM params; an `onerror` guard covers deleted photos |
| **Cold-start banner** | appears only after **1.5 s** of real waiting, retries `/api/health` with backoff to 5 s, hides on first answer, and re-fires the search the user already asked for |

Thumbnails are hotlinked at `?w=480&auto=format&q=75`; the detail view uses `?w=1400`.
Local-library results skip those imgix params (`isRemote()`), which was the only
frontend change the second corpus required.

**On the cold-start delay being a `setTimeout` rather than a check inside the retry
loop:** the loop spends most of its time asleep in the backoff, so an inline check would
only notice the deadline on the *next* iteration — showing a "please wait" notice several
seconds after the wait it explains had already begun.

---

## 9. Scripts reference

Every script is run with `uv run python <path>`.

| script | purpose | key flags |
|---|---|---|
| `00_hello_clip.py` | Learning script: encodes your own photos + texts, prints the full similarity matrix | `--path FOLDER` |
| `01_prepare_dataset.py` | Download Unsplash Lite, parse EXIF, write `photos.parquet` + a data-quality report | — |
| `02_build_index.py` | **The indexer.** Chunked, resumable, concurrent download → CLIP encode → `embeddings.npy` | `--sample N` (measure + ETA), `--sanity QUERY` |
| `03_search_cli.py` | Terminal REPL against the real index | `--time` |
| `04_ingest_chroma.py` | Load the index into a persistent Chroma collection | `--reset`, `--verify`, `--probes N`, `--seed N` |
| `05_index_folder.py` | Index your own photo folder, incrementally | `--path`, `--dry-run`, `--stats`, `--reset`, `--data-dir` |
| `06_build_space_artifacts.py` | fp16 + slim parquet for the Space/release, with verification | `--verify-only`, `--out-dir` |
| `07_build_encoder.py` | Build the ONNX text encoder for Render | `--sweep` (reproduces the encoder table), `--quantize` (block-wise 8-bit fallback) |
| `download_artifacts.py` | **Quickstart:** fetch the ~26 MB index into `data/` | `--into`, `--force` |
| `fetch_deploy_artifacts.py` | Fetch index **and** encoder (~210 MB); run by Render at build time | `--into`, `--force` |
| `sync_space.py` | Copy deployable files into a local clone of the Space repo — never commits, never pushes | `--space PATH`, `--check` |

### Notes on the indexer (`02_build_index.py`)

Three properties worth understanding, because they generalize to essentially every batch
inference pipeline:

1. **Downsized fetches.** Images are requested at `?w=336&q=80`. CLIP resizes every input
   to 224×224 internally, so pixels beyond ~336 px are decoded and immediately discarded.
   This turns a tens-of-GB download into ~1–1.5 GB with **zero** effect on the
   embeddings — the cheapest 20× win in the project.
2. **Chunked, resumable, failure-tolerant.** 500-row chunks, each written to its own
   `.npz`; on restart, existing chunks are skipped. A 25k-item job over the network *will*
   hit 404s (deleted photos) and timeouts, so failures are recorded rather than fatal. A
   failure at photo 24,000 costs one chunk, not the run.
3. **Order discipline.** Downloads are submitted as `{pool.submit(fetch, url): position}`
   and reassembled in dataframe order *before* encoding, because concurrent downloads
   finish out of order. The run ends with an element-wise assert
   `(photo_ids == photos.photo_id.values).all()`.

### Notes on `sync_space.py`

A Space is its own git repo expecting `app.py` at *its* root, so the script flattens
`src/photosearch/` → `photosearch/` beside `app.py`, copies artifacts into `data/`,
ensures `.gitattributes` tracks `*.npy`/`*.parquet` in LFS, prints a diff — **and then
stops.** It never commits and never pushes, because everything it copies becomes public,
and a tool that pushes on your behalf turns *"what exactly did I republish?"* into a
question you answer after the fact.

Two transcription details bite when writing the Space's `requirements.txt`: PyPI has no
`+cpu` local-version tag (write `torch==2.13.0`, not `torch==2.13.0+cpu`), and `gradio`
is a local `--group space` dependency invisible to the FastAPI app, so it must be added
by hand.

---

## 10. Configuration

All configuration is environment variables; there is no config file to keep in sync.

| variable | values | default | effect |
|---|---|---|---|
| `PHOTOSEARCH_STORE` | `numpy` · `chroma` · `library` | `numpy` | which back-end answers |
| `PHOTOSEARCH_ENCODER` | `clip` · `onnx` | `clip` | sentence-transformers, or the torch-free ONNX text tower |
| `PHOTOSEARCH_DATA_DIR` | path | `<repo>/data` | where artifacts live — a deploy points this at its unpacked release |
| `PHOTOSEARCH_ONNX_DIR` | path | `<repo>/data/encoder` | the ONNX graph + tokenizer |
| `PHOTOSEARCH_ONNX_THREADS` | int | runtime default | intra-op threads; set to `1` on 0.1 vCPU, where extra threads only add contention |

The active combination is echoed back in every response (`store`, `source`, `encoder`),
so which one answered is observable rather than assumed.

`.claude/launch.json` carries four ready-made profiles: `photosearch-api` (default),
`photosearch-chroma` (port 8011), `photosearch-render` (the deploy config, port 8012),
and `photosearch-space` (Gradio, port 7860).

---

## 11. How to run it

### 11.1 Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — the only prerequisite. It manages the Python
  interpreter itself, so no separate Python install is needed.
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- ~2 GB free disk (dependencies + the CLIP model cache + the index).
- Windows one-timers: enable **Developer Mode** (so the Hugging Face cache can use
  symlinks) and **long paths**.

### 11.2 Quickstart — searching in about ten minutes

```bash
git clone https://github.com/greisikertuka/semantic-photo-search
cd semantic-photo-search
uv sync
uv run python scripts/download_artifacts.py
uv run fastapi dev src/photosearch/api.py
```

Open <http://127.0.0.1:8000>. The first query additionally downloads the CLIP model
(~600 MB, cached under `~/.cache/huggingface`).

`uv run <cmd>` is the universal prefix — it creates and syncs the environment first, so
there is no virtualenv to activate (which on Windows collides with PowerShell execution
policy).

### 11.3 Building the index from scratch (the overnight path)

```bash
uv run python scripts/01_prepare_dataset.py
uv run python scripts/02_build_index.py --sample 100    # measure, get an ETA
uv run python scripts/02_build_index.py                 # the real run
uv run python scripts/02_build_index.py --sanity "a dog"
```

Optionally load it into Chroma and prove the two stores agree:

```bash
uv run python scripts/04_ingest_chroma.py
uv run python scripts/04_ingest_chroma.py --verify
```

### 11.4 The terminal REPL

```bash
uv run python scripts/03_search_cli.py --time
```

### 11.5 Indexing your own photos

```bash
uv add pillow-heif
uv run python scripts/05_index_folder.py --path "D:\Photos" --dry-run
uv run python scripts/05_index_folder.py --path "D:\Photos"
uv run python scripts/05_index_folder.py --stats
```

Re-running embeds only new or modified files and removes deleted ones. The UI then
offers an **Unsplash / My library** toggle that swaps corpora without a restart.

### 11.6 Running the deploy configuration locally

Always do this before pushing a deploy — never debug via push-and-wait-for-container-build.

```bash
PHOTOSEARCH_ENCODER=onnx PHOTOSEARCH_DATA_DIR=data/space uv run uvicorn photosearch.api:app --port 8012
uv run --group space python space/app.py     # the Gradio Space, on :7860
```

---

## 12. Testing

```bash
uv run pytest          # 181 tests, ~8 seconds
uv run ruff check .    # lint
```

| file | covers |
|---|---|
| `test_exif.py` | the string → number parsers, including garbage and empty input |
| `test_store.py` | a 6×4 fixture matrix with hand-computed neighbours: top-k order, FilterSpec masking |
| `test_chroma_store.py` | cosine space, the distance → similarity conversion, `where=` compilation |
| `test_api.py` | routes, response shape, 422 validation, the 503-when-not-loaded path |
| `test_library.py` | real EXIF coercion against JPEGs the test itself writes; the decompression-bomb ceiling restore |
| `test_onnx_encoder.py` | the ONNX text tower's parity and `supports_images = False` |
| `test_baseline.py`, `test_evaluation.py` | BM25 scoring; P@K / R@K / MRR / pooling arithmetic |

**The whole suite is model-free and artifact-free by construction.** API tests override
the `get_service` dependency with a `StubEncoder` and a 6×4 synthetic store, so the entire
HTTP layer is testable in seconds with zero AI dependency. That is why CI needs no data
download and no 600 MB model.

Two subtleties that were learned the hard way and are worth preserving:

- **Test fixtures construct `TestClient(app)` *without* the `with` context manager**, so
  FastAPI's `lifespan` never runs. Once index artifacts exist on disk, a
  `with TestClient(app)` would call `build_service()` → load the real model, turning a
  3-second suite into 98 seconds and breaking the "index-not-loaded → 503" test. CI never
  hit this (because `data/` is gitignored, the model load failed fast there) — but the
  dev machine did.
- **`test_library.py` builds its own JPEGs** rather than mocking Pillow, because the EXIF
  bugs it guards against (rationals, tuples) only appear in real files.

---

## 13. Evaluation

```bash
uv run python eval/run_eval.py                 # metrics + latency, both systems
uv run python eval/run_eval.py --system clip   # clip | bm25 | deploy
uv run python eval/run_eval.py --store chroma
uv run python eval/run_eval.py --failures 5    # the worst queries, with results
uv run python eval/run_eval.py --markdown      # README-ready tables
uv run python eval/run_eval.py --no-latency
```

### 13.1 Methodology

- **24 queries** across four buckets: easy (one concrete subject), compositional
  (several clauses), abstract (mood, no object), and photography jargon.
- **679 relevance judgments**, pooled from the union of CLIP top-12, two hand-written
  rephrasings at top-6 each, and BM25 top-12 (~28 candidates per query).
- **Unjudged photos count as not relevant** — the standard convention, which makes the
  pool's composition load-bearing: a system whose results were never looked at would
  score zero by construction, so *both* systems' top hits must be in the pool.
- **The relevance policy was written before labeling** (`eval/POLICY.md`): binary
  judgments, every clause counts, negation means negation, abstract queries judged on
  evoked feeling, undecidable → not relevant. Without it, *"is a sunrise a golden-hour
  match?"* gets re-decided at photo 300 and the labels drift into noise.
- **Unanswerable queries are excluded, not scored zero.** *"A black cat sitting on a
  windowsill"* has no match anywhere in 25k photos; every system scores 0, which is a
  fact about the **corpus**, not about ranking.

### 13.2 Results

| system | P@10 | R@10 | MRR |
|---|---|---|---|
| **CLIP** (ViT-B/32) | **0.691** | **0.517** | **0.873** |
| BM25 over captions | 0.283 | 0.241 | 0.623 |

| bucket | CLIP | BM25 |
|---|---|---|
| easy | 0.733 | 0.467 |
| compositional | 0.460 | 0.100 |
| abstract | 0.767 | 0.267 |
| jargon | 0.767 | 0.267 |

The comparison *is* the finding. On easy queries keyword search reaches 0.467 — captions
often literally say the thing — and BM25 **beats** CLIP on two compositional queries. The
gap opens on mood and photographic technique, where there is no lexical foothold at all.

### 13.3 Known failure modes

1. **Attribute binding** — *"a man in a red jacket on a mountain"* (P@10 = 0.20). The
   colour detaches from the object: red-ish, jacket-ish, mountain-ish, rarely the
   conjunction.
2. **Clause dropping** — *"shallow depth of field portrait with creamy bokeh"*
   (P@10 = 0.10). It nails the texture and forgets the subject: nine of ten results are
   beautifully out-of-focus photographs of no one.
3. **Multi-clause scenes** — *"two people sitting on a bench by the water"* (P@10 = 0.00).
   The classic "bag of concepts" limitation of a contrastive model with a single global
   embedding.

Negation (*"a street without any people"*) is the textbook CLIP weakness that did **not**
appear — it scored 0.90, because empty streets are what "street" retrieves in this corpus
anyway. The failure modes you predict are not always the ones you measure.

### 13.4 Stated limitations

- Pooling from one's own systems makes **Recall@10 optimistic**; a relevant photo none of
  the arms surfaced is invisible to the metric.
- One labeller, 24 queries, no inter-annotator agreement, and no confidence intervals
  tight enough to argue about a two-point difference.

---

## 14. Deployment

### 14.1 Hugging Face Space (Gradio, ZeroGPU tier)

**Account requirement:** free Space hosting requires an account **30+ days old** with a
verified email. Free accounts get up to **2 Gradio-SDK Spaces on ZeroGPU**; CPU and
Docker Spaces require PRO ($9/month).

**The GPU is never used.** ZeroGPU allocates a GPU slice *per decorated call*, so running
everything undecorated on the Space's host CPU consumes **zero visitor GPU quota** — and
no GPU is needed: the only model that runs at query time is the CLIP *text* encoder, tens
of milliseconds on CPU. Exactly one `@spaces.GPU`-decorated function exists and is never
called, as insurance against the platform's "no GPU function detected" startup validation.

Steps:

```bash
uv run python scripts/06_build_space_artifacts.py
uv run --group space python space/app.py                 # test locally FIRST
git clone https://huggingface.co/spaces/<you>/<space> ../latent-space
uv run python scripts/sync_space.py --space ../latent-space --check
uv run python scripts/sync_space.py --space ../latent-space
cd ../latent-space && git add -A && git commit -m "…" && git push
```

The Space sleeps after ~48 h idle; the first visit after that takes a minute or so.

### 14.2 Render (the full FastAPI app, free tier)

**Constraints:** 512 MB RAM, 0.1 vCPU, spin-down after 15 minutes idle (~60 s cold start).

Steps:

1. Push the repo to GitHub with [`render.yaml`](../render.yaml) at the root.
2. Ensure the deploy artifacts are published as a GitHub Release (tag
   `deploy-artifacts-v1`): `index-artifacts.tar.gz` and `text-encoder.tar.gz`.
3. In the Render dashboard: **New → Blueprint Instance**, connect the repo, select
   branch `main`, **Apply**.
4. The build runs `scripts/fetch_deploy_artifacts.py --into .` (before `pip install`,
   which is why that script is stdlib-only), then installs `deploy/requirements.txt`.
5. Health check: `GET /api/health`.

First build takes roughly 5–10 minutes, most of it the 210 MB artifact download.

**Why a GitHub Release rather than LFS or a rebuild:** committing 210 MB would bloat every
clone forever and exceed GitHub's 100 MB per-file limit; Git LFS on a public repo has
bandwidth quotas that a build-on-every-push burns through; rebuilding the encoder during
the build would need `onnx` and ~1 GB of peak RAM to rewrite a 254 MB graph — on the tier
being fitted into. A release asset is a plain cacheable URL, and **the tag pins exactly
which index the deployed app is serving**.

The fetch script extracts with `filter="data"` (refusing absolute paths, `..`, symlinks
and device files). These are our own archives, but an extractor that trusts its input is
a habit worth not having.

### 14.3 The three absences that make 512 MB work

| absent | why it can be | consequence |
|---|---|---|
| **PyTorch** | image vectors were computed offline; the text tower runs on onnxruntime | `import torch` alone costs hundreds of MB of RSS before a weight loads |
| **Chroma** | filtering is a NumPy boolean mask, because `FilterSpec` is store-agnostic | no DB, no SQLite file, no HNSW in the container |
| **Vision tower** | only text is encoded at query time | `POST /api/search/by-image` returns **501**, and `/api/health` advertises `supports_images: false` so the UI hides the upload button rather than lying |

The package is pure Python, so `PYTHONPATH=src` is the entire "install" — deliberately
**not** `pip install -e .`, which would drag in torch, sentence-transformers and chromadb.

---

## 15. Performance characteristics

### 15.1 Measured latency

| operation | time | notes |
|---|---|---|
| text encode (CLIP, local CPU) | **34 ms** | dominates end-to-end cost |
| text encode (ONNX, mmapped fp32) | **25 ms** | on the deploy tier |
| search — NumPy exact | **5.10 ms** | 25M multiply-adds over 51 MB |
| search — Chroma HNSW | **5.33 ms** | *not faster* at this size |
| end-to-end search | ~39 ms | |
| indexing throughput (CPU) | ~1.7 img/s | ≈ 4 h for 25k photos |

**Encoding dominates search by roughly 7×.** Any optimization effort belongs on the
encoder, not the search — which is exactly the kind of conclusion that only a measurement
produces.

### 15.2 Memory

| configuration | peak RSS |
|---|---|
| local dev (sentence-transformers + fp32 index) | ~1.2 GB |
| Render deploy (ONNX external weights + fp16→fp32 index) | **400 MB** |
| Render deploy with weights inline (the naive load) | 598 MB — does not fit |

mmapped pages are file-backed, so under memory pressure the kernel evicts them rather
than the OOM killer taking the process — a property the raw numbers hide.

### 15.3 Where this design would have to change

- **~1M+ vectors** — brute force stops being milliseconds; the ANN index starts earning
  its keep.
- **Multi-instance serving** — an embedded store (Chroma/SQLite) is single-process by
  design; this is the point at which Qdrant or a server-mode DB becomes the answer.
- **Payload-indexed filtering at scale** — filtering millions of rows by metadata wants
  real secondary indexes, not a boolean mask.

Knowing *when* you would switch is the useful knowledge; switching at 25k vectors would
have been cosplay.

---

## 16. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `503` from every endpoint | index artifacts missing — `lifespan` couldn't build the service | `uv run python scripts/download_artifacts.py` |
| `FileNotFoundError: none of embeddings.npy, embeddings.f16.npy found` | wrong data directory | check `PHOTOSEARCH_DATA_DIR` |
| `ValueError: row alignment broken` | embeddings and parquet are from different builds | rebuild both, or re-download the release artifacts |
| A filter returns **zero results** and nothing looks wrong in the logs | EXIF values ingested as strings, not numbers | re-run `04_ingest_chroma.py --reset`; `$lte` silently matches nothing on strings |
| Scores look wrong (~0–2 instead of ~0.2–0.35) | the Chroma collection was created with L2, not cosine | recreate it with `metadata={"hnsw:space": "cosine"}` |
| `/api/health` is missing `store` / `exif_count` / `encoder` | a stale server process is still running old code | stop it and restart `uv run fastapi dev src/photosearch/api.py` |
| Search-by-image returns **501** | you are on the Render deploy — no vision tower | use the local app, which has it |
| Hugging Face symlink warning on Windows | Developer Mode off | enable Settings → System → For developers |
| `uv python install` fails on Windows | virtualized AppData | set `UV_PYTHON_INSTALL_DIR` to a path outside AppData |
| First query takes several seconds | CLIP model downloading (~600 MB) | one-time; it caches in `~/.cache/huggingface` |
| Render app slow on first hit | free-tier cold start (~60 s) | expected — the UI shows a "waking up" banner and retries |

---

## 17. Licensing

**Source code:** MIT — see [`LICENSE`](../LICENSE).

**The dataset is a separate matter, handled explicitly.** The Unsplash Lite terms permit
use but prohibit redistributing the Licensed Data, so:

- `data/` is gitignored in its entirety; the TSV and the photographs are never committed.
- What is published is (a) **model outputs** — embeddings are derived data, not the
  dataset — and (b) the **display minimum** needed to render a result and credit its
  photographer, which Unsplash's attribution requirements make mandatory: photo id, image
  URL, photo page URL, photographer name, dimensions, blur hash, numeric EXIF.
- The `photo_description` and `ai_description` caption columns are **dropped** from the
  slim parquet, and no TSV is ever exported. Captions are plainly dataset content rather
  than display necessity, and nothing renders them — so dropping them makes *"the corpus
  is not reconstructible from what we publish"* an accurate statement rather than a
  hopeful one.
- Photographs are **hotlinked** from Unsplash's imgix CDN with the required UTM
  attribution params, which is how Unsplash asks to be used.
- A takedown contact is stated in the README and the Space's README.

**Model weights:** CLIP ViT-B/32 is MIT (OpenAI), used via sentence-transformers.

A documented judgment call reads as engineering judgment; the identical files with no
commentary read as carelessness.

---

## 18. Glossary

| term | meaning |
|---|---|
| **ANN** | Approximate Nearest Neighbour — trades a little recall for a lot of speed at scale |
| **BlurHash** | ~30-character string encoding a blurry preview of an image; decoded client-side |
| **BM25** | The classic keyword-ranking algorithm at the heart of Elasticsearch; the baseline here |
| **Chroma** | Embedded vector database (SQLite + HNSW), no server process |
| **CLIP** | Contrastive Language–Image Pre-training — the frozen model providing the shared space |
| **Cosine similarity** | Cosine of the angle between two vectors; equals the dot product for unit vectors |
| **Embedding** | A learned vector representing content; 512 floats here |
| **EXIF** | Camera metadata embedded in photo files: aperture, ISO, focal length, exposure, make/model |
| **FilterSpec** | This project's store-agnostic filter language — the architectural spine |
| **HNSW** | Hierarchical Navigable Small World — the graph structure behind most ANN indexes |
| **Lifespan** | FastAPI's startup/shutdown context manager; where the model is loaded once |
| **MRR** | Mean Reciprocal Rank — 1/rank of the first relevant hit, averaged |
| **ONNX** | Framework-neutral model format; runs on `onnxruntime` with no PyTorch |
| **Parquet** | Compressed, typed, columnar file format — types survive the round trip, unlike CSV |
| **P@K / R@K** | Precision@K (fraction of top-K that is relevant) / Recall@K (fraction of all relevant found) |
| **Pooling** | Building relevance judgments from the union of several systems' top results |
| **Quantization** | Storing weights at lower precision (8-bit, 4-bit) to shrink a model |
| **uv** | The Python package/project manager used here — `pyproject.toml` ≈ `package.json` |
| **ZeroGPU** | Hugging Face's free tier that allocates GPU per decorated call |

---

## Further reading in this repository

| file | what it holds |
|---|---|
| [`README.md`](../README.md) | the portfolio front door — pitch, demo links, results |
| [`DECISIONS.md`](../DECISIONS.md) | the engineering log: every non-obvious choice and why it beat the alternatives, written as the decisions were made |
| [`PLAN.md`](../PLAN.md) | the original 13-session build plan, with concepts explained per session |
| [`eval/POLICY.md`](../eval/POLICY.md) | the relevance-judgment rules, written before labeling started |

`DECISIONS.md` is the one to read if you want the reasoning rather than the result. It
covers, among others: sentence-transformers over raw `transformers`; ViT-B/32 over L/14;
the NumPy → Chroma → Qdrant graduation ladder; `FilterSpec` as the store seam; pre- vs
post-filtering; L2-vs-cosine score scales; fp16 storage versus fp32 compute; why the
quantized ONNX encoder was rejected on retrieval quality rather than cosine; typed
metadata at ingest; and the licensing call.
