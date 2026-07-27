# Decisions

An engineering log — every non-obvious choice, and *why* it beat the alternatives. Written as decisions are made, not reconstructed after the fact. (This file is deliberately part of the portfolio: it's the "why" behind the code.)

---

## Session 0 — Toolchain & project skeleton

### uv as the package manager
**Decision:** Use [uv](https://docs.astral.sh/uv/) (v0.11.x) for dependency management, virtual environments, Python installation, and command running.

**Why:** It is the mainstream Python tool in 2026 and maps cleanly onto tooling I already know:
`pyproject.toml` ≈ `package.json`, `uv.lock` ≈ `package-lock.json`, `.venv` ≈ `node_modules`, `uv run` ≈ `npx`. It also manages the Python interpreter itself, so there is no separate Python install step. Both `pyproject.toml` and `uv.lock` are committed; the lock is never hand-edited.

**Alternatives:** pip + venv + pyenv + pip-tools is the traditional stack — four tools where uv is one, and slower. Poetry/PDM are closer but less fast and less dominant now.

### Python 3.12 (pinned)
**Decision:** Pin Python to 3.12 via `.python-version` and `requires-python = ">=3.12"`.

**Why:** 3.12 is the newest version supported by the *entire* stack **and** it matches the Hugging Face Space runtime (ZeroGPU pins Python 3.12.12), giving local/deploy parity. 3.13 works locally but buys nothing here and loses that parity.

### PyTorch CPU index pin — the one non-obvious config
**Decision:** Pin torch to the CPU-only wheel index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

**Why:** On Windows, PyPI serves CPU-only torch wheels anyway — but on **Linux** (any deploy target: HF Space, Render) the default resolves to multi-GB CUDA builds. This app never uses a GPU, so those builds are pure bloat that would blow the free-tier RAM/disk budgets. Pinning the CPU index on day one keeps every future Linux build small. A classic "works on my machine" trap, inverted — the machine that would break is the *server*, not the laptop.
Verified: the resolved install reports `torch==2.13.0+cpu`.

---

## Session 2 — Dataset

### Download the Unsplash Lite TSV directly (not the HF dataset)
**Decision:** Fetch `https://unsplash.com/data/lite/latest` (a ~320 MB zip, no signup) and load `photos.tsv000` with pandas — instead of `load_dataset("jamescalam/unsplash-25k-photos")`.

**Why:** That HF dataset is *script-based*, and the `datasets` library removed script support in v4.0. Its script only ever wrapped this same official TSV, so we cut out a broken dependency and load the source directly. One fewer moving part.

### Corpus snapshot pinned: 25 Jun 2026
**Decision:** Record the zip's `Last-Modified` — **Thu, 25 Jun 2026 22:29:53 GMT** — as the frozen corpus snapshot.

**Why:** `/latest` is a moving target. Embeddings (Session 3) and eval labels (Session 10) must all refer to one fixed corpus, or metrics drift silently when the dataset is refreshed. Saved to `data/raw/DATASET_SNAPSHOT.txt` and surfaced in the README later.

### Parse EXIF strings to numeric types at ingest
**Decision:** In `photosearch.exif`, parse the useful EXIF fields from strings into clean numeric dtypes now: `aperture`/`focal_length`/`exposure_s` → float, `iso` → nullable `Int64`, `camera_make` title-cased, `camera_model` case-preserved. Unparseable → NaN.

**Why:** Chroma's `$lt/$gte` filter operators (Session 7) only work on values *stored as numbers* — strings silently match nothing. Typing at ingest is the fix, and doing it in the shared `exif.py` means the Session 9 local-folder pipeline reuses the exact same parsers (and their tests).

**Observed EXIF coverage** (25,000 photos): aperture 85.2%, focal_length 85.5%, exposure 86.8%, iso 86.8%, camera make/model ~88%. So ~12–15% of photos have no EXIF and can never match a numeric filter — a fact the filter UX must surface (Session 7/8 expose "searching N photos with EXIF data").

### Never commit the data
**Decision:** `data/` is gitignored; the TSV and images are never committed.

**Why:** The Lite dataset's terms permit use but prohibit *redistributing* the data. We ship only derived/display-minimum artifacts later (Session 11/12), credit photographers in the UI, and hotlink images via Unsplash's CDN (which is how Unsplash wants images used).

---

## Session 3 — The indexer

### Fetch downsized images, not originals
**Decision:** Request each photo at `?w=336&q=80` (Unsplash imgix CDN params) rather than full resolution.

**Why:** CLIP resizes every input to 224×224 internally, so any pixels beyond ~336px are decoded and immediately thrown away. Downsizing at the CDN turns a ~tens-of-GB download into ~1–1.5 GB with zero effect on the embeddings — the cheapest 20× win in the project.

### Store L2-normalized float32 vectors
**Decision:** `model.encode(..., normalize_embeddings=True)`, saved as float32.

**Why:** If every vector has length 1, cosine similarity *is* the dot product — so the entire search becomes one matrix-vector product (Session 4), no per-query normalization. float32 keeps the matrix at 25k×512×4 B ≈ 51 MB, small enough to hold in RAM on any free tier. (fp16 storage is a Session 11 deploy optimization, not the working format — NumPy has no fast fp16 matmul.)

### Chunked, resumable pipeline with per-chunk checkpoints
**Decision:** Process in 500-row chunks; write each chunk's embeddings + ids to its own `.npz`; on restart, skip chunks that already exist; record (never crash on) per-photo download failures.

**Why:** A 25k-item job over the network *will* hit 404s (deleted photos) and timeouts. Checkpointing per chunk means a failure at photo 24,000 costs one chunk, not the whole run — and the job can run unattended and survive a laptop sleep. This is the shape of essentially every real batch-inference pipeline.

### Index-keyed futures preserve row alignment; validate element-wise
**Decision:** Submit downloads as `{pool.submit(fetch, url): position}` and reassemble each chunk in dataframe order before encoding. At the end, assert `(photo_ids == photos.photo_id.values).all()` — element-wise, not just shape.

**Why:** Concurrent downloads finish out of order. If row *i* of the embedding matrix stops being the same photo as row *i* of the metadata, search returns *plausible-looking but wrong* photos with **no error anywhere** — the nastiest bug class in the system. A shape check can't catch a scrambled order; an ID-equality check can. The invariant is worth a hard assert on every build.

---

## Session 4 — Search core (NumPy) & the FilterSpec seam

### Brute-force NumPy search before any vector DB
**Decision:** v1 search is `embeddings @ query_vec` over the full matrix, top-k via `np.argpartition`.

**Why:** 25k×512 ≈ 25M multiply-adds over 51 MB ≈ a few ms on a laptop CPU — measured, not assumed. A vector DB's ANN index earns its keep at ~1M+ vectors; adopting one here would be complexity with no payoff. Writing brute force by hand is also the thing that lets me actually *explain* vector search in an interview. `argpartition` finds the top k in O(n) then sorts only those k, instead of an O(n log n) full sort — the right habit even when n is small.

### FilterSpec: one filter language, two back-ends
**Decision:** A frozen `FilterSpec` dataclass (aperture_max, iso_max, focal_min/max, camera_make) is the *store-agnostic* filter interface. `NumpyStore` compiles it to a boolean mask; Session 7's `ChromaStore` will compile the *same* object to a Chroma `where=` clause.

**Why:** This seam is the architectural spine of the project. It means the store is swappable by config, tests run against NumPy, and the deployed HF Space (Session 11) can do filtered search with NumPy alone — no Chroma in the container. Designing the interface *before* the second implementation exists is what keeps the two honest.

### Pre-filter (mask), never post-filter
**Decision:** Apply the filter mask to the candidate set *first*, then rank the survivors.

**Why:** Post-filtering (rank top-k, then drop non-matches) breaks under selective filters — ask for top-50, filter to f/1.8, and you might keep 2 results. Pre-filtering restricts the candidate set, then ranks, so k is honoured. Rows with no EXIF have NaN in the filter columns, and every comparison against NaN is False — so "no EXIF ⇒ excluded when a filter is active" falls out for free, which is the correct behaviour (and drives the "searching N photos with EXIF" UI note later).

### Encoder as an injectable interface
**Decision:** `SearchService` depends on an `encoder` object with `encode_text`/`encode_image`, not on SentenceTransformer directly.

**Why:** That one seam gives model-free tests (a fixed-vector `StubEncoder`) *and* the Session 11b path to swap in an ONNX text encoder — same interface, different weights. The dependency injection is the test seam made visible.

---

## Session 5 — API & CI

### Load the model once, at startup (lifespan), not per request
**Decision:** Build the single `SearchService` in FastAPI's `lifespan` context manager and warm the encoder with a throwaway query before serving.

**Why:** The model is ~600 MB; loading it per request would be absurd. `lifespan` is Python's try-with-resources — startup code before `yield`, shutdown after — and is the modern replacement for the deprecated `@app.on_event("startup")`. Warming the encoder means visitor #1 doesn't pay the first-call setup cost.

### Sync `def` for the search endpoint, not `async def`
**Decision:** The search endpoint is a plain `def`.

**Why:** CLIP inference is CPU-bound. FastAPI runs a sync endpoint in its thread pool, keeping the event loop free; marking it `async` would block the loop on CPU work and help nothing. Knowing *when not* to reach for async is the senior move.

### Tests (and CI) are model-free and artifact-free by construction
**Decision:** API tests override the `get_service` dependency with a `StubEncoder` + the 6×4 synthetic store; CI runs only `ruff check` + `pytest`, no data download, no model.

**Why:** CI that needs a 600 MB model or a 51 MB index is slow and flaky. Designing the encoder as an injectable interface (Session 4) means the entire HTTP layer — routing, 422 validation, response shape — is testable in seconds with zero AI dependency. The green CI badge is the cheapest strong signal of engineering culture the repo carries. (One config note: ruff's B008 flags FastAPI's `Depends()`/`Query()`-in-defaults idiom as a false positive, so they're whitelisted via `extend-immutable-calls`.)

### Tests must not boot the real service — skip the lifespan
**Decision:** The API test fixtures construct `TestClient(app)` **without** the `with` context manager, so FastAPI's `lifespan` never runs during tests.

**Why:** Once the index artifacts exist on disk (i.e. after Session 3, on the dev machine), a `with TestClient(app)` would run `lifespan`, which calls `build_service()` → loads the real 600 MB CLIP model — turning a 3-second suite into a 98-second one and breaking the "index-not-loaded → 503" test. Skipping the lifespan keeps `app.state.service` unset, so the dependency override is the only thing wiring a service in. (CI never hit this because `data/` is gitignored, so the model load failed fast there — but the dev machine did. The fix makes both paths identical and fast.)

---

## Session 6 — The web UI

### Vanilla HTML/JS/CSS, no build step
**Decision:** Hand-write `web/index.html + style.css + app.js`, served by FastAPI `StaticFiles` mounted at `/` (after the `/api/*` routes so they win). No framework, no bundler.

**Why:** I'm a frontend dev — a hand-rolled grid is *less* work than standing up a framework, has zero build/deploy complexity, and looks better in a portfolio than a template. The mount is guarded by `WEB_DIR.is_dir()` so the app still imports headless (CI, or before this session existed).

### Score calibration: the distributions overlap, so flag conservatively
**Decision:** Show the **raw** cosine score on every card (e.g. `.316`), color-coded by band (strong ≥ 0.28 ember, decent ≥ 0.24 gold, else grey), and show a "⚠ no strong matches — showing the closest anyway" banner only when the *best* hit is below **0.26**.

**Why:** I measured top-1 scores over 16 queries (10 sensible, 6 absurd) against the live index. Sensible queries landed **0.264–0.340**; absurd/gibberish landed **0.246–0.292**. The ranges **overlap**: "purple elephant playing chess" scores 0.292 (CLIP genuinely finds purple imagery) — *higher* than the legitimate "a bowl of ramen" at 0.264. So there is **no threshold that cleanly separates sense from nonsense**, and any "did you mean nothing?" feature that claims otherwise is lying. The honest design is: always show raw scores (not fake confidence %s), and reserve the warning for a *clearly* weak best match (< 0.26, which catches pure gibberish like "xqzptn wobble" at 0.246 while leaving real-but-weak queries alone). This overlap is itself the interview-worthy finding — CLIP scores are *relative rankings, not probabilities*.

### BlurHash placeholders decoded client-side
**Decision:** Decode each photo's `blur_hash` (shipped in the dataset) into a tiny `<canvas>` behind the image, and crossfade the real CDN image in on load; images are `loading="lazy"`.

**Why:** The grid should never flash empty grey boxes. BlurHash is ~30 chars → an instant, color-accurate blur, so the layout paints meaningfully before any image arrives — photographer-grade polish for almost nothing. A per-card `padding-bottom` reserves the true aspect ratio (from the photo's width/height), so masonry never reflows when images load. Thumbnails are hotlinked from the Unsplash CDN at `?w=480&auto=format&q=75` (detail view `?w=1400`), with UTM attribution params and an `onerror` guard for the occasional deleted photo.
