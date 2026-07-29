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

---

## Session 7 — EXIF-aware search with Chroma

### Create the collection with cosine space, and convert distance → similarity
**Decision:** `create_collection(..., metadata={"hnsw:space": "cosine"})`, and in `ChromaStore` convert every returned distance with `similarity = 1 - distance`.

**Why:** Chroma's default metric is **L2**, and `query()` returns *distances* (lower = better, scale ~0–2) — but the whole app speaks cosine *similarity* (higher = better, ~0.2–0.35): the Session 6 score bands and the "no strong matches" threshold are all calibrated in that space. On normalized vectors L2 and cosine give the *same ranking* but different numbers, so an L2 collection would rank correctly yet silently poison every score badge and the weak-match warning. Fixing it at the source (cosine space) plus the one-line conversion keeps both stores returning the *same numbers*, not just the same order. "Check what number your DB actually returns" is a lesson far cheaper here than in production.

### Only filterable EXIF goes into Chroma; display fields stay in the parquet
**Decision:** Ingest just the numeric/filter fields (aperture, iso, focal_length, exposure_s, camera_make/model) as metadata. URLs, photographer, blur_hash, descriptions stay in `photos.parquet`; `ChromaStore` joins results back by `photo_id`.

**Why:** One source of truth for rendering data. The vector DB's job is vector-search-plus-filter; the parquet's job is display. Duplicating URLs into Chroma would mean two places to keep correct. `camera_make` is stored **lowercased** so Chroma's exact `$eq` matches the case-insensitive FilterSpec, and per-photo NaN fields are *omitted* — a document with no `aperture` key can't match an `aperture` `$lte` clause, which reproduces NumpyStore's "no EXIF ⇒ excluded when filtering" rule exactly. A `has_exif` bool is always added so Chroma never sees an empty metadata dict (which it rejects). Shared `exif_metadata()` in `photosearch.store` builds this for both the ingest script and the tests, so the mapping is defined once.

### Filters must be typed numbers at ingest
**Decision:** Ingest aperture/focal/exposure as `float`, iso as `int`.

**Why:** Chroma's `$lt/$lte/$gte` operators only work on values *stored as numbers* — a stringy `"1.8"` silently matches nothing, with no error. This is the downstream payoff of Session 2 parsing `"f/1.8"` → `1.8`: typed-at-ingest is what makes the whole EXIF differentiator function. (A war story worth telling: the failure mode is "the filter returns zero results and nothing is wrong in the logs.")

### The store is swappable by config, proven by a parity check
**Decision:** `load_store()` selects `NumpyStore` or `ChromaStore` from the `PHOTOSEARCH_STORE` env var (default `numpy`); both implement the identical `search(query_vec, k, filters)` shape, and `scripts/04_ingest_chroma.py --verify` proves they agree.

**Why:** This is the FilterSpec seam paying off — the API, CLI, and every test are written against the interface, not a back-end, so the deployed HF Space (Session 11) can filter with NumPy alone while local dev can exercise the real vector DB. The parity check is deliberately honest about **exact vs approximate**: NumpyStore is brute-force exact, ChromaStore rides an approximate HNSW index, so the invariant is *not* bit-identical top-10 lists. It decomposes into two guarantees that test different things — (1) for a photo **both** stores return, the scores must match to floating-point precision (~1e-6), which proves cosine-space + the distance conversion are correct and is immune to recall; (2) top-k **set overlap** measures ANN recall, which is <1.0 by design (HNSW swaps items at the rank boundary, and genuine score ties from duplicate photos make the exact order ambiguous anyway). Measured over 15 probe queries × 3 filter conditions: **max same-id score gap 1.07e-06, mean top-10 overlap 0.996**. A demanded-identical check would have "failed" on those duplicates and taught the wrong lesson; this one proves the seam *and* names the approximation.

### Expose the EXIF-bearing corpus size in the API
**Decision:** Every search response carries `corpus` (total indexed), `exif_count` (how many have filterable EXIF), and `store` (which back-end answered).

**Why:** ~12–13% of photos have no EXIF and can *never* match a numeric filter, so an active filter searches a smaller universe than the corpus. Surfacing `exif_count` lets the UI say "searching 21,852 of 24,994 frames with EXIF" (Session 8) instead of leaving the shrunken candidate set mysterious. `store` makes the swappable seam visible in the response itself — handy for the parity demo in Swagger.

---

## Session 8 — Filter UI & image-to-image search

### "More like this" needs no encoder — the query vector is already stored
**Decision:** `GET /api/similar/{photo_id}` looks the photo's own embedding up in the store (`get_embedding`) and searches with it, dropping result #1 (a photo is always its own nearest neighbour).

**Why:** This is the session's whole lesson: an image and a text query are *both* just vectors in the shared CLIP space, so "photos like this photo" is the identical dot product with a different query vector — and for an *already-indexed* photo that vector is sitting in the store, so no model call happens at all. Filters still apply ("like this one, but shot wide open"). The one subtlety is asking for `k+1` and filtering the seed out, since it scores 1.0 against itself.

### Upload search runs the indexing-side encoder at query time
**Decision:** `POST /api/search/by-image` takes a multipart upload → PIL → `encoder.encode_image()` → same search path (filters included). A decode failure is a 400, not a 500.

**Why:** The *indexing-side* encoder (`encode_image`, Session 3/4) is exactly what a query-time upload needs — same 512-dim space, same `search()`. The endpoint is a plain `def` so FastAPI runs it in the thread pool (both the upload read and the CLIP encode are blocking); the sync `file.file.read()` is correct there. A client's un-decodable file is a client error, so it maps to 400 with a clean message rather than surfacing a 500.

### One filter panel drives every search mode
**Decision:** The web UI keeps a single `activeMode` (`text` | `similar` | `image`); changing any filter re-runs *whatever* is currently on screen through the same FilterSpec params. The panel shows an active-filter count, a clear button, and the "N of M frames carry EXIF" note.

**Why:** The FilterSpec is one language on the backend, so the frontend should treat filters as one orthogonal axis too — "shot wide open" should mean the same thing whether you're doing a text search, browsing similar frames, or searching by an uploaded photo. Modelling the current query as state (rather than only reacting to keystrokes) is what lets a filter change re-issue an image or similar search without re-uploading logic scattered per mode. A request-id guard drops superseded responses so fast filter toggles never render stale results.

---

## Session 9 — Your own photo library

### The library is a new *source*, not a new code path
**Decision:** Local photos land in the same `Result` shape as Unsplash ones: `photosearch/library.py` writes a manifest parquet carrying every display column `build_result()` already reads, with `photo_image_url` set to `/api/photo/{id}/thumb` instead of a CDN URL. `LibraryStore` subclasses `ChromaStore` over a `library` collection.

**Why:** This is the payoff of the seams built in Sessions 4 and 7. Nothing downstream — not `FilterSpec`, not `SearchService`, not the result renderer, not the "more like this" endpoint — learns that a second corpus exists. The only genuinely new code is *ingestion*, which is where a new source should be confined. The frontend needed one helper (`isRemote()`) to stop appending imgix sizing params to our own URLs; everything else rendered unchanged on the first try, which is the test of whether the abstraction was real.

### Path-derived photo ids, mtime+size for the incremental diff
**Decision:** `photo_id = sha1(absolute path)[:16]` (case-folded on Windows); a file is "modified" when its mtime or size differs from the manifest. Deletions are scoped to the folder currently being scanned.

**Why:** A *content* hash would give a re-edited photo a new id, orphaning the old vector and losing the association with the file. A path hash keeps identity stable across edits, and mtime+size is what marks the row dirty — so `upsert` replaces the vector in place. Scoping deletions to the scanned root is what makes a multi-folder library possible: without it, indexing `D:\Photos` would silently wipe everything indexed from `E:\Archive`. mtime+size is cheap and wrong only for a same-size edit that preserves the timestamp; a content hash is the upgrade path if that ever bites.

### Real EXIF is rationals, tuples and bytes — coerce before parsing
**Decision:** `_rational()` converts `IFDRational`, `(numerator, denominator)` tuples and floats to a plain number *before* the Session 2 string parsers see it; ISO goes through `_num()` because `ISOSpeedRatings` is a SHORT array, not a fraction.

**Why:** The bug this prevents is silent and total. Pillow returns `ExposureTime` as a bare `(1, 500)` tuple on some files; taking element `[0]` yields `1.0` — a one-second exposure recorded for every 1/500s frame, with no error anywhere. `str(IFDRational(9, 5))` is `"9/5"`, which a string parser reads as `9.0` — an f/1.8 lens recorded as f/9. Both were caught only by asserting against a file whose EXIF we wrote ourselves, which is why `tests/test_library.py` builds its own JPEGs rather than mocking Pillow.

### Raise Pillow's decompression-bomb ceiling for local files only
**Decision:** `load_photo` lifts `Image.MAX_IMAGE_PIXELS` to 300 MP for the duration of one open, in a `try/finally` that always puts the previous value back. The upload endpoint in `api.py` keeps Pillow's ~179 MP default.

**Why:** Found by running the indexer over a real 5,919-photo archive rather than a fixture folder: two stitched phone panoramas (199,756,800 px) were refused as suspected decompression bombs. The guard is correct for `POST /api/search/by-image`, which decodes bytes from strangers, and wrong for a folder the user explicitly pointed the indexer at — there, a 200 MP panorama is data. So the lift is scoped to the local path.

The `finally` is the part worth keeping: `MAX_IMAGE_PIXELS` is **global to the process**, and `api.py` imports this module for `LibraryStore`. Setting it and forgetting to restore it would silently disarm the upload endpoint's bomb guard for the life of the server — a security regression introduced by an ingestion convenience, with nothing failing to reveal it. `tests/test_library.py::TestDecompressionBombCeiling` asserts the value is restored on both the success and the exception path. 300 MP (≈900 MB decoded) is a deliberate ceiling rather than `None`: room for panoramas, still bounded.

### File endpoints take an id, never a path
**Decision:** `/api/photo/{id}/thumb` and `/full` resolve the id through the manifest to a path server-side; there is no filename in any URL.

**Why:** The safest file-serving endpoint is one that cannot be told which file to serve. Path traversal isn't defended against here — it's *unrepresentable*, because the only paths reachable are the ones the user's own indexing run recorded. Thumbnails are generated at index time (640px long edge) so the grid never touches originals, and originals never leave the machine except through the explicit `/full` route.

---

## Session 10 — Evaluation

### A metric without a baseline measures nothing
**Decision:** Ship BM25-over-captions (`photosearch/baseline.py`) as a first-class system with the same `search(query, k)` shape as `SearchService`, and run both through the identical harness.

**Why:** "P@10 = 0.69" is unfalsifiable on its own — good compared to what? The comparison is the finding: CLIP 0.691 vs BM25 0.283, and, more interestingly, *where*. On easy single-subject queries keywords reach 0.467 (captions often literally say the thing), and BM25 **beats** CLIP on two compositional queries. The gap opens on mood and photographic technique, where there is no lexical foothold at all. The baseline is deliberately un-crippled — no stopword list, no stemming, both caption columns, zero-score documents dropped rather than padded — because a baseline you tuned down proves nothing.

### Pool from every system, and say what that costs
**Decision:** Candidates per query = union of CLIP top-12, two hand-written rephrasings top-6 each, and BM25 top-12 (~28 avg, 679 total). Unjudged photos count as not relevant.

**Why:** "Unjudged = irrelevant" is the standard convention, and it makes the pool's composition load-bearing: a system whose results were never looked at would score zero by construction, so *both* systems' top hits must be in the pool for the comparison to be fair. The cost is stated rather than hidden — pooling from your own systems makes **Recall@10 optimistic**, because a relevant photo none of the arms surfaced is invisible. That limitation is in the README, not just in `eval/POLICY.md`.

### Write the relevance policy before labeling
**Decision:** `eval/POLICY.md` fixes the rules (binary judgments, every clause counts, negation means negation, abstract queries judged on evoked feeling, undecidable → not relevant) and records the close calls, before a single judgment was made.

**Why:** Without it, "is a sunrise a golden-hour match?" gets re-decided at photo 300 and the labels drift into noise. Labeling ran off numbered contact sheets (`eval/label.py sheets`) — thirty photos in one glance instead of thirty clicks — which is the difference between an eval that exists and one that doesn't.

### Queries with no relevant photo are excluded, not scored zero
**Decision:** `drop_unanswerable()` splits off any query whose pool contains nothing relevant; it's reported separately instead of dragging the averages down.

**Why:** *"A black cat sitting on a windowsill"* has no match anywhere in 25k photos. Every system scores 0, which says nothing about ranking — it's a fact about the **corpus**. Averaging it in would understate every system equally and hide the real finding. Excluding it is standard TREC practice, and printing it explicitly is what keeps the exclusion honest rather than convenient.

### Measure latency before claiming an index is needed
**Decision:** The harness times encode-ms and search-ms separately for both stores over the eval queries.

**Why:** It converts a Session 4 guess into a number: encoding is **34 ms**, search is **5 ms**, and Chroma's HNSW (5.33 ms) is *not faster* than exact brute force (5.10 ms) at 25k vectors. So the vector DB is justified by metadata filtering and incremental add/delete — the two things Sessions 7 and 9 actually needed — and not by speed. Being able to say "I measured brute force first and it was 5 ms, so an ANN index would have been premature" is worth more than any ANN benchmark.

---

## Session 11 — Deployment (Hugging Face Space)

### The Space ships the *NumPy* store, and no vector DB at all
**Decision:** `space/app.py` builds a `NumpyStore` over the shipped embeddings and passes it the same `FilterSpec` the FastAPI app uses. Chroma is not installed in the container; `scripts/sync_space.py` copies only six modules (`models`, `store`, `encoder`, `search`, `exif`, `__init__`) out of the package.

**Why:** This is the Session 4 seam collecting its payment. The deployed demo does **aperture-and-ISO-filtered semantic search with a boolean mask over a 51 MB array** — no database, no server, no HNSW — because `FilterSpec` was designed as the store-agnostic language *before* the second store existed. Had the filters been written as Chroma `where=` clauses at the time, the free tier would now require shipping chromadb and a 35 MB SQLite file to get the project's headline feature onto the internet. Instead the whole payload is **29.3 MB**. A smoke test asserts `"chromadb" not in sys.modules` after importing the app, so the boundary can't rot silently.

### float16 on disk, float32 in RAM
**Decision:** Ship `embeddings.f16.npy` (25.6 MB, half the fp32 file) and call `.astype(np.float32)` immediately at load.

**Why:** Halving the artifact halves the Git-LFS push, the container's cold-start download, and the Session 12 release asset — for a quantization error measured at **max |Δscore| = 4.7e-05** over 40 random unit-vector probes, which is two orders of magnitude below the score gap between adjacent search results. The trap is the second half of the sentence: **fp16 is a storage format, not a compute format.** NumPy has no fast half-precision matmul and would emulate it in software, turning a 5 ms search into a few hundred ms — a "deploy optimization" that silently makes the app 50× slower at the thing it exists to do. The conversion costs 51 MB of RAM, which the free tier has.

Verification is deliberately model-free (`scripts/06_build_space_artifacts.py`): random unit vectors probe the same geometry real queries do, with a fixed seed, so the check runs in CI-like conditions and reproduces exactly. It reports one number that surprised me and is worth keeping honest — **36/40 probes had a bit-identical top-10 order**, not 40/40. Random probes produce near-tied scores in the tail of the top-10, so fp16 occasionally swaps ranks 9 and 10. Real queries have far wider score gaps, but "the ranking never changes" would have been a claim the measurement doesn't support.

### ZeroGPU, used entirely on the CPU
**Decision:** Target the free Gradio-on-ZeroGPU tier but run everything undecorated, on the Space's host CPU. Include exactly one `@spaces.GPU`-decorated function that is never called.

**Why:** In 2026 the only free HF compute is up to 2 Gradio Spaces on ZeroGPU (account 30+ days old); CPU and Docker Spaces need PRO. ZeroGPU allocates a GPU slice *per decorated call*, so by never calling one we consume **zero visitor GPU quota** — and we don't need a GPU: the only model that runs at query time is the CLIP *text* encoder, which is tens of milliseconds on a CPU. The 25k images are never touched by the server at all; the browser hotlinks Unsplash's CDN. The unused decorated function is insurance against the platform's "no GPU function detected" startup validation, and is documented as inert when uncalled.

### Ship precomputed; publish the display minimum — the licensing call, made explicitly
**Decision:** The public Space carries three files: the fp16 embeddings, `photo_ids.npy`, and a **slim** display parquet with 13 columns — id, image URL, photo page URL, photographer name, dimensions, blur hash, and the numeric EXIF. The Unsplash `photo_description` and `ai_description` columns are **dropped**, and no TSV is published.

**Why:** The Lite dataset's terms permit use but prohibit republishing the Licensed Data, so this needed a reasoned position rather than a shrug. What ships is (a) *model outputs* — embeddings are derived data, not the dataset — and (b) the minimum required to render a result and **credit its photographer**, which Unsplash's own attribution requirements make mandatory. Images are hotlinked from Unsplash's CDN, which is how Unsplash asks to be used. The captions were the one category that is plainly dataset content rather than display necessity, and the Space never rendered them, so dropping them costs nothing and makes "the corpus is not reconstructible from what we publish" an accurate statement instead of a hopeful one. The same reasoning is stated in the Space's README with a takedown contact. A documented judgment call reads as engineering judgment; the identical files with no commentary read as carelessness.

`photo_ids.npy` ships even though the parquet has the same column, at a cost of 1.1 MB: it's what keeps `NumpyStore`'s element-wise alignment assert a *real* check at Space startup. A half-synced deploy — new embeddings pushed against a stale parquet — then dies loudly on boot instead of serving confidently mismatched photos, which is the Session 3 nightmare with a public URL attached.

### `sync_space.py` copies; a human pushes
**Decision:** A Space is its own git repo expecting `app.py` at *its* root, so `scripts/sync_space.py` flattens `src/photosearch/` → `photosearch/` beside `app.py`, copies the artifacts into `data/`, ensures `.gitattributes` tracks `*.npy`/`*.parquet` in LFS, prints a diff, and then **stops**. It never commits and never pushes.

**Why:** Everything this script copies becomes public, including the licensing decision above. A tool that pushes on your behalf turns "what exactly did I republish?" into a question you answer *after* the fact. The `--check` mode prints the payload and byte count and writes nothing, so the review step is cheap enough to actually do. LFS isn't optional either — the Hub rejects non-LFS files over 10 MB, and a 26 MB `.npy` fails that at push time, i.e. the least convenient moment.

Two transcription details bite here and are worth stating: the Space installs from PyPI, so the local `torch==2.13.0+cpu` pin must lose its `+cpu` local-version tag (it doesn't exist on PyPI), and `gradio` — a `--group space` dependency locally, invisible to the FastAPI app — has to be added by hand.

### The demo mirrors the real UI, deliberately
**Decision:** The Gradio app re-implements `web/style.css`'s palette, type, masonry grid, score badge and hover credit rather than accepting Gradio defaults, and shares the Session 6 score calibration constants (0.28 / 0.24 / 0.26) verbatim.

**Why:** The live demo is the first thing a recruiter clicks and the README GIF is the second. If they look like different products, the GIF reads as a mockup of something that doesn't exist. Sharing the calibration matters more than the CSS: a *different* "no strong matches" threshold in the demo would quietly contradict the finding Session 6 measured. (Gradio 6 gotcha, found by running it: `css` and `theme` moved off the `Blocks` constructor onto `.launch()` — passing them the old way is a `UserWarning` and silently unstyled output.)

---

## Session 11b — The real API on Render (ONNX, 512 MB)

### The plan said "quantize to 64 MB". Measuring said the memory problem was somewhere else entirely.
**Decision:** Deploy the **fp32** CLIP text encoder, unquantized, with its weights re-saved to an external `.data` file so ONNX Runtime memory-maps them instead of copying them into RAM.

**Why:** The session opened with a plausible plan — Render's free tier is 512 MB, the fp32 text encoder is 254 MB, so use the ready-made 64 MB int8 export. Both halves of that turned out to be wrong, in an order worth recording.

*First*, the 64 MB export is broken for this model. Measured against fp32 on the same twelve queries, it agrees at **cosine 0.88** and returns the same top-1 photo **8% of the time**; the top-10 overlap is 4.2/10. The cause is *per-tensor* quantization — one scale factor for an entire weight matrix — meeting a transformer's activation outliers. Re-quantizing it myself per-channel made it *worse* (0.84), which killed the idea that this was a packaging accident.

*Then*, profiling the whole process showed the model's size was never the binding constraint. ONNX stores weights **inside** the graph protobuf by default, so loading a 254 MB model means parsing 254 MB and materializing every initializer again — the app peaked at **598 MB**. Moving those same weights to a sidecar file lets ORT map them from disk:

| encoder | app peak RSS | encode | agreement with fp32 |
|---|---|---|---|
| fp32, weights inline (the naive load) | **598 MB** — OOM | 26 ms | — |
| ready-made int8, 64 MB (the plan's pick) | 366 MB | 5 ms | cos 0.88, top-1 8% |
| ready-made 4-bit `q4f16`, 72 MB | 366 MB | 45 ms | cos 0.988, **−0.087 P@10** |
| block-wise 8-bit, built here, 141 MB | 288 MB | 305 ms | cos 0.9999 |
| **fp32, weights external (shipped)** | **400 MB** | **25 ms** | **exact** |

The shipped row is the fastest, the most accurate, and fits — *without quantizing anything*. Quantization was an answer to a question nobody had measured. It also has a property raw numbers hide: mmapped pages are file-backed, so under memory pressure the kernel evicts them instead of the OOM killer taking the process.

`scripts/07_build_encoder.py --sweep` reproduces every row. `--quantize` still builds the block-wise 8-bit model, kept as a documented fallback — and as the demonstration that **the fix was blocks, not bits**: same 8 bits as the broken export, ~200× more scale factors, cosine 0.88 → 0.9999.

### "Cosine 0.988 is basically identical" — no. Ask the gold set.
**Decision:** Gate the deploy encoder on **P@10 against Session 10's judgments** (`eval/run_eval.py --system deploy`), not on cosine similarity to the reference vectors.

**Why:** 0.988 cosine *sounds* like a rounding error. Run the same 23 labeled queries through it and it costs **P@10 0.691 → 0.604** — a seventh of the retrieval quality, hidden behind a number that looked like agreement. Cosine between two query vectors is a measure of the embedding; P@10 is a measure of the product. Having built a gold set in Session 10, the marginal cost of asking it this question was one command, and it converted a judgment call into a measurement. The shipped encoder scores **0.691 / 0.517 / 0.873 — identical to the full model in every bucket**, which is a far stronger claim than any cosine.

### No PyTorch, no Chroma, no vision tower — and each absence is load-bearing
**Decision:** The Render service installs `deploy/requirements.txt` (FastAPI, uvicorn, numpy, pandas, pyarrow, onnxruntime, tokenizers) and puts `src/` on `PYTHONPATH`, rather than `pip install -e .`.

**Why:** The project's own dependencies include torch, sentence-transformers and chromadb — gigabytes, and `import torch` alone costs hundreds of MB of RSS before a single weight loads. None are needed to *serve*: image vectors were computed offline in Session 3, filtering is a NumPy boolean mask because `FilterSpec` was designed store-agnostic in Session 4, and the text tower runs on onnxruntime. The package is pure Python, so `PYTHONPATH=src` is the entire "install". Every seam this leans on was built for a different reason in an earlier session, which is the argument for building seams.

The vision tower's absence is a *user-visible* API decision: `POST /api/search/by-image` returns **501 Not Implemented**, and `/api/health` advertises `supports_images: false` so the frontend hides the upload affordance rather than letting someone drag a photo in and receive an error. 501 is the honest code — this server doesn't implement it; the local one does.

### Artifacts by GitHub Release, not by LFS or by rebuild
**Decision:** The 210 MB payload (index + encoder) is published as release `deploy-artifacts-v1`; `scripts/fetch_deploy_artifacts.py` (stdlib only) downloads and unpacks it in Render's build step.

**Why:** Committing it would bloat every clone forever and blow GitHub's 100 MB per-file limit; Git LFS on a public repo has bandwidth quotas that a build-on-every-push burns through. Rebuilding the encoder during the build would need `onnx` and ~1 GB of peak RAM to rewrite a 254 MB graph — on the tier we're trying to fit inside. A release asset is a plain cacheable URL, and **the tag pins exactly which index the deployed app is serving**. The script is stdlib-only because it runs *before* `pip install`, and it extracts with `filter="data"` — these are our own archives, but an extractor that trusts its input is a habit worth not having.

### Say "waking up", and mean it
**Decision:** A cold-start banner that appears only after **1.5 s** of waiting, retries `/api/health` with exponential backoff to 5 s, hides itself the moment the backend answers — and, on a *network* failure mid-session, re-fires the search the user already asked for.

**Why:** The free tier spins down after 15 minutes idle; waking takes about a minute. The naive version of this feature shows the notice immediately, which is a lie on every warm load and locally — hence the delay, so the banner only ever appears when there is genuinely something to explain. The subtler half is the failure path: the realistic cold start isn't page load (Render holds that request while booting), it's a **search fired from a tab that stayed open while the container fell asleep**. That surfaces as a `TypeError`, not an HTTP status, and the first implementation here showed "search failed" for something the user cannot act on. Now it explains, waits, and retries the original query. Backoff caps at 5 s deliberately: a booting container is loading a 254 MB model on a tenth of a CPU, and retry traffic competes with it for exactly that.

One implementation detail worth the comment it carries: the delay is a `setTimeout`, not a check inside the retry loop. The loop spends most of its time asleep in the backoff, so an inline check would only notice the deadline on the *next* iteration — showing a "please wait" notice several seconds after the wait it explains had begun.

---

## Session 12 — Portfolio polish: reproducibility, licensing, documentation

### The quickstart downloads the index the *deploy* already pins
**Decision:** `scripts/download_artifacts.py` pulls **only** `index-artifacts.tar.gz` (~26 MB) from the same `deploy-artifacts-v1` release the Render build uses, unpacks it into `data/`, and imports its download/extract helpers from `fetch_deploy_artifacts.py` rather than copying them.

**Why:** `data/` is gitignored and rebuilding the index is an overnight job, so without this the README quickstart would be a lie — the single most common failure of portfolio repos. Two smaller calls inside it are the interesting ones. First, it unpacks into `data/`, not `data/space/`, because `NumpyStore.load` already falls back to the fp16/slim names when the full-precision pair is absent (Session 11b) — so a fresh clone needs **no env var**, and the quickstart stays four commands. Second, the release tag lives in exactly one file: sharing `BASE`/`TAG` via a `sys.path` import of the sibling script means the quickstart and the deploy can never drift onto different indexes, which is a bug that would present as "the demo and my local copy disagree" and take an hour to find.

What it deliberately does *not* fetch is the 184 MB ONNX encoder. A reader cloning the repo wants to search; they don't want the deploy's text tower, and making them wait for it would trade the headline number (10 minutes to first search) for nothing.

### LICENSE covers the code, and says out loud what it doesn't cover
**Decision:** MIT for the source, with a second section in the `LICENSE` file itself naming the Unsplash Lite terms, the Unsplash License, and the CLIP weights — and stating that `data/` is not redistributed.

**Why:** An MIT file alone, sitting in a repo whose whole subject is someone else's photographs, makes a claim broader than the one we're entitled to make. Putting the boundary *in the license file* — not only in the README, which is the file people skim — means the limitation travels with the code when someone vendors it. The licensing reasoning from Session 11 stays in DECISIONS.md and the README; the LICENSE just refuses to be misread.

### A separate technical document, because the README has a different job
**Decision:** Ship `docs/DOCUMENTATION.md` — module reference, full API reference, artifact table, configuration, troubleshooting, glossary — and keep the README to the 90-second skim plus the evaluation results.

**Why:** These two audiences want opposite things and the usual mistake is serving neither. A recruiter skimming for 90 seconds needs the pitch, the demo link, and the one number that proves it works; an engineer who has decided to look properly needs the request flow, the env vars, and the reason a filter silently returns zero results. Merging them produces a README that is too long to skim and too shallow to work from. Splitting them also means the troubleshooting table can be *specific* — "a filter returns zero results and nothing is wrong in the logs" is the single best entry in it, and it would never have survived a README edit for length.

The three files now divide cleanly: `README.md` is the front door, `DOCUMENTATION.md` is the manual, `DECISIONS.md` is the *why*. Nothing is duplicated between them except the evaluation table, which earns its place in both.

### The build log stays honest about Session 1
**Decision:** Mark Session 1 done but annotate it as a learning script pointed at your own photos, rather than silently checking it off with the rest.

**Why:** Every other session produced code that CI exercises; `00_hello_clip.py` produces a printed matrix on a folder that isn't in the repo. Checking it off identically would be a small lie of the kind that, if a reviewer catches it, retroactively taxes the credibility of every other measured claim in the README — and there are a lot of those. The cheap fix is a parenthesis.
