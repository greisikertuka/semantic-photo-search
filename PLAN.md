# Build Plan — Semantic Photo Search

*A web app that searches 25,000 Unsplash photos by meaning, not tags: type "golden hour by the sea" and get ranked, relevant photos — powered by CLIP embeddings and vector search.*

**Plan written:** 2026-07-27. All version numbers, free-tier policies, and URLs below were verified against live sources on this date, and the full draft was reviewed by independent technical/pedagogy/portfolio critique passes.

---

## How to use this plan

- It's split into **sessions 0–12** (two split into a/b), each sized for one sitting of roughly 1–3 hours. Do them in order; each builds on the last.
- Every session has: **Goal → Why it matters → Concepts** (plain-language explanations of the AI/Python ideas you'll meet) **→ What we build → Definition of done**. Don't move on until the definition of done passes — that's your regression safety net.
- You'll write the code *with* Claude session by session, but this document is the map. When a session says "we'll write X", that's the scope contract for that sitting.
- Testing is a thread, not a session: Sessions 2, 4, and 5 each add a small pytest file, and CI arrives in Session 5. By the end the repo has the test culture a senior reviewer expects — without a boring "testing chapter."

---

## Reality checks (things the original brief assumed that are no longer true)

Research against live sources (July 2026) surfaced four corrections. Knowing these up front saves you from following stale tutorials:

1. **Don't use `load_dataset("jamescalam/unsplash-25k-photos")`.** That Hugging Face dataset is a *script-based* dataset, and the `datasets` library removed script support in v4.0 (current is 5.0). It errors with "Dataset scripts are no longer supported." The dataset was only ever a thin wrapper anyway — its script just downloads the official Unsplash Lite TSV. We'll download that zip directly (~305 MB, no signup: `https://unsplash.com/data/lite/latest`) and load it with pandas. Simpler, and one less broken dependency.
2. **Hugging Face Spaces' free tier changed (July 2026).** Free accounts can no longer create CPU or Docker Spaces — the only free compute is up to **2 Gradio-SDK Spaces on ZeroGPU**, and your account must be **30+ days old with a verified email**. → **Day-1 action in Session 0: create your HF account now so the 30-day clock is running by the time you deploy.** (Escape hatch if you can't wait: PRO is $9/month and unlocks classic CPU Spaces.)
3. **Streamlit is deprecated as a Spaces SDK.** It's out. Our UI story instead: a plain HTML/JS frontend served by FastAPI (you're a frontend dev — this is *less* work for you than learning Streamlit), plus a thin Gradio wrapper only for the deployed Space.
4. **Your machine has no Python installed.** That's fine — `uv` (the tool we'll use) downloads and manages Python interpreters itself. No separate Python install step.

---

## Architecture (what you're building)

```
                       ┌─────────────────────────────────────────────┐
  OFFLINE (runs once)  │  INDEXING PIPELINE  (scripts/build_index.py) │
                       │                                             │
  Unsplash Lite TSV ──▶│  fetch each photo (small size, via URL)     │
  (25k rows, metadata) │        │                                    │
                       │        ▼                                    │
                       │  CLIP image encoder ──▶ 25,000 × 512 floats │
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
                       FastAPI + HTML/JS UI      Gradio app (uses the
                       (local; Render deploy     NumPy store + FilterSpec)
                        in Session 11b)          (HF Space — live demo)
```

The key mental model: **the CLIP model is frozen**. We never train anything. Indexing is just "run every image through the encoder once and save the numbers." Search is "run the query text through the encoder and find the nearest saved numbers."

---

## The stack (decided, with reasons — seed material for DECISIONS.md)

| Choice | What | Why (over alternatives) |
|---|---|---|
| Package manager | **uv** (latest, ~0.11.x) | The mainstream Python tool in 2026. `pyproject.toml` ≈ `package.json`, `uv.lock` ≈ `package-lock.json`, `.venv` ≈ `node_modules`, `uv run` ≈ `npx`. Manages Python versions itself. |
| Python | **3.12** (pinned) | Newest version supported by the whole stack *and* it matches the HF Space runtime (ZeroGPU pins Python 3.12.12). 3.13 works locally but buys nothing and loses deploy parity. |
| CLIP library | **sentence-transformers** (`clip-ViT-B-32`) | One model object encodes *both* images and text via one `.encode()` call. Raw `transformers` CLIPModel is ~3× the code for control we don't need; `open_clip` is a training/research tool. |
| CLIP checkpoint | **ViT-B-32** (512-dim, ~600 MB) | Fast enough on CPU (~tens of ms per encode) for free hosting. ViT-L-14 is ~10–20× slower and 1.7 GB — wrong trade for this app. Upgrade path documented in DECISIONS.md. |
| Vector store v1 | **NumPy brute force** | 25k × 512 floats = 51 MB; one matrix-vector product ≈ 1–5 ms. Teaches exactly what vector DBs abstract. |
| Vector store v2 | **Chroma** (~1.5.x, embedded) | Adds combined vector search + numeric metadata filters (`{"aperture": {"$lt": 2.0}}`) with persistence and incremental adds. Embedded/SQLite, no server to run. |
| Graduation story | **Qdrant** (name-drop, not deployed) | Free cloud tier suspends after 1 week idle — bad for a portfolio demo. But knowing *when* you'd switch (≈1M+ vectors, multi-instance, payload-indexed filtering) is the interview answer. |
| API | **FastAPI** + uvicorn | Industry standard; you know Spring Boot, this will feel familiar (typed DTOs via Pydantic, DI-ish patterns, auto OpenAPI docs). |
| UI | **Static HTML/JS served by FastAPI** + **Gradio** (Space only) | You're already a frontend dev; a hand-rolled grid is faster for you than learning a UI framework, and looks better in a portfolio. Gradio exists only because it's the free deploy path on HF. |
| Tests & CI | **pytest + ruff + GitHub Actions** | Thin thread through Sessions 2/4/5. An AI repo with zero tests is the most predictable senior-review ding; ~1–2 h total buys it off. |
| Deployment | **HF Space (Gradio, ZeroGPU tier, model on CPU)** + **Render free + ONNX (Session 11b)** | The only free HF compute in 2026, plus a real FastAPI deployment on a 512 MB tier via a 64 MB quantized ONNX text encoder — the best engineering story in the project. |

---

# The Sessions

---

## Session 0 — Toolchain, project skeleton, and the 30-day clock

**Goal:** A working `uv`-managed Python project with an importable `photosearch` package, git repo, and the accounts you'll need later.

**Why it matters:** Python's packaging story has historically been the #1 source of beginner pain. `uv` collapses it into something that feels like npm. Getting this right once means you never fight your environment again.

**Concepts:**
- **`pyproject.toml` / `uv.lock` / `.venv`** — direct analogs of `package.json` / `package-lock.json` / `node_modules`. Declared dependency ranges live in `pyproject.toml`; the exact resolved versions live in `uv.lock` (commit both; never hand-edit the lock). `.venv` is gitignored like `node_modules`.
- **`uv run <cmd>`** — runs a command inside the project environment, auto-creating/syncing it first. We use this as the universal prefix for *everything* (`uv run python …`, `uv run fastapi dev …`, `uv run pytest`). This deliberately sidesteps venv activation, which on Windows collides with PowerShell execution policy.
- **Your own code as an installable package.** This is the Python idea with no Node/Java equivalent in your muscle memory: code in `src/photosearch/` is **not** importable by scripts elsewhere in the repo until the project itself is installed into the environment as a package (Node resolves relative `require`s; Java has the classpath; Python has neither by default). `uv init --package` sets this up: your project becomes a package named `photosearch`, installed in *editable* mode (like `npm link` — imports always read your live source). Scripts then just say `import photosearch` and it works, from anywhere.
- **Install name ≠ import name.** A Python quirk to know once: you install `pillow` but `import PIL`. (Same deal later: install `pillow-heif`, import `pillow_heif`.)
- **The PyTorch CPU index pin** — the one non-obvious config. On Windows, PyPI serves CPU-only torch wheels anyway, but on Linux (i.e., any deploy target) the default resolves to multi-GB CUDA builds. Pinning the CPU index in `pyproject.toml` on day one keeps every future Linux build small. A classic "works on my machine" trap, inverted.

**What we build:**
1. Install uv (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (or `winget install astral-sh.uv`).
2. In the project folder: `uv init --package --name photosearch .` (creates the `src/photosearch/` layout **with** build config so the package is importable), then `uv python pin 3.12`.
3. Add to `pyproject.toml`:
   ```toml
   [[tool.uv.index]]
   name = "pytorch-cpu"
   url = "https://download.pytorch.org/whl/cpu"
   explicit = true

   [tool.uv.sources]
   torch = [{ index = "pytorch-cpu" }]
   ```
4. `uv add torch sentence-transformers pandas pyarrow pillow "fastapi[standard]"` and `uv add --dev pytest ruff` — watch it resolve and write `uv.lock`.
5. `git init` (if `uv init` didn't), first commit. Create the repo on GitHub.
6. Windows one-timers: enable **Developer Mode** (Settings → System → For developers) so Hugging Face's cache can use symlinks (otherwise you get a scary-looking but benign warning and duplicated cache files), and enable **long paths** (Settings → System → Advanced → "Enable long paths" or the `LongPathsEnabled` registry key).
7. **Create/verify your Hugging Face account today** — free Space hosting requires the account to be 30+ days old. Also create a Render account for Session 11b.
8. Project skeleton:
   ```
   semantic-photo-search/
   ├── pyproject.toml, uv.lock, .python-version, .gitignore
   ├── PLAN.md  DECISIONS.md  README.md
   ├── data/            # gitignored: TSV, parquet, embeddings
   ├── scripts/         # one-off pipeline scripts
   ├── src/photosearch/ # the importable core library
   ├── tests/           # pytest (grows from Session 2 on)
   ├── web/             # static frontend
   ├── eval/            # evaluation queries + harness
   └── space/           # the Gradio app for HF deployment
   ```
   Start `DECISIONS.md` immediately with entries for uv, Python 3.12, and the CPU-index pin — writing decisions down *as you make them* is what makes the file authentic.

**Definition of done:** `uv run python -c "import photosearch; import torch; print(torch.__version__)"` prints a version ending in `+cpu` with no ImportError. Repo pushed to GitHub. HF account exists.

*Est. 1–1.5 h.*

---

## Session 1 — Embeddings 101: your first CLIP experiment

**Goal:** A tiny throwaway script that proves to *you* that text and images can land in the same vector space — before any app code exists.

**Why it matters:** This is the core concept of the entire project (and of most modern AI retrieval systems). If you deeply get this session, everything else is plumbing. It's also exactly what you'll whiteboard in interviews.

**Concepts (the big ones — read slowly):**
- **Embedding.** A learned list of numbers (a *vector*) representing a piece of content — here, 512 floats. Think of it as coordinates in a 512-dimensional space where *position encodes meaning*: things with similar meaning sit near each other. The numbers themselves are meaningless individually; only distances/angles between vectors matter.
- **Cosine similarity.** "How similar are two vectors?" measured as the cosine of the angle between them: 1.0 = same direction (very similar), 0 = unrelated, negative = opposed. Crucially: if you *normalize* vectors to length 1 first, cosine similarity is just the dot product — one multiply-add per dimension. That's why we'll store normalized vectors: search becomes a single matrix multiplication.
- **CLIP.** OpenAI's 2021 model, trained on ~400M (image, caption) pairs with a *contrastive* objective: two separate encoders — one for images, one for text — trained jointly so that matching image/caption pairs get high cosine similarity and mismatched pairs get low similarity. The result: **one shared vector space for two different media types.** That's the magic — a text query can be compared directly against image vectors.
- **Zero-shot.** CLIP works on our photos with no training, no fine-tuning, no labels. We only ever run *inference* (a forward pass). "Indexing" sounds fancy but it's just: run each image through the frozen encoder once, save the 512 numbers.
- **Tokens (one-liner).** Text goes into CLIP as ~word-chunks called tokens; CLIP truncates at 77 tokens. Irrelevant for short queries, but know it exists.

**What we build:** `scripts/00_hello_clip.py`. Put 4–6 of your own photos in a folder. The script:
```python
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")   # first run downloads ~600 MB to C:\Users\<you>\.cache\huggingface
img_embs = model.encode([Image.open(p) for p in paths], normalize_embeddings=True)
txt_embs = model.encode(["a dog", "a mountain at sunset", "a red car"], normalize_embeddings=True)
scores = txt_embs @ img_embs.T                 # cosine similarity via dot product — that's it
```
Print the score matrix with filenames as columns. Then play: try adjectives, styles ("a blurry photo", "a photo taken at night"), wrong descriptions. Watch scores move. Note the *absolute* score range you see (good matches typically land ~0.25–0.35, not 0.9 — CLIP scores are relative rankings, not probabilities; this matters for Session 6's "no good matches" feature).

**Definition of done:** For each of your test texts, the highest-scoring image is the right one — and you can explain to a rubber duck why one matrix multiply computed all pairwise similarities.

*Est. 1.5–2 h (including the model download and playing time — the playing is the point).*

---

## Session 2 — The dataset: download, explore, clean (+ your first tests)

**Goal:** The Unsplash Lite metadata loaded, understood, and saved as a clean typed `photos.parquet` — with the EXIF parsers unit-tested.

**Why it matters:** Every real AI system is ~50% data plumbing. This session is also where the EXIF differentiator gets its foundation: the raw EXIF columns are *strings with many nulls*, and turning them into filterable numbers now is what makes Session 7 possible.

**Concepts:**
- **pandas DataFrame** — Python's in-memory table; think "a SQL table you script against." You know SQL, so we'll lean on that: `df[df.exif_iso < 800]` is `WHERE`, `df.groupby` is `GROUP BY`.
- **Nullable numbers.** pandas represents missing values as `NaN`, which only exists for floats — so `astype(int)` on a column with missing values *raises*. Use the nullable `"Int64"` dtype (capital I) for ISO, or just keep it float. (First Python-specific landmine, defused.)
- **Parquet** — a compressed, *typed*, columnar file format. Unlike CSV/TSV, types survive the round trip (your parsed floats stay floats).
- **pytest in one paragraph.** Python's JUnit, minus ceremony: any function named `test_*` in `tests/` is a test; plain `assert` statements are the assertions; `uv run pytest` runs everything. The EXIF parsers are pure functions — perfect first-test material.
- **Data licensing hygiene** — the Lite dataset's terms allow use but prohibit *redistributing* the data. The photos themselves are Unsplash-licensed (hotlinking is not just allowed, it's how Unsplash wants images used, via their imgix CDN URLs). Practical rules we'll follow: never commit the TSV or any images to the public repo (`data/` is gitignored); anything published ships only *derived or display-minimum* artifacts (a documented judgment call we'll make explicitly in Session 11); credit photographers in the UI; note the license in the README. Also record the zip's `Last-Modified` date — the `/latest` download is a moving target, and this date pins which corpus snapshot everything downstream (embeddings, eval labels) refers to.

**What we build:** `scripts/01_prepare_dataset.py`, `src/photosearch/exif.py` (the parsers live in the package so both this script and Session 9 can reuse them), and `tests/test_exif.py`.
1. Download `https://unsplash.com/data/lite/latest` (~305 MB zip); note its Last-Modified date in DECISIONS.md; extract **only** `photos.tsv000` into `data/raw/`.
2. Load: `pd.read_csv("data/raw/photos.tsv000", sep="\t")` → expect ~25,000 rows × 31 columns. Explore: `df.columns`, `df.head()`, null counts per column. (If the row count isn't ~25k, revisit quoting: `quoting=csv.QUOTE_NONE`.)
3. Clean into a typed frame, keeping what the app needs:
   - identity/display: `photo_id`, `photo_image_url`, `photo_url`, `photographer_first_name/last_name`, `photo_width/height`, `blur_hash`, `photo_description`, `ai_description`
   - EXIF via `photosearch.exif` (string → numeric, `errors="coerce"` so failures become NaN): `exif_aperture_value` → float (strip a leading `f/` if present), `exif_focal_length` → float, `exif_iso` → `"Int64"`, `exif_exposure_time` "1/250" → float seconds (parse the fraction), plus `exif_camera_make/model` normalized (strip/titlecase).
4. Tests: `parse_aperture("f/1.8") == 1.8`, `parse_exposure("1/250") == 0.004`, garbage → `None`/NaN, empty → NaN.
5. Save `data/photos.parquet`. Print a small "data quality report": % nulls per EXIF field, value ranges (sanity: apertures ~0.95–32, ISO ~25–25600).

**Definition of done:** `photos.parquet` exists; `uv run pytest` is green; a 5-line script can answer "how many photos were shot at f/1.8 or wider?" — and you know the actual EXIF coverage percentages (a large minority of rows have no EXIF — this number shapes the filter UX later).

*Est. 2 h.*

---

## Session 3 — The indexer: embed 25,000 photos

**Goal:** `data/embeddings.npy` + `data/photo_ids.npy` — a 25,000 × 512 float32 matrix and its row-aligned ID list, consistent with `photos.parquet`.

**Why it matters:** This is the "indexing phase" from the project brief — the one-time offline batch job that makes search possible. It's also your first real batch-inference pipeline: chunking, retries, checkpointing, and *order discipline*. Pipelines like this are half of applied AI engineering.

**Deps for this session:** `uv add httpx`

**Concepts:**
- **Batch inference.** Models are much faster fed batches (e.g., 32 images at once) than one-at-a-time — fixed per-call overhead amortizes and the math vectorizes. `model.encode(images, batch_size=32)` handles it.
- **Why we fetch images over HTTP.** The dataset has *no image bytes* — only CDN URLs. We request downsized copies (`?w=336&q=80` — imgix params on the Unsplash CDN): CLIP resizes to 224×224 internally anyway, so full-res downloads would be pure waste (tens of GB vs ~1–1.5 GB).
- **Row alignment is sacred.** The entire system rests on one invariant: *row `i` of the embedding matrix is the same photo as row `i` of the metadata.* Concurrent downloads return **out of order** (whichever finishes first) — so results must be re-assembled in dataframe order before encoding, and we persist the aligned `photo_ids` next to the embeddings as the canonical join key. A shape check can't catch scrambled order; an ID-equality check can. This is the bug class that produces "search returns random photos" with no error message anywhere.
- **Idempotent, resumable pipelines.** A 25k-item job over a network *will* hit failures (some photos are deleted from Unsplash → 404s; timeouts happen). Design: process in chunks (e.g., 500), save progress after each chunk, skip already-done work on restart, record failures instead of crashing.
- **GPU vs CPU here.** sentence-transformers auto-uses a CUDA GPU if present. On CPU expect the encode itself to take roughly 30 min–2 h for 25k (order of magnitude — we'll time a 100-image sample first and extrapolate before committing to the full run).

**What we build:** `scripts/02_build_index.py`:
- Reads `photos.parquet`; for each chunk of ~500 rows: download concurrently (`ThreadPoolExecutor` + httpx, ~8–16 workers, timeout + 1 retry) with **index-keyed futures** (`{executor.submit(fetch, url): row_index}`), re-assemble the chunk **in dataframe order**, record failures; `model.encode(imgs, batch_size=32, normalize_embeddings=True, show_progress_bar=True)`; append embeddings *and their photo_ids* to on-disk accumulators; checkpoint after each chunk.
- First: a `--sample 100` mode to measure download and encode throughput, then extrapolate and print an ETA before you commit to the full run (which can just run while you make dinner).
- After the run: drop failed rows from the parquet (keep the failure list), then validate **element-wise**: `(photo_ids == photos.photo_id.values).all()` — not just shape equality.

**Definition of done:** ID-alignment check passes; `np.linalg.norm(embeddings[0])` ≈ 1.0 (normalized); a 10-line sanity script encodes the text "a dog" and the top-scoring photo URL, opened in your browser, is… a dog. (This moment is genuinely fun. Screenshot it.)

*Est. 1.5–2 h of coding + an unattended run.*

---

## Session 4 — The search engine (NumPy core) + CLI

**Goal:** A real, importable search module — `search("foggy empty street", k=12)` returns ranked results in milliseconds — with the filter interface designed in from the start, plus a tiny CLI and unit tests.

**Why it matters:** This is the retrieval heart of the app, built from primitives so you actually understand it. When an interviewer asks "how does vector search work?", your answer comes from having written this file, not from a library README.

**Concepts:**
- **Brute-force search and why it's fine here.** Scoring = one matrix-vector product: 25,000 × 512 ≈ 25.6 million multiply-adds over a 51 MB matrix ≈ **1–5 ms** on a modern CPU. Know this sizing math cold — "I measured brute force first and it was 3 ms, so an ANN index would have been premature complexity" is a *great* interview sentence (Session 10 turns it into a measured number).
- **Top-K selection.** Full sort is O(n log n); `np.argpartition` gets the top K in O(n) then sorts just those K. Small thing, right habit.
- **Design for two stores: the FilterSpec.** We define one small typed object — `FilterSpec` (aperture_max, iso_max, focal_min/max, camera_make; a `dataclass`, Python's version of a Java record) — as the *store-agnostic* language for EXIF filters. `NumpyStore` compiles a FilterSpec into a **boolean mask** (a True/False array marking eligible rows — pre-filtering with array math); Session 7's `ChromaStore` will compile the *same* FilterSpec into a `where=` clause. One interface, two backends — this seam is the architecture lesson of the whole project, and it's what later lets the deployed Space do filtered search with NumPy alone.
- **ANN (approximate nearest neighbor) — concept only.** At millions of vectors, brute force gets slow; structures like HNSW (a layered graph you greedily walk) find *approximately* nearest neighbors in ~logarithmic time, trading a little recall for a lot of speed. That's what Chroma/Qdrant/FAISS do inside. We name it now, use it in Session 7, and measure it in Session 10.

**What we build:**
- `src/photosearch/models.py` — `FilterSpec` and `Result` dataclasses.
- `src/photosearch/store.py` — `NumpyStore`: loads `embeddings.npy` + `photo_ids.npy` + `photos.parquet` (verifying alignment at load), exposes `search(query_vec, k, filters: FilterSpec | None) -> list[Result]`; internally compiles filters → mask → masked argpartition.
- `src/photosearch/encoder.py` — thin wrapper owning the SentenceTransformer instance; `encode_text(str)`, `encode_image(PIL.Image)`.
- `src/photosearch/search.py` — `SearchService` composing encoder + store; results carry photo id, URLs, photographer, score, EXIF.
- `scripts/03_search_cli.py` — REPL loop: type a query, print top 10 with scores + URLs; `--time` flag prints encode-ms and search-ms separately.
- `tests/test_store.py` — a 6×4 fixture matrix with hand-computed nearest neighbors: assert top-k order, assert FilterSpec masking excludes the right rows. Deterministic, no model download needed — this is what runs in CI.

**Definition of done:** Queries from the brief — "golden hour by the sea", "man in a red jacket on a mountain", "foggy empty street" — return obviously-right photos with search latency ≤ ~10 ms (text encoding will dominate at ~20–100 ms); `uv run pytest` green; you can explain where every millisecond goes.

*Est. 2–2.5 h.*

---

## Session 5 — The API: FastAPI backend (+ CI)

**Goal:** `GET /api/search?q=foggy+street&k=12` returns clean JSON; interactive docs at `/docs`; GitHub Actions runs your tests on every push.

**Why it matters:** This wraps the AI core in the shape the rest of the software world consumes — and the CI badge is the cheapest strong signal of engineering culture the repo can carry.

**Concepts:**
- **FastAPI ↔ Spring Boot mapping.** Decorators ≈ annotations (`@app.get` ≈ `@GetMapping`); **Pydantic models** ≈ DTOs with built-in Bean-Validation (declare `k: int = Field(12, le=50)` and bad requests get automatic 422s); the OpenAPI/Swagger UI comes free at `/docs`.
- **Lifespan startup, and `yield`.** The model (~600 MB) must load **once at process start**, never per-request. FastAPI's `lifespan` is a *context manager* — Python's try-with-resources: code before the `yield` statement runs at startup, code after it at shutdown, and `yield` marks the "now the app runs" point in between. (The `@app.on_event("startup")` you'll see in old tutorials is deprecated.) We'll also encode a dummy query at startup so user #1 isn't the model warmup.
- **Async — the honest version.** Model inference is CPU-bound, so we declare the search endpoint as a plain `def` (FastAPI runs it in a thread pool) rather than pretending `async` helps. Knowing *when not to* use async is the senior move.

**What we build:**
- `src/photosearch/api.py` — FastAPI app with `/api/search` (q, k, and the FilterSpec params: `aperture_max`, `iso_max`, `focal_min/max`, `camera_make`), `/api/health`, CORS config, Pydantic response models, lifespan that builds one `SearchService`. Run: `uv run fastapi dev src/photosearch/api.py`.
- `tests/test_api.py` — FastAPI's `TestClient` with a **stubbed encoder** (returns a fixed vector — no model in CI): asserts response shape, 422 on bad params. The stub *is* the dependency-injection seam made visible.
- `.github/workflows/ci.yml` (~20 lines): checkout → install uv → `uv sync` → `uv run ruff check .` → `uv run pytest`. Badge in the README.

**Definition of done:** `/docs` renders and searches work from the Swagger UI; nonsense params return a clean 422; CI is green on GitHub with the badge showing.

*Est. 2–2.5 h.*

---

## Session 6 — The UI: search grid

**Goal:** A clean, fast web UI: search box → responsive photo grid, best match first, with scores, loading state, and a "no strong matches" treatment.

**Why it matters:** This is what people *see* — including recruiters who will never read the code. The three UX cases the brief demands (loading, no-matches, scores) are also where you first confront how CLIP scores actually behave.

**Concepts:**
- **What is a "good" cosine score?** With CLIP ViT-B-32, strong matches typically land around 0.28–0.35, decent ones 0.22–0.28, and noise below ~0.20 — but these bands are *empirical, not universal*. We'll calibrate: run 15–20 queries (half sensible, half absurd like "purple elephant playing chess"), look at the top-1 score distributions, and pick a threshold below which the UI says "no strong matches — showing closest anyway." Displaying raw scores plus a calibrated verdict is more honest (and more impressive) than fake percentages.
- **BlurHash.** The dataset ships a `blur_hash` per photo — a ~30-char string encoding a blurry preview. Decode it client-side (tiny JS lib) into instant placeholders while CDN images load. Photographer-grade polish, nearly free.

**What we build:** `web/index.html + app.js + style.css` (vanilla or a sprinkle of Alpine — your call, you're the frontend dev; no build step): debounced search box, CSS grid of results with score badge, photographer credit linking to the Unsplash photo page (attribution etiquette + required UTM params), skeleton/BlurHash loading state, empty-state message, and an `onerror` fallback for the occasional 404'd photo. Serve via FastAPI `StaticFiles`. Thumbnails via `{photo_image_url}?w=400&auto=format&q=75`; a click opens a larger `w=1200` view with full EXIF displayed.

**Definition of done:** Demo-able end-to-end MVP: type → grid updates → scores visible; absurd queries trigger the no-matches treatment; throttled network shows BlurHash placeholders. **This is the MVP milestone — tag it `v1.0` in git.** Record a first demo GIF now as insurance.

*Est. 2–3 h.*

---

## Session 7 — The differentiator: EXIF-aware search with Chroma (backend)

**Goal:** The same `SearchService` running on a real vector DB: semantic query **AND** aperture ≤ 2.0 **AND** ISO ≤ 800 via Chroma's fused vector-plus-metadata query — swappable with the NumPy store by config, returning *identical scores*.

**Why it matters:** This is the feature that separates your project from every "CLIP search in 50 lines" tutorial — it fuses your photography domain expertise with the AI skill. It's also your hands-on vector-DB experience (interviews *will* ask "have you used a vector database?").

**Deps for this session:** `uv add chromadb`

**Concepts:**
- **What a vector DB actually adds** over our NumPy file: persistence, incremental add/delete, **metadata filtering fused with vector search**, and ANN indexing (HNSW) — plus, in server-mode DBs, scaling beyond one machine. We adopt Chroma *specifically* for the filtering and incremental adds (Session 9 needs them); at 25k vectors the ANN part is incidental.
- **Pre-filtering vs post-filtering.** Post-filter (search top-K, then drop non-matching) breaks with selective filters: ask for top-50 then filter to f/1.8 and you may keep 2 results. Pre-filter (restrict candidate set *first*, then rank) is correct; Chroma's `where=` does this. You already built the same idea as `NumpyStore`'s mask — same concept, DB-grade.
- **Distance ≠ similarity — configure your space.** Chroma's default distance metric is **L2**, and `query()` returns *distances* (lower = better, scale ~0–2), while our whole app speaks *cosine similarity* (higher = better, ~0.2–0.35). Same ranking on normalized vectors, different numbers — which would silently break the Session 6 threshold and score badges. Fix at the source: create the collection with cosine space (`metadata={"hnsw:space": "cosine"}`) and convert in the store: `similarity = 1 - distance`. "Check what number your DB actually returns" is a lesson cheaper learned here than in production.
- **Filters must be typed at ingest.** Chroma's `$lt/$gte` operators only work on values stored as numbers — which is why Session 2 parsed `"f/1.8"` into `1.8`. Strings silently match nothing; typed ingest is the fix (a war story for DECISIONS.md).

**What we build:**
- `scripts/04_ingest_chroma.py` — `PersistentClient(path="data/chroma")`, one collection created with cosine space; add embeddings + **only the filterable fields** (aperture, iso, focal_length, exposure_s, camera_make/model) as metadata, in batches of ~5,000 (Chroma caps batch size), skipping NaN fields per-photo. Display fields (URLs, photographer, blur_hash) stay in `photos.parquet` — `ChromaStore` joins results back by `photo_id`, keeping one source of truth for rendering data.
- `src/photosearch/store.py` — add `ChromaStore` implementing the same `search(query_vec, k, filters)` interface: FilterSpec → `where={"$and": [...]}`, distances → `1 - d`, join to parquet for display fields. Store selected by config/env var.
- API: the FilterSpec params from Session 5 now route through whichever store is active. Handle the null problem explicitly: photos without EXIF can't match numeric filters (they're excluded when a filter is active) — expose "searching N photos with EXIF data" in the API response so Session 8's UI can show it.

**Definition of done:** In the Swagger UI, "night street" with and without `aperture_max=2.0` returns visibly different, correct results; **the same unfiltered query returns the same top-10 with ~equal scores from both stores** (the parity check that proves the seam); you can articulate pre- vs post-filtering and L2-vs-cosine without notes.

*Est. 2–2.5 h.*

---

## Session 8 — Filter UI + image-to-image search

**Goal:** The photographer's filter panel in the web UI, then two search features: "more like this" on every result, and upload-an-image search.

**Why it matters:** The filter panel is the *screenshot* that goes at the top of the README — the visible form of the differentiator. And image-to-image demolishes any lingering magic: images and text are *both* just vectors, so "find photos like this photo" is the same dot product with a different query vector.

**Concepts:** none new — that's the lesson. "More like this" doesn't even need the encoder: the query vector is the clicked photo's *stored* embedding (its nearest neighbor is itself — skip result #1). Upload search runs the image through `encoder.encode_image()` — the *indexing-side* encoder — at query time.

**What we build:**
- UI: collapsible "photographer's filters" panel (aperture ≤, ISO ≤, focal range, camera make) wired to the FilterSpec params; the "searching N photos with EXIF data" corpus note from Session 7's API, so the shrunken candidate set is visible, not mysterious.
- `GET /api/similar/{photo_id}` (stored-embedding lookup) + a "more like this" button on every result card.
- `POST /api/search/by-image` (multipart upload → PIL → encode → same search path, *including* filters — "photos like this one, but shot wide open") + a drag-and-drop upload zone.

**Definition of done:** "night street" toggling aperture ≤ 2.0 visibly changes the grid and the displayed EXIF respects the filter; "more like this" on a portrait yields portraits; uploading one of your own photos returns stylistically sensible matches. Update the demo GIF — this version has the differentiator in it.

*Est. 2–2.5 h.*

---

## Session 9a — Your own photo library: the ingestion pipeline

**Goal:** `scripts/05_index_folder.py --path "D:\Photos"` — scan a real folder, read real EXIF, thumbnail, and incrementally embed into its own Chroma collection.

**Why it matters:** Personal utility was a stated goal — and "I run it on my own photo archive" beats any canned demo in an interview. It also forces the ingestion code to generalize beyond one dataset, which is where the architecture proves itself.

**Deps for this session:** `uv add pillow-heif` (vanilla Pillow cannot open HEIC — one `register_heif_opener()` call at startup and iPhone photos work everywhere, both for EXIF reading and embedding).

**Concepts:**
- **EXIF from real files.** No TSV this time — we read EXIF straight from JPEG/HEIC via Pillow and map the camera tags (FNumber, ISOSpeedRatings, FocalLength, ExposureTime, Make/Model, DateTimeOriginal) into the *same* schema Session 2 defined — reusing `photosearch/exif.py`. Real EXIF is messier than the TSV (rationals, missing tags, weird vendor values): the parsers earn their tests here. This is the `PhotoSource` seam: same downstream code, new source.
- **Incremental indexing.** Re-embedding everything when 50 photos were added is silly. Track file path + mtime (or content hash); embed only new/changed; remove deleted. Chroma's add/delete-by-id makes this natural — and it's the honest answer to "how would you keep a production index fresh?"
- **Thumbnails.** No CDN now: generate small thumbnails at index time (Pillow `.thumbnail()`), so the grid stays fast; originals never leave your machine.

**What we build:** recursive scan → EXIF extraction → thumbnail generation → incremental embed into a `library` Chroma collection (file path + mtime tracked in the collection metadata or a sidecar parquet).

**Definition of done:** First run indexes your archive; dropping 5 new photos into the folder and re-running embeds *only those 5*; deleting one removes it from the index.

*Est. 2–3 h (plus unattended indexing time proportional to your archive size).*

---

## Session 9b — Your own photo library: serving + source toggle

**Goal:** Search *your* archive in the same UI, switchable between Unsplash and My Library.

**What we build:** `/api/photo/{id}/thumb` and `/full` endpoints serving local files (FastAPI `FileResponse`); a `source` toggle (config + UI) selecting which store/collection backs the search; result cards render local thumbs instead of CDN URLs when in library mode (EXIF filters work identically — same FilterSpec, same schema).

**Definition of done:** "that beach sunset" finds your beach sunsets; aperture filters work from real camera metadata; the toggle swaps corpora without a restart.

*Est. 1.5–2 h.*

---

## Session 10 — Evaluation: prove it works (and know its limits)

**Goal:** An honest retrieval-quality harness: a labeled query set, Precision@10 / Recall@10 / MRR — for CLIP **and for a keyword baseline** — plus a latency benchmark. One command, results in the README.

**Why it matters:** "How did you evaluate it?" is the question that separates people who *shipped a demo* from people who *did AI engineering*. And a metric without a baseline can't answer the question interviewers actually probe: *how do you know semantic search beats plain keyword search?*

**Deps for this session:** `uv add rank-bm25`

**Concepts:**
- **Relevance judgments.** For each test query, a hand-labeled set of photo IDs that count as correct. Building even a small "gold set" teaches you why eval is expensive and why relevance is genuinely fuzzy (is a sunrise a "golden hour" match? Decide, write the policy down).
- **The metrics.** **Precision@K**: of the top K results, what fraction is relevant? (User's view of quality.) **Recall@K**: of all relevant photos, what fraction appeared in top K? (Coverage — needs to know "all relevant", which at 25k photos we approximate by **pooling**.) **MRR** (mean reciprocal rank): 1/rank of the first relevant hit, averaged — "how far down is the first good result?"
- **Pooling, concretely for this build:** for each query, label the union of top-20 results from (a) the CLIP query itself, (b) one or two manual rephrasings of it, and (c) a keyword match over `ai_description`. Honest caveat to note in the README: pooling from your own systems makes Recall@K *optimistic* — anything none of them surfaced is invisible. Knowing this limitation is itself a concept interviewers respect.
- **The baseline.** **BM25** is the classic keyword-ranking algorithm (the heart of Elasticsearch): term-frequency scoring over the photo's `ai_description`/`photo_description` text. It's ~30 lines with `rank-bm25`, runs through the identical harness, and turns your README from "I measured myself" into "CLIP beats a BM25-over-captions baseline by X points — and here are the compositional queries where the gap is largest."
- **Failure analysis** — the underrated half: for the worst queries, *look at what CLIP got wrong* (counting, negation "street without people", text-in-image, fine-grained distinctions are classic CLIP weaknesses). A README section on known failure modes signals maturity louder than good scores do.

**What we build:** `eval/queries.jsonl` (~25 queries across easy/compositional/abstract/photography-jargon buckets); a tiny labeling helper (shows a query's pooled top-20 in a grid, Y/N per photo → saves judgments); `eval/run_eval.py` printing a metrics table for `--system clip|bm25` and `--store numpy|chroma`; a latency pass (median encode-ms and search-ms per store across the eval queries — the `--time` plumbing from Session 4 already exists). Results — metrics table, latency table, 3 documented failure cases — go into the README.

**Definition of done:** One command prints CLIP-vs-BM25 metrics and the NumPy-vs-Chroma latency table; you know your P@10 and by how much CLIP beats keywords; you have three failure examples you can discuss fluently.

*Est. 2.5–3 h (labeling is most of it).*

---

## Session 11 — Deployment: the live demo (HF Space)

**Goal:** A public URL anyone can try: a Gradio app on a Hugging Face Space (free ZeroGPU tier), searching the 25k Unsplash corpus — visually consistent with your real UI.

**Why it matters:** "Live demo link" is a portfolio requirement, and deployment constraints are where AI engineering gets real: model size vs RAM, cold starts, what gets shipped vs recomputed.

**Deps for this session:** `uv add --group space gradio` (a dependency *group* — like npm devDependencies — since the main app doesn't need it).

**Concepts:**
- **The 2026 free-hosting landscape (verified):** HF free accounts get up to **2 Gradio-SDK Spaces on ZeroGPU** (account 30+ days old). ZeroGPU allocates GPU *per-decorated-call* — but we simply **don't use the GPU**: undecorated code runs on the Space's host CPU, text encoding takes ~tens of ms there, and we consume zero visitor GPU quota. (Insurance: include one trivial `@spaces.GPU`-decorated no-op the app never calls — the decorator is documented as effect-free when unused, and this sidesteps any "no GPU function detected" startup validation.) Docker/CPU Spaces need PRO ($9/mo); Railway/Fly have no free tier anymore; Render's free tier survives (Session 11b).
- **A Space is its own git repo.** Your `space/` folder doesn't magically deploy — the Space expects `app.py` at *its* repo root with its own `requirements.txt`. Our strategy: a tiny `scripts/sync_space.py` that copies `app.py`, the `photosearch/` package source, `embeddings.f16.npy`, and the slim display parquet into a local clone of the Space repo; you review and `git push` there. (Requirements caveats: Spaces install from PyPI, so strip the `+cpu` local tag when transcribing pins from uv.lock — `torch==2.13.0`, not `torch==2.13.0+cpu` — and remember `gradio` itself.)
- **Ship precomputed, compute only the tiny thing.** The Space gets the embeddings as **float16** (halves 51 MB → ~26 MB, negligible cosine error — easy DECISIONS.md entry) plus a display parquet trimmed to the minimum: photo_id, image URL, photo page URL, photographer name, blur_hash, numeric EXIF. At query time only the *text* encoder runs; the 25k images never touch the server — the browser hotlinks the Unsplash CDN. **One subtlety: fp16 is a storage format, not a compute format** — NumPy has no fast fp16 matmul, so load with `.astype(np.float32)` at startup (still only ~51 MB RAM) or searches go from milliseconds to hundreds of milliseconds.
- **The licensing judgment call — made explicitly.** The Lite dataset's terms prohibit republishing the *Licensed Data*; the public Space ships derived embeddings plus that minimal display parquet. Handle it like a senior engineer: document the reasoning in DECISIONS.md and the README's licensing section — what's published and why (model outputs + the minimum needed to render attribution; images hotlinked per Unsplash's own requirement; no TSV export; corpus not reconstructible), plus a takedown contact note. A reasoned, documented call reads as judgment; the same files with no commentary read as carelessness.
- **Cold starts.** Free Spaces sleep after ~48 h idle; first visitor waits ~1–3 min while the container rebuilds and the model loads. Mitigation: a "may take a minute to wake" note next to the demo link.

**What we build:** `space/app.py` — Gradio Blocks reusing the `photosearch` core with `NumpyStore` + FilterSpec (filters work without Chroma — the Session 4 seam pays off): search box, filter sliders, results via `gr.HTML` **styled to mirror the web UI's cards** (same layout, score badge, attribution) so the live demo matches the README GIF instead of reading as a bait-and-switch; the fp16 conversion + parquet-trimming script; `scripts/sync_space.py`; deploy and test.

**Definition of done:** `uv run python space/app.py` works **locally first** (never debug via push-and-wait-for-container-build); then the Space URL works in an incognito browser on your phone, search + filters return correct results, attribution links work; README carries the link + wake-up note; the licensing DECISIONS entry exists.

*Est. 2.5–3 h.*

---

## Session 11b — The "real API" deploy: FastAPI on Render with an ONNX encoder

**Goal:** Your *actual* FastAPI + HTML/JS app — the one in the GIF — publicly deployed on Render's free tier.

**Why it matters:** This is arguably the best engineering story in the whole project, and it deploys the polished UI recruiters otherwise never touch. Treat it as a full session, not an afterthought.

**Concepts:**
- **Quantization.** Storing model weights as 8-bit integers instead of 32-bit floats: ~4× smaller, tiny accuracy cost. It's how a 254 MB text encoder becomes 64 MB — the difference between OOM and fitting a 512 MB free tier.
- **ONNX.** A framework-neutral saved-model format: export once, run anywhere with `onnxruntime` — **no PyTorch at all** (torch alone would blow Render's RAM budget). Reading a model repo's file listing (`Xenova/clip-vit-base-patch32/onnx/` → `text_model_quantized.onnx`, 64 MB) like a menu is a real skill.
- **The constraint math:** Render free = 512 MB RAM / 0.1 vCPU / 15-min idle spin-down (~60 s cold start). ONNX text encoder (64 MB) + fp32 embeddings (51 MB) + FastAPI ≈ comfortable; encode takes a few hundred ms on 0.1 vCPU — fine.

**What we build:** an alternative `OnnxTextEncoder` (onnxruntime + `tokenizers` — same 512-dim space, verified against the sentence-transformers encoder on a few queries: cosine agreement > 0.99); a Dockerfile or Render Python service config; a frontend "waking the server…" state for cold starts; README gets both links, honestly labeled (*Gradio demo — instant-ish; full app on Render — may take ~1 min to wake*).

**Definition of done:** The Render URL serves the real UI end-to-end; the encoder-parity check passes; cold-start UX is handled, not suffered.

*Est. 2–3 h.*

---

## Session 12 — Portfolio polish: README, DECISIONS, reproducibility, the story

**Goal:** The repo reads like a portfolio piece in a 90-second recruiter skim *and* holds up under a senior engineer's 15-minute read — and a stranger can actually run it.

**What we build:**

**Reproducibility (the missing link most portfolio repos fail):** `data/` is gitignored and rebuilding the index takes hours — so "clone and run" needs an artifact path. Publish `embeddings.f16.npy` + the slim display parquet (the exact files the Space already ships publicly) as a **GitHub Release asset**, and add `scripts/download_artifacts.py`. Now the README quickstart is honest: *clone → `uv sync` → `uv run python scripts/download_artifacts.py` → `uv run fastapi dev` → searching in under 10 minutes* (with "rebuild the full index yourself" documented as the overnight alternative). Record the dataset snapshot date (the zip's Last-Modified, from Session 2) in the README so the eval labels are tied to a frozen corpus.

**README.md** (structure):
1. One-paragraph pitch + **demo GIF** (search → results → aperture filter changes them; ~15 s; ScreenToGif on Windows) + both live links (Gradio Space; Render full app with wake note) + badge row (CI, license, demo).
2. "How it works" — the architecture diagram from this plan, plus a 5-sentence explanation of CLIP + cosine search written *in your own words* (write it from memory as a self-test, then fact-check it).
3. Features table (semantic search, EXIF filters ★, image-to-image, own-library mode, eval harness).
4. Evaluation — CLIP vs BM25 metrics table, latency table (NumPy vs Chroma), 3 failure cases with thumbnails, the pooling caveat.
5. Quickstart (the 3-command artifact path above; tested on a clean clone).
6. Dataset & licensing section (Unsplash Lite terms, what's published and why, hotlinking, takedown note).
7. Tech + "read DECISIONS.md for the why".

**Repo hygiene (15 minutes, outsized signal):** MIT `LICENSE` file; GitHub About description + topics (`clip`, `semantic-search`, `vector-search`, `embeddings`, `fastapi`, `chromadb`); pin the demo GIF so it renders above the fold.

**DECISIONS.md** — you've been writing it since Session 0; final pass for narrative quality. The entries that carry interview weight: sentence-transformers over raw transformers; ViT-B-32 over L-14 (deploy-constraint math); NumPy-first → Chroma-for-filters → Qdrant-at-scale (the graduation ladder); FilterSpec as the store seam; pre- vs post-filtering; L2-vs-cosine score scales; fp16 storage vs fp32 compute; the ONNX-quantized encoder for the 512 MB tier; typed-metadata-at-ingest; the licensing call.

**The 2-minute interview story** — write it down, say it out loud: *problem → shared embedding space → frozen model, index once → cosine = dot product, measured at ~3 ms over 25k → the EXIF twist nobody else has → evaluated against a BM25 baseline at P@10=X with known failure modes → deployed twice under free-tier constraints (fp16 shipping, ONNX quantization, cold-start UX).* Every noun in that sentence is something you built with your hands.

**Definition of done:** A stranger goes from `git clone` to local search in under 10 minutes using only the README; CI badge green; the GIF plays at the top; both live links work; you can deliver the story without looking.

*Est. 2–3 h.*

---

## After v1: where this goes next (optional roadmap)

- **Better model, same architecture:** re-index with an open_clip LAION checkpoint (e.g. ViT-B-32 trained on LAION-2B — noticeably better retrieval, same 512-dim interface) and let your eval harness *prove* the improvement. The cleanest possible demonstration of why eval matters.
- **Hybrid ranking:** blend cosine score with BM25-over-captions (you built both halves in Session 10) — hybrid search is the current industry default, and you'd be implementing it from parts you understand.
- **Natural-language filters:** parse "at f/1.8 on my 35mm" into a FilterSpec (regex first; an LLM call is the sledgehammer version).
- **Scale story:** load-test, then actually migrate to Qdrant and write the before/after post.

---

*Build order recap: 0 setup → 1 CLIP hello → 2 data + first tests → 3 index → 4 search core + FilterSpec → 5 API + CI → 6 UI (=MVP, tag v1.0) → 7 Chroma backend → 8 filter UI + image-to-image → 9a/9b own library → 10 eval + baseline → 11 HF Space → 11b Render/ONNX → 12 polish.*
