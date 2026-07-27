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
