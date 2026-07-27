# Semantic Photo Search

Search 25,000 Unsplash photos by **meaning, not tags**: type *"golden hour by the sea"* and get ranked, relevant photos — powered by CLIP embeddings and vector search.

> 🚧 **Work in progress.** Built session-by-session following [`PLAN.md`](PLAN.md). This README grows into the full portfolio write-up in Session 12; for now it's a signpost.

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
- [ ] Session 1 — CLIP hello-world experiment
- [ ] Sessions 2–12 — data, indexing, search, API, UI, filters, eval, deploy, polish

## License

MIT (see `LICENSE`, added in Session 12). Dataset & photo licensing documented there — the Unsplash Lite data is **not** redistributed in this repo.
