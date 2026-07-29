---
title: latent · semantic photo search
emoji: 🌅
colorFrom: orange
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Search 25k Unsplash photos by meaning, with EXIF filters.
---

# latent — semantic photo search

Type a scene, a mood, a feeling. [CLIP](https://openai.com/research/clip) ViT-B/32
encodes your words into the same 512-dimensional space as 25,000 Unsplash
photographs, and cosine similarity ranks every frame against it — including ones
nobody ever tagged.

The twist this demo adds over the usual CLIP search: **EXIF filters**. Ask for a
mood *and* a shooting style — "night street" at f/2.0 or wider, ISO ≤ 800.

## How it works

Indexing ran once, offline: every photo through the frozen image encoder, 512
floats saved. Searching runs the *text* encoder on your query (~tens of ms on this
CPU) and takes one matrix–vector product against the stored matrix (~5 ms for all
25k). No training, no fine-tuning, no GPU.

The embeddings ship as **float16** (26 MB instead of 51 MB) and are converted back
to float32 at load — fp16 is a storage format, not a compute one; NumPy has no fast
half-precision matmul.

Filters are pre-applied, not post-applied: the candidate set is restricted *first*,
then ranked, so asking for 30 results at f/1.8 returns 30 rather than whatever
survives. A photo with no aperture recorded can never match an aperture filter, so
an active filter searches a smaller universe than the full corpus — the panel says
how much smaller.

## Source & write-up

Full project — FastAPI backend, web UI, Chroma store, own-library indexing, and an
evaluation harness comparing CLIP against a BM25 keyword baseline:
**https://github.com/greisikertuka/semantic-photo-search**

## Photographs & licensing

Images come from the [Unsplash Lite dataset](https://unsplash.com/data) and are
**hotlinked from Unsplash's own CDN** — that is how Unsplash asks for them to be
used, and no image bytes are stored or served here. The dataset's terms permit use
but prohibit republishing the Licensed Data, so this Space ships only *derived*
artifacts: the CLIP embeddings (model outputs) plus the minimum metadata needed to
render a result and credit its photographer — id, image URL, photo page URL, name,
blur hash, dimensions, and numeric EXIF. No TSV, no captions, no bulk export; the
corpus is not reconstructible from what is published here. Every result links back
to the photographer's page on Unsplash. Takedown or attribution requests: open an
issue on the GitHub repo above.

*This Space sleeps after ~48 h idle — the first visit after that may take a minute
to wake.*
