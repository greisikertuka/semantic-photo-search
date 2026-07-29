"""Session 11 — the public demo: a Gradio app on a Hugging Face Space.

Run it locally first, always:

    uv run --group space python space/app.py

The whole point of this file is how *little* of it is new. It imports the same
``photosearch`` package the FastAPI app uses, builds the same ``NumpyStore``, and
speaks the same ``FilterSpec`` — so the deployed demo filters by aperture and ISO
with **no Chroma in the container at all**. That is the Session 4 seam paying off:
the store is a config choice, and the smallest one fits the free tier.

What deployment actually forced us to change (each one a DECISIONS.md entry):

* **fp16 on disk, fp32 in RAM.** The embeddings ship as float16 (26 MB instead of
  51 MB) and are converted back at load. NumPy has no fast fp16 matmul, so keeping
  them half-precision in memory would turn a 5 ms search into a few hundred ms.
* **Ship precomputed; compute only the tiny thing.** The 25k images never touch
  this server — the browser hotlinks Unsplash's CDN, exactly as Unsplash asks. The
  only model that runs here is the CLIP *text* encoder, on CPU, ~tens of ms.
* **ZeroGPU without using the GPU.** Free Spaces in 2026 are Gradio-on-ZeroGPU
  only. Undecorated code runs on the host CPU, which is all we need, so visitors
  burn zero GPU quota. The one ``@spaces.GPU`` function below is never called — it
  exists so the platform's "no GPU function found" startup validation stays happy.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

# On the Space, ``photosearch/`` sits next to this file (scripts/sync_space.py puts
# it there). In the repo it lives under ``src/``. Support both so "works locally"
# means the same code, not a lookalike.
_HERE = Path(__file__).resolve().parent
for candidate in (_HERE, _HERE.parent / "src"):
    if (candidate / "photosearch" / "__init__.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from photosearch.models import FilterSpec
from photosearch.store import NumpyStore

# --- ZeroGPU insurance -------------------------------------------------------
# Deliberately never invoked: we want the host CPU, not a GPU slice. The decorator
# is documented as inert when the function isn't called, and its mere presence
# satisfies the Space's startup check.
try:  # pragma: no cover - only importable on a Space
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_probe() -> None:
        """Never called. See the module docstring."""

except ImportError:
    pass

# --- calibration (identical to web/app.js — one calibration, two frontends) ---
STRONG = 0.28  # ember badge
DECENT = 0.24  # gold badge
WEAK_TOP = 0.26  # if the *best* hit is below this, say so

IMG_GRID = "?w=480&auto=format&q=75"
UTM = "utm_source=latent_photo_search&utm_medium=referral"

RESULTS = 30
EXAMPLES = [
    "golden hour by the sea",
    "foggy empty street",
    "a man in a red jacket on a mountain",
    "rain on a neon window at night",
    "quiet minimalist interior",
    "long exposure water, shot wide open",
]

APERTURES = ["any", "1.4", "1.8", "2", "2.8", "4", "5.6", "8", "11", "16"]
ISOS = ["any", "100", "200", "400", "800", "1600", "3200", "6400"]
MAKES = ["any", "Canon", "Nikon", "Sony", "Fujifilm", "Panasonic", "Olympus",
         "Leica", "Apple", "Google", "DJI"]


def artifact_dir() -> Path:
    """Where the three shipped artifacts live — Space layout or repo layout."""
    override = os.environ.get("PHOTOSEARCH_SPACE_DATA")
    for candidate in (
        Path(override) if override else None,
        _HERE / "data",  # the Space: artifacts sit beside app.py
        _HERE.parent / "data" / "space",  # the repo: what 06_build_space_artifacts.py wrote
    ):
        if candidate is not None and (candidate / "embeddings.f16.npy").is_file():
            return candidate
    raise SystemExit(
        "no Space artifacts found — run:\n"
        "  uv run python scripts/06_build_space_artifacts.py"
    )


def load_store() -> NumpyStore:
    """fp16 off disk, fp32 into RAM, then the ordinary NumpyStore.

    ``.astype(np.float32)`` is the load-bearing line: fp16 halves the download and
    the repo's release asset, but NumPy would emulate a half-precision matmul in
    software and search would go from milliseconds to hundreds of them.
    """
    data = artifact_dir()
    embeddings = np.load(data / "embeddings.f16.npy").astype(np.float32)
    photo_ids = np.load(data / "photo_ids.npy", allow_pickle=True)
    photos = pd.read_parquet(data / "photos.slim.parquet")
    # NumpyStore's constructor re-asserts row alignment element-wise, which is why
    # photo_ids.npy ships at all: a half-synced deploy dies here, loudly, instead of
    # serving confidently mismatched photos.
    return NumpyStore(embeddings, photo_ids, photos)


print("[space] loading artifacts...")
STORE = load_store()
print(f"[space] {STORE.count():,} photos ({STORE.exif_count:,} with EXIF)")

print("[space] loading CLIP text encoder...")
# Imported *after* the artifacts load on purpose: a missing/mismatched index should
# fail in seconds, not after a 600 MB model download.
from photosearch.encoder import Encoder

ENCODER = Encoder()
ENCODER.encode_text("warmup")  # visitor #1 should not pay for the first-call setup
print("[space] ready")


# --- rendering ---------------------------------------------------------------
def verdict(score: float) -> str:
    if score >= STRONG:
        return "strong"
    if score >= DECENT:
        return "decent"
    return "weak"


def shutter(seconds: float | None) -> str | None:
    """Exposure seconds -> the string a photographer would actually say."""
    if seconds is None:
        return None
    if seconds >= 1:
        return f"{seconds:g}s"
    return f"1/{round(1 / seconds)}s"


def escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def exif_line(r) -> str:
    """The one-line exposure summary under each frame, blanks omitted."""
    bits = []
    if r.aperture is not None:
        bits.append(f"f/{r.aperture:g}")
    shot = shutter(r.exposure_s)
    if shot:
        bits.append(shot)
    if r.iso is not None:
        bits.append(f"ISO&nbsp;{r.iso}")
    if r.focal_length is not None:
        bits.append(f"{r.focal_length:g}mm")
    return " · ".join(bits)


def card_html(r) -> str:
    """One result card — the same anatomy as the web UI's: badge, frame, credit."""
    ratio = 66.7
    if r.width and r.height:
        ratio = max(40.0, min(160.0, r.height / r.width * 100))
    v = verdict(r.score)
    score_text = f"{r.score:.3f}".lstrip("0")
    photo_page = f"{r.photo_url}?{UTM}"
    exif = exif_line(r)
    return f"""
<figure class="ps-card">
  <a class="ps-frame" href="{escape(photo_page)}" target="_blank" rel="noopener"
     style="padding-bottom:{ratio:.1f}%">
    <img src="{escape(r.photo_image_url + IMG_GRID)}" alt="" loading="lazy"
         onerror="this.closest('.ps-card').style.display='none'">
    <span class="ps-badge ps-{v}" title="cosine similarity {r.score:.4f} — {v} match"
      >{score_text}</span>
    <figcaption class="ps-credit">
      <span class="ps-by">by</span> {escape(r.photographer)}
      {f'<span class="ps-exif">{exif}</span>' if exif else ''}
    </figcaption>
  </a>
</figure>"""


def status_html(query: str, results: list, filtered: bool, encode_ms: float,
                search_ms: float) -> str:
    if not results:
        return (
            '<div class="ps-status"><span class="ps-warn">no frames match those '
            "filters — try relaxing them</span></div>"
        )
    parts = []
    if results[0].score < WEAK_TOP:
        parts.append(
            '<span class="ps-warn">⚠ no strong matches — showing the closest '
            "frames anyway</span>"
        )
    parts.append(
        f'<span class="ps-stat"><b>{len(results)}</b> frames · '
        f"encode <b>{encode_ms:.0f}</b>ms · search <b>{search_ms:.1f}</b>ms</span>"
    )
    if filtered:
        parts.append(
            f'<span class="ps-stat">filtered · searching '
            f"<b>{STORE.exif_count:,}</b> of {STORE.count():,} frames with EXIF</span>"
        )
    return f'<div class="ps-status">{"".join(parts)}</div>'


EMPTY_STATE = """
<div class="ps-empty">
  Type a scene, a mood, a feeling — <span>golden hour by the sea</span>,
  <span>foggy empty street</span>. CLIP finds the closest frames across
  25,000 photographs, including ones no one ever tagged.
</div>"""


# --- the search callback -----------------------------------------------------
def as_float(value: str) -> float | None:
    return None if value in (None, "", "any") else float(value)


def run_search(query: str, aperture: str, iso: str, focal_min, focal_max,
               make: str) -> str:
    query = (query or "").strip()
    if not query:
        return EMPTY_STATE

    filters = FilterSpec(
        aperture_max=as_float(aperture),
        iso_max=int(float(iso)) if iso not in (None, "", "any") else None,
        focal_min=float(focal_min) if focal_min else None,
        focal_max=float(focal_max) if focal_max else None,
        camera_make=None if make in (None, "", "any") else make,
    )

    t0 = time.perf_counter()
    query_vec = ENCODER.encode_text(query)
    t1 = time.perf_counter()
    results = STORE.search(query_vec, k=RESULTS, filters=filters)
    t2 = time.perf_counter()

    status = status_html(query, results, filters.is_active(),
                         (t1 - t0) * 1000, (t2 - t1) * 1000)
    if not results:
        return status
    cards = "".join(card_html(r) for r in results)
    return f'{status}<div class="ps-grid">{cards}</div>'


def clear_filters():
    return "any", "any", None, None, "any"


# --- the interface -----------------------------------------------------------
# Editorial darkroom, same palette and type as web/style.css, so the live demo and
# the README GIF are recognisably the same product rather than a bait-and-switch.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..700;1,9..144,300..600&family=Archivo:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root, .gradio-container {
  --ps-bg: #0e0c0a; --ps-surface: #1a1611; --ps-surface-2: #221d16;
  --ps-line: #2c261d; --ps-line-2: #3a3327;
  --ps-text: #f4efe3; --ps-muted: #9c9384; --ps-faint: #6d6557;
  --ps-ember: #e8863f; --ps-ember-deep: #c26a2a; --ps-gold: #cba24c; --ps-grey: #877f70;
}
.gradio-container, body, gradio-app {
  background: radial-gradient(120% 90% at 50% -10%, #131009 0%, var(--ps-bg) 55%) fixed, var(--ps-bg) !important;
  color: var(--ps-text) !important;
  font-family: "Archivo", system-ui, sans-serif !important;
}
.gradio-container { max-width: 1360px !important; }
footer { display: none !important; }

.ps-head { padding: 26px 0 6px; }
.ps-wordmark { font-family: "Fraunces", Georgia, serif; font-size: 22px; font-weight: 600;
  letter-spacing: -.02em; color: var(--ps-text); }
.ps-wordmark span { color: var(--ps-ember); }
.ps-title { font-family: "Fraunces", Georgia, serif; font-weight: 300;
  font-size: clamp(30px, 5vw, 52px); line-height: 1.04; letter-spacing: -.025em;
  margin: 18px 0 14px; color: var(--ps-text); }
.ps-title em { font-style: italic; font-weight: 400; color: var(--ps-ember); }
.ps-sub { color: var(--ps-muted); max-width: 56ch; margin: 0 0 4px; font-size: 16px; }
.ps-sub a { color: var(--ps-muted); border-bottom: 1px solid var(--ps-line-2); text-decoration: none; }
.ps-sub a:hover { color: var(--ps-ember); }

/* search field — Fraunces, ember focus ring, matching the real app */
#ps-query textarea, #ps-query input {
  background: var(--ps-surface) !important; color: var(--ps-text) !important;
  border: 1px solid var(--ps-line) !important; border-radius: 4px !important;
  font-family: "Fraunces", Georgia, serif !important; font-weight: 300 !important;
  font-size: clamp(18px, 2.2vw, 23px) !important; padding: 15px 17px !important;
  box-shadow: none !important;
}
#ps-query textarea:focus, #ps-query input:focus {
  border-color: var(--ps-ember) !important;
  box-shadow: 0 0 0 1px var(--ps-ember), 0 18px 50px -24px rgba(232,134,63,.5) !important;
  background: var(--ps-surface-2) !important;
}
#ps-query textarea::placeholder { color: var(--ps-faint) !important; font-style: italic; }

#ps-go { background: var(--ps-ember) !important; color: #14100b !important;
  border: 0 !important; border-radius: 4px !important; font-weight: 600 !important;
  font-family: "Archivo", sans-serif !important; letter-spacing: .01em; }
#ps-go:hover { background: var(--ps-ember-deep) !important; }

/* filter panel */
.ps-filters { background: var(--ps-surface) !important; border: 1px solid var(--ps-line) !important;
  border-radius: 4px !important; }
.ps-filters label span, .ps-filters .gr-form label { font-family: "Space Mono", monospace !important;
  font-size: 10.5px !important; letter-spacing: .06em !important; text-transform: uppercase;
  color: var(--ps-faint) !important; }
.ps-filters input, .ps-filters select, .ps-filters .wrap-inner, .ps-filters .secondary-wrap {
  background: #131009 !important; color: var(--ps-text) !important;
  border-color: var(--ps-line-2) !important; }
.ps-clear { background: transparent !important; color: var(--ps-muted) !important;
  border: 1px solid var(--ps-line-2) !important; border-radius: 100px !important;
  font-size: 12px !important; }
.ps-clear:hover { color: var(--ps-ember) !important; border-color: var(--ps-ember-deep) !important; }
.ps-note { font-family: "Space Mono", monospace; font-size: 11px; color: var(--ps-faint); }

/* example chips */
.ps-examples { gap: 9px !important; flex-wrap: wrap !important; margin-top: 12px; }
.ps-chip, .ps-chip button {
  background: transparent !important; color: var(--ps-muted) !important;
  border: 1px solid var(--ps-line) !important; border-radius: 100px !important;
  font-family: "Archivo", sans-serif !important; font-size: 13px !important;
  font-weight: 400 !important; padding: 7px 15px !important;
  min-width: 0 !important; width: auto !important; box-shadow: none !important; }
.ps-chip:hover, .ps-chip button:hover {
  color: var(--ps-text) !important; border-color: var(--ps-ember-deep) !important;
  background: rgba(232,134,63,.07) !important; }

/* status line */
.ps-status { display: flex; flex-wrap: wrap; gap: 14px; min-height: 22px;
  margin: 10px 0 20px; font-family: "Space Mono", monospace; font-size: 12px;
  letter-spacing: .04em; color: var(--ps-faint); }
.ps-status .ps-warn { color: var(--ps-gold); }
.ps-status .ps-stat { color: var(--ps-muted); }
.ps-status b { color: var(--ps-text); font-weight: 700; }

/* the masonry grid — CSS columns, same as the real UI */
.ps-grid { column-count: 4; column-gap: 16px; }
@media (max-width: 1100px) { .ps-grid { column-count: 3; } }
@media (max-width: 720px)  { .ps-grid { column-count: 2; column-gap: 12px; } }
@media (max-width: 420px)  { .ps-grid { column-count: 1; } }

.ps-card { position: relative; margin: 0 0 16px; break-inside: avoid; border-radius: 4px;
  overflow: hidden; background: var(--ps-surface); border: 1px solid var(--ps-line);
  animation: ps-rise .5s cubic-bezier(.2,.7,.2,1) both; }
@keyframes ps-rise { from { opacity: 0; transform: translateY(8px); } }
.ps-card:hover { border-color: var(--ps-line-2); }
.ps-frame { position: relative; display: block; width: 100%; height: 0; }
.ps-frame img { position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: cover; display: block; }

.ps-badge { position: absolute; top: 10px; left: 10px; z-index: 2;
  font-family: "Space Mono", monospace; font-size: 11px; font-weight: 700;
  padding: 4px 8px; border-radius: 3px; background: rgba(14,12,10,.72);
  backdrop-filter: blur(6px); color: var(--ps-grey);
  border: 1px solid rgba(255,255,255,.08); }
.ps-badge.ps-strong { color: var(--ps-ember); border-color: rgba(232,134,63,.45); }
.ps-badge.ps-decent { color: var(--ps-gold); border-color: rgba(203,162,76,.4); }

.ps-credit { position: absolute; left: 0; right: 0; bottom: 0; z-index: 2;
  padding: 28px 12px 10px; font-size: 12px; color: #ece6da;
  background: linear-gradient(transparent, rgba(10,8,6,.88));
  opacity: 0; transform: translateY(6px);
  transition: opacity .25s ease, transform .25s ease; }
.ps-card:hover .ps-credit { opacity: 1; transform: translateY(0); }
.ps-credit .ps-by { color: var(--ps-muted); }
.ps-credit .ps-exif { display: block; margin-top: 3px;
  font-family: "Space Mono", monospace; font-size: 10.5px; color: var(--ps-muted); }

.ps-empty { font-family: "Fraunces", Georgia, serif; font-style: italic; font-weight: 300;
  font-size: 21px; line-height: 1.5; color: var(--ps-muted); padding: 34px 0; max-width: 52ch; }
.ps-empty span { color: var(--ps-text); font-style: normal; }

.ps-foot { border-top: 1px solid var(--ps-line); margin-top: 34px; padding: 22px 0 30px;
  display: flex; flex-wrap: wrap; gap: 10px 24px; justify-content: space-between;
  font-size: 12px; color: var(--ps-faint); }
.ps-foot a { color: var(--ps-muted); border-bottom: 1px solid var(--ps-line-2);
  text-decoration: none; }
.ps-foot a:hover { color: var(--ps-ember); }
.ps-foot .ps-tech { font-family: "Space Mono", monospace; letter-spacing: .04em; }

@media (prefers-reduced-motion: reduce) { * { animation-duration: .001ms !important;
  transition-duration: .001ms !important; } }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=["Archivo", "system-ui", "sans-serif"],
).set(body_background_fill="#0e0c0a", block_background_fill="transparent",
      block_border_width="0px", panel_background_fill="transparent")


# Gradio 6 moved `css` and `theme` off the Blocks constructor and onto `.launch()`
# (they're passed at the bottom of this file). `title` still belongs here.
with gr.Blocks(title="latent · semantic photo search", analytics_enabled=False) as demo:
    gr.HTML(
        '<div class="ps-head"><span class="ps-wordmark">latent<span>.</span></span>'
        '<h1 class="ps-title">Search photographs by <em>meaning</em>,<br>'
        "not by keyword.</h1>"
        '<p class="ps-sub">CLIP ViT-B/32 over 25,000 Unsplash photographs — plus '
        "EXIF filters, so you can ask for a mood <em>and</em> a shooting style. "
        'Cosine search runs in about 5&nbsp;ms; the wait is the text encoder.</p></div>'
    )

    with gr.Row():
        query = gr.Textbox(
            placeholder="golden hour by the sea…", label="", lines=1,
            elem_id="ps-query", scale=8, autofocus=True, submit_btn=False,
        )
        go = gr.Button("Search", variant="primary", elem_id="ps-go", scale=1)

    with gr.Accordion("Photographer's filters", open=False), gr.Group(elem_classes="ps-filters"):
        with gr.Row():
            aperture = gr.Dropdown(APERTURES, value="any", label="Aperture ≤")
            iso = gr.Dropdown(ISOS, value="any", label="ISO ≤")
            focal_min = gr.Number(label="Focal min (mm)", value=None, minimum=0)
            focal_max = gr.Number(label="Focal max (mm)", value=None, minimum=0)
            make = gr.Dropdown(MAKES, value="any", label="Camera make")
        with gr.Row():
            gr.HTML(
                f'<span class="ps-note">{STORE.exif_count:,} of {STORE.count():,} '
                "frames carry EXIF — a photo with no aperture recorded can never "
                "match an aperture filter.</span>"
            )
            clear = gr.Button("Clear filters", size="sm", elem_classes="ps-clear", scale=0)

    # Plain buttons rather than gr.Examples: an example that only *fills* the box and
    # waits for a second click is a dead end on a demo people spend 20 seconds on.
    # These set the query and run it, exactly like the web UI's chips.
    with gr.Row(elem_classes="ps-examples"):
        chips = [gr.Button(text, size="sm", elem_classes="ps-chip") for text in EXAMPLES]

    out = gr.HTML(EMPTY_STATE)

    inputs = [query, aperture, iso, focal_min, focal_max, make]
    for chip, text in zip(chips, EXAMPLES, strict=True):
        chip.click(lambda t=text: t, None, query).then(run_search, inputs, out)
    # Every control re-runs the current query, so a filter change re-ranks what's on
    # screen instead of silently going stale — the same rule the web UI follows.
    query.submit(run_search, inputs, out)
    go.click(run_search, inputs, out)
    for control in (aperture, iso, make):
        control.change(run_search, inputs, out)
    for control in (focal_min, focal_max):
        control.submit(run_search, inputs, out)
    clear.click(clear_filters, None, [aperture, iso, focal_min, focal_max, make]).then(
        run_search, inputs, out
    )

    gr.HTML(
        '<div class="ps-foot"><span>Photographs via '
        '<a href="https://unsplash.com/data" target="_blank" rel="noopener">Unsplash '
        "Lite</a> · hotlinked from Unsplash's CDN, never redistributed. This demo "
        "ships only derived embeddings plus the minimum needed to credit each "
        'photographer.</span>'
        '<span class="ps-tech">CLIP ViT-B/32 · fp16 storage, fp32 compute · '
        "brute-force cosine over 25k vectors</span></div>"
    )


if __name__ == "__main__":
    # A Space runs this file as a script, so this is the launch the platform uses too.
    demo.launch(css=CSS, theme=THEME)
