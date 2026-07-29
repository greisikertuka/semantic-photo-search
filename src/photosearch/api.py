"""Session 5 — the FastAPI backend that wraps the search core in HTTP + JSON.

Maps cleanly onto Spring Boot if that's your background: ``@app.get`` ≈
``@GetMapping``, the Pydantic response models ≈ DTOs with Bean-Validation (bad
params get automatic 422s), and ``/docs`` is Swagger UI for free.

Two deliberate design notes:

* **The model loads once, at startup** — never per request. That's the ``lifespan``
  context manager below: code before ``yield`` runs at startup (build the one
  SearchService, warm the encoder), code after runs at shutdown. Loading a 600 MB
  model per request would be absurd; this is the correct place for it.
* **The search endpoint is a plain ``def``, not ``async def``.** Inference is
  CPU-bound, so FastAPI runs a sync endpoint in its thread pool — honest, and it
  keeps the event loop free. Pretending it's ``async`` would help nothing.

Run locally:  ``uv run fastapi dev src/photosearch/api.py``
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from photosearch.models import FilterSpec, Result
from photosearch.search import SearchService

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"

# The 25k Unsplash corpus is always there; "library" appears only if Session 9a has
# indexed a local folder. The toggle is a *corpus* switch, not a store switch.
DEFAULT_SOURCE = "unsplash"
LIBRARY_SOURCE = "library"


class ResultModel(BaseModel):
    """One search hit, shaped for the frontend."""

    photo_id: str
    score: float
    photo_image_url: str
    photo_url: str
    photographer: str
    width: int | None
    height: int | None
    blur_hash: str | None
    description: str | None
    ai_description: str | None
    aperture: float | None
    focal_length: float | None
    exposure_s: float | None
    iso: int | None
    camera_make: str | None
    camera_model: str | None

    @classmethod
    def from_result(cls, r: Result) -> ResultModel:
        return cls(**r.__dict__)


class SearchResponse(BaseModel):
    query: str
    k: int
    count: int
    filtered: bool
    corpus: int  # total photos indexed
    exif_count: int  # of those, how many carry filterable EXIF (the filter's real universe)
    store: str  # which back-end answered — "numpy" or "chroma" (the seam, made visible)
    source: str  # which corpus answered — "unsplash" or "library"
    encode_ms: float
    search_ms: float
    results: list[ResultModel]


class HealthResponse(BaseModel):
    status: str
    indexed: int
    exif_count: int
    store: str
    source: str
    sources: list[str]


def build_service() -> SearchService:
    """Load the configured store + CLIP encoder. Separate so lifespan can try/except it.

    The store is chosen by ``PHOTOSEARCH_STORE`` (``numpy`` default, or ``chroma``) —
    the FilterSpec seam means neither the endpoints nor the encoder know or care which
    back-end answered.
    """
    from photosearch.encoder import Encoder
    from photosearch.store import load_store

    store = load_store()
    encoder = Encoder()
    return SearchService(encoder, store)


def build_services() -> dict[str, SearchService]:
    """Every corpus this process can search, keyed by source name.

    One ``Encoder`` is shared across all of them — the model is the expensive thing,
    and a query vector means the same thing in every collection, which is exactly why
    a second corpus costs almost nothing to add. The local library is optional: if
    Session 9a hasn't run, the toggle simply isn't offered.
    """
    unsplash = build_service()
    services = {DEFAULT_SOURCE: unsplash}
    try:
        from photosearch.library import LibraryStore

        library = LibraryStore.load()
        if library.count():
            services[LIBRARY_SOURCE] = SearchService(
                unsplash.encoder, library, source=LIBRARY_SOURCE
            )
            print(f"[api] local library available — {library.count():,} photos")
    except Exception as exc:  # noqa: BLE001 - "no library indexed" is the normal case
        print(f"[api] no local photo library ({type(exc).__name__}: {exc})")
    return services


def store_kind(store: object) -> str:
    """"NumpyStore" -> "numpy", "ChromaStore" -> "chroma" — for the response's `store`."""
    return type(store).__name__.removesuffix("Store").lower()


def make_response(
    query: str,
    k: int,
    results: list[Result],
    timing,
    filters: FilterSpec,
    service: SearchService,
) -> SearchResponse:
    """Shape a store result list into the wire response — shared by all search paths."""
    store = service.store
    return SearchResponse(
        query=query,
        k=k,
        count=len(results),
        filtered=filters.is_active(),
        corpus=store.count(),
        exif_count=getattr(store, "exif_count", 0),
        store=store_kind(store),
        source=getattr(service, "source", DEFAULT_SOURCE),
        encode_ms=round(timing.encode_ms, 2),
        search_ms=round(timing.search_ms, 3),
        results=[ResultModel.from_result(r) for r in results],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        services = build_services()
        primary = services[DEFAULT_SOURCE]
        # Warm the encoder so visitor #1 doesn't eat the first-call JIT/setup cost.
        primary.search("warmup query", k=1)
        app.state.services = services
        store = primary.store
        print(
            f"[api] ready — {store.count():,} photos indexed "
            f"({store_kind(store)} store, {getattr(store, 'exif_count', 0):,} with EXIF) "
            f"| sources: {', '.join(services)}"
        )
    except FileNotFoundError as exc:
        # Index not built yet (or running without artifacts, e.g. in CI). Start anyway;
        # /api/search returns 503 until the artifacts exist. Tests override the service.
        app.state.services = {}
        print(f"[api] WARNING: index artifacts not found ({exc}); search disabled")
    yield


app = FastAPI(
    title="Semantic Photo Search",
    description="Search 25k Unsplash photos by meaning, with EXIF filters.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public read-only API; tighten if this ever gets a write path
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def available_sources() -> dict[str, SearchService]:
    return getattr(app.state, "services", None) or {}


def get_service(
    source: str = Query(DEFAULT_SOURCE, description='corpus to search: "unsplash" or "library"'),
) -> SearchService:
    """DI seam: the endpoints depend on this, tests override it with a stub service.

    Since Session 9b it also resolves the ``?source=`` toggle. Both corpora are loaded
    at startup, so switching between them is a query param — no restart, no reload.
    """
    services = available_sources()
    if not services:
        raise HTTPException(status_code=503, detail="search index not loaded")
    service = services.get(source)
    if service is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown source {source!r} (available: {', '.join(services)})",
        )
    return service


def get_library():
    """The local-library store, for the file-serving endpoints. 404 if none is indexed."""
    service = available_sources().get(LIBRARY_SOURCE)
    if service is None:
        raise HTTPException(status_code=404, detail="no local photo library indexed")
    return service.store


@app.get("/api/health", response_model=HealthResponse)
def health(service: SearchService = Depends(get_service)) -> HealthResponse:
    store = service.store
    return HealthResponse(
        status="ok",
        indexed=store.count(),
        exif_count=getattr(store, "exif_count", 0),
        store=store_kind(store),
        source=getattr(service, "source", DEFAULT_SOURCE),
        sources=list(available_sources()) or [DEFAULT_SOURCE],
    )


# The EXIF FilterSpec params, declared once and reused by every search path. Returning
# a dependency object keeps the three endpoints' signatures short and identical.
def filter_params(
    aperture_max: float | None = Query(None, gt=0, description="keep f/<= this"),
    iso_max: int | None = Query(None, gt=0, description="keep ISO <= this"),
    focal_min: float | None = Query(None, gt=0, description="keep focal >= this (mm)"),
    focal_max: float | None = Query(None, gt=0, description="keep focal <= this (mm)"),
    camera_make: str | None = Query(None, description="exact camera make (case-insensitive)"),
) -> FilterSpec:
    return FilterSpec(
        aperture_max=aperture_max,
        iso_max=iso_max,
        focal_min=focal_min,
        focal_max=focal_max,
        camera_make=camera_make,
    )


@app.get("/api/search", response_model=SearchResponse)
def search(
    service: SearchService = Depends(get_service),
    q: str = Query(..., min_length=1, description="natural-language query"),
    k: int = Query(12, ge=1, le=50, description="number of results"),
    filters: FilterSpec = Depends(filter_params),
) -> SearchResponse:
    results, timing = service.search_timed(q, k=k, filters=filters)
    return make_response(q, k, results, timing, filters, service)


@app.get("/api/similar/{photo_id}", response_model=SearchResponse)
def similar(
    photo_id: str,
    service: SearchService = Depends(get_service),
    k: int = Query(12, ge=1, le=50, description="number of results"),
    filters: FilterSpec = Depends(filter_params),
) -> SearchResponse:
    """"More like this" — nearest neighbours of a photo by its own stored embedding."""
    results, timing = service.similar_timed(photo_id, k=k, filters=filters)
    if results is None:
        raise HTTPException(status_code=404, detail=f"photo {photo_id!r} not in the index")
    return make_response(f"similar:{photo_id}", k, results, timing, filters, service)


@app.post("/api/search/by-image", response_model=SearchResponse)
def search_by_image(
    service: SearchService = Depends(get_service),
    file: UploadFile = File(..., description="an image to search by"),
    k: int = Query(12, ge=1, le=50, description="number of results"),
    filters: FilterSpec = Depends(filter_params),
) -> SearchResponse:
    """Upload an image → encode it → same search path (filters included).

    The endpoint is a plain ``def``: reading the upload and CLIP-encoding it are both
    blocking, so FastAPI runs it in its thread pool (the sync ``file.file.read()`` is
    correct here — no event loop to await).
    """
    from PIL import Image

    raw = file.file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.load()
    except Exception as exc:  # any decode failure is a 400 (client's bad file), not a 500
        raise HTTPException(status_code=400, detail="could not read image file") from exc

    results, timing = service.search_by_image_timed(image, k=k, filters=filters)
    label = f"image:{file.filename or 'upload'}"
    return make_response(label, k, results, timing, filters, service)


# --- local library files (Session 9b) ---------------------------------------------
# The URL only ever carries a *photo id*; the server maps it to a path through the
# manifest the indexer wrote. A client can't name a file, so there is no traversal to
# defend against — the safest kind of file endpoint is one that takes no filename.
CACHE_A_DAY = {"Cache-Control": "public, max-age=86400"}


@app.get("/api/photo/{photo_id}/thumb")
def photo_thumb(photo_id: str, library=Depends(get_library)) -> FileResponse:
    """The index-time thumbnail — this is what fills the grid in library mode."""
    path = library.thumbnail_path(photo_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no thumbnail for {photo_id!r}")
    return FileResponse(path, media_type="image/jpeg", headers=CACHE_A_DAY)


@app.get("/api/photo/{photo_id}/full")
def photo_full(photo_id: str, library=Depends(get_library)) -> FileResponse:
    """The original file, straight off disk — used by the detail lightbox."""
    path = library.original_path(photo_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{photo_id!r} is not in the library")
    return FileResponse(path, headers=CACHE_A_DAY)


# Serve the static frontend (web/) at the root. Mounted LAST so the /api/* routes
# above take precedence; the mount is a catch-all for everything else. html=True
# makes "/" serve index.html. Guarded so importing the app without a web/ dir (e.g.
# in CI, or before Session 6) still works.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
