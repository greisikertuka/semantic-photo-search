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

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from photosearch.models import FilterSpec, Result
from photosearch.search import SearchService


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
    encode_ms: float
    search_ms: float
    results: list[ResultModel]


class HealthResponse(BaseModel):
    status: str
    indexed: int


def build_service() -> SearchService:
    """Load the real store + CLIP encoder. Kept separate so lifespan can try/except it."""
    from photosearch.encoder import Encoder
    from photosearch.store import NumpyStore

    store = NumpyStore.load()
    encoder = Encoder()
    return SearchService(encoder, store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = build_service()
        # Warm the encoder so visitor #1 doesn't eat the first-call JIT/setup cost.
        service.search("warmup query", k=1)
        app.state.service = service
        indexed = len(service.store.photo_ids)
        print(f"[api] ready — {indexed:,} photos indexed")
    except FileNotFoundError as exc:
        # Index not built yet (or running without artifacts, e.g. in CI). Start anyway;
        # /api/search returns 503 until the artifacts exist. Tests override the service.
        app.state.service = None
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


def get_service() -> SearchService:
    """DI seam: the endpoints depend on this, tests override it with a stub service."""
    service = getattr(app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="search index not loaded")
    return service


@app.get("/api/health", response_model=HealthResponse)
def health(service: SearchService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(status="ok", indexed=len(service.store.photo_ids))


@app.get("/api/search", response_model=SearchResponse)
def search(
    service: SearchService = Depends(get_service),
    q: str = Query(..., min_length=1, description="natural-language query"),
    k: int = Query(12, ge=1, le=50, description="number of results"),
    aperture_max: float | None = Query(None, gt=0, description="keep f/<= this"),
    iso_max: int | None = Query(None, gt=0, description="keep ISO <= this"),
    focal_min: float | None = Query(None, gt=0, description="keep focal >= this (mm)"),
    focal_max: float | None = Query(None, gt=0, description="keep focal <= this (mm)"),
    camera_make: str | None = Query(None, description="exact camera make (case-insensitive)"),
) -> SearchResponse:
    filters = FilterSpec(
        aperture_max=aperture_max,
        iso_max=iso_max,
        focal_min=focal_min,
        focal_max=focal_max,
        camera_make=camera_make,
    )
    results, timing = service.search_timed(q, k=k, filters=filters)
    return SearchResponse(
        query=q,
        k=k,
        count=len(results),
        filtered=filters.is_active(),
        encode_ms=round(timing.encode_ms, 2),
        search_ms=round(timing.search_ms, 3),
        results=[ResultModel.from_result(r) for r in results],
    )
