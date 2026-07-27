"""API tests with a stubbed encoder — no model download, no index artifacts.

The whole point is the dependency-injection seam made visible: we override
``get_service`` with a SearchService built from a *fixed-vector* encoder and the
same 6x4 synthetic store from test_store. So CI exercises the real HTTP layer
(routing, param validation, response shape, 422s) without any AI dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_store import EMB, IDS, PHOTOS

from photosearch.api import app, get_service
from photosearch.search import SearchService
from photosearch.store import NumpyStore


class StubEncoder:
    """Returns a fixed query vector so tests are deterministic and model-free."""

    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.vec


@pytest.fixture
def client() -> TestClient:
    store = NumpyStore(EMB.copy(), IDS.copy(), PHOTOS.copy())
    # Query aligned with p0 — same ordering the store tests assert on.
    service = SearchService(StubEncoder(np.array([1.0, 0.0, 0.0, 0.0])), store)
    app.dependency_overrides[get_service] = lambda: service
    # NB: no `with` — we deliberately DON'T run the lifespan, which would load the
    # real 600 MB model whenever the index artifacts happen to exist on disk. The
    # DI override is the whole point; the real service must never boot in tests.
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestSearchEndpoint:
    def test_returns_ranked_results(self, client: TestClient) -> None:
        resp = client.get("/api/search", params={"q": "anything", "k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert body["filtered"] is False
        assert [r["photo_id"] for r in body["results"]] == ["p0", "p4", "p5"]
        assert body["results"][0]["score"] == pytest.approx(1.0)

    def test_result_shape_has_display_fields(self, client: TestClient) -> None:
        body = client.get("/api/search", params={"q": "x", "k": 1}).json()
        top = body["results"][0]
        assert top["photographer"] == "Ada Byte"
        assert top["photo_image_url"] == "http://img/0"
        assert top["camera_make"] == "Canon"

    def test_filter_applies_and_flags_filtered(self, client: TestClient) -> None:
        body = client.get(
            "/api/search", params={"q": "x", "k": 10, "aperture_max": 2.0}
        ).json()
        assert body["filtered"] is True
        assert [r["photo_id"] for r in body["results"]] == ["p0", "p5", "p3"]

    def test_camera_make_filter(self, client: TestClient) -> None:
        body = client.get(
            "/api/search", params={"q": "x", "k": 10, "camera_make": "canon"}
        ).json()
        assert {r["photo_id"] for r in body["results"]} == {"p0", "p3", "p5"}

    def test_timings_present(self, client: TestClient) -> None:
        body = client.get("/api/search", params={"q": "x"}).json()
        assert "encode_ms" in body and "search_ms" in body


class TestValidation:
    def test_missing_query_is_422(self, client: TestClient) -> None:
        assert client.get("/api/search").status_code == 422

    def test_empty_query_is_422(self, client: TestClient) -> None:
        assert client.get("/api/search", params={"q": ""}).status_code == 422

    def test_k_over_max_is_422(self, client: TestClient) -> None:
        assert client.get("/api/search", params={"q": "x", "k": 999}).status_code == 422

    def test_k_zero_is_422(self, client: TestClient) -> None:
        assert client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422

    def test_negative_aperture_is_422(self, client: TestClient) -> None:
        resp = client.get("/api/search", params={"q": "x", "aperture_max": -1})
        assert resp.status_code == 422


class TestHealth:
    def test_health_reports_indexed_count(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body == {"status": "ok", "indexed": 6}


def test_search_unavailable_without_service() -> None:
    # No override and no lifespan (no `with`): app.state has no `service`, so
    # get_service sees None and returns 503 instead of crashing.
    app.dependency_overrides.clear()
    client = TestClient(app)
    assert client.get("/api/search", params={"q": "x"}).status_code == 503
