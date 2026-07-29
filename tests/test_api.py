"""API tests with a stubbed encoder — no model download, no index artifacts.

The whole point is the dependency-injection seam made visible: we override
``get_service`` with a SearchService built from a *fixed-vector* encoder and the
same 6x4 synthetic store from test_store. So CI exercises the real HTTP layer
(routing, param validation, response shape, 422s) without any AI dependency.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_store import EMB, IDS, PHOTOS

from photosearch.api import app, get_service
from photosearch.search import SearchService
from photosearch.store import NumpyStore


class StubEncoder:
    """Returns a fixed query vector so tests are deterministic and model-free.

    Both ``encode_text`` and ``encode_image`` return the same fixed vector — the point
    is that the *store* path is identical for text, image-upload, and "more like this".
    """

    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        return self.vec

    def encode_image(self, image: object) -> np.ndarray:
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

    def test_response_carries_corpus_and_store(self, client: TestClient) -> None:
        body = client.get("/api/search", params={"q": "x"}).json()
        assert body["corpus"] == 6  # the 6-photo fixture
        assert body["exif_count"] == 5  # p4 has no EXIF
        assert body["store"] == "numpy"


class TestSimilar:
    def test_more_like_this_excludes_the_photo_itself(self, client: TestClient) -> None:
        # p0's own embedding is [1,0,0,0]; its neighbours are p4 then p5, and p0
        # (itself, similarity 1.0) must be dropped from its own "more like this".
        body = client.get("/api/similar/p0", params={"k": 3}).json()
        ids = [r["photo_id"] for r in body["results"]]
        assert "p0" not in ids
        assert ids == ["p4", "p5", "p1"] or ids[:2] == ["p4", "p5"]
        assert body["query"] == "similar:p0"

    def test_unknown_photo_is_404(self, client: TestClient) -> None:
        assert client.get("/api/similar/nope").status_code == 404

    def test_similar_respects_filters(self, client: TestClient) -> None:
        # neighbours of p0 with f/<=2.0: p5(2.0),p3(1.4) qualify; p1(2.8) excluded.
        body = client.get("/api/similar/p0", params={"k": 5, "aperture_max": 2.0}).json()
        ids = {r["photo_id"] for r in body["results"]}
        assert "p1" not in ids
        assert ids <= {"p3", "p5"}


class TestSearchByImage:
    def _png_bytes(self) -> bytes:
        # a tiny valid PNG so PIL can decode it — content is irrelevant (encoder is stubbed)
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (120, 80, 40)).save(buf, format="PNG")
        return buf.getvalue()

    def test_by_image_runs_the_same_search_path(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/by-image",
            files={"file": ("probe.png", self._png_bytes(), "image/png")},
            params={"k": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        # stub encodes to [1,0,0,0] → same ranking as the text query
        assert [r["photo_id"] for r in body["results"]] == ["p0", "p4", "p5"]
        assert body["query"].startswith("image:")

    def test_by_image_applies_filters(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/by-image",
            files={"file": ("probe.png", self._png_bytes(), "image/png")},
            params={"k": 10, "aperture_max": 2.0},
        )
        assert resp.json()["filtered"] is True
        assert [r["photo_id"] for r in resp.json()["results"]] == ["p0", "p5", "p3"]

    def test_garbage_upload_is_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/by-image",
            files={"file": ("bad.png", b"not an image", "image/png")},
        )
        assert resp.status_code == 400


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
        assert body == {
            "status": "ok",
            "indexed": 6,
            "exif_count": 5,
            "store": "numpy",
            "source": "unsplash",
            "sources": ["unsplash"],
            "encoder": "stubencoder",
            "supports_images": True,
        }


class TestTextOnlyDeploy:
    """Session 11b: Render runs the CLIP *text* tower only — no vision model in RAM.

    The frontend reads ``supports_images`` from /api/health and hides the upload zone,
    so the 501 is a backstop rather than something a user is expected to hit.
    """

    @pytest.fixture
    def text_only_client(self) -> TestClient:
        encoder = StubEncoder(np.array([1.0, 0.0, 0.0, 0.0]))
        encoder.supports_images = False
        store = NumpyStore(EMB.copy(), IDS.copy(), PHOTOS.copy())
        app.dependency_overrides[get_service] = lambda: SearchService(encoder, store)
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()

    def test_health_advertises_no_image_support(self, text_only_client: TestClient) -> None:
        assert text_only_client.get("/api/health").json()["supports_images"] is False

    def test_by_image_is_501_not_500(self, text_only_client: TestClient) -> None:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        resp = text_only_client.post(
            "/api/search/by-image",
            files={"file": ("probe.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 501
        assert "text-only" in resp.json()["detail"]

    def test_text_search_is_unaffected(self, text_only_client: TestClient) -> None:
        body = text_only_client.get("/api/search", params={"q": "x", "k": 3}).json()
        assert [r["photo_id"] for r in body["results"]] == ["p0", "p4", "p5"]


class StubLibraryStore(NumpyStore):
    """A NumpyStore that also answers the two file lookups the endpoints need.

    Standing in for ``LibraryStore`` keeps these tests free of Chroma while still
    exercising the real ``get_service`` / ``get_library`` resolution.
    """

    def __init__(self, thumb, original, **kwargs) -> None:
        super().__init__(**kwargs)
        self._thumb, self._original = thumb, original

    def thumbnail_path(self, photo_id: str):
        return self._thumb if photo_id == "p0" else None

    def original_path(self, photo_id: str):
        return self._original if photo_id == "p0" else None


@pytest.fixture
def two_source_client(tmp_path) -> TestClient:
    """Both corpora registered on app.state — the real Session 9b wiring, no stubs."""
    from PIL import Image

    thumb = tmp_path / "p0.jpg"
    Image.new("RGB", (12, 8), (10, 20, 30)).save(thumb, "JPEG")
    original = tmp_path / "p0_original.jpg"
    Image.new("RGB", (40, 30), (10, 20, 30)).save(original, "JPEG")

    encoder = StubEncoder(np.array([1.0, 0.0, 0.0, 0.0]))
    unsplash = SearchService(encoder, NumpyStore(EMB.copy(), IDS.copy(), PHOTOS.copy()))
    library = SearchService(
        encoder,
        StubLibraryStore(
            thumb, original,
            embeddings=EMB.copy(), photo_ids=IDS.copy(), photos=PHOTOS.copy(),
        ),
        source="library",
    )
    app.dependency_overrides.clear()
    app.state.services = {"unsplash": unsplash, "library": library}
    client = TestClient(app)
    yield client
    app.state.services = {}


class TestSourceToggle:
    def test_default_source_is_unsplash(self, two_source_client: TestClient) -> None:
        body = two_source_client.get("/api/search", params={"q": "x"}).json()
        assert body["source"] == "unsplash"

    def test_source_param_selects_the_library_corpus(self, two_source_client: TestClient) -> None:
        body = two_source_client.get("/api/search", params={"q": "x", "source": "library"}).json()
        assert body["source"] == "library"
        assert body["store"] == "stublibrary"  # store kind is independent of source

    def test_health_lists_every_available_source(self, two_source_client: TestClient) -> None:
        body = two_source_client.get("/api/health").json()
        assert body["sources"] == ["unsplash", "library"]

    def test_unknown_source_is_404(self, two_source_client: TestClient) -> None:
        resp = two_source_client.get("/api/search", params={"q": "x", "source": "nope"})
        assert resp.status_code == 404

    def test_similar_and_by_image_honour_the_source(self, two_source_client: TestClient) -> None:
        body = two_source_client.get("/api/similar/p0", params={"source": "library"}).json()
        assert body["source"] == "library"


class TestLibraryFiles:
    def test_thumb_is_served_from_disk(self, two_source_client: TestClient) -> None:
        resp = two_source_client.get("/api/photo/p0/thumb")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert "max-age" in resp.headers["cache-control"]
        assert len(resp.content) > 0

    def test_full_is_the_original_not_the_thumb(self, two_source_client: TestClient) -> None:
        thumb = two_source_client.get("/api/photo/p0/thumb").content
        full = two_source_client.get("/api/photo/p0/full").content
        assert full != thumb

    def test_unknown_id_is_404(self, two_source_client: TestClient) -> None:
        assert two_source_client.get("/api/photo/nope/thumb").status_code == 404
        assert two_source_client.get("/api/photo/nope/full").status_code == 404

    def test_404_when_no_library_is_indexed(self, client: TestClient) -> None:
        # `client` overrides get_service only, so app.state has no library registered
        app.state.services = {}
        assert client.get("/api/photo/p0/thumb").status_code == 404


def test_search_unavailable_without_service() -> None:
    # No override and no lifespan (no `with`): app.state has no `service`, so
    # get_service sees None and returns 503 instead of crashing.
    app.dependency_overrides.clear()
    client = TestClient(app)
    assert client.get("/api/search", params={"q": "x"}).status_code == 503
