"""ChromaStore parity tests — the Session 7 "same seam, two back-ends" proof.

Runs entirely in-process against an **ephemeral** Chroma client (no persistence, no
network, no model), built from the same 6x4 fixture as test_store. So CI proves the
real thing the definition-of-done asks for: a FilterSpec compiles to *both* stores,
the collection speaks cosine (not L2), and ``1 - distance`` yields the same scores
NumpyStore computes by hand — without any AI dependency.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest
from test_store import EMB, IDS, PHOTOS, QUERY

from photosearch.models import FilterSpec
from photosearch.store import (
    COLLECTION_NAME,
    ChromaStore,
    NumpyStore,
    exif_metadata,
)

chromadb = pytest.importorskip("chromadb")


@pytest.fixture
def chroma_store() -> ChromaStore:
    client = chromadb.EphemeralClient()
    # EphemeralClient is a process-level singleton, so wipe any collection left by a
    # prior test before rebuilding a clean one.
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # the decision that keeps scores comparable
        embedding_function=None,
    )
    metadatas = [exif_metadata(row) for _, row in PHOTOS.iterrows()]
    collection.add(
        ids=list(IDS.astype(str)),
        embeddings=EMB.tolist(),
        metadatas=metadatas,
    )
    return ChromaStore(collection, PHOTOS.copy())


@pytest.fixture
def numpy_store() -> NumpyStore:
    return NumpyStore(EMB.copy(), IDS.copy(), PHOTOS.copy())


class TestParity:
    """The seam: identical top-k and ~equal scores from both stores (small n → exact)."""

    def test_unfiltered_ranking_matches_numpy(
        self, chroma_store: ChromaStore, numpy_store: NumpyStore
    ) -> None:
        n_res = numpy_store.search(QUERY, k=3)
        c_res = chroma_store.search(QUERY, k=3)
        assert [r.photo_id for r in c_res] == [r.photo_id for r in n_res] == ["p0", "p4", "p5"]

    def test_scores_are_cosine_similarity_not_distance(
        self, chroma_store: ChromaStore
    ) -> None:
        # If the collection were L2 (or we forgot 1-distance), these would be wrong:
        # cosine sim of [1,0,0,0] with itself is 1.0, with [0.8,0.6,0,0] is 0.8.
        res = chroma_store.search(QUERY, k=3)
        assert res[0].score == pytest.approx(1.0, abs=1e-5)
        assert res[1].score == pytest.approx(0.8, abs=1e-5)
        assert res[2].score == pytest.approx(0.6, abs=1e-5)

    def test_scores_match_numpy_elementwise(
        self, chroma_store: ChromaStore, numpy_store: NumpyStore
    ) -> None:
        n_res = numpy_store.search(QUERY, k=6)
        c_scores = {r.photo_id: r.score for r in chroma_store.search(QUERY, k=6)}
        for r in n_res:
            assert c_scores[r.photo_id] == pytest.approx(r.score, abs=1e-5)

    def test_results_carry_display_fields_from_parquet(
        self, chroma_store: ChromaStore
    ) -> None:
        # ChromaStore only stores filterable EXIF; display fields must be joined back.
        top = chroma_store.search(QUERY, k=1)[0]
        assert top.photo_id == "p0"
        assert top.photographer == "Ada Byte"
        assert top.photo_image_url == "http://img/0"
        assert top.camera_make == "Canon"


class TestFilterParity:
    """FilterSpec -> where-clause must exclude exactly what the NumPy mask excludes."""

    def test_aperture_max(self, chroma_store: ChromaStore, numpy_store: NumpyStore) -> None:
        f = FilterSpec(aperture_max=2.0)
        n_ids = [r.photo_id for r in numpy_store.search(QUERY, k=10, filters=f)]
        c_ids = [r.photo_id for r in chroma_store.search(QUERY, k=10, filters=f)]
        assert c_ids == n_ids == ["p0", "p5", "p3"]

    def test_no_exif_row_excluded_when_filtering(self, chroma_store: ChromaStore) -> None:
        # p4 (no EXIF metadata) must vanish under any numeric filter, just like NumPy.
        ids = {r.photo_id for r in chroma_store.search(QUERY, k=10, filters=FilterSpec(aperture_max=99.0))}
        assert "p4" not in ids
        assert ids == {"p0", "p1", "p2", "p3", "p5"}

    def test_iso_max(self, chroma_store: ChromaStore) -> None:
        ids = {r.photo_id for r in chroma_store.search(QUERY, k=10, filters=FilterSpec(iso_max=400))}
        assert ids == {"p0", "p1", "p5"}

    def test_focal_range(self, chroma_store: ChromaStore) -> None:
        f = FilterSpec(focal_min=30.0, focal_max=60.0)
        ids = {r.photo_id for r in chroma_store.search(QUERY, k=10, filters=f)}
        assert ids == {"p0", "p1", "p5"}

    def test_camera_make_case_insensitive(self, chroma_store: ChromaStore) -> None:
        # FilterSpec make is lowercased on both sides; metadata was stored lowercased.
        ids = {r.photo_id for r in chroma_store.search(QUERY, k=10, filters=FilterSpec(camera_make="CANON"))}
        assert ids == {"p0", "p3", "p5"}

    def test_combined_filters_and_together(self, chroma_store: ChromaStore) -> None:
        f = FilterSpec(camera_make="Canon", aperture_max=1.9)
        ids = {r.photo_id for r in chroma_store.search(QUERY, k=10, filters=f)}
        assert ids == {"p0", "p3"}

    def test_impossible_filter_returns_empty(self, chroma_store: ChromaStore) -> None:
        assert chroma_store.search(QUERY, k=10, filters=FilterSpec(aperture_max=0.5)) == []


class TestChromaIntrospection:
    def test_count_and_exif_count(self, chroma_store: ChromaStore) -> None:
        assert chroma_store.count() == 6
        assert chroma_store.exif_count == 5

    def test_get_embedding_round_trips(self, chroma_store: ChromaStore) -> None:
        vec = chroma_store.get_embedding("p0")
        assert vec is not None
        np.testing.assert_allclose(vec, EMB[0], atol=1e-6)

    def test_get_embedding_unknown_id_is_none(self, chroma_store: ChromaStore) -> None:
        assert chroma_store.get_embedding("nope") is None


class TestExifMetadata:
    def test_skips_nan_and_flags_no_exif(self) -> None:
        # p4 row: every EXIF field NaN -> only has_exif=False remains.
        row = PHOTOS.iloc[4]
        assert exif_metadata(row) == {"has_exif": False}

    def test_types_and_lowercased_make(self) -> None:
        m = exif_metadata(PHOTOS.iloc[0])
        assert m["aperture"] == pytest.approx(1.8)
        assert isinstance(m["iso"], int) and m["iso"] == 100
        assert m["camera_make"] == "canon"  # lowercased for exact $eq matching
        assert m["has_exif"] is True

    def test_all_values_are_chroma_scalar_types(self) -> None:
        # Chroma only accepts str/int/float/bool metadata values.
        for _, row in PHOTOS.iterrows():
            for key, value in exif_metadata(row).items():
                assert isinstance(value, (str, int, float, bool)), (key, type(value))


def test_photos_fixture_columns_present() -> None:
    # guard: exif_metadata reads these by name; keep the fixture honest.
    for col in ("aperture", "iso", "focal_length", "exposure_s", "camera_make", "camera_model"):
        assert col in PHOTOS.columns
    assert isinstance(PHOTOS, pd.DataFrame)
