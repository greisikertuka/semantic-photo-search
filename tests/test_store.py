"""Deterministic unit tests for NumpyStore — no model, no network, no artifacts.

A hand-built 6x4 fixture (six photos, four-dim unit vectors) with nearest neighbors
you can verify by eye. This is the suite that runs in CI: it exercises ranking
order, FilterSpec masking (including the "no-EXIF rows drop out when filtering"
rule), pre-filtering correctness, and the sacred alignment guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from photosearch.models import FilterSpec
from photosearch.store import NumpyStore, top_k_indices

# Six unit vectors in 4-space. Chosen so nearest neighbors are obvious and tie-free:
# for the query [1,0,0,0] the order is p0 (1.0) > p4 (0.8) > p5 (0.6) > the rest (0).
EMB = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # p0
        [0.0, 1.0, 0.0, 0.0],  # p1
        [0.0, 0.0, 1.0, 0.0],  # p2
        [0.0, 0.0, 0.0, 1.0],  # p3
        [0.8, 0.6, 0.0, 0.0],  # p4
        [0.6, 0.0, 0.8, 0.0],  # p5
    ],
    dtype=np.float32,
)
IDS = np.array([f"p{i}" for i in range(6)], dtype=object)

# EXIF chosen so each filter has a hand-checkable answer. p4 has NO exif on purpose:
# it must vanish from results whenever any filter is active.
PHOTOS = pd.DataFrame(
    {
        "photo_id": [f"p{i}" for i in range(6)],
        "photo_image_url": [f"http://img/{i}" for i in range(6)],
        "photo_url": [f"http://page/{i}" for i in range(6)],
        "photographer_first_name": ["Ada", "Bo", "Cy", "Di", "Eve", "Fu"],
        "photographer_last_name": ["Byte", "Lin", "Ng", "Rao", "Sol", None],
        "photo_width": [4000, 4000, 4000, 4000, 4000, 4000],
        "photo_height": [3000, 3000, 3000, 3000, 3000, 3000],
        "blur_hash": ["L0", "L1", "L2", "L3", "L4", "L5"],
        "photo_description": ["a", "b", "c", "d", None, "f"],
        "ai_description": ["a", "b", "c", "d", "e", "f"],
        "aperture": [1.8, 2.8, 4.0, 1.4, np.nan, 2.0],
        "focal_length": [35.0, 50.0, 85.0, 24.0, np.nan, 35.0],
        "exposure_s": [0.004, 0.008, 0.016, 0.002, np.nan, 0.004],
        "iso": pd.array([100, 400, 800, 1600, None, 200], dtype="Int64"),
        "camera_make": ["Canon", "Nikon", "Sony", "Canon", None, "Canon"],
        "camera_model": ["R5", "Z6", "A7", "R6", None, "R5"],
    }
)

QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def store() -> NumpyStore:
    return NumpyStore(EMB.copy(), IDS.copy(), PHOTOS.copy())


class TestRanking:
    def test_top_k_order_is_by_similarity(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=3)
        assert [r.photo_id for r in results] == ["p0", "p4", "p5"]

    def test_scores_are_the_dot_products(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=3)
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(0.8)
        assert results[2].score == pytest.approx(0.6)

    def test_k_larger_than_corpus_returns_all_sorted(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=99)
        assert len(results) == 6
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_carries_display_and_exif_fields(self, store: NumpyStore) -> None:
        top = store.search(QUERY, k=1)[0]
        assert top.photo_id == "p0"
        assert top.photographer == "Ada Byte"
        assert top.camera_make == "Canon"
        assert top.aperture == pytest.approx(1.8)
        assert top.iso == 100

    def test_missing_lastname_still_builds_photographer(self, store: NumpyStore) -> None:
        # p5 has no last name; query [1,0,1,0]-ish would rank it, but check directly.
        p5 = next(r for r in store.search(QUERY, k=6) if r.photo_id == "p5")
        assert p5.photographer == "Fu"

    def test_no_exif_row_yields_none_not_nan(self, store: NumpyStore) -> None:
        p4 = next(r for r in store.search(QUERY, k=6) if r.photo_id == "p4")
        assert p4.aperture is None
        assert p4.iso is None
        assert p4.camera_make is None
        assert p4.description is None


class TestFilters:
    def test_aperture_max_prefilters_then_ranks(self, store: NumpyStore) -> None:
        # eligible at f/<=2.0: p0(1.8), p3(1.4), p5(2.0). Ranked by the query: p0>p5>p3.
        results = store.search(QUERY, k=10, filters=FilterSpec(aperture_max=2.0))
        assert [r.photo_id for r in results] == ["p0", "p5", "p3"]

    def test_no_exif_row_excluded_when_any_filter_active(self, store: NumpyStore) -> None:
        # A permissive filter that everything with EXIF passes — p4 (no EXIF) must not.
        results = store.search(QUERY, k=10, filters=FilterSpec(aperture_max=99.0))
        ids = {r.photo_id for r in results}
        assert "p4" not in ids
        assert ids == {"p0", "p1", "p2", "p3", "p5"}

    def test_iso_max(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=10, filters=FilterSpec(iso_max=400))
        assert {r.photo_id for r in results} == {"p0", "p1", "p5"}

    def test_focal_range(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=10, filters=FilterSpec(focal_min=30.0, focal_max=60.0))
        # focal in [30,60]: p0(35), p1(50), p5(35). p2(85), p3(24) out; p4 no EXIF.
        assert {r.photo_id for r in results} == {"p0", "p1", "p5"}

    def test_camera_make_is_case_insensitive(self, store: NumpyStore) -> None:
        results = store.search(QUERY, k=10, filters=FilterSpec(camera_make="canon"))
        assert {r.photo_id for r in results} == {"p0", "p3", "p5"}

    def test_combined_filters_and_together(self, store: NumpyStore) -> None:
        results = store.search(
            QUERY, k=10, filters=FilterSpec(camera_make="Canon", aperture_max=1.9)
        )
        # Canon AND f/<=1.9: p0(1.8) yes, p3(1.4) yes, p5(2.0) no.
        assert {r.photo_id for r in results} == {"p0", "p3"}

    def test_impossible_filter_returns_empty(self, store: NumpyStore) -> None:
        assert store.search(QUERY, k=10, filters=FilterSpec(aperture_max=0.5)) == []

    def test_empty_filterspec_is_inactive(self, store: NumpyStore) -> None:
        assert not FilterSpec().is_active()
        assert len(store.search(QUERY, k=10, filters=FilterSpec())) == 6


class TestAlignmentGuard:
    def test_scrambled_ids_raise(self) -> None:
        bad = np.array([f"x{i}" for i in range(6)], dtype=object)
        with pytest.raises(ValueError, match="alignment"):
            NumpyStore(EMB.copy(), bad, PHOTOS.copy())

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            NumpyStore(EMB[:5].copy(), IDS.copy(), PHOTOS.copy())


class TestTopKIndices:
    def test_returns_best_first(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.3])
        assert list(top_k_indices(scores, 2)) == [1, 2]

    def test_k_zero_is_empty(self) -> None:
        assert top_k_indices(np.array([0.1, 0.2]), 0).size == 0

    def test_k_ge_n_sorts_all(self) -> None:
        scores = np.array([0.1, 0.9, 0.5])
        assert list(top_k_indices(scores, 10)) == [1, 2, 0]
