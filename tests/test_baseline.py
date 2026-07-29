"""Session 10 — the BM25 keyword baseline, on a tiny hand-built corpus.

The point of a baseline is that it's honestly implemented: if it's accidentally
crippled, "CLIP beats keywords" measures nothing. So these tests check that BM25
actually ranks by term overlap, that rare terms outweigh common ones (that's IDF),
and that it returns the same ``Result`` objects everything else in the app speaks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from photosearch.baseline import Bm25Baseline, tokenize


def photos(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(photo_id, ai_description, photo_description) -> a display-shaped frame."""
    return pd.DataFrame(
        [
            {
                "photo_id": pid,
                "ai_description": ai,
                "photo_description": desc,
                "photo_image_url": f"http://img/{pid}",
                "photo_url": f"http://page/{pid}",
                "photographer_first_name": "Ada",
                "photographer_last_name": "Byte",
                "photo_width": 100,
                "photo_height": 100,
                "blur_hash": None,
                "aperture": None,
                "focal_length": None,
                "exposure_s": None,
                "iso": None,
                "camera_make": None,
                "camera_model": None,
            }
            for pid, ai, desc in rows
        ]
    )


CORPUS = photos(
    [
        ("p0", "a brown dog running on a sandy beach", ""),
        ("p1", "a dog sleeping on a sofa", "indoor pet portrait"),
        ("p2", "waves on an empty beach at sunset", ""),
        ("p3", "snow covered mountain peaks", "alpine"),
        ("p4", "", ""),  # no caption at all — the ~0 signal case
    ]
)


@pytest.fixture
def bm25() -> Bm25Baseline:
    return Bm25Baseline(CORPUS)


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self) -> None:
        assert tokenize("Golden-Hour, by the SEA!") == ["golden", "hour", "by", "the", "sea"]

    def test_drops_single_characters(self) -> None:
        assert tokenize("a dog") == ["dog"]

    def test_handles_missing_text(self) -> None:
        assert tokenize("") == []


class TestRanking:
    def test_matches_on_shared_terms(self, bm25: Bm25Baseline) -> None:
        top = bm25.search("dog on a beach", k=3)
        assert top[0].photo_id == "p0"  # matches both "dog" and "beach"

    def test_rare_terms_outrank_common_ones(self, bm25: Bm25Baseline) -> None:
        """"mountain" appears once, so it should dominate — that's IDF doing its job."""
        assert bm25.search("mountain", k=1)[0].photo_id == "p3"

    def test_searches_both_caption_columns(self, bm25: Bm25Baseline) -> None:
        # "alpine" only exists in photo_description, "indoor" only in the other row's
        assert bm25.search("alpine", k=1)[0].photo_id == "p3"
        assert bm25.search("indoor portrait", k=1)[0].photo_id == "p1"

    def test_zero_score_documents_are_not_returned(self, bm25: Bm25Baseline) -> None:
        """The honest failure mode: no shared term means no result, not a padded page."""
        results = bm25.search("helicopter", k=10)
        assert results == []

    def test_no_semantic_generalisation(self, bm25: Bm25Baseline) -> None:
        # THE point of the baseline: "puppy" and "dog" are unrelated tokens to BM25.
        assert bm25.search("puppy", k=5) == []

    def test_respects_k(self, bm25: Bm25Baseline) -> None:
        assert len(bm25.search("beach", k=1)) == 1

    def test_returns_full_result_objects(self, bm25: Bm25Baseline) -> None:
        top = bm25.search("beach", k=1)[0]
        assert top.photo_image_url == "http://img/p0"
        assert top.photographer == "Ada Byte"
        assert top.score > 0

    def test_accepts_and_ignores_filters(self, bm25: Bm25Baseline) -> None:
        from photosearch.models import FilterSpec

        # same signature as a store so the harness can call either one
        assert bm25.search("beach", k=2, filters=FilterSpec(aperture_max=1.4))


class TestCorpusIntrospection:
    def test_counts_documents_and_empty_captions(self, bm25: Bm25Baseline) -> None:
        assert bm25.count() == 5
        assert bm25.empty_docs == 1  # p4 has nothing to match on, ever
