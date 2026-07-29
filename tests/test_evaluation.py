"""Session 10 — the metrics, checked against hand-computed answers.

Evaluation code is uniquely dangerous: a Recall@10 that divides by K instead of by
the number of relevant photos produces numbers that look completely reasonable and
are simply wrong. So every metric here is asserted against a value worked out by
hand, not against whatever the implementation happens to return.
"""

from __future__ import annotations

import pytest

from photosearch.evaluation import (
    by_bucket,
    drop_unanswerable,
    mean,
    score_query,
    summarize,
)

JUDGED = {"query": "a dog on a beach", "bucket": "easy", "relevant": ["a", "b", "c", "d"]}


def ranked(*ids: str) -> list[str]:
    return list(ids)


class TestScoreQuery:
    def test_perfect_first_page(self) -> None:
        # 4 relevant photos exist and 4 of the 10 slots hit them
        s = score_query("q", JUDGED, ranked("a", "b", "c", "d", *"efghij"), k=10)
        assert s.precision == pytest.approx(0.4)  # 4 hits / 10 slots
        assert s.recall == pytest.approx(1.0)  # 4 hits / 4 relevant
        assert s.reciprocal_rank == pytest.approx(1.0)  # first hit at rank 1

    def test_precision_divides_by_k_not_by_results_returned(self) -> None:
        """A system that returns 2 results has failed to fill the page, not aced it."""
        s = score_query("q", JUDGED, ranked("a", "b"), k=10)
        assert s.precision == pytest.approx(0.2)
        assert s.recall == pytest.approx(0.5)

    def test_reciprocal_rank_is_one_over_the_first_hit(self) -> None:
        s = score_query("q", JUDGED, ranked("x", "y", "a", "z"), k=10)
        assert s.reciprocal_rank == pytest.approx(1 / 3)

    def test_no_hits_scores_zero_everywhere(self) -> None:
        s = score_query("q", JUDGED, ranked(*"xyzuvw"), k=10)
        assert (s.precision, s.recall, s.reciprocal_rank) == (0.0, 0.0, 0.0)

    def test_only_the_first_k_count(self) -> None:
        # the one relevant photo sits at rank 11 — off the page, so it doesn't exist
        s = score_query("q", JUDGED, ranked(*"xxxxxxxxxx", "a"), k=10)
        assert s.precision == 0.0
        assert s.reciprocal_rank == 0.0

    def test_unjudged_photos_count_as_not_relevant(self) -> None:
        s = score_query("q", JUDGED, ranked("never-pooled", "a"), k=10)
        assert s.hits == [False, True]
        assert s.precision == pytest.approx(0.1)

    def test_empty_relevant_set_does_not_divide_by_zero(self) -> None:
        s = score_query("q", {**JUDGED, "relevant": []}, ranked("a"), k=10)
        assert s.recall == 0.0


class TestAggregation:
    def _scores(self):
        return [
            score_query("q1", JUDGED, ranked("a", "b", *"xxxxxxxx"), k=10),  # P=.2 RR=1
            score_query(
                "q2", {**JUDGED, "bucket": "jargon"}, ranked("x", "x", "a", *"xxxxxxx"), k=10
            ),  # P=.1 RR=1/3
        ]

    def test_summary_macro_averages_over_queries(self) -> None:
        summary = summarize(self._scores(), k=10)
        assert summary["P@10"] == pytest.approx(0.15)
        assert summary["MRR"] == pytest.approx((1.0 + 1 / 3) / 2)
        assert summary["n"] == 2

    def test_bucket_split_keeps_queries_in_their_own_bucket(self) -> None:
        buckets = by_bucket(self._scores(), ["easy", "jargon"], k=10)
        assert buckets["easy"]["P@10"] == pytest.approx(0.2)
        assert buckets["jargon"]["P@10"] == pytest.approx(0.1)

    def test_missing_bucket_is_omitted_not_zero(self) -> None:
        assert "abstract" not in by_bucket(self._scores(), ["abstract"], k=10)

    def test_mean_of_nothing_is_zero_not_an_error(self) -> None:
        assert mean([]) == 0.0


class TestDropUnanswerable:
    def test_splits_off_queries_with_no_relevant_photo(self) -> None:
        judgments = {
            "good": {"relevant": ["a"], "query": "x", "bucket": "easy"},
            "empty": {"relevant": [], "query": "y", "bucket": "compositional"},
        }
        kept, dropped = drop_unanswerable(judgments)
        assert list(kept) == ["good"]
        assert dropped == ["empty"]

    def test_keeps_everything_when_every_query_has_a_hit(self) -> None:
        judgments = {"a": {"relevant": ["1"]}, "b": {"relevant": ["2"]}}
        kept, dropped = drop_unanswerable(judgments)
        assert len(kept) == 2 and dropped == []
