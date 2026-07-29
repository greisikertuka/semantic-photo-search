"""Session 10 — the retrieval metrics, as pure functions.

Kept in the package (rather than inside ``eval/run_eval.py``) so they can be unit
tested against hand-computed examples. Metric code is exactly the kind that looks
right and is off by one: a Recall@10 that divides by ``K`` instead of by the number
of relevant photos will happily report plausible numbers forever.

Convention throughout: **unjudged means not relevant**. That's the standard
assumption in pooled evaluation, and it's why the pool has to contain every system's
top hits — see eval/POLICY.md for what that costs us.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_K = 10


@dataclass
class QueryScore:
    """One query's scores against one system."""

    query_id: str
    query: str
    bucket: str
    precision: float
    recall: float
    reciprocal_rank: float
    n_relevant: int
    hits: list[bool]


def score_query(
    query_id: str,
    judged: dict,
    ranked_ids: list[str],
    k: int = DEFAULT_K,
) -> QueryScore:
    """Score one ranked result list against one query's judgments.

    * **Precision@k** divides by ``k``, not by the number returned — a system that
      returns 3 results for a 10-slot page has failed to fill the page, and saying
      "3/3 correct = 1.0" would hide that.
    * **Recall@k** divides by the number of *known relevant* photos.
    * **Reciprocal rank** is ``1/position`` of the first hit, or 0 if there is none.
    """
    relevant = set(judged["relevant"])
    top = ranked_ids[:k]
    hits = [pid in relevant for pid in top]
    n_hits = sum(hits)
    first = next((i + 1 for i, hit in enumerate(hits) if hit), None)
    return QueryScore(
        query_id=query_id,
        query=judged.get("query", query_id),
        bucket=judged.get("bucket", "unknown"),
        precision=n_hits / k,
        # guarded: a query nobody judged relevant would divide by zero. Such queries
        # are dropped upstream, but the metric should not be a landmine either way.
        recall=(n_hits / len(relevant)) if relevant else 0.0,
        reciprocal_rank=(1.0 / first) if first else 0.0,
        n_relevant=len(relevant),
        hits=hits,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(scores: list[QueryScore], k: int = DEFAULT_K) -> dict:
    """Macro-average across queries — every query counts equally, regardless of size."""
    return {
        f"P@{k}": mean(s.precision for s in scores),
        f"R@{k}": mean(s.recall for s in scores),
        "MRR": mean(s.reciprocal_rank for s in scores),
        "n": len(scores),
    }


def by_bucket(scores: list[QueryScore], buckets: list[str], k: int = DEFAULT_K) -> dict[str, dict]:
    """Per-bucket summaries — where the aggregate number hides the interesting story."""
    out = {}
    for bucket in buckets:
        subset = [s for s in scores if s.bucket == bucket]
        if subset:
            out[bucket] = summarize(subset, k)
    return out


def drop_unanswerable(judgments: dict) -> tuple[dict, list[str]]:
    """Split off queries with no relevant photo in the pool.

    They score 0 for every system and say nothing about ranking, so the standard
    practice (TREC) is to exclude them from the averages and report them separately.
    Finding one is a fact about the *corpus* — nothing in 25k photos satisfies the
    query — not a defect in the retriever.
    """
    unanswerable = [qid for qid, judged in judgments.items() if not judged["relevant"]]
    keep = {qid: judged for qid, judged in judgments.items() if judged["relevant"]}
    return keep, unanswerable
