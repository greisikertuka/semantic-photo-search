"""Session 10 — the keyword baseline every semantic-search claim needs.

A retrieval metric without a baseline can't answer the question an interviewer will
actually ask: *how do you know CLIP beats plain keyword search?* So we implement the
classic answer — **BM25**, the term-frequency/inverse-document-frequency ranker at
the heart of Elasticsearch — over the captions Unsplash ships with each photo, and
run it through the identical harness.

What BM25 scores, in one paragraph: a document ranks high for a query when it
contains the query's terms *often* (term frequency, with diminishing returns), those
terms are *rare* across the corpus (inverse document frequency), and the document
isn't padding its counts by being long (length normalization). It has no idea what
words mean — "a dog" and "a puppy" are unrelated tokens — which is exactly the
weakness CLIP is supposed to fix, and exactly what the eval should measure.

The interface is deliberately ``search(query, k)`` — the same shape as
:class:`~photosearch.search.SearchService`, so ``run_eval.py`` can treat "a CLIP
system" and "a keyword system" as the same kind of thing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from photosearch.models import FilterSpec, Result
from photosearch.store import build_result

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop single characters.

    Deliberately naive — no stemming, no stopword list. BM25's IDF term already
    discounts words that appear everywhere, so a stopword list buys almost nothing,
    and every extra step is one more thing that makes the baseline "unfairly weak".
    A baseline you tuned down is not a baseline.
    """
    return [t for t in _TOKEN.findall(str(text).lower()) if len(t) > 1]


class Bm25Baseline:
    """BM25 over each photo's caption text. Same ``search(query, k)`` shape as CLIP."""

    def __init__(self, photos: pd.DataFrame) -> None:
        from rank_bm25 import BM25Okapi

        self.photos = photos.reset_index(drop=True)
        self.documents = [
            tokenize(f"{a} {b}")
            for a, b in zip(
                self.photos["ai_description"].fillna(""),
                self.photos["photo_description"].fillna(""),
                strict=True,
            )
        ]
        self.bm25 = BM25Okapi(self.documents)
        self.empty_docs = sum(1 for d in self.documents if not d)

    @classmethod
    def load(cls, data_dir) -> Bm25Baseline:
        from pathlib import Path

        return cls(pd.read_parquet(Path(data_dir) / "photos.parquet"))

    def count(self) -> int:
        return len(self.photos)

    def search(self, query: str, k: int = 12, filters: FilterSpec | None = None) -> list[Result]:
        """Top-k by BM25 score. ``filters`` is accepted and ignored — see below.

        The EXIF filter seam is a *store* concern; the baseline exists to isolate the
        ranking signal, and the eval runs both systems unfiltered so the comparison is
        about relevance, not about who can mask rows.
        """
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)
        if not scores.size:
            return []
        top = np.argsort(-scores)[:k]
        # A zero score means "shares no query term at all" — returning those as ranked
        # hits would flatter the baseline's recall with pure noise.
        return [
            build_result(self.photos.iloc[i], float(scores[i]))
            for i in top
            if scores[i] > 0
        ]
