"""The search service: compose an encoder with a store and time each half.

This is the object the API and CLI both hold. It does nothing clever — encode the
text, hand the vector to the store — but keeping it as a seam means the encoder and
store are each swappable (stub encoder in tests; Chroma store in Session 7) without
either side knowing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from photosearch.models import FilterSpec, Result


@dataclass
class SearchTiming:
    """Where the milliseconds went — encode vs. store lookup, kept separate."""

    encode_ms: float
    search_ms: float


class SearchService:
    """Encoder + store, wired together. Both are injected — that's the test seam."""

    def __init__(self, encoder: object, store: object) -> None:
        self.encoder = encoder
        self.store = store

    def search(
        self,
        query: str,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> list[Result]:
        results, _ = self.search_timed(query, k, filters)
        return results

    def search_timed(
        self,
        query: str,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> tuple[list[Result], SearchTiming]:
        """Same as :meth:`search` but also returns encode/search timings (for --time)."""
        t0 = time.perf_counter()
        query_vec = self.encoder.encode_text(query)
        t1 = time.perf_counter()
        results = self.store.search(query_vec, k=k, filters=filters)
        t2 = time.perf_counter()
        timing = SearchTiming(
            encode_ms=(t1 - t0) * 1000.0,
            search_ms=(t2 - t1) * 1000.0,
        )
        return results, timing
