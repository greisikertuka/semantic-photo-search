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

    def __init__(self, encoder: object, store: object, source: str = "unsplash") -> None:
        self.encoder = encoder
        self.store = store
        # Which *corpus* this service searches (Session 9b): "unsplash" or "library".
        # Orthogonal to which *store* backs it — one is the data, the other the engine.
        self.source = source

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

    def search_by_image_timed(
        self,
        image: object,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> tuple[list[Result], SearchTiming]:
        """Search by an uploaded image (Session 8): encode it, then the same path.

        The lesson is that there's no new concept — an image and a text query are
        *both* just vectors in the shared CLIP space, so "photos like this photo"
        is the identical dot product with a different query vector (and filters
        still apply — "like this one, but shot wide open").
        """
        t0 = time.perf_counter()
        query_vec = self.encoder.encode_image(image)
        t1 = time.perf_counter()
        results = self.store.search(query_vec, k=k, filters=filters)
        t2 = time.perf_counter()
        timing = SearchTiming(
            encode_ms=(t1 - t0) * 1000.0,
            search_ms=(t2 - t1) * 1000.0,
        )
        return results, timing

    def similar_timed(
        self,
        photo_id: str,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> tuple[list[Result] | None, SearchTiming]:
        """"More like this" (Session 8): the query vector IS the photo's stored one.

        No encoder call at all — we look the embedding up in the store and search
        with it, then drop result #1 (the photo is always its own nearest neighbour).
        Returns ``None`` for an unknown id so the API can answer 404. The lookup time
        is reported in ``encode_ms``'s slot (there's nothing to encode).
        """
        t0 = time.perf_counter()
        query_vec = self.store.get_embedding(photo_id)
        t1 = time.perf_counter()
        if query_vec is None:
            return None, SearchTiming(encode_ms=(t1 - t0) * 1000.0, search_ms=0.0)
        # ask for k+1 because the photo itself will come back first, then filter it out
        results = self.store.search(query_vec, k=k + 1, filters=filters)
        results = [r for r in results if r.photo_id != str(photo_id)][:k]
        t2 = time.perf_counter()
        timing = SearchTiming(
            encode_ms=(t1 - t0) * 1000.0,
            search_ms=(t2 - t1) * 1000.0,
        )
        return results, timing
