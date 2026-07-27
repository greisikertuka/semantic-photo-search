"""Vector stores: the thing that turns a query vector into ranked results.

``NumpyStore`` is the from-scratch v1 — brute-force cosine search over the whole
25k x 512 matrix. Scoring is a single matrix-vector product (~25M multiply-adds
over a 51 MB matrix, a few milliseconds on a laptop CPU), which is exactly why a
vector DB is *premature* at this size. Writing it by hand is the point: when an
interviewer asks "how does vector search actually work?", the answer is this file.

Session 7 adds ``ChromaStore`` alongside, implementing the same ``search`` shape so
they're swappable by config. Both speak :class:`~photosearch.models.FilterSpec`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from photosearch.models import FilterSpec, Result

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best-first.

    Uses ``argpartition`` (O(n) to find the top k) then sorts only those k, instead
    of a full O(n log n) sort of all 25k — the right habit even when n is small.
    Falls back to a plain sort when k covers everything.
    """
    n = scores.shape[0]
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    if k >= n:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


class NumpyStore:
    """Brute-force cosine store backed by the on-disk embedding artifacts."""

    def __init__(
        self,
        embeddings: np.ndarray,
        photo_ids: np.ndarray,
        photos: pd.DataFrame,
    ) -> None:
        if len(embeddings) != len(photo_ids) or len(embeddings) != len(photos):
            raise ValueError(
                f"length mismatch: embeddings={len(embeddings)}, "
                f"photo_ids={len(photo_ids)}, photos={len(photos)}"
            )
        # THE alignment guard — element-wise, not just shapes. A scrambled order is
        # the "search returns random photos" bug, and it has no other symptom.
        parquet_ids = photos["photo_id"].to_numpy().astype(str)
        if not np.array_equal(parquet_ids, photo_ids.astype(str)):
            raise ValueError("row alignment broken: photo_ids != photos.photo_id order")

        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.photo_ids = photo_ids.astype(str)
        self.photos = photos.reset_index(drop=True)

        # Pre-extract the filterable columns as float arrays once, so mask
        # compilation is pure vectorized NumPy. Missing values become NaN, and
        # every comparison against NaN is False — which is exactly how "no EXIF =>
        # excluded when filtering" falls out for free.
        self._aperture = self._float_col("aperture")
        self._iso = self._float_col("iso")
        self._focal = self._float_col("focal_length")
        # Object array of lowercased makes with missing -> None (None != any string,
        # so no-make rows never match a camera_make filter).
        make = self.photos["camera_make"].astype("string").str.lower()
        self._make_lower = make.to_numpy(dtype=object, na_value=None)

    def _float_col(self, name: str) -> np.ndarray:
        return pd.to_numeric(self.photos[name], errors="coerce").to_numpy(dtype=float)

    @classmethod
    def load(cls, data_dir: Path | str = DATA_DIR) -> NumpyStore:
        """Load ``embeddings.npy`` + ``photo_ids.npy`` + ``photos.parquet`` from disk."""
        data_dir = Path(data_dir)
        embeddings = np.load(data_dir / "embeddings.npy")
        photo_ids = np.load(data_dir / "photo_ids.npy", allow_pickle=True)
        photos = pd.read_parquet(data_dir / "photos.parquet")
        return cls(embeddings, photo_ids, photos)

    def _compile_mask(self, filters: FilterSpec) -> np.ndarray:
        """FilterSpec -> boolean array marking rows eligible for the result set."""
        mask = np.ones(len(self.embeddings), dtype=bool)
        if filters.aperture_max is not None:
            mask &= self._aperture <= filters.aperture_max
        if filters.iso_max is not None:
            mask &= self._iso <= filters.iso_max
        if filters.focal_min is not None:
            mask &= self._focal >= filters.focal_min
        if filters.focal_max is not None:
            mask &= self._focal <= filters.focal_max
        if filters.camera_make is not None:
            mask &= self._make_lower == filters.camera_make.strip().lower()
        return mask

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> list[Result]:
        """Rank photos by cosine similarity to ``query_vec``, best first.

        With an active filter we PRE-filter (restrict the candidate set, then rank),
        never post-filter — asking for top-50 and then dropping non-matches can leave
        you with 2 results for a selective filter. Chroma's ``where=`` does the same
        thing server-side in Session 7.
        """
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self.embeddings @ query_vec  # cosine sim, because both are unit-length

        if filters is not None and filters.is_active():
            candidates = np.nonzero(self._compile_mask(filters))[0]
            if candidates.size == 0:
                return []
            local = top_k_indices(scores[candidates], k)
            top = candidates[local]
        else:
            top = top_k_indices(scores, k)

        return [self._result(int(i), float(scores[i])) for i in top]

    def _result(self, i: int, score: float) -> Result:
        row = self.photos.iloc[i]
        first = _clean_str(row.get("photographer_first_name"))
        last = _clean_str(row.get("photographer_last_name"))
        photographer = " ".join(part for part in (first, last) if part) or "Unknown"
        return Result(
            photo_id=str(row["photo_id"]),
            score=score,
            photo_image_url=str(row["photo_image_url"]),
            photo_url=str(row["photo_url"]),
            photographer=photographer,
            width=_opt_int(row.get("photo_width")),
            height=_opt_int(row.get("photo_height")),
            blur_hash=_clean_str(row.get("blur_hash")),
            description=_clean_str(row.get("photo_description")),
            ai_description=_clean_str(row.get("ai_description")),
            aperture=_opt_float(row.get("aperture")),
            focal_length=_opt_float(row.get("focal_length")),
            exposure_s=_opt_float(row.get("exposure_s")),
            iso=_opt_int(row.get("iso")),
            camera_make=_clean_str(row.get("camera_make")),
            camera_model=_clean_str(row.get("camera_model")),
        )


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _opt_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: object) -> int | None:
    f = _opt_float(value)
    return None if f is None else round(f)
