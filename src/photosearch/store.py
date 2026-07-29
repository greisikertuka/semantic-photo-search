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

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from photosearch.models import FilterSpec, Result

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# The one Chroma collection holding the Unsplash Lite index (Session 7).
COLLECTION_NAME = "unsplash_lite"


def exif_metadata(row: pd.Series) -> dict:
    """The filterable EXIF fields of one photo, shaped for Chroma metadata.

    Only *filterable* fields go in the vector DB (display fields stay in the
    parquet, joined back by ``photo_id`` — one source of truth for rendering).
    NaN/blank fields are omitted entirely: a document with no ``aperture`` key
    simply won't match an ``aperture`` ``$lte`` clause, which is exactly the
    "no EXIF ⇒ excluded when filtering" rule NumpyStore gets from NaN comparisons.
    ``camera_make`` is stored lowercased so Chroma's exact ``$eq`` matches the
    case-insensitive FilterSpec. ``has_exif`` guarantees non-empty metadata
    (Chroma rejects ``{}``) and lets a caller isolate the EXIF-bearing corpus.
    """
    m: dict = {}
    ap = row.get("aperture")
    if pd.notna(ap):
        m["aperture"] = float(ap)
    iso = row.get("iso")
    if pd.notna(iso):
        m["iso"] = int(iso)
    focal = row.get("focal_length")
    if pd.notna(focal):
        m["focal_length"] = float(focal)
    exposure = row.get("exposure_s")
    if pd.notna(exposure):
        m["exposure_s"] = float(exposure)
    make = row.get("camera_make")
    if pd.notna(make) and str(make).strip():
        m["camera_make"] = str(make).strip().lower()
    model = row.get("camera_model")
    if pd.notna(model) and str(model).strip():
        m["camera_model"] = str(model).strip()
    m["has_exif"] = len(m) > 0
    return m


def build_result(row: pd.Series, score: float, photo_id: str | None = None) -> Result:
    """Turn one display-parquet row + a score into a :class:`Result`.

    Shared by both stores: NumpyStore passes a positional row (``photo_id`` in the
    row), ChromaStore passes a row indexed *by* ``photo_id`` and supplies it here.
    """
    pid = photo_id if photo_id is not None else str(row["photo_id"])
    # The Unsplash parquet splits the name in two; the Session 9 library manifest
    # supplies one ready-made ``photographer`` string. Either shape renders the same.
    photographer = _clean_str(row.get("photographer"))
    if photographer is None:
        first = _clean_str(row.get("photographer_first_name"))
        last = _clean_str(row.get("photographer_last_name"))
        photographer = " ".join(part for part in (first, last) if part) or "Unknown"
    return Result(
        photo_id=pid,
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


def _exif_count(photos: pd.DataFrame) -> int:
    """Photos with *any* numeric EXIF (aperture/iso/focal) — the filterable corpus.

    ~85–88% of the Unsplash Lite rows; the rest can never match a numeric filter,
    which is why the UI surfaces "searching N photos with EXIF data" (Session 8).
    """
    have = np.zeros(len(photos), dtype=bool)
    for col in ("aperture", "iso", "focal_length"):
        if col in photos.columns:
            have |= pd.to_numeric(photos[col], errors="coerce").notna().to_numpy()
    return int(have.sum())


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

        # photo_id -> row index, for the "more like this" stored-embedding lookup.
        self._id_to_idx = {pid: i for i, pid in enumerate(self.photo_ids)}
        self.exif_count = _exif_count(self.photos)

    def count(self) -> int:
        """Number of indexed photos — the store-agnostic corpus size."""
        return len(self.photo_ids)

    def get_embedding(self, photo_id: str) -> np.ndarray | None:
        """The stored (already-normalized) vector for a photo, or None if unknown.

        This is the whole trick behind "more like this" (Session 8): the query
        vector *is* the clicked photo's own embedding — no encoder call needed.
        """
        i = self._id_to_idx.get(str(photo_id))
        return None if i is None else self.embeddings[i]

    def _float_col(self, name: str) -> np.ndarray:
        return pd.to_numeric(self.photos[name], errors="coerce").to_numpy(dtype=float)

    @classmethod
    def load(cls, data_dir: Path | str = DATA_DIR) -> NumpyStore:
        """Load ``embeddings.npy`` + ``photo_ids.npy`` + ``photos.parquet`` from disk.

        Falls back to the Session 11 deploy artifacts (``embeddings.f16.npy`` +
        ``photos.slim.parquet``) when the full-precision pair isn't there, so the
        29.3 MB payload the Space and Render both ship loads through this same
        class. float16 is a *storage* format only — the constructor widens it back
        to float32, because NumPy has no fast half-precision matmul.
        """
        data_dir = Path(data_dir)
        embeddings = _load_first(data_dir, "embeddings.npy", "embeddings.f16.npy")
        photo_ids = np.load(data_dir / "photo_ids.npy", allow_pickle=True)
        photos = _read_first_parquet(data_dir, "photos.parquet", "photos.slim.parquet")
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

        return [build_result(self.photos.iloc[i], float(scores[i])) for i in top]


class ChromaStore:
    """Chroma-backed store — same ``search`` shape as NumpyStore, filters fused in.

    What the vector DB buys us over the NumPy file: persistence, incremental
    add/delete (Session 9), and — the reason we adopt it here — *metadata filtering
    fused with the vector query*. Chroma's ``where=`` pre-filters the candidate set
    server-side before ranking, the DB-grade version of NumpyStore's boolean mask.

    Two numbers to keep straight (both configured away below):

    * The collection is created with **cosine** space (``hnsw:space``), not Chroma's
      default L2 — otherwise the ranking is the same but the *distances* aren't the
      similarities the rest of the app speaks.
    * ``query()`` returns cosine **distance** (lower = better); we convert to the
      cosine **similarity** the UI expects with ``similarity = 1 - distance``.

    Display fields (URLs, photographer, blur_hash) live only in the parquet; results
    are joined back by ``photo_id`` so there's one source of truth for rendering.
    """

    def __init__(self, collection: Collection, photos: pd.DataFrame) -> None:
        self.collection = collection
        self.photos = photos.reset_index(drop=True)
        self._by_id = self.photos.set_index("photo_id")
        self.exif_count = _exif_count(self.photos)

    @classmethod
    def load(
        cls,
        data_dir: Path | str = DATA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> ChromaStore:
        """Open the persistent Chroma collection + the display parquet from disk."""
        import chromadb

        data_dir = Path(data_dir)
        client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        collection = client.get_collection(collection_name, embedding_function=None)
        photos = pd.read_parquet(data_dir / "photos.parquet")
        return cls(collection, photos)

    def count(self) -> int:
        return self.collection.count()

    def get_embedding(self, photo_id: str) -> np.ndarray | None:
        res = self.collection.get(ids=[str(photo_id)], include=["embeddings"])
        embeddings = res.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return np.asarray(embeddings[0], dtype=np.float32)

    @staticmethod
    def _compile_where(filters: FilterSpec) -> dict | None:
        """FilterSpec -> Chroma ``where=`` clause (the same intent as the NumPy mask).

        ``$lte``/``$gte`` only work on values stored as *numbers*, which is why
        Session 2 parsed ``"f/1.8"`` into ``1.8`` — strings would silently match
        nothing. Multiple criteria combine with ``$and``.
        """
        clauses: list[dict] = []
        if filters.aperture_max is not None:
            clauses.append({"aperture": {"$lte": filters.aperture_max}})
        if filters.iso_max is not None:
            clauses.append({"iso": {"$lte": filters.iso_max}})
        if filters.focal_min is not None:
            clauses.append({"focal_length": {"$gte": filters.focal_min}})
        if filters.focal_max is not None:
            clauses.append({"focal_length": {"$lte": filters.focal_max}})
        if filters.camera_make is not None:
            clauses.append({"camera_make": {"$eq": filters.camera_make.strip().lower()}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(
        self,
        query_vec: np.ndarray,
        k: int = 12,
        filters: FilterSpec | None = None,
    ) -> list[Result]:
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        where = None
        if filters is not None and filters.is_active():
            where = self._compile_where(filters)

        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=k,
            where=where,
            include=["distances"],
        )
        ids = res["ids"][0]
        distances = res["distances"][0]

        results: list[Result] = []
        for pid, distance in zip(ids, distances, strict=True):
            try:
                row = self._by_id.loc[pid]
            except KeyError:
                continue  # id in the DB but not the display parquet — skip defensively
            results.append(build_result(row, 1.0 - float(distance), photo_id=pid))
        return results


def _pick(data_dir: Path, *names: str) -> Path:
    """First of ``names`` that exists in ``data_dir``; raises naming all of them."""
    for name in names:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"none of {', '.join(names)} found in {data_dir}")


def _load_first(data_dir: Path, *names: str) -> np.ndarray:
    return np.load(_pick(data_dir, *names))


def _read_first_parquet(data_dir: Path, *names: str) -> pd.DataFrame:
    return pd.read_parquet(_pick(data_dir, *names))


def data_dir_from_env(default: Path | str = DATA_DIR) -> Path:
    """``PHOTOSEARCH_DATA_DIR`` if set — the deploy points it at the unpacked release."""
    return Path(os.environ.get("PHOTOSEARCH_DATA_DIR") or default)


def load_store(kind: str | None = None, data_dir: Path | str | None = None):
    """Return the store selected by ``kind`` (or the ``PHOTOSEARCH_STORE`` env var).

    ``"numpy"`` (default) or ``"chroma"`` — the config seam that lets the API swap
    back-ends without either endpoint knowing which one answered.
    """
    kind = (kind or os.environ.get("PHOTOSEARCH_STORE", "numpy")).strip().lower()
    data_dir = data_dir_from_env() if data_dir is None else Path(data_dir)
    if kind == "numpy":
        return NumpyStore.load(data_dir)
    if kind == "chroma":
        return ChromaStore.load(data_dir)
    if kind == "library":
        from photosearch.library import LibraryStore  # local import: avoids a cycle

        return LibraryStore.load(data_dir)
    raise ValueError(f"unknown store kind {kind!r} (expected 'numpy', 'chroma' or 'library')")


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
