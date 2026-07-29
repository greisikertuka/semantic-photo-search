"""Session 9 — your own photo folder as a second corpus.

Everything downstream of this module is unchanged: the same ``FilterSpec``, the
same ``Result``, the same store interface, the same UI. That is the whole point —
a *new source* plugs in, and the search core never learns it exists.

Three things are genuinely different from the Unsplash path, and each is a lesson:

* **EXIF comes from the files, not a TSV.** Real cameras write rationals
  (``FNumber`` is ``9/5``, not ``"f/1.8"``), sometimes tuples, sometimes bytes,
  and half the tags are missing. We coerce to plain numbers *first* and then feed
  the Session 2 parsers, so both sources land in one schema. (Coercing first
  matters: ``str(IFDRational(9, 5))`` is ``"9/5"``, which a string parser would
  happily read as ``9.0``.)
* **Indexing is incremental.** Re-embedding 20,000 photos because you imported 5
  is silly. We keep a manifest of ``path -> (mtime, size)``; only new or modified
  files get embedded, and files that vanished get deleted from the collection.
  This is the honest answer to "how would you keep a production index fresh?"
* **There is no CDN.** We generate a thumbnail at index time and serve it from the
  API (Session 9b); the originals never leave the machine. So ``photo_image_url``
  points at ``/api/photo/{id}/thumb`` instead of imgix — the frontend cannot tell.

Layout under ``data/library/``: ``manifest.parquet`` (the sidecar index) and
``thumbs/<photo_id>.jpg``. Vectors live in the ``library`` Chroma collection,
alongside ``unsplash_lite`` in the same ``data/chroma`` client.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from photosearch.exif import (
    normalize_make,
    normalize_model,
    parse_aperture,
    parse_exposure,
    parse_focal_length,
    parse_iso,
)
from photosearch.store import DATA_DIR, ChromaStore

LIBRARY_COLLECTION = "library"
THUMB_MAX = 640  # long edge, px — plenty for the grid, tiny on disk
JPEG_QUALITY = 82

# Pillow refuses images over ~179 MP as suspected "decompression bombs". That guard
# is exactly right for the *upload* endpoint, which decodes bytes from strangers —
# but a 200 MP stitched panorama sitting in your own photo folder is data, not an
# attack, and the first real archive run silently skipped two of them. So the
# ceiling is raised only on this path, which reads files the user explicitly pointed
# the indexer at. ~300 MP is ≈900 MB decoded: room for phone panoramas, still bounded.
LOCAL_MAX_PIXELS = 300_000_000

# Formats Pillow can open once pillow-heif is registered. RAW is deliberately out
# of scope: it needs rawpy/libraw and a demosaic step, which is a different project.
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tif", ".tiff", ".bmp", ".avif",
}

# EXIF tag numbers. Make/Model live in IFD0; the exposure triangle lives in the
# Exif sub-IFD (pointer tag 0x8769), which is why reading EXIF is a two-step walk.
EXIF_IFD = 0x8769
TAG_MAKE = 0x010F
TAG_MODEL = 0x0110
TAG_EXPOSURE = 0x829A
TAG_FNUMBER = 0x829D
TAG_ISO = 0x8827
TAG_FOCAL = 0x920A
TAG_DATETIME_ORIGINAL = 0x9003

# The manifest doubles as the display parquet, so it carries every column
# ``build_result`` reads plus the bookkeeping the incremental diff needs.
MANIFEST_COLUMNS = [
    "photo_id", "path", "mtime", "size",
    "photo_image_url", "photo_url", "photographer",
    "photo_width", "photo_height", "blur_hash",
    "photo_description", "ai_description",
    "aperture", "focal_length", "exposure_s", "iso",
    "camera_make", "camera_model", "taken_at",
]
_FLOAT_COLUMNS = ("mtime", "aperture", "focal_length", "exposure_s")
_INT_COLUMNS = ("size", "photo_width", "photo_height")

_heif_registered = False


def register_heif() -> None:
    """Teach Pillow to open HEIC/HEIF — one call, then iPhone photos just work.

    Idempotent, and a missing ``pillow-heif`` degrades to "HEIC files fail to
    open" rather than taking the whole run down.
    """
    global _heif_registered
    if _heif_registered:
        return
    _heif_registered = True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:  # pragma: no cover - dependency is declared; belt and braces
        print("[library] WARNING: pillow-heif not installed; HEIC files will be skipped")


def library_dir(data_dir: Path | str = DATA_DIR) -> Path:
    return Path(data_dir) / "library"


def manifest_path(data_dir: Path | str = DATA_DIR) -> Path:
    return library_dir(data_dir) / "manifest.parquet"


def thumb_dir(data_dir: Path | str = DATA_DIR) -> Path:
    return library_dir(data_dir) / "thumbs"


def thumb_path(photo_id: str, data_dir: Path | str = DATA_DIR) -> Path:
    return thumb_dir(data_dir) / f"{photo_id}.jpg"


def photo_id_for(path: Path | str) -> str:
    """A stable id for a file: 16 hex chars of SHA-1 over its absolute path.

    Path-derived (not content-derived) so that *re-editing* a photo updates the
    same row instead of orphaning one — mtime/size in the manifest is what marks
    it dirty. Case-folded on Windows, where two spellings are the same file.
    """
    key = str(Path(path).resolve())
    if os.name == "nt":
        key = key.lower()
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class ScannedFile:
    """One image file on disk, with the two stat fields the diff compares."""

    path: str
    mtime: float
    size: int
    photo_id: str


def scan_folder(root: Path | str) -> list[ScannedFile]:
    """Recursively find indexable images under ``root``, sorted by path.

    Skips dot-directories and our own ``thumbs/`` output, so pointing the indexer
    at a folder that contains the library dir can't feed thumbnails back in.
    """
    root = Path(root).resolve()
    out: list[ScannedFile] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if "thumbs" in path.parts and path.parent.parent.name == "library":
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        out.append(
            ScannedFile(
                path=str(path),
                mtime=round(stat.st_mtime, 3),
                size=int(stat.st_size),
                photo_id=photo_id_for(path),
            )
        )
    return out


def _rational(value: object) -> float | None:
    """A RATIONAL EXIF value -> plain float. This is the one that bites you.

    Aperture, shutter and focal length are stored as *fractions* on disk, and
    depending on the file and the Pillow version you get back an ``IFDRational``,
    a plain float, or a bare ``(numerator, denominator)`` tuple. All three have to
    become the same number: a 1/500s exposure read as the tuple's first element is
    ``1.0`` — a one-second exposure — and nothing downstream would ever notice.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            num, den = value
            return float(num) / float(den) if den else None
        value = value[0] if len(value) else None
    return _num(value)


def _num(value: object) -> float | None:
    """A scalar (or first-of-array) EXIF value -> plain float, or None."""
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore").strip("\x00").strip()
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _text(value: object) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    if value is None:
        return None
    text = str(value).strip("\x00").strip()
    return text or None


def parse_taken_at(value: object) -> str | None:
    """``"2024:07:14 18:32:01"`` (the EXIF format, colons in the date) -> ISO string."""
    text = _text(value)
    if text is None:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            # Deliberately naive: DateTimeOriginal carries no zone (OffsetTime is a
            # separate, usually absent, tag). Attaching one here would be inventing data.
            return datetime.strptime(text, fmt).isoformat(sep=" ")  # noqa: DTZ007
        except ValueError:
            continue
    return None


def read_exif(image: object) -> dict:
    """Map a PIL image's EXIF onto the Session 2 schema (``None`` for anything absent).

    Every value goes through :mod:`photosearch.exif`, the same parsers the Unsplash
    TSV uses — that shared funnel is what makes one ``FilterSpec`` work over two
    completely different sources.
    """
    empty = {
        "aperture": None, "focal_length": None, "exposure_s": None, "iso": None,
        "camera_make": None, "camera_model": None, "taken_at": None,
    }
    try:
        exif = image.getexif()
    except Exception:  # noqa: BLE001 - a corrupt/absent EXIF block is "no metadata"
        return empty
    if not exif:
        return empty

    try:
        sub = exif.get_ifd(EXIF_IFD)
    except Exception:  # noqa: BLE001 - same
        sub = {}

    def pick(tag: int) -> object:
        # Most cameras write these into the Exif sub-IFD; a few leave them in IFD0.
        value = sub.get(tag)
        return exif.get(tag) if value is None else value

    return {
        "aperture": parse_aperture(_rational(pick(TAG_FNUMBER))),
        "focal_length": parse_focal_length(_rational(pick(TAG_FOCAL))),
        "exposure_s": parse_exposure(_rational(pick(TAG_EXPOSURE))),
        # ISOSpeedRatings is a SHORT *array*, not a rational — first value wins.
        "iso": parse_iso(_num(pick(TAG_ISO))),
        "camera_make": normalize_make(_text(exif.get(TAG_MAKE))),
        "camera_model": normalize_model(_text(exif.get(TAG_MODEL))),
        "taken_at": parse_taken_at(pick(TAG_DATETIME_ORIGINAL)),
    }


def load_photo(path: Path | str):
    """Open one file -> ``(rgb_image, exif_dict, (width, height))``.

    EXIF is read *before* ``exif_transpose`` (which consumes the orientation tag),
    and the image is fully decoded before the file handle closes.
    """
    from PIL import Image, ImageOps

    register_heif()
    # ``MAX_IMAGE_PIXELS`` is global module state, so the raise is scoped with a
    # try/finally: leaving it lifted would quietly weaken the *upload* endpoint's
    # bomb guard in the same process, since the API imports this module for
    # ``LibraryStore``. Raise it for one local file, then put it back.
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = LOCAL_MAX_PIXELS
    try:
        with Image.open(path) as opened:
            exif = read_exif(opened)
            upright = ImageOps.exif_transpose(opened)
            rgb = upright.convert("RGB")
            rgb.load()  # force the decode while the file is still open
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    return rgb, exif, rgb.size


def write_thumbnail(image, dest: Path) -> None:
    """Downscale a copy to ``THUMB_MAX`` on the long edge and save it as JPEG."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = image.copy()
    thumb.thumbnail((THUMB_MAX, THUMB_MAX))
    thumb.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)


def library_row(scanned: ScannedFile, exif: dict, width: int, height: int) -> dict:
    """One manifest row, shaped so ``build_result`` renders it like any other photo.

    The URLs are our own endpoints rather than a CDN, and ``photographer`` is the
    containing folder — for a personal archive, "Iceland 2024" is the useful
    caption where a photographer's name would go.
    """
    path = Path(scanned.path)
    return {
        "photo_id": scanned.photo_id,
        "path": str(path),
        "mtime": scanned.mtime,
        "size": scanned.size,
        "photo_image_url": f"/api/photo/{scanned.photo_id}/thumb",
        "photo_url": f"/api/photo/{scanned.photo_id}/full",
        "photographer": path.parent.name or "Local library",
        "photo_width": width,
        "photo_height": height,
        "blur_hash": None,  # no blurhash for local files; the grid falls back cleanly
        "photo_description": path.name,
        "ai_description": None,
        "aperture": exif["aperture"],
        "focal_length": exif["focal_length"],
        "exposure_s": exif["exposure_s"],
        "iso": exif["iso"],
        "camera_make": exif["camera_make"],
        "camera_model": exif["camera_model"],
        "taken_at": exif["taken_at"],
    }


def empty_manifest() -> pd.DataFrame:
    return _coerce_manifest(pd.DataFrame(columns=MANIFEST_COLUMNS))


def _coerce_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Pin dtypes so an all-missing column never flips type between runs.

    Without this, a first run where no photo has an aperture yields an ``object``
    column, and the next run's ``$lte`` filter silently matches nothing.
    """
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[MANIFEST_COLUMNS].copy()
    for col in _FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    df["iso"] = pd.to_numeric(df["iso"], errors="coerce").astype("Int64")
    for col in ("photo_id", "path", "photo_image_url", "photo_url", "photographer",
                "blur_hash", "photo_description", "ai_description",
                "camera_make", "camera_model", "taken_at"):
        df[col] = df[col].astype("object")
    return df.reset_index(drop=True)


def load_manifest(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """The sidecar index, or an empty (correctly typed) frame if nothing is indexed."""
    path = manifest_path(data_dir)
    if not path.exists():
        return empty_manifest()
    return _coerce_manifest(pd.read_parquet(path))


def save_manifest(df: pd.DataFrame, data_dir: Path | str = DATA_DIR) -> None:
    path = manifest_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_manifest(df).to_parquet(path, index=False)


@dataclass
class IndexPlan:
    """What one incremental run has to do — the whole point of Session 9a."""

    new: list[ScannedFile]
    changed: list[ScannedFile]
    unchanged: list[ScannedFile]
    deleted_ids: list[str]

    @property
    def to_embed(self) -> list[ScannedFile]:
        return self.new + self.changed

    def is_noop(self) -> bool:
        return not self.new and not self.changed and not self.deleted_ids


def plan_changes(manifest: pd.DataFrame, scanned: list[ScannedFile], root: Path) -> IndexPlan:
    """Diff the folder against the manifest: what's new, modified, gone, untouched.

    Deletions are scoped to ``root``: a manifest row from *another* indexed folder
    is left alone, so a multi-folder library doesn't wipe itself every run.
    """
    root = Path(root).resolve()
    known = {
        str(row.photo_id): (float(row.mtime), int(row.size), str(row.path))
        for row in manifest.itertuples()
    }

    new: list[ScannedFile] = []
    changed: list[ScannedFile] = []
    unchanged: list[ScannedFile] = []
    for item in scanned:
        prior = known.get(item.photo_id)
        if prior is None:
            new.append(item)
        elif abs(prior[0] - item.mtime) > 1e-3 or prior[1] != item.size:
            changed.append(item)
        else:
            unchanged.append(item)

    seen = {item.photo_id for item in scanned}
    deleted_ids = [
        pid
        for pid, (_, _, path) in known.items()
        if pid not in seen and _is_under(Path(path), root)
    ]
    return IndexPlan(new=new, changed=changed, unchanged=unchanged, deleted_ids=deleted_ids)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (ValueError, OSError):
        return False
    return True


def open_collection(data_dir: Path | str = DATA_DIR, create: bool = False):
    """The ``library`` Chroma collection — cosine space, same as the Unsplash one."""
    import chromadb

    data_dir = Path(data_dir)
    chroma_dir = data_dir / "chroma"
    if create:
        chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if create:
        return client.get_or_create_collection(
            LIBRARY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    return client.get_collection(LIBRARY_COLLECTION, embedding_function=None)


class LibraryStore(ChromaStore):
    """ChromaStore over the ``library`` collection, plus id -> file-path lookups.

    The path lookups are what make Session 9b's file endpoints safe: the URL only
    ever carries a photo id, and the *server* decides which bytes that maps to.
    A path can never arrive from the client, so there is no traversal to defend.
    """

    def __init__(self, collection, photos: pd.DataFrame, data_dir: Path | str = DATA_DIR) -> None:
        super().__init__(collection, photos)
        self.data_dir = Path(data_dir)

    @classmethod
    def load(cls, data_dir: Path | str = DATA_DIR, **_: object) -> LibraryStore:
        collection = open_collection(data_dir)
        return cls(collection, load_manifest(data_dir), data_dir)

    def original_path(self, photo_id: str) -> Path | None:
        """The on-disk original for a photo id, if it's indexed and still there."""
        try:
            path = Path(str(self._by_id.loc[str(photo_id), "path"]))
        except KeyError:
            return None
        return path if path.is_file() else None

    def thumbnail_path(self, photo_id: str) -> Path | None:
        if str(photo_id) not in self._by_id.index:
            return None
        path = thumb_path(str(photo_id), self.data_dir)
        return path if path.is_file() else None
