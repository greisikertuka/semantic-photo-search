"""Session 2 — download the Unsplash Lite metadata, clean it, save photos.parquet.

Why download the zip directly instead of `load_dataset("jamescalam/unsplash-25k-photos")`?
That HF dataset is script-based, and the `datasets` library dropped script support in
v4. The script only ever wrapped this same official TSV anyway. So we fetch
https://unsplash.com/data/lite/latest (~305 MB, no signup) and load it with pandas.

The TSV stores EXIF as strings with many blanks; we parse the photographically useful
fields into clean numeric types (via photosearch.exif) so Session 7 can filter on them.

Run:  uv run python scripts/01_prepare_dataset.py
Data licensing: the TSV and images are never committed (data/ is gitignored).
"""

from __future__ import annotations

import csv
import sys
import urllib.request
import zipfile
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

LITE_URL = "https://unsplash.com/data/lite/latest"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "unsplash-lite-latest.zip"
TSV_NAME = "photos.tsv000"
TSV_PATH = RAW_DIR / TSV_NAME
OUT_PATH = PROJECT_ROOT / "data" / "photos.parquet"
SNAPSHOT_PATH = RAW_DIR / "DATASET_SNAPSHOT.txt"

# The `/latest` endpoint is a moving target; record which snapshot we pinned so the
# embeddings and eval labels downstream all refer to one frozen corpus.
_UA = "semantic-photo-search/0.1 (dataset prep; +https://github.com/)"


def download_zip() -> str | None:
    """Download the Lite zip if absent. Returns the server's Last-Modified string."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"[skip] zip already present: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.0f} MB)")
        return None

    print(f"[download] {LITE_URL}  -> {ZIP_PATH}")
    request = urllib.request.Request(LITE_URL, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request) as response:
        last_modified = response.headers.get("Last-Modified")
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_mark = 20_000_000
        tmp = ZIP_PATH.with_suffix(".zip.part")
        with tmp.open("wb") as out:
            while chunk := response.read(1 << 20):  # 1 MB chunks
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_mark:
                    pct = f"{downloaded / total * 100:4.0f}%" if total else "  ? "
                    print(f"    {downloaded / 1e6:6.0f} MB  {pct}")
                    next_mark += 20_000_000
        tmp.replace(ZIP_PATH)
    print(f"[download] done: {downloaded / 1e6:.0f} MB. Last-Modified: {last_modified}")
    if last_modified:
        SNAPSHOT_PATH.write_text(f"Unsplash Lite snapshot (Last-Modified): {last_modified}\n")
    return last_modified


def extract_tsv() -> None:
    """Extract only photos.tsv000 from the zip (we ignore keywords/collections/etc.)."""
    if TSV_PATH.exists():
        print(f"[skip] tsv already extracted: {TSV_PATH}")
        return
    print(f"[extract] {TSV_NAME} from {ZIP_PATH.name}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        if TSV_NAME not in names:
            sys.exit(f"[error] {TSV_NAME} not found in zip. Contents: {names}")
        with zf.open(TSV_NAME) as src, TSV_PATH.open("wb") as dst:
            dst.write(src.read())
    print(f"[extract] done: {TSV_PATH} ({TSV_PATH.stat().st_size / 1e6:.0f} MB)")


def load_tsv() -> pd.DataFrame:
    """Load the TSV, with a quoting fallback if the naive read scrambles rows."""
    df = pd.read_csv(TSV_PATH, sep="\t")
    if not (24_000 <= len(df) <= 26_000):
        print(f"[warn] naive read gave {len(df)} rows; retrying with QUOTE_NONE")
        df = pd.read_csv(TSV_PATH, sep="\t", quoting=csv.QUOTE_NONE)
    print(f"[load] {len(df):,} rows x {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Select display + EXIF fields; parse EXIF strings into numeric dtypes."""
    out = pd.DataFrame(
        {
            # identity / display
            "photo_id": df["photo_id"],
            "photo_image_url": df["photo_image_url"],
            "photo_url": df["photo_url"],
            "photographer_first_name": df["photographer_first_name"],
            "photographer_last_name": df["photographer_last_name"],
            "photo_width": df["photo_width"],
            "photo_height": df["photo_height"],
            "blur_hash": df["blur_hash"],
            "photo_description": df["photo_description"],
            "ai_description": df["ai_description"],
        }
    )
    # EXIF -> numeric (None -> NaN); iso uses the nullable Int64 dtype.
    out["aperture"] = df["exif_aperture_value"].map(parse_aperture).astype("float64")
    out["focal_length"] = df["exif_focal_length"].map(parse_focal_length).astype("float64")
    out["exposure_s"] = df["exif_exposure_time"].map(parse_exposure).astype("float64")
    out["iso"] = df["exif_iso"].map(parse_iso).astype("Int64")
    out["camera_make"] = df["exif_camera_make"].map(normalize_make)
    out["camera_model"] = df["exif_camera_model"].map(normalize_model)
    return out


def quality_report(df: pd.DataFrame) -> None:
    n = len(df)
    print("\n===== data quality report =====")
    print(f"rows: {n:,}")
    print("\nEXIF coverage (non-null):")
    for col in ("aperture", "focal_length", "exposure_s", "iso", "camera_make", "camera_model"):
        have = int(df[col].notna().sum())
        print(f"  {col:14s} {have:6,d}  ({have / n * 100:5.1f}%)")

    print("\nvalue ranges (sanity — apertures ~0.95-32, iso ~25-25600):")
    for col in ("aperture", "focal_length", "exposure_s", "iso"):
        s = df[col].dropna()
        if len(s):
            print(f"  {col:14s} min={s.min():>10}  max={s.max():>10}")

    wide = int((df["aperture"] <= 1.8).sum())
    print(f"\nphotos shot at f/1.8 or wider: {wide:,}")
    print("===============================\n")


def main() -> None:
    download_zip()
    extract_tsv()
    df_raw = load_tsv()
    df = clean(df_raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[save] {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    quality_report(df)


if __name__ == "__main__":
    main()
