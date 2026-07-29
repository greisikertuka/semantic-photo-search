"""Session 11 — build the artifacts the public Space (and the Session 12 release) ships.

    uv run python scripts/06_build_space_artifacts.py
    uv run python scripts/06_build_space_artifacts.py --verify-only

Two transformations, each a deliberate deploy decision:

* **float32 -> float16 embeddings.** 51 MB -> 26 MB for a cosine error around 1e-3,
  which is three orders of magnitude below the gap between adjacent search results.
  The subtlety worth knowing: fp16 is a *storage* format, not a compute format —
  NumPy has no fast fp16 matmul, so the Space loads it and immediately calls
  ``.astype(np.float32)``. Skipping that turns a 5 ms search into a ~300 ms one.
* **The full parquet -> a slim display parquet.** Only the columns needed to *render
  a result and credit its photographer* survive: ids, the two URLs, the name,
  blur_hash, dimensions, and the numeric EXIF the filters run on. The Unsplash
  descriptions are deliberately dropped — the app never displayed them in the Space,
  and not republishing them keeps the "derived artifacts only, corpus not
  reconstructible" licensing argument (DECISIONS.md) an honest one.

``photo_ids.npy`` ships alongside even though the parquet has the same column: it
is what keeps :class:`~photosearch.store.NumpyStore`'s alignment assert a *real*
check at Space startup, so a half-synced deploy (new embeddings, stale parquet)
fails loudly instead of serving plausible-looking wrong photos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "space"

# Everything ``build_result`` reads to render a card, and nothing else. ``photographer``
# is pre-joined from the dataset's two name columns so the Space ships one string.
SLIM_COLUMNS = [
    "photo_id",
    "photo_image_url",
    "photo_url",
    "photographer",
    "photo_width",
    "photo_height",
    "blur_hash",
    "aperture",
    "focal_length",
    "exposure_s",
    "iso",
    "camera_make",
    "camera_model",
]

PROBE_SEED = 20260729
PROBE_COUNT = 40
PROBE_K = 10


def build_slim_parquet(photos: pd.DataFrame) -> pd.DataFrame:
    """Trim the working parquet to the display minimum, names pre-joined."""
    slim = pd.DataFrame(index=photos.index)
    first = photos["photographer_first_name"].astype("string").fillna("")
    last = photos["photographer_last_name"].astype("string").fillna("")
    slim["photographer"] = (first + " " + last).str.strip().replace("", "Unknown")
    for col in SLIM_COLUMNS:
        if col == "photographer":
            continue
        slim[col] = photos[col]
    return slim[SLIM_COLUMNS].reset_index(drop=True)


def verify(embeddings: np.ndarray, half: np.ndarray) -> None:
    """Prove the fp16 round trip is harmless *before* trusting it in production.

    No CLIP model needed: random unit vectors probe the same geometry real queries
    do, and a fixed seed makes the numbers reproducible. Two things get measured —
    how far the scores move (should be ~1e-3) and whether the top-10 *ranking* moves
    at all (it should not, because the score gaps between neighbouring results are
    far larger than the quantization error).
    """
    rng = np.random.default_rng(PROBE_SEED)
    probes = rng.standard_normal((PROBE_COUNT, embeddings.shape[1])).astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)

    # This is the line the Space repeats at startup — fp16 storage, fp32 compute.
    restored = half.astype(np.float32)

    norm_err = np.abs(np.linalg.norm(restored, axis=1) - 1.0).max()
    max_score_delta = 0.0
    identical_topk = 0
    for probe in probes:
        exact = embeddings @ probe
        approx = restored @ probe
        max_score_delta = max(max_score_delta, float(np.abs(exact - approx).max()))
        top_exact = np.argsort(-exact)[:PROBE_K]
        top_approx = np.argsort(-approx)[:PROBE_K]
        identical_topk += int(np.array_equal(top_exact, top_approx))

    print(f"[verify] {PROBE_COUNT} random unit-vector probes, top-{PROBE_K}")
    print(f"[verify]   max |score_fp32 - score_fp16| : {max_score_delta:.2e}")
    print(f"[verify]   max deviation of ||v|| from 1 : {norm_err:.2e}")
    print(f"[verify]   probes with an identical top-{PROBE_K} order: "
          f"{identical_topk}/{PROBE_COUNT}")
    if max_score_delta > 1e-2:
        raise SystemExit(f"fp16 error {max_score_delta:.2e} is too large — refusing to ship")


def human(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Space/release artifacts.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--verify-only", action="store_true", help="check, write nothing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    embeddings = np.load(data_dir / "embeddings.npy").astype(np.float32)
    photo_ids = np.load(data_dir / "photo_ids.npy", allow_pickle=True).astype(str)
    photos = pd.read_parquet(data_dir / "photos.parquet")

    # The invariant, re-asserted at the one point where the artifacts get rewritten.
    if not np.array_equal(photos["photo_id"].to_numpy().astype(str), photo_ids):
        raise SystemExit("row alignment broken: photo_ids != photos.photo_id order")
    print(f"[load] {len(photo_ids):,} photos x {embeddings.shape[1]} dims (aligned)")

    half = embeddings.astype(np.float16)
    verify(embeddings, half)
    if args.verify_only:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.f16.npy"
    ids_path = out_dir / "photo_ids.npy"
    slim_path = out_dir / "photos.slim.parquet"

    np.save(emb_path, half)
    np.save(ids_path, photo_ids)
    build_slim_parquet(photos).to_parquet(slim_path, index=False)

    src_emb = (data_dir / "embeddings.npy").stat().st_size
    src_parquet = (data_dir / "photos.parquet").stat().st_size
    total = sum(p.stat().st_size for p in (emb_path, ids_path, slim_path))
    print(f"[write] {emb_path.name:<22} {human(emb_path)}  (from {src_emb / 1e6:.1f} MB fp32)")
    print(f"[write] {ids_path.name:<22} {human(ids_path)}")
    print(f"[write] {slim_path.name:<22} {human(slim_path)}  (from {src_parquet / 1e6:.1f} MB)")
    print(f"[done]  {total / 1e6:.1f} MB total in {out_dir}")
    print("[note]  the Space loads the fp16 file and immediately .astype(float32)s it — "
          "fp16 is a storage format, not a compute one.")


if __name__ == "__main__":
    main()
