"""Session 3 — embed all 25,000 photos into a CLIP vector index.

This is the one-time offline batch job that makes search possible. Output:
  data/embeddings.npy   float32, shape (N, 512), L2-normalized, one row per photo
  data/photo_ids.npy    str, shape (N,), row-aligned join key
  data/photos.parquet   pruned to only the photos that were successfully indexed

The dataset ships no image bytes, only CDN URLs, so we fetch a *downsized* copy of
each photo (CLIP resizes to 224x224 internally, so full-res would be pure waste),
encode in batches, and checkpoint per chunk so a network hiccup never loses progress.

THE SACRED INVARIANT: row i of the embedding matrix must be the same photo as row i
of photo_ids and of photos.parquet. Concurrent downloads finish out of order, so we
reassemble each chunk in dataframe order before encoding, and validate ID equality
(not just shape) at the end. A shape check can't catch scrambled rows; ID equality can.

Usage:
  uv run python scripts/02_build_index.py --sample 100   # time it, print an ETA
  uv run python scripts/02_build_index.py                # full run (resumable)
  uv run python scripts/02_build_index.py --sanity "a dog"   # DoD sanity check
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "photos.parquet"
CHUNK_DIR = DATA_DIR / "index_chunks"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "photo_ids.npy"
FAILED_PATH = DATA_DIR / "index_chunks" / "failed_ids.txt"

MODEL_NAME = "clip-ViT-B-32"
IMAGE_PARAMS = "?w=336&q=80"  # imgix params on the Unsplash CDN; CLIP only needs ~224px
CHUNK_SIZE = 500
DOWNLOAD_WORKERS = 12
ENCODE_BATCH = 32
HTTP_TIMEOUT = 15.0


def fetch_image(client: httpx.Client, url: str) -> Image.Image | None:
    """Download one image (1 retry). Returns an RGB PIL image, or None on failure."""
    full_url = url + IMAGE_PARAMS
    for attempt in range(2):
        try:
            resp = client.get(full_url, timeout=HTTP_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            image.load()  # force decode now, while the byte buffer is alive
            return image
        except Exception:  # noqa: BLE001 - deleted photos 404, timeouts happen; both are "skip"
            if attempt == 1:
                return None
    return None


def download_in_order(
    client: httpx.Client, urls: list[str]
) -> list[Image.Image | None]:
    """Fetch a chunk concurrently but return results in the ORIGINAL order.

    Futures are keyed by their position so out-of-order completion can't scramble
    the row alignment — the bug class that yields "search returns random photos".
    """
    results: list[Image.Image | None] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        future_to_pos = {pool.submit(fetch_image, client, url): i for i, url in enumerate(urls)}
        for future, pos in future_to_pos.items():
            results[pos] = future.result()
    return results


def load_model():
    from sentence_transformers import SentenceTransformer

    print(f"[model] loading {MODEL_NAME} (first run downloads ~600 MB)...")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    print(f"[model] ready in {time.perf_counter() - t0:.1f}s")
    return model


def encode(model, images: list[Image.Image]) -> np.ndarray:
    embeddings = model.encode(
        images,
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def process_chunk(
    model, client: httpx.Client, chunk: pd.DataFrame
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (embeddings, success_ids, failed_ids) for one chunk, in dataframe order."""
    urls = chunk["photo_image_url"].tolist()
    ids = chunk["photo_id"].tolist()
    images = download_in_order(client, urls)

    keep_images: list[Image.Image] = []
    keep_ids: list[str] = []
    failed_ids: list[str] = []
    for img, pid in zip(images, ids, strict=True):
        if img is None:
            failed_ids.append(pid)
        else:
            keep_images.append(img)
            keep_ids.append(pid)

    if keep_images:
        embeddings = encode(model, keep_images)
    else:
        embeddings = np.empty((0, 512), dtype=np.float32)
    return embeddings, keep_ids, failed_ids


def run_sample(n: int) -> None:
    """Time downloads + encoding on the first n rows and extrapolate an ETA."""
    df = pd.read_parquet(PARQUET_PATH).head(n)
    model = load_model()
    client = httpx.Client()

    t0 = time.perf_counter()
    images = download_in_order(client, df["photo_image_url"].tolist())
    t_download = time.perf_counter() - t0
    ok = [im for im in images if im is not None]
    n_ok = len(ok)
    n_fail = len(images) - n_ok

    t1 = time.perf_counter()
    if ok:
        encode(model, ok)
    t_encode = time.perf_counter() - t1
    client.close()

    total = len(pd.read_parquet(PARQUET_PATH, columns=["photo_id"]))
    dl_rate = n / t_download if t_download else 0
    enc_rate = n_ok / t_encode if t_encode else 0
    eta_min = (total / dl_rate + total / enc_rate) / 60 if dl_rate and enc_rate else float("nan")

    print("\n===== sample timing =====")
    print(f"sample size:      {n}  ({n_ok} ok, {n_fail} failed)")
    print(f"download:         {t_download:6.1f}s  -> {dl_rate:6.1f} img/s ({DOWNLOAD_WORKERS} workers)")
    print(f"encode (CPU):     {t_encode:6.1f}s  -> {enc_rate:6.1f} img/s (batch {ENCODE_BATCH})")
    print(f"full corpus:      {total:,} photos")
    print(f"estimated total:  ~{eta_min:.0f} min  (download + encode, run unattended)")
    print("=========================\n")


def run_full() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    n = len(df)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    client = httpx.Client()
    print(f"[index] {n:,} photos in {n_chunks} chunks of {CHUNK_SIZE}")

    t_start = time.perf_counter()
    for c in range(n_chunks):
        chunk_path = CHUNK_DIR / f"chunk_{c:04d}.npz"
        if chunk_path.exists():
            continue  # resume: already done
        chunk = df.iloc[c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
        embeddings, ok_ids, failed_ids = process_chunk(model, client, chunk)
        np.savez(
            chunk_path,
            embeddings=embeddings,
            photo_ids=np.array(ok_ids, dtype=object),
            failed_ids=np.array(failed_ids, dtype=object),
        )
        done = c + 1
        elapsed = time.perf_counter() - t_start
        rate = done / elapsed if elapsed else 0
        eta = (n_chunks - done) / rate / 60 if rate else 0
        print(
            f"[chunk {done:>3}/{n_chunks}] +{len(ok_ids)} ok, {len(failed_ids)} failed "
            f"| {elapsed / 60:.1f} min elapsed, ~{eta:.0f} min left"
        )

    client.close()
    finalize(df, n_chunks)


def finalize(df: pd.DataFrame, n_chunks: int) -> None:
    """Concatenate chunks in order, validate ID alignment, save final artifacts."""
    emb_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    failed: list[str] = []
    for c in range(n_chunks):
        data = np.load(CHUNK_DIR / f"chunk_{c:04d}.npz", allow_pickle=True)
        if data["embeddings"].shape[0]:
            emb_parts.append(data["embeddings"])
            id_parts.append(data["photo_ids"].astype(str))
        failed.extend(data["failed_ids"].astype(str).tolist())

    embeddings = np.concatenate(emb_parts, axis=0).astype(np.float32)
    photo_ids = np.concatenate(id_parts, axis=0).astype(str)

    # Prune parquet to indexed rows, preserving dataframe order == embedding order.
    ok_set = set(photo_ids.tolist())
    pruned = df[df["photo_id"].isin(ok_set)].reset_index(drop=True)

    # THE alignment check — element-wise IDs, not just shapes.
    assert len(pruned) == len(photo_ids), f"count mismatch: {len(pruned)} vs {len(photo_ids)}"
    assert (pruned["photo_id"].to_numpy().astype(str) == photo_ids).all(), (
        "ROW ALIGNMENT BROKEN: photo_ids do not match parquet order"
    )

    norm = float(np.linalg.norm(embeddings[0]))
    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, photo_ids)
    pruned.to_parquet(PARQUET_PATH, index=False)
    FAILED_PATH.write_text("\n".join(failed) + ("\n" if failed else ""))

    print("\n===== index built =====")
    print(f"indexed:   {len(photo_ids):,} photos  (shape {embeddings.shape}, {embeddings.dtype})")
    print(f"failed:    {len(failed):,}  (ids saved to {FAILED_PATH.name})")
    print(f"norm[0]:   {norm:.4f}  (should be ~1.0)")
    print(f"saved:     {EMB_PATH.name}, {IDS_PATH.name}, photos.parquet (pruned)")
    print("alignment: OK (element-wise id check passed)")
    print("=======================\n")


def run_sanity(query: str) -> None:
    """DoD check: encode a text query, print the top-5 matching photo URLs."""
    embeddings = np.load(EMB_PATH)
    photo_ids = np.load(IDS_PATH)
    df = pd.read_parquet(PARQUET_PATH).set_index("photo_id")
    model = load_model()

    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    scores = embeddings @ q
    top = np.argsort(-scores)[:5]
    print(f'\ntop 5 for "{query}":')
    for rank, i in enumerate(top, 1):
        pid = str(photo_ids[i])
        print(f"  {rank}. {scores[i]:.3f}  {df.loc[pid, 'photo_image_url']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CLIP photo index.")
    parser.add_argument("--sample", type=int, metavar="N", help="time N rows and print an ETA")
    parser.add_argument("--sanity", type=str, metavar="QUERY", help="encode QUERY, show top-5")
    args = parser.parse_args()

    if args.sanity:
        run_sanity(args.sanity)
    elif args.sample:
        run_sample(args.sample)
    else:
        run_full()


if __name__ == "__main__":
    main()
