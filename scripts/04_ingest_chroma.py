"""Session 7 — ingest the CLIP index into an embedded Chroma vector DB.

Takes the artifacts Session 3 produced (``embeddings.npy`` + ``photo_ids.npy`` +
``photos.parquet``) and loads them into a persistent Chroma collection at
``data/chroma``, so ``ChromaStore`` can do vector search *fused* with EXIF metadata
filters — the differentiator this whole project is built around.

Two decisions bake in here (see DECISIONS.md):

* The collection is created with **cosine** space, not Chroma's default L2 — so the
  distances it returns convert cleanly to the cosine similarities the app speaks
  (``similarity = 1 - distance``), keeping the Session 6 score bands honest.
* Only the **filterable** EXIF fields go in as metadata (aperture, iso,
  focal_length, exposure_s, camera_make/model), typed as *numbers* so ``$lte``/
  ``$gte`` work. Display fields stay in the parquet — one source of truth, joined
  back by photo_id. Per-photo NaN fields are dropped, so no-EXIF photos correctly
  fall out of any numeric filter.

Usage:
  uv run python scripts/04_ingest_chroma.py            # build (idempotent; --reset to rebuild)
  uv run python scripts/04_ingest_chroma.py --reset    # drop & rebuild the collection
  uv run python scripts/04_ingest_chroma.py --verify   # model-free parity vs NumpyStore
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from photosearch.models import FilterSpec
from photosearch.store import (
    COLLECTION_NAME,
    ChromaStore,
    NumpyStore,
    exif_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "photo_ids.npy"
PARQUET_PATH = DATA_DIR / "photos.parquet"

# Chroma caps a single add() at ~5.5k records; stay comfortably under it.
ADD_BATCH = 5000


def build_metadatas(photos: pd.DataFrame) -> list[dict]:
    """One metadata dict per row, in dataframe (== embedding) order."""
    return [exif_metadata(row) for _, row in photos.iterrows()]


def ingest(reset: bool) -> None:
    import chromadb

    embeddings = np.load(EMB_PATH).astype(np.float32)
    photo_ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    photos = pd.read_parquet(PARQUET_PATH)

    # The sacred invariant, re-checked at the DB boundary: row i of the embedding
    # matrix must be the same photo as row i of the parquet. A scrambled ingest
    # would poison every filtered query with no error anywhere.
    assert len(embeddings) == len(photo_ids) == len(photos), "length mismatch"
    assert (photos["photo_id"].to_numpy().astype(str) == photo_ids).all(), (
        "ROW ALIGNMENT BROKEN: photo_ids != parquet order"
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[chroma] dropped existing collection {COLLECTION_NAME!r}")
        except Exception as exc:  # noqa: BLE001 - fine if it didn't exist yet
            print(f"[chroma] (no existing collection to drop: {exc})")

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # NOT the L2 default — see module docstring
        embedding_function=None,  # we always supply precomputed vectors; no model needed
    )

    already = collection.count()
    if already and not reset:
        print(
            f"[chroma] collection already holds {already:,} vectors — "
            f"pass --reset to rebuild. Nothing to do."
        )
        return

    ids = photo_ids.tolist()
    metadatas = build_metadatas(photos)
    n = len(ids)
    n_with_exif = sum(1 for m in metadatas if m["has_exif"])
    print(f"[chroma] ingesting {n:,} vectors ({n_with_exif:,} with EXIF metadata)...")

    t0 = time.perf_counter()
    for start in range(0, n, ADD_BATCH):
        end = min(start + ADD_BATCH, n)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        print(f"[chroma]   added {end:,}/{n:,}")

    final = collection.count()
    assert final == n, f"count mismatch after ingest: {final} != {n}"
    print(
        f"[chroma] done — {final:,} vectors in {time.perf_counter() - t0:.1f}s "
        f"at {CHROMA_DIR}"
    )


# Parity thresholds. NumpyStore is *exact* brute force; ChromaStore rides an
# *approximate* HNSW index — so the honest invariant is NOT bit-identical id lists.
# We decompose parity into two guarantees that test different things:
#   * SAME-ID SCORE agreement: for a photo BOTH stores return, the score must match
#     to floating-point precision. This is the pure correctness check — it proves the
#     collection is cosine (not L2) and distance→similarity is converted right, and
#     it is immune to recall differences. A metric/conversion bug would show up here
#     as a gap of order ~0.1–1.0 (L2 distances live on a different scale entirely).
#   * TOP-K OVERLAP measures ANN recall. It's <1.0 by design: HNSW can swap items
#     near the rank-k boundary, and genuine score ties / duplicate photos make the
#     exact order ambiguous anyway. We gate it loosely — a *low* overlap would mean
#     a real bug (e.g. a filter compiled wrong), not mere approximation.
SCORE_GAP_MAX = 1e-4  # |numpy.score - chroma.score| for an id both stores returned
OVERLAP_MIN = 0.85  # mean |n_ids ∩ c_ids| / k across all probes


def verify(n_probes: int, seed: int) -> None:
    """Prove both stores are the same seam: equal score profiles, high recall overlap.

    Model-free — we use *stored image embeddings* as the query vectors, so no CLIP
    download is needed. This is the Session 7 definition-of-done parity check, run
    as a script instead of by hand in Swagger. See the threshold notes above for why
    "identical top-10" is the wrong bar for an exact-vs-approximate comparison.
    """
    numpy_store = NumpyStore.load(DATA_DIR)
    chroma_store = ChromaStore.load(DATA_DIR)
    print(
        f"[verify] numpy={numpy_store.count():,}  chroma={chroma_store.count():,}  "
        f"exif={numpy_store.exif_count:,}"
    )

    rng = random.Random(seed)
    probes = rng.sample(range(numpy_store.count()), min(n_probes, numpy_store.count()))
    k = 10
    filters_to_try = [None, FilterSpec(aperture_max=2.8), FilterSpec(iso_max=800)]

    max_score_gap = 0.0
    overlaps: list[float] = []
    checks = 0
    for i in probes:
        vec = numpy_store.embeddings[i]
        for filters in filters_to_try:
            n_res = numpy_store.search(vec, k=k, filters=filters)
            c_res = chroma_store.search(vec, k=k, filters=filters)
            checks += 1

            n_scores = {r.photo_id: r.score for r in n_res}
            c_scores = {r.photo_id: r.score for r in c_res}
            # same-id score agreement: only ids BOTH stores returned
            for pid in n_scores.keys() & c_scores.keys():
                max_score_gap = max(max_score_gap, abs(n_scores[pid] - c_scores[pid]))

            denom = max(len(n_scores), 1)
            overlaps.append(len(n_scores.keys() & c_scores.keys()) / denom)

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(
        f"[verify] {checks} query pairs | max same-id score gap: {max_score_gap:.2e} "
        f"(<= {SCORE_GAP_MAX:.0e}) | mean top-{k} overlap: {mean_overlap:.3f} "
        f"(>= {OVERLAP_MIN})"
    )
    if max_score_gap <= SCORE_GAP_MAX and mean_overlap >= OVERLAP_MIN:
        print(
            "[verify] PARITY OK — same scores from both stores; the small ranking "
            "differences are HNSW approximation + score ties, exactly as expected."
        )
    else:
        print("[verify] PARITY FAILED — investigate before trusting the swap.")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the CLIP index into Chroma.")
    parser.add_argument("--reset", action="store_true", help="drop & rebuild the collection")
    parser.add_argument("--verify", action="store_true", help="parity-check vs NumpyStore")
    parser.add_argument("--probes", type=int, default=8, help="verify: number of probe queries")
    parser.add_argument("--seed", type=int, default=7, help="verify: RNG seed")
    args = parser.parse_args()

    if args.verify:
        verify(args.probes, args.seed)
    else:
        ingest(args.reset)


if __name__ == "__main__":
    main()
