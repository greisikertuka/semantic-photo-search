"""Session 9a — index your own photo folder into a searchable `library` collection.

    uv run python scripts/05_index_folder.py --path "D:\\Photos"
    uv run python scripts/05_index_folder.py --path "D:\\Photos" --dry-run   # plan only
    uv run python scripts/05_index_folder.py --stats                        # what's indexed
    uv run python scripts/05_index_folder.py --path "D:\\Photos" --reset     # from scratch

The interesting property is that re-running is *cheap*. The first run embeds
everything; the second embeds only what changed since. That is the difference
between a demo script and an index you could actually keep pointed at a live
folder — and it is why Session 7 adopted a vector DB in the first place, since
add/upsert/delete-by-id are exactly the operations a NumPy file can't do.

The scan is deliberately three separate phases (scan -> plan -> apply) with the
plan printable via ``--dry-run``: an indexer you can't ask "what would you do?"
is one you won't trust to point at 20,000 personal photos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from photosearch.library import (
    LIBRARY_COLLECTION,
    IndexPlan,
    ScannedFile,
    library_dir,
    library_row,
    load_manifest,
    load_photo,
    open_collection,
    plan_changes,
    save_manifest,
    scan_folder,
    thumb_path,
    write_thumbnail,
)
from photosearch.store import exif_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EMBED_BATCH = 16  # images per CLIP forward pass on CPU
REPORT_EVERY = 50


def print_plan(plan: IndexPlan, root: Path) -> None:
    print(f"[scan] {root}")
    print(
        f"[plan] {len(plan.new)} new, {len(plan.changed)} modified, "
        f"{len(plan.unchanged)} unchanged, {len(plan.deleted_ids)} to remove"
    )


def embed_batch(encoder, batch: list[ScannedFile], data_dir: Path) -> tuple[list, list, list]:
    """Load + thumbnail + embed one batch -> (ids, vectors, manifest rows).

    Files that fail to open (corrupt, unsupported, permission denied) are skipped
    with a note rather than aborting a multi-hour run over a personal archive.
    """
    images = []
    rows: list[dict] = []
    for item in batch:
        try:
            image, exif, (width, height) = load_photo(item.path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            print(f"[skip] {item.path}: {type(exc).__name__}: {exc}")
            continue
        try:
            write_thumbnail(image, thumb_path(item.photo_id, data_dir))
        except Exception as exc:  # noqa: BLE001 - same
            print(f"[skip] {item.path}: thumbnail failed: {exc}")
            continue
        images.append(image)
        rows.append(library_row(item, exif, width, height))

    if not images:
        return [], [], []
    vectors = encoder.encode_images(images, batch_size=len(images))
    return [r["photo_id"] for r in rows], vectors, rows


def index_folder(root: Path, data_dir: Path, dry_run: bool, reset: bool) -> None:
    if not root.is_dir():
        raise SystemExit(f"not a folder: {root}")

    manifest = load_manifest(data_dir)
    if reset:
        manifest = manifest.iloc[0:0]

    scanned = scan_folder(root)
    if not scanned:
        print(f"[scan] no indexable images under {root}")
        return
    plan = plan_changes(manifest, scanned, root)
    print_plan(plan, root)

    if dry_run:
        for item in plan.new[:10]:
            print(f"  + {item.path}")
        for item in plan.changed[:10]:
            print(f"  ~ {item.path}")
        print("[dry-run] nothing written")
        return

    if plan.is_noop():
        print("[done] index already up to date — nothing embedded")
        return

    collection = open_collection(data_dir, create=True)
    if reset:
        existing = collection.get(include=[])["ids"]
        if existing:
            collection.delete(ids=existing)
            print(f"[reset] dropped {len(existing):,} vectors from {LIBRARY_COLLECTION!r}")

    from photosearch.encoder import Encoder

    todo = plan.to_embed
    print("[model] loading CLIP (first run downloads ~600 MB)...")
    encoder = Encoder()

    new_rows: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start : start + EMBED_BATCH]
        ids, vectors, rows = embed_batch(encoder, batch, data_dir)
        if ids:
            # upsert, not add: a modified file keeps its id and replaces its vector
            collection.upsert(
                ids=ids,
                embeddings=[v.tolist() for v in vectors],
                metadatas=[exif_metadata(pd.Series(r)) for r in rows],
            )
            new_rows.extend(rows)
        done += len(batch)
        if done % REPORT_EVERY < EMBED_BATCH or done == len(todo):
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"[embed] {done:,}/{len(todo):,}  ({rate:.1f} img/s)")

    if plan.deleted_ids:
        collection.delete(ids=plan.deleted_ids)
        for pid in plan.deleted_ids:
            thumb_path(pid, data_dir).unlink(missing_ok=True)
        print(f"[prune] removed {len(plan.deleted_ids)} vanished photo(s)")

    # Rebuild the manifest: drop what we re-indexed or deleted, append the fresh rows.
    touched = {r["photo_id"] for r in new_rows} | set(plan.deleted_ids)
    kept = manifest[~manifest["photo_id"].isin(touched)]
    updated = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else kept
    save_manifest(updated, data_dir)

    print(
        f"[done] embedded {len(new_rows):,} "
        f"({len(plan.new)} new, {len(plan.changed)} modified) in "
        f"{time.perf_counter() - t0:.1f}s | collection now holds {collection.count():,} "
        f"vectors | manifest {len(updated):,} rows"
    )


def show_stats(data_dir: Path) -> None:
    manifest = load_manifest(data_dir)
    if manifest.empty:
        print(f"[stats] nothing indexed yet ({library_dir(data_dir)} is empty)")
        return
    try:
        count = open_collection(data_dir).count()
    except Exception as exc:  # noqa: BLE001 - manifest without a collection is worth reporting
        count = -1
        print(f"[stats] WARNING: could not open the collection: {exc}")

    with_exif = manifest[["aperture", "iso", "focal_length"]].notna().any(axis=1).sum()
    print(f"[stats] {len(manifest):,} photos in the manifest, {count:,} vectors in Chroma")
    print(f"[stats] {with_exif:,} carry filterable EXIF ({with_exif / len(manifest):.0%})")
    makes = manifest["camera_make"].dropna().value_counts().head(5)
    for make, n in makes.items():
        print(f"[stats]   {make}: {n:,}")
    roots = sorted({str(Path(p).parent) for p in manifest["path"].head(2000)})
    print(f"[stats] {len(roots)} folder(s), e.g. {roots[0] if roots else '-'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a local photo folder for search.")
    parser.add_argument("--path", type=str, help="folder to scan (recursively)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="where the index lives")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    parser.add_argument("--reset", action="store_true", help="drop the library index and rebuild")
    parser.add_argument("--stats", action="store_true", help="show what's currently indexed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.stats:
        show_stats(data_dir)
        return
    if not args.path:
        parser.print_help()
        sys.exit("\nerror: --path is required (or use --stats)")
    index_folder(Path(args.path), data_dir, args.dry_run, args.reset)


if __name__ == "__main__":
    main()
