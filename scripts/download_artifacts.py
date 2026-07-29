"""Session 12 — the quickstart artifact path: clone, sync, download, search.

    uv run python scripts/download_artifacts.py

Rebuilding the index from scratch means downloading 25,000 photos and running them
all through CLIP — an overnight job (see ``scripts/02_build_index.py``). ``data/`` is
gitignored, so without this script a fresh clone has nothing to search and the README
quickstart would be a lie.

This pulls the **index only** (~26 MB): the float16 embeddings, the row-aligned photo
ids, and the slim display parquet — the same three files the public Space ships, from
the same GitHub Release the Render deploy builds from. Unpacked into ``data/``,
``NumpyStore.load`` finds them without any env var, because it falls back to the
fp16/slim names when the full-precision pair isn't there.

What this does *not* download: the 184 MB ONNX text encoder (that's the deploy's
business — ``scripts/fetch_deploy_artifacts.py``), and any photographs. The images are
hotlinked from Unsplash's CDN at render time and never touch this machine.

Stdlib only, and it shares its download/extract helpers with the deploy fetcher so
there is one definition of "which release tag is the index pinned to".
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Sibling script, not a package — put scripts/ on the path and reuse its helpers
# rather than keeping two copies of the release tag in sync by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_deploy_artifacts import BASE, TAG, download, extract

ARCHIVE = "index-artifacts.tar.gz"
SENTINEL = "embeddings.f16.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--into",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="directory to unpack into (default: <repo>/data)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    target: Path = args.into
    if (target / SENTINEL).is_file() and not args.force:
        print(f"[skip] {target / SENTINEL} already present (--force to re-download)")
    elif (target / "embeddings.npy").is_file() and not args.force:
        # A full local build outranks the shipped fp16 copy; NumpyStore prefers it too.
        print(f"[skip] {target / 'embeddings.npy'} exists - you already built the index")
    else:
        print(f"[index] release {TAG}")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / ARCHIVE
            download(f"{BASE}/{ARCHIVE}", archive)
            extract(archive, target)

    print("\n[done] index ready. Start the app with:\n")
    print("    uv run fastapi dev src/photosearch/api.py\n")
    print("Then open http://127.0.0.1:8000 . The first query downloads the CLIP model")
    print("(~600 MB, cached in ~/.cache/huggingface) and takes a few seconds; after")
    print("that, encoding is ~34 ms and search ~5 ms.")


if __name__ == "__main__":
    main()
