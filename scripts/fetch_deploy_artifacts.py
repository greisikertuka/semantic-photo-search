"""Session 11b — pull the index + encoder a deploy needs, without git-lfs or a model hub.

    python scripts/fetch_deploy_artifacts.py --into .

Run by Render at build time (see ``render.yaml``). Stdlib only, on purpose: this runs
*before* ``pip install``, so it cannot assume a single third-party package exists.

Why a GitHub Release and not the repo? The payload is ~210 MB of binary — embeddings,
the ONNX graph, its weights. Committing that would bloat every clone forever and hit
GitHub's 100 MB per-file limit; Git LFS on a public repo has bandwidth quotas that a
build-on-every-push would burn through. A release asset is a plain, cacheable,
versioned URL, and the tag pins exactly which index the deployed app is serving.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "greisikertuka/semantic-photo-search"
TAG = "deploy-artifacts-v1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"

# archive -> (destination subdirectory, a file that proves it's already unpacked)
BUNDLES = {
    "index-artifacts.tar.gz": ("data/space", "embeddings.f16.npy"),
    "text-encoder.tar.gz": ("data/encoder", "text_model.onnx.data"),
}


def download(url: str, destination: Path) -> None:
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"[fetch] {destination.stat().st_size / 1e6:.1f} MB")


def extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # filter="data" refuses absolute paths, "..", symlinks and device files. These
        # are our own archives, but an extractor that trusts its input is a habit worth
        # not having — and it's one keyword.
        tar.extractall(into, filter="data")
    print(f"[unpack] {into}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--into", type=Path, default=Path.cwd(),
                        help="project root to unpack into (default: cwd)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    for archive_name, (subdir, sentinel) in BUNDLES.items():
        target = args.into / subdir
        if (target / sentinel).is_file() and not args.force:
            print(f"[skip] {subdir} already present")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / archive_name
            download(f"{BASE}/{archive_name}", archive)
            extract(archive, target)

    print("\n[done] deploy artifacts ready")
    for subdir, _ in BUNDLES.values():
        for path in sorted((args.into / subdir).iterdir()):
            print(f"       {path.stat().st_size / 1e6:8.1f} MB  {subdir}/{path.name}")


if __name__ == "__main__":
    main()
