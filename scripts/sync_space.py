"""Session 11 — copy the deployable files into a local clone of the HF Space repo.

    git clone https://huggingface.co/spaces/<you>/latent-photo-search ../latent-space
    uv run python scripts/sync_space.py --space ../latent-space
    uv run python scripts/sync_space.py --space ../latent-space --check   # diff only

Then *you* review the diff and `git push` from that folder. This script deliberately
never commits or pushes: a Space is a public artifact, and the licensing call it
embodies (derived embeddings + the minimum needed to credit photographers, no TSV,
images hotlinked) deserves a human looking at the file list before it goes live.

Why a copy at all: a Space is **its own git repo**, and it expects ``app.py`` at
*its* root next to its own ``requirements.txt``. Our ``photosearch`` package lives
under ``src/`` and is installed editable locally — none of which the Space knows
about. So we flatten: ``src/photosearch/`` becomes ``photosearch/`` beside
``app.py``, which is exactly the layout the sys.path shim in ``space/app.py``
already handles.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_SRC = PROJECT_ROOT / "space"
PACKAGE_SRC = PROJECT_ROOT / "src" / "photosearch"
ARTIFACT_SRC = PROJECT_ROOT / "data" / "space"

ARTIFACTS = ["embeddings.f16.npy", "photo_ids.npy", "photos.slim.parquet"]

# Modules the Space genuinely needs. Chroma, the library ingester, the BM25 baseline
# and the eval harness are all local-only concerns — shipping them would drag
# chromadb and rank-bm25 into a container that has no use for either.
PACKAGE_MODULES = ["__init__.py", "models.py", "store.py", "encoder.py", "search.py", "exif.py"]

# Git LFS is not optional here: a plain-git push of a 26 MB .npy is rejected by the
# Hub's 10 MB limit for non-LFS files.
GITATTRIBUTES = "*.npy filter=lfs diff=lfs merge=lfs -text\n*.parquet filter=lfs diff=lfs merge=lfs -text\n"


def plan_copies(space: Path) -> list[tuple[Path, Path]]:
    """(source, destination) for every file the Space needs — the whole payload."""
    pairs: list[tuple[Path, Path]] = [
        (SPACE_SRC / "app.py", space / "app.py"),
        (SPACE_SRC / "requirements.txt", space / "requirements.txt"),
        (SPACE_SRC / "README.md", space / "README.md"),
    ]
    pairs += [(PACKAGE_SRC / name, space / "photosearch" / name) for name in PACKAGE_MODULES]
    pairs += [(ARTIFACT_SRC / name, space / "data" / name) for name in ARTIFACTS]
    return pairs


def state(src: Path, dest: Path) -> str:
    if not src.is_file():
        return "MISSING"
    if not dest.is_file():
        return "new"
    return "same" if filecmp.cmp(src, dest, shallow=False) else "changed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the deployable files into a Space clone.")
    parser.add_argument("--space", required=True, help="path to your local clone of the Space repo")
    parser.add_argument("--check", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    space = Path(args.space).resolve()
    if not space.is_dir():
        raise SystemExit(f"not a folder: {space}\nClone the Space repo there first.")
    if not (space / ".git").exists():
        print(f"[warn] {space} is not a git repo — a Space clone normally is")

    pairs = plan_copies(space)
    missing = [src for src, _ in pairs if not src.is_file()]
    if missing:
        print("[error] these sources do not exist yet:")
        for src in missing:
            print(f"  - {src.relative_to(PROJECT_ROOT)}")
        if any(ARTIFACT_SRC in src.parents for src in missing):
            print("\nBuild the artifacts first:\n"
                  "  uv run python scripts/06_build_space_artifacts.py")
        raise SystemExit(1)

    changed = 0
    total_bytes = 0
    for src, dest in pairs:
        status = state(src, dest)
        size = src.stat().st_size
        total_bytes += size
        if status != "same":
            changed += 1
        mark = {"new": "+", "changed": "~", "same": " "}[status]
        print(f" {mark} {dest.relative_to(space).as_posix():<34} {size / 1e6:7.2f} MB  {status}")
        if args.check or status == "same":
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # Make sure the two big binaries are tracked by LFS before anyone stages them.
    if not args.check:
        attrs = space / ".gitattributes"
        current = attrs.read_text(encoding="utf-8") if attrs.is_file() else ""
        for line in GITATTRIBUTES.splitlines():
            if line not in current:
                current = f"{current.rstrip(chr(10))}\n{line}\n".lstrip("\n")
        attrs.write_text(current, encoding="utf-8")

    verb = "would change" if args.check else "copied"
    print(f"\n[{'check' if args.check else 'sync'}] {changed} file(s) {verb}; "
          f"{total_bytes / 1e6:.1f} MB payload total")
    if args.check:
        return
    print("\nNext — review, then push *from the Space clone*:")
    print(f"  cd {space}")
    print("  git lfs install && git add -A && git status")
    print('  git commit -m "Deploy latent photo search" && git push')
    print("\nBefore you push, sanity-check the app locally:")
    print("  uv run --group space python space/app.py")


if __name__ == "__main__":
    main()
