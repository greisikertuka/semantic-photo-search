"""Session 1 — Embeddings 101: prove text and images share one vector space.

This is a throwaway *learning* script, not app code. Drop 4-6 of your own photos
in a folder and run it: it encodes each image and a handful of text descriptions
with the SAME CLIP model, then prints the full text x image cosine-similarity
matrix. The whole point of the project fits in one screen of output here.

    uv run python scripts/00_hello_clip.py --path "C:\\Users\\greisi\\Pictures\\clip-test"

Then *play*: edit TEXTS below, try adjectives and styles ("a blurry photo",
"a photo taken at night"), try descriptions that are wrong on purpose, and watch
the scores move. Note the ABSOLUTE range you see — good matches land ~0.25-0.35,
not 0.9. CLIP scores are relative rankings, not probabilities. (That fact drives
Session 6's "no strong matches" threshold.)

Definition of done: for each text, the highest-scoring image is the right one,
and you can explain to a rubber duck why one matrix multiply computed every
pairwise similarity at once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Edit these freely — that's the exercise. Aim them at whatever your test photos show.
TEXTS = [
    "a dog",
    "a mountain at sunset",
    "a red car",
    "a photo taken at night",
    "a blurry photo",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIP text-vs-image similarity demo.")
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="folder containing 4-6 of your own test photos",
    )
    args = parser.parse_args()

    folder: Path = args.path
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")
    paths = find_images(folder)
    if not paths:
        raise SystemExit(f"no images ({', '.join(sorted(IMAGE_SUFFIXES))}) found in {folder}")

    # Imported here so --help stays instant and doesn't pay the import cost.
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    print("[model] loading clip-ViT-B-32 (first run downloads ~600 MB)...")
    model = SentenceTransformer("clip-ViT-B-32")

    images = [Image.open(p).convert("RGB") for p in paths]
    # normalize_embeddings=True makes every vector length 1, so the dot product
    # below IS the cosine similarity — no separate normalization step needed.
    img_embs = model.encode(images, normalize_embeddings=True, convert_to_numpy=True)
    txt_embs = model.encode(TEXTS, normalize_embeddings=True, convert_to_numpy=True)

    # ONE matrix multiply computes all len(TEXTS) x len(paths) pairwise similarities.
    scores = txt_embs @ img_embs.T  # shape (n_texts, n_images)

    names = [p.name for p in paths]
    col_w = max(len(n) for n in names + ["query"]) if names else 8
    col_w = min(col_w, 22)

    def cell(text: str) -> str:
        return text[:col_w].rjust(col_w)

    header = "  ".join(cell(n) for n in names)
    print(f"\n{'query'.ljust(28)}  {header}")
    for row, text in enumerate(TEXTS):
        best = int(scores[row].argmax())
        cells = []
        for col in range(len(names)):
            mark = "*" if col == best else " "  # star the top image for this text
            cells.append(cell(f"{scores[row, col]:.3f}{mark}"))
        print(f"{text.ljust(28)}  {'  '.join(cells)}")

    print("\n(* = best image for that row. Good matches ~0.25-0.35, not ~0.9.)")


if __name__ == "__main__":
    main()
