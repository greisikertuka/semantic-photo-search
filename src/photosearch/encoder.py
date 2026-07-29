"""Thin wrapper owning the one CLIP model instance.

Two reasons this is its own class rather than loose functions:

1. The model is ~600 MB and must be loaded exactly once (Session 5's API loads it
   at startup, never per-request). An object makes that lifetime explicit.
2. It defines the ``Encoder`` *interface* (``encode_text`` / ``encode_image``) that
   the search service depends on — so tests inject a fixed-vector stub instead of
   downloading a model, and Session 11b can swap in an ONNX text encoder. Both are
   the same dependency-injection seam.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_NAME = "clip-ViT-B-32"


class Encoder:
    """Owns a SentenceTransformer CLIP model; encodes text and images to unit vectors."""

    # Both towers are present, so search-by-image works. The ONNX deployment
    # encoder sets this False; the API checks it instead of type-sniffing.
    supports_images = True

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Imported lazily so importing this module (e.g. in tests using a stub) does
        # not drag in torch + sentence-transformers.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one query string to a normalized float32 vector (shape ``(512,)``)."""
        vec = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vec.astype(np.float32)

    def encode_image(self, image: object) -> np.ndarray:
        """Encode one PIL image to a normalized float32 vector — the Session 8 path."""
        return self.encode_images([image])[0]

    def encode_images(self, images: list, batch_size: int = 16) -> np.ndarray:
        """Encode many PIL images at once — shape ``(n, 512)``, rows in input order.

        Batching is what makes bulk indexing (Session 9) tolerable on a CPU: the
        per-call overhead is amortized and the matmuls get wide. Order is preserved,
        which is the alignment invariant every indexing path in this project rests on.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)
        vecs = self.model.encode(
            images,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


def load_encoder(kind: str | None = None):
    """Return the encoder selected by ``kind`` (or the ``PHOTOSEARCH_ENCODER`` env var).

    ``"clip"`` (default) is the full sentence-transformers model — both towers, used
    for indexing, search-by-image and every local run. ``"onnx"`` is the text-only
    ONNX encoder Session 11b deploys, which fits a 512 MB box because no PyTorch is
    ever imported. The config seam mirrors ``load_store``: same shape, chosen by env.
    """
    kind = (kind or os.environ.get("PHOTOSEARCH_ENCODER", "clip")).strip().lower()
    if kind == "clip":
        return Encoder()
    if kind == "onnx":
        from photosearch.onnx_encoder import OnnxTextEncoder

        return OnnxTextEncoder()
    raise ValueError(f"unknown encoder kind {kind!r} (expected 'clip' or 'onnx')")
