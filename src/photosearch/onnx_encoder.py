"""Session 11b — a CLIP text encoder with no PyTorch in the process.

Why this exists: Render's free tier gives 512 MB of RAM. ``import torch`` alone
costs a few hundred MB of RSS before a single weight is loaded, and the
sentence-transformers stack pulls in transformers on top of it. The deployed API
only ever needs **one** direction of CLIP — text in, 512-dim vector out — because
the *image* vectors were computed offline in Session 3 and ship as a file.

So we drop the framework: ONNX Runtime executes the exported graph, ``tokenizers``
does the BPE, and nothing else is imported. Same 512-dim space, same ``encode_text``
signature the rest of the app already depends on — this is the DI seam
:mod:`photosearch.encoder` was written for, finally being used.

The graph is the smallest possible one: one input (``input_ids``), one output
(``text_embeds``, already through CLIP's text projection). No attention mask —
CLIP pools at the EOS token position, which is found from the ids themselves.

**The weights are memory-mapped, and that is the whole trick.** The model this loads
is built by ``scripts/07_build_encoder.py``, which re-saves the graph with its weights
in a sidecar ``.data`` file. ONNX Runtime then maps them from disk instead of parsing
and copying them into RAM: peak RSS for the whole app drops from 598 MB to 400 MB with
*zero* accuracy loss, which is what makes the 512 MB tier viable. Quantizing — the
plan's original idea — turned out to be an answer to the wrong question; see that
script's docstring for the measurements that redirected it.

Nothing here talks to the network: the model and tokenizer are read from disk, so a
cold start is bounded by disk I/O, not by whether the Hub is up.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Written by scripts/07_build_encoder.py; fetched at build time on Render.
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "encoder"
ONNX_FILE = "text_model.onnx"
TOKENIZER_FILE = "tokenizer.json"

# CLIP's text context length. Longer queries are truncated, exactly as
# sentence-transformers does — matching it is part of what makes parity exact.
MAX_TOKENS = 77

MODEL_DIR_ENV = "PHOTOSEARCH_ONNX_DIR"


def resolve_model(
    model_dir: Path | str | None = None, filename: str | None = None
) -> tuple[Path, Path]:
    """Locate the encoder's model + tokenizer on disk, or say exactly what's missing.

    A module-level function rather than a method so it is testable without
    onnxruntime installed — which is what keeps CI model-free.
    """
    directory = Path(model_dir or os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    filename = filename or os.environ.get("PHOTOSEARCH_ONNX_FILE") or ONNX_FILE
    model = directory / filename
    tokenizer = directory / TOKENIZER_FILE
    missing = [p.name for p in (model, tokenizer) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {', '.join(missing)} — build it with:\n"
            f"  uv run --group render python scripts/07_build_encoder.py"
        )
    return model, tokenizer


class OnnxTextEncoder:
    """Text-only CLIP encoder over ONNX Runtime. Same interface as :class:`Encoder`.

    ``supports_images`` is ``False``: search-by-image (Session 8) needs the *vision*
    tower, which this deployment deliberately does not carry. The API surfaces that
    as a 501 rather than pretending the endpoint exists.
    """

    supports_images = False

    def __init__(
        self,
        filename: str | None = None,
        model_dir: Path | str | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path, tokenizer_path = resolve_model(model_dir, filename)
        self.model_path = model_path

        options = ort.SessionOptions()
        # One vCPU tenth on the free tier: extra threads only add contention.
        options.intra_op_num_threads = int(os.environ.get("PHOTOSEARCH_ONNX_THREADS", "1"))
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one query string to a normalized float32 vector (shape ``(512,)``).

        The export does not L2-normalize, so we do it here — every consumer in this
        project assumes unit vectors, because that is what turns the store's dot
        product into a cosine similarity.
        """
        ids = np.asarray([self.tokenizer.encode(text).ids], dtype=np.int64)
        vec = self.session.run(["text_embeds"], {"input_ids": ids})[0][0]
        vec = vec.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        # An all-zero embedding is not a thing CLIP produces, but dividing by zero
        # would poison the whole result list with NaNs rather than failing loudly.
        if norm == 0.0:
            raise ValueError(f"encoder produced a zero vector for {text!r}")
        return vec / norm

    def encode_image(self, image: object) -> np.ndarray:
        raise NotImplementedError(
            "this deployment ships the CLIP text tower only; "
            "search-by-image needs the full model (run the app locally)"
        )

    def encode_images(self, images: list, batch_size: int = 16) -> np.ndarray:
        raise NotImplementedError(
            "this deployment ships the CLIP text tower only; "
            "indexing needs the full model (run the app locally)"
        )
