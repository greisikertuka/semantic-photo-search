"""Session 13 — the device seam, and why it is not cosmetic.

The first deploy of the HF Space served 30 results per query with every cosine score
at 0.000, and nothing raised. Cause: ZeroGPU reports ``torch.cuda.is_available() ==
True`` so libraries configure for a GPU, but a slice only exists inside a
``@spaces.GPU`` call — which this project deliberately never makes. Left to
auto-detect, sentence-transformers put CLIP on CUDA and every forward pass came back
a zero vector. ``embeddings @ 0`` is all-zero, so top-k returned rows in storage
order and the UI rendered them confidently.

These tests pin the two things that would have caught it: the device actually
reaches the model, and a degenerate encode is detectable rather than silent.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from photosearch.encoder import Encoder


class _FakeST:
    """Stand-in for SentenceTransformer that records the device it was handed."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, model_name, device=None, **kwargs):
        _FakeST.last_kwargs = {"model_name": model_name, "device": device}
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 512), dtype=np.float32) / np.sqrt(512)


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Install a stub module so these tests never download 600 MB of CLIP."""
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    _FakeST.last_kwargs = {}
    return mod


def test_device_reaches_the_model(fake_sentence_transformers):
    """An explicit device must be forwarded — this is the ZeroGPU fix."""
    Encoder(device="cpu")
    assert _FakeST.last_kwargs["device"] == "cpu"


def test_device_defaults_to_auto(fake_sentence_transformers):
    """No device given and no env var set means auto-detect, as before."""
    enc = Encoder()
    assert enc.device is None
    assert _FakeST.last_kwargs["device"] is None


def test_device_from_env(fake_sentence_transformers, monkeypatch):
    """PHOTOSEARCH_DEVICE mirrors the other seams (STORE/ENCODER/DATA_DIR)."""
    monkeypatch.setenv("PHOTOSEARCH_DEVICE", "cpu")
    enc = Encoder()
    assert enc.device == "cpu"
    assert _FakeST.last_kwargs["device"] == "cpu"


def test_explicit_device_beats_env(fake_sentence_transformers, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DEVICE", "cuda")
    assert Encoder(device="cpu").device == "cpu"


def test_encoded_vector_is_unit_length(fake_sentence_transformers):
    """The boot guard's invariant: a healthy encode has L2 norm ~1, never 0.

    The ZeroGPU failure produced norm 0.0. Asserting ~1.0 is what separates 'the
    model ran' from 'the model returned a shape of the right size'.
    """
    vec = Encoder(device="cpu").encode_text("a foggy forest at sunrise")
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-3)


def test_zero_vector_would_poison_every_score():
    """Why the failure was invisible: a zero query ranks without ever erroring.

    Documents the actual mechanism — all-zero scores still produce a full, ordered,
    plausible-looking result set. Nothing in the search path can notice.
    """
    embeddings = np.random.default_rng(0).normal(size=(100, 512)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = embeddings @ np.zeros(512, dtype=np.float32)
    assert np.all(scores == 0.0)
    # argsort on a constant array is stable, so "top 30" is just rows 0..29 —
    # identical for every query, which is exactly what the live Space returned.
    assert list(np.argsort(-scores, kind="stable")[:5]) == [0, 1, 2, 3, 4]
