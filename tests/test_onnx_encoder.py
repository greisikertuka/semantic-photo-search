"""Session 11b — the deploy seams, tested without onnxruntime or a model on disk.

CI installs only the ``dev`` group, so ``onnxruntime`` isn't even importable here.
That's deliberate and it's the reason these tests are about *resolution and wiring*
rather than inference: the numerical claim ("the deployed encoder retrieves
identically to the real one") is an evaluation, not a unit test, and it lives in
``eval/run_eval.py --system deploy`` where it can be measured against the gold set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from photosearch import onnx_encoder
from photosearch.encoder import load_encoder
from photosearch.onnx_encoder import resolve_model
from photosearch.store import NumpyStore, data_dir_from_env


class TestResolveModel:
    """Where the deploy looks for its model, and what it says when it isn't there."""

    def test_finds_model_and_tokenizer(self, tmp_path):
        (tmp_path / "text_model.onnx").write_bytes(b"not really a model")
        (tmp_path / "tokenizer.json").write_text("{}")
        model, tokenizer = resolve_model(tmp_path)
        assert model.name == "text_model.onnx"
        assert tokenizer.name == "tokenizer.json"

    def test_env_var_overrides_the_default_dir(self, tmp_path, monkeypatch):
        (tmp_path / "text_model.onnx").write_bytes(b"x")
        (tmp_path / "tokenizer.json").write_text("{}")
        monkeypatch.setenv(onnx_encoder.MODEL_DIR_ENV, str(tmp_path))
        model, _ = resolve_model()
        assert model.parent == tmp_path

    def test_missing_files_are_named_with_the_command_that_makes_them(self, tmp_path):
        # A deploy that can't find its model should fail at startup with a fix, not a
        # stack trace 300 ms into the first user's request.
        with pytest.raises(FileNotFoundError) as exc:
            resolve_model(tmp_path)
        message = str(exc.value)
        assert "text_model.onnx" in message
        assert "tokenizer.json" in message
        assert "07_build_encoder.py" in message

    def test_partial_directory_names_only_what_is_missing(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        with pytest.raises(FileNotFoundError) as exc:
            resolve_model(tmp_path)
        assert "text_model.onnx" in str(exc.value)
        assert "tokenizer.json" not in str(exc.value)


class TestEncoderSeam:
    """``PHOTOSEARCH_ENCODER`` picks the model the same way ``PHOTOSEARCH_STORE`` picks
    the back-end — one config seam per swappable thing."""

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown encoder kind"):
            load_encoder("tensorflow")

    def test_onnx_kind_routes_to_the_onnx_encoder(self, monkeypatch):
        built = {}

        class Fake:
            def __init__(self):
                built["yes"] = True

        monkeypatch.setattr(onnx_encoder, "OnnxTextEncoder", Fake)
        load_encoder("onnx")
        assert built == {"yes": True}

    def test_env_var_is_the_default_source(self, monkeypatch):
        monkeypatch.setenv("PHOTOSEARCH_ENCODER", "nonsense")
        with pytest.raises(ValueError, match="nonsense"):
            load_encoder()


class TestDeployArtifactLoading:
    """The 29.3 MB deploy payload loads through the same NumpyStore as the full index."""

    @staticmethod
    def _write(directory, embeddings_name, parquet_name):
        vectors = np.eye(3, 4, dtype=np.float32)
        np.save(directory / embeddings_name, vectors.astype(np.float16))
        np.save(directory / "photo_ids.npy", np.array(["a", "b", "c"], dtype=object))
        pd.DataFrame(
            {
                "photo_id": ["a", "b", "c"],
                "photo_image_url": ["u"] * 3,
                "photo_url": ["v"] * 3,
                "photographer": ["p"] * 3,
                "photo_width": [1] * 3,
                "photo_height": [1] * 3,
                "blur_hash": [None] * 3,
                "aperture": [1.8, None, 8.0],
                "focal_length": [35.0, None, None],
                "exposure_s": [0.01, None, None],
                "iso": [100, None, None],
                "camera_make": ["canon", None, None],
                "camera_model": ["r6", None, None],
            }
        ).to_parquet(directory / parquet_name)

    def test_fp16_and_slim_parquet_are_accepted(self, tmp_path):
        self._write(tmp_path, "embeddings.f16.npy", "photos.slim.parquet")
        store = NumpyStore.load(tmp_path)
        assert store.count() == 3
        # float16 is a *storage* format: it must be widened, because NumPy has no
        # fast half-precision matmul and every score would silently change dtype.
        assert store.embeddings.dtype == np.float32

    def test_full_precision_artifacts_win_when_both_exist(self, tmp_path):
        self._write(tmp_path, "embeddings.f16.npy", "photos.slim.parquet")
        np.save(tmp_path / "embeddings.npy", np.full((3, 4), 0.5, dtype=np.float32))
        store = NumpyStore.load(tmp_path)
        assert np.allclose(store.embeddings, 0.5)

    def test_missing_artifacts_name_every_candidate(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="embeddings.npy, embeddings.f16.npy"):
            NumpyStore.load(tmp_path)

    def test_slim_parquet_drops_descriptions_without_breaking_results(self, tmp_path):
        # The deploy parquet omits photo_description/ai_description for licensing
        # reasons — Result must still build, with those fields simply None.
        self._write(tmp_path, "embeddings.f16.npy", "photos.slim.parquet")
        store = NumpyStore.load(tmp_path)
        result = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)[0]
        assert result.photo_id == "a"
        assert result.description is None
        assert result.ai_description is None
        assert result.aperture == pytest.approx(1.8)


class TestDataDirEnv:
    def test_defaults_to_the_repo_data_dir(self):
        assert data_dir_from_env().name == "data"

    def test_env_var_points_the_deploy_at_its_unpacked_release(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PHOTOSEARCH_DATA_DIR", str(tmp_path))
        assert data_dir_from_env() == tmp_path
