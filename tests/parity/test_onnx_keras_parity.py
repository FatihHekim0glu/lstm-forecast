"""Parity + served-path forward determinism for the committed ONNX LSTM.

Two layers:

- ``test_committed_artifact_serves_deterministically`` (DEFAULT suite, onnxruntime
  ONLY): the shipped ``artifacts/*.onnx`` MUST load and produce finite, stable
  output across repeated forward passes. This exercises the SERVED model the
  backend runs every request, so CI covers the served path without TensorFlow.
- ``test_onnx_matches_keras_within_tolerance`` (``slow``): the exported ONNX graph
  must reproduce the trained Keras forward pass to ``1e-5``. Marked ``slow`` because
  it needs the ``[train]`` extra (TensorFlow + tf2onnx) to produce the Keras
  reference; skipped in the default lean run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytestmark = [pytest.mark.parity]

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_HAS_TF2ONNX = importlib.util.find_spec("tf2onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@pytest.mark.skipif(not _HAS_ORT, reason="served forward-determinism check needs onnxruntime.")
def test_committed_artifact_serves_deterministically() -> None:
    """DEFAULT-suite served-path check: the shipped ONNX LSTM loads + is stable.

    onnxruntime ONLY (no TensorFlow). Loads the committed default artifact, runs a
    forward pass on a seeded batch through the serve wrapper the backend uses, and
    asserts the output is the right shape, finite, NON-trivial (a real trained LSTM
    moves off zero on non-zero input), and bit-stable across a repeat run.
    """
    from lstmforecast.models.onnx_runtime import OnnxForecaster, default_artifact_path
    from lstmforecast.train import _n_features

    artifact = default_artifact_path()
    if not artifact.is_file():
        pytest.skip("shipped artifact not present in this checkout")

    look_back, n_features = 60, _n_features()
    x = (
        np.random.default_rng(20260617)
        .standard_normal((8, look_back, n_features))
        .astype("float64")
    )

    forecaster = OnnxForecaster(artifact)
    y1 = forecaster.predict(x)
    assert y1.shape == (8,)
    assert np.isfinite(y1).all()
    # A real trained LSTM produces a non-trivial response on non-zero input
    # (this is what makes the served beats_naive comparison non-vacuous).
    assert not np.allclose(y1, 0.0)
    # Forward determinism: re-running the SAME session reproduces the output exactly.
    y2 = forecaster.predict(x)
    np.testing.assert_array_equal(y1, y2)
    # A freshly-loaded session over the SAME artifact agrees too.
    y3 = OnnxForecaster(artifact).predict(x)
    np.testing.assert_allclose(y1, y3, atol=0.0, rtol=0.0)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAS_TF and _HAS_TF2ONNX and _HAS_ORT),
    reason="ONNX-vs-Keras parity requires the [train] extra (tensorflow + tf2onnx) and onnxruntime.",
)
@pytest.mark.parametrize("architecture", ["vanilla", "attention"])
def test_onnx_matches_keras_within_tolerance(architecture: str, tmp_path: Path) -> None:
    from lstmforecast.models.lstm import LstmConfig, build_model, export_onnx
    from lstmforecast.models.onnx_runtime import OnnxForecaster

    cfg = LstmConfig(
        architecture=architecture,  # type: ignore[arg-type]
        units=6,
        look_back=12,
        n_features=3,
        epochs=1,
    )
    model = build_model(cfg)
    x = (
        np.random.default_rng(7)
        .standard_normal((9, cfg.look_back, cfg.n_features))
        .astype("float32")
    )
    keras_out = np.asarray(model.predict(x, verbose=0)).reshape(-1)

    artifact = tmp_path / f"parity_{architecture}.onnx"
    written = export_onnx(model, str(artifact), config=cfg)
    assert Path(written) == artifact

    onnx_out = OnnxForecaster(str(artifact)).predict(x.astype("float64"))
    assert onnx_out.shape == keras_out.shape
    np.testing.assert_allclose(onnx_out, keras_out, atol=1e-5)
