"""Parity: the exported ONNX forward pass must match Keras to 1e-5.

The exported artifact is the model the container serves via onnxruntime, so the
ONNX graph MUST reproduce the trained Keras forward pass. This test builds a
small Keras LSTM, exports it to ONNX via tf2onnx, runs the same batch through
both engines, and asserts agreement to ``1e-5``. Marked ``slow`` because it
requires the ``[train]`` extra (TensorFlow + tf2onnx) to produce the Keras
reference; it is skipped in the default lean test run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytestmark = [pytest.mark.parity, pytest.mark.slow]

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_HAS_TF2ONNX = importlib.util.find_spec("tf2onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


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
