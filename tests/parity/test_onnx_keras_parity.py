"""Parity: the exported ONNX forward pass must match Keras to 1e-5.

Placeholder until ``models.lstm.build_model`` / ``export_onnx`` and
``models.onnx_runtime.OnnxForecaster`` are implemented and an artifact is
committed. Marked ``slow`` because it requires the ``[train]`` extra (TensorFlow)
to produce the Keras reference; it is skipped in the default lean test run.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = [pytest.mark.parity, pytest.mark.slow]

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


@pytest.mark.skipif(
    not (_HAS_TF and _HAS_ORT),
    reason="ONNX-vs-Keras parity requires the [train] extra (tensorflow) and onnxruntime.",
)
def test_onnx_matches_keras_within_tolerance() -> None:
    # TODO(models): build a small Keras LSTM, export to ONNX, and assert the
    # forward passes agree to 1e-5 on a batch of sequences once the model layer
    # is implemented.
    pytest.skip("Pending models.lstm / models.onnx_runtime implementation.")
