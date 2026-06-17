"""Integration: import purity + the random-walk anti-leakage guard scaffold.

Two guards:

1. ``import lstmforecast`` must NOT import TensorFlow or onnxruntime (the package
   is import-pure; heavy deps load lazily inside functions only). This runs now.

2. The headline anti-leakage guard — on synthetic random-walk data the LSTM must
   NOT beat persistence (``MASE >= ~1``, DM insignificant) — is scaffolded here
   and activated once the walk-forward + model layer is implemented. If a future
   change makes the LSTM "beat" persistence on a random walk, leakage has
   re-entered and this test must fail.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_import_lstmforecast_does_not_import_tensorflow_or_onnxruntime() -> None:
    # Run in a fresh interpreter so prior test imports cannot mask a leak.
    code = (
        "import sys; import lstmforecast; "
        "assert 'tensorflow' not in sys.modules, 'TensorFlow imported at package load'; "
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported at package load'; "
        "assert 'keras' not in sys.modules, 'Keras imported at package load'; "
        "print('IMPORT_PURE_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_PURE_OK" in proc.stdout


def test_public_api_is_importable() -> None:
    import lstmforecast

    # A representative slice of the curated public API must be present.
    for name in (
        "random_walk_prices",
        "derive_verdict",
        "run_walk_forward",
        "WalkForwardConfig",
        "forecast_metrics",
        "PersistenceForecaster",
        "RunManifest",
    ):
        assert hasattr(lstmforecast, name), name


@pytest.mark.slow
@pytest.mark.skip(reason="Pending walkforward.engine + models.lstm implementation.")
def test_lstm_does_not_beat_persistence_on_random_walk() -> None:
    # TODO(walkforward+models): run an end-to-end fold on a synthetic random walk
    # and assert MASE >= ~1 and the Diebold-Mariano test is insignificant, so
    # `derive_verdict(...).beats_naive` is False. This is the leakage guard.
    ...
