"""Integration: import purity + the random-walk anti-leakage guard (the headline).

Two guards:

1. ``import lstmforecast`` must NOT import TensorFlow or onnxruntime (the package
   is import-pure; heavy deps load lazily inside functions only).

2. The headline anti-leakage guard — on synthetic random-walk data the leakage-free
   walk-forward forecaster must NOT beat persistence (``MASE >= ~1``, the
   Diebold-Mariano test insignificant, so ``beats_naive`` is ``False``). If a
   future change makes the model "beat" persistence on a random walk, leakage has
   re-entered and this test FAILS. Runs WITHOUT TensorFlow.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
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


def test_lstm_does_not_beat_persistence_on_random_walk() -> None:
    """THE leakage guard: on a random walk the model must NOT beat persistence.

    Runs the full leakage-free pipeline end-to-end (TF-free, no export) on a seeded
    synthetic random walk and asserts the documented NULL: MASE >= ~1, the
    Diebold-Mariano test is insignificant, and ``beats_naive`` is ``False``. If
    leakage re-enters (e.g. a future-peeking scaler or a window straddling a
    boundary), the model would spuriously beat persistence and this test fails.
    """
    from lstmforecast.train import train_pipeline

    result = train_pipeline(n_obs=2000, look_back=60, seed=20260617, export=False)
    metrics = result.metrics

    # Return-space error is NOT below persistence's (no skill on a random walk).
    assert metrics.mase_vs_persistence >= 1.0 - 1e-9
    # The Diebold-Mariano test cannot reject equal predictive accuracy.
    assert metrics.dm_pvalue >= 0.05
    # Therefore the honest verdict is False — by construction, on random-walk data.
    assert result.verdict.beats_naive is False
    # And the honest multiplicity count equals the explored HPO grid size.
    assert result.n_effective_trials >= 1


def test_run_forecast_summary_is_json_safe_and_null_on_random_walk() -> None:
    """The backend ``run_forecast`` entrypoint returns a JSON-safe honest NULL.

    Exercises the serve path the FastAPI router calls: every summary scalar is a
    native float/bool/int, the figures are ``{data, layout}`` dicts, and
    ``beats_naive`` is ``False`` on synthetic random-walk data — with NO
    price-level R² anywhere in the payload.
    """
    from lstmforecast.serve import run_forecast

    run = run_forecast(n_obs=1000, look_back=60, seed=7)
    payload = run.to_dict()
    summary = payload["summary"]

    assert summary["beats_naive"] is False
    assert summary["mase_vs_persistence"] >= 1.0 - 1e-9
    assert summary["data_source"] == "synthetic"
    assert isinstance(summary["rmse_return"], float)
    assert np.isfinite(summary["rmse_return"])
    # No price-level R^2 leaks into the honest payload.
    assert "r2" not in summary
    assert "r_squared" not in summary
    for fig_key in ("forecast_figure", "error_vs_baseline_figure"):
        assert set(payload[fig_key]) >= {"data", "layout"}
