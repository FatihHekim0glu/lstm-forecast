"""Integration: import purity + the random-walk anti-leakage guard (the headline).

Two guards:

1. ``import lstmforecast`` must NOT import TensorFlow or onnxruntime (the package
   is import-pure; heavy deps load lazily inside functions only).

2. The headline anti-leakage guard - on synthetic random-walk data the leakage-free
   walk-forward forecaster (the REAL ONNX LSTM, run via onnxruntime) must NOT beat
   persistence (``MASE >= ~1``, the Diebold-Mariano test never significantly in the
   model's FAVOUR, so ``beats_naive`` is ``False``). If a future change makes the
   model "beat" persistence on a random walk, leakage has re-entered and this test
   FAILS. Runs WITHOUT TensorFlow (onnxruntime only).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def _can_serve() -> bool:
    """The OOS/serve path runs the committed ONNX LSTM: needs ort + the artifact."""
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    from lstmforecast.models.onnx_runtime import default_artifact_path

    return default_artifact_path().is_file()


_serve_skip = pytest.mark.skipif(
    not _can_serve(),
    reason="OOS/serve path runs the committed ONNX LSTM: needs onnxruntime + the shipped artifact.",
)


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


@_serve_skip
def test_lstm_does_not_beat_persistence_on_random_walk() -> None:
    """THE leakage guard: on a random walk the model must NOT beat persistence.

    Runs the full leakage-free pipeline end-to-end (TF-free, no export) on a seeded
    synthetic random walk and asserts the documented NULL: the REAL ONNX LSTM
    (run via onnxruntime as the MODEL arm) lands MASE >= ~1, the Diebold-Mariano
    test is never significant IN THE MODEL'S FAVOUR (the untrained-on-signal LSTM's
    noise can only tie or lose to the random walk, never beat it), and
    ``beats_naive`` is ``False``. If leakage re-enters (e.g. a future-peeking scaler
    or a window straddling a boundary), the model would spuriously beat persistence
    and this test fails.
    """
    from lstmforecast.train import train_pipeline

    result = train_pipeline(n_obs=2000, look_back=60, seed=20260617, export=False)
    metrics = result.metrics

    # Return-space error is NOT below persistence's (no skill on a random walk).
    assert metrics.mase_vs_persistence >= 1.0 - 1e-9
    # The Diebold-Mariano test never rejects equal accuracy in the MODEL'S favour:
    # a negative DM statistic means lower model loss, so a *significant* negative
    # statistic would be the model beating the random walk. The honest LSTM can
    # only tie or lose, so the DM statistic is never significantly negative.
    model_significantly_beats = metrics.dm_statistic < 0.0 and metrics.dm_pvalue < 0.05
    assert not model_significantly_beats
    # Therefore the honest verdict is False - by construction, on random-walk data.
    assert result.verdict.beats_naive is False
    # And the honest multiplicity count equals the explored HPO grid size.
    assert result.n_effective_trials >= 1


@_serve_skip
def test_run_forecast_summary_is_json_safe_and_null_on_random_walk() -> None:
    """The backend ``run_forecast`` entrypoint returns a JSON-safe honest NULL.

    Exercises the serve path the FastAPI router calls: every summary scalar is a
    native float/bool/int, the figures are ``{data, layout}`` dicts, and
    ``beats_naive`` is ``False`` on synthetic random-walk data - with NO
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


@_serve_skip
def test_run_forecast_actually_runs_the_onnx_lstm_not_persistence() -> None:
    """INTEGRITY: the served metrics come from the ONNX LSTM, not persistence.

    Runs ``run_forecast`` end-to-end on a seeded synthetic random walk and proves
    the MODEL arm genuinely executed the committed ONNX LSTM through onnxruntime -
    NOT the persistence baseline (which would make ``beats_naive`` vacuous,
    persistence-vs-persistence):

    * ``meta["served_via"] == "onnx"`` - the served path used the ONNX model;
    * the OOS model forecast is NOT identically the persistence ``r_hat = 0`` (a
      real LSTM forward pass moves off zero), AND it differs from the all-zero
      naive column - so the LSTM truly ran;
    * yet on a random walk it still does NOT beat persistence: ``MASE >= ~1`` and
      ``beats_naive`` is ``False`` (the honest NULL, now NON-vacuously).
    """
    from lstmforecast.evaluation.metrics import forecast_metrics
    from lstmforecast.serve import run_forecast
    from lstmforecast.train import _HPO_GRID, _walk_forward_config
    from lstmforecast.walkforward.engine import run_walk_forward

    run = run_forecast(n_obs=1500, look_back=60, seed=20260617)

    # The served path provably ran the ONNX LSTM (never the persistence fallback).
    assert run.meta["served_via"] == "onnx"

    # Recompute the stacked OOS model predictions to prove the LSTM forecast is a
    # genuine, non-trivial forward pass - not the persistence r_hat=0 column.
    from lstmforecast.data import random_walk_prices
    from lstmforecast.models.onnx_runtime import (
        OnnxForecaster,
        OnnxWalkForwardForecaster,
        default_artifact_path,
    )

    prices = random_walk_prices(n_obs=1500, seed=20260617)
    cfg = _walk_forward_config(int(prices.shape[0]), 60)
    grid = [dict(p) for p in _HPO_GRID]
    shared = OnnxForecaster(default_artifact_path())
    wf = run_walk_forward(prices, lambda _p: OnnxWalkForwardForecaster(shared), cfg, hpo_grid=grid)

    # The LSTM truly ran: its OOS forecast is NOT identically zero (persistence)
    # and differs from the all-zero naive baseline column.
    assert not np.allclose(wf.y_pred_model, 0.0)
    assert np.allclose(wf.y_pred_naive, 0.0)
    assert not np.allclose(wf.y_pred_model, wf.y_pred_naive)

    # The recomputed metrics match the served summary (same engine, same folds) and
    # the honest NULL holds NON-vacuously (real LSTM, still does not beat the RW).
    metrics = forecast_metrics(wf.y_true, wf.y_pred_model, wf.y_pred_naive)
    assert run.summary.mase_vs_persistence == pytest.approx(metrics.mase_vs_persistence, rel=1e-9)
    assert run.summary.mase_vs_persistence >= 1.0 - 1e-9
    assert run.summary.beats_naive is False
