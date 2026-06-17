"""Unit tests for the end-to-end train pipeline, the serve entrypoints, and the
TensorFlow-free native ONNX artifact builder.

These exercise the wiring the backend depends on WITHOUT TensorFlow:

- ``lstmforecast.train.train_pipeline`` — synthetic/CSV -> leakage-free
  walk-forward -> return-space metrics -> honest verdict -> (optional) ONNX export
  + RunManifest. On random-walk data the verdict MUST be ``beats_naive=False``.
- ``lstmforecast.serve.run_forecast`` / ``forecast_from_onnx`` — the FastAPI
  entrypoints: a JSON-safe summary (no price-level R^2), two ``{data, layout}``
  figures, and onnxruntime-only inference over the committed artifact.
- ``lstmforecast.models.onnx_export.build_native_lstm_onnx`` — builds a tiny,
  seeded, runnable LSTM-shaped ONNX graph via the ``onnx`` builder (skipped when
  ``onnx`` is unavailable).

All data is synthetic/seeded; nothing touches the network or imports TensorFlow.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lstmforecast._exceptions import ArtifactError
from lstmforecast.models.lstm import LstmConfig
from lstmforecast.serve import (
    ForecastRun,
    ForecastSummary,
    forecast_from_onnx,
    run_forecast,
)
from lstmforecast.train import TrainResult, _n_features, _walk_forward_config, train_pipeline

pytestmark = pytest.mark.unit

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_HAS_ONNX = importlib.util.find_spec("onnx") is not None


# --------------------------------------------------------------------------- #
# train_pipeline                                                              #
# --------------------------------------------------------------------------- #
def test_train_pipeline_no_export_reports_honest_null() -> None:
    result = train_pipeline(n_obs=1000, look_back=60, seed=7, export=False)
    assert isinstance(result, TrainResult)
    assert result.data_source == "synthetic"
    # The documented NULL: no return-space improvement, insignificant DM test.
    assert result.metrics.mase_vs_persistence >= 1.0 - 1e-9
    assert result.metrics.dm_pvalue >= 0.05
    assert result.verdict.beats_naive is False
    # Honest multiplicity count == the explored HPO grid size.
    assert result.n_effective_trials == 4
    # No export requested -> no artifact path, manifest still stamped.
    assert result.artifact_path == ""
    assert result.meta["export_backend"] == "skipped"
    assert result.manifest.seed == 7


def test_train_pipeline_to_dict_is_json_safe() -> None:
    import json

    result = train_pipeline(n_obs=900, look_back=60, seed=3, export=False)
    payload = result.to_dict()
    # Round-trips through JSON with no numpy/pandas leaks.
    text = json.dumps(payload)
    assert "mase_vs_persistence" in text
    assert payload["verdict"]["beats_naive"] is False
    assert payload["n_effective_trials"] == 4


@pytest.mark.skipif(not (_HAS_ORT and _HAS_ONNX), reason="export needs the onnx builder + ort.")
def test_train_pipeline_native_export_builds_servable_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    result = train_pipeline(n_obs=1000, look_back=60, seed=7, export=True, artifact_path=artifact)
    assert result.artifact_path == str(artifact)
    assert artifact.is_file()
    # Tiny artifact (well under the 5MB ship budget).
    assert artifact.stat().st_size < 5_000_000
    assert result.meta["export_backend"] == "native"
    # The freshly-built artifact serves through onnxruntime.
    x = np.zeros((3, 60, _n_features()), dtype="float64")
    out = forecast_from_onnx(x, artifact_path=artifact)
    assert out.shape == (3,)
    assert np.isfinite(out).all()


def test_train_pipeline_loads_csv(tmp_path: Path) -> None:
    # A real-data path: write a strictly-positive close CSV and retrain on it.
    from lstmforecast.data import random_walk_prices

    prices = random_walk_prices(n_obs=800, seed=11)
    csv = tmp_path / "prices.csv"
    pd.DataFrame({"date": prices.index, "close": prices.to_numpy()}).to_csv(csv, index=False)

    result = train_pipeline(data_path=csv, look_back=60, seed=11, export=False)
    assert result.data_source == "csv"
    assert result.verdict.beats_naive is False


def test_walk_forward_config_yields_at_least_one_fold() -> None:
    from lstmforecast.walkforward.engine import make_folds

    for n_obs in (700, 1000, 2000):
        cfg = _walk_forward_config(n_obs, 60)
        folds = make_folds(n_obs, cfg)
        assert len(folds) >= 1
        # Purge is clamped to >= look_back so no window straddles a boundary.
        assert cfg.effective_purge >= 60


def test_n_features_matches_default_feature_spec() -> None:
    from lstmforecast.data import random_walk_prices
    from lstmforecast.features.engineer import FeatureSpec, engineer_features

    frame = engineer_features(random_walk_prices(n_obs=300, seed=1), FeatureSpec())
    assert _n_features() == frame.shape[1]


def test_final_training_tensors_are_well_formed_and_scaled() -> None:
    # The final-fit tensors (used on the [train] export branch) are leakage-free
    # pure pandas/numpy and must be buildable without TensorFlow.
    from lstmforecast.data import random_walk_prices
    from lstmforecast.train import _final_training_tensors

    cfg = LstmConfig(look_back=30, n_features=_n_features(), seed=7)
    prices = random_walk_prices(n_obs=500, seed=7)
    x, y = _final_training_tensors(prices, cfg)
    assert x.ndim == 3
    assert x.shape[1:] == (30, _n_features())
    assert x.shape[0] == y.shape[0]
    assert np.isfinite(x).all() and np.isfinite(y).all()


@pytest.mark.skipif(not (_HAS_ORT and _HAS_ONNX), reason="export needs the onnx builder + ort.")
def test_export_artifact_native_backend(tmp_path: Path) -> None:
    from lstmforecast.data import random_walk_prices
    from lstmforecast.train import _export_artifact

    cfg = LstmConfig(look_back=20, n_features=_n_features(), seed=7)
    prices = random_walk_prices(n_obs=400, seed=7)
    out = tmp_path / "exp.onnx"
    written, backend = _export_artifact(prices, cfg, out, seed=7)
    assert written == str(out)
    # TF absent in this environment -> the native onnx-builder path is taken.
    assert backend == "native"
    assert out.is_file()


# --------------------------------------------------------------------------- #
# serve.run_forecast                                                          #
# --------------------------------------------------------------------------- #
def test_run_forecast_returns_summary_and_two_figures() -> None:
    run = run_forecast(n_obs=900, look_back=60, seed=7)
    assert isinstance(run, ForecastRun)
    assert isinstance(run.summary, ForecastSummary)
    # Honest NULL surfaces in the summary the backend forwards.
    assert run.summary.beats_naive is False
    assert run.summary.mase_vs_persistence >= 1.0 - 1e-9
    assert run.summary.n_effective_trials == 4
    # Two return-space figures, each a {data, layout} mapping.
    for fig in (run.forecast_figure, run.error_vs_baseline_figure):
        assert set(fig) >= {"data", "layout"}
    assert len(run.error_vs_baseline_figure["data"]) == 2


def test_run_forecast_payload_has_no_price_level_r2() -> None:
    payload = run_forecast(n_obs=800, look_back=60, seed=2).to_dict()
    summary = payload["summary"]
    # The whole point: never a price-level R^2.
    assert not any(k.lower() in {"r2", "r_squared", "rsq"} for k in summary)
    assert summary["served_via" if "served_via" in summary else "data_source"]  # smoke


def test_run_forecast_served_via_reports_artifact_presence(tmp_path: Path) -> None:
    # With a non-existent artifact path the serve layer falls back to persistence.
    missing = tmp_path / "absent.onnx"
    run = run_forecast(n_obs=800, look_back=60, seed=5, artifact_path=missing)
    assert run.meta["served_via"] == "persistence"


def test_forecast_summary_to_dict_types() -> None:
    summary = ForecastSummary(
        rmse_return=0.01,
        mae_return=0.008,
        mase_vs_persistence=1.0,
        directional_accuracy=0.5,
        dm_pvalue=1.0,
        beats_naive=False,
        n_effective_trials=4,
        data_source="synthetic",
        n_obs=100,
        verdict="no_significant_difference",
        rationale="MASE=1.000>=1",
    )
    d = summary.to_dict()
    assert d["beats_naive"] is False
    assert isinstance(d["rmse_return"], float)
    assert isinstance(d["n_effective_trials"], int)
    assert d["data_source"] == "synthetic"


# --------------------------------------------------------------------------- #
# forecast_from_onnx serve wrapper                                            #
# --------------------------------------------------------------------------- #
def test_forecast_from_onnx_missing_artifact_raises_artifact_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        forecast_from_onnx(
            np.zeros((2, 60, 1), dtype="float64"),
            artifact_path=tmp_path / "nope.onnx",
        )


@pytest.mark.skipif(not _HAS_ORT, reason="serve needs onnxruntime ([serve]).")
def test_forecast_from_onnx_serves_committed_default_artifact() -> None:
    # The shipped default artifact (synthetic-trained) must serve via onnxruntime.
    from lstmforecast.models.onnx_runtime import default_artifact_path

    if not default_artifact_path().is_file():
        pytest.skip("shipped artifact not present in this checkout")
    x = np.zeros((4, 60, _n_features()), dtype="float64")
    out = forecast_from_onnx(x)
    assert out.shape == (4,)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# native ONNX builder (TF-free)                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_HAS_ONNX and _HAS_ORT), reason="needs the onnx builder + ort.")
def test_build_native_lstm_onnx_is_small_and_runs(tmp_path: Path) -> None:
    from lstmforecast.models.onnx_export import build_native_lstm_onnx
    from lstmforecast.models.onnx_runtime import OnnxForecaster

    cfg = LstmConfig(units=8, look_back=12, n_features=3, seed=7)
    out = tmp_path / "native.onnx"
    written = build_native_lstm_onnx(cfg, out, seed=7)
    assert written == str(out)
    assert out.stat().st_size < 5_000_000

    x = np.random.default_rng(0).standard_normal((6, 12, 3)).astype("float64")
    forecaster = OnnxForecaster(out)
    y1 = forecaster.predict(x)
    assert y1.shape == (6,)
    # Deterministic: a second predict (reusing the session) matches.
    assert np.allclose(y1, forecaster.predict(x))


@pytest.mark.skipif(not (_HAS_ONNX and _HAS_ORT), reason="needs the onnx builder + ort.")
def test_build_native_lstm_onnx_is_seed_reproducible(tmp_path: Path) -> None:
    from lstmforecast.models.onnx_export import build_native_lstm_onnx
    from lstmforecast.models.onnx_runtime import OnnxForecaster

    cfg = LstmConfig(units=4, look_back=8, n_features=2, seed=7)
    a = tmp_path / "a.onnx"
    b = tmp_path / "b.onnx"
    build_native_lstm_onnx(cfg, a, seed=123)
    build_native_lstm_onnx(cfg, b, seed=123)
    x = np.random.default_rng(1).standard_normal((5, 8, 2)).astype("float64")
    # Same seed => byte-identical weights => identical forward pass.
    assert np.allclose(OnnxForecaster(a).predict(x), OnnxForecaster(b).predict(x))
