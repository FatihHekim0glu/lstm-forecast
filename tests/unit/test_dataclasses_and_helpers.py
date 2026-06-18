"""Unit tests for the implemented frozen dataclasses and small pure helpers.

Even where the heavy compute is stubbed, the configuration dataclasses carry real
validation + ``to_dict`` logic (the contracts parallel authors build against), so
they are exercised here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast import ValidationError, WalkForwardConfig
from lstmforecast.evaluation.metrics import ForecastMetrics
from lstmforecast.evaluation.metrics import _norm_sf as norm_sf
from lstmforecast.features.engineer import FeatureSpec
from lstmforecast.models.lstm import LstmConfig
from lstmforecast.models.onnx_runtime import OnnxForecaster, default_artifact_path
from lstmforecast.walkforward.engine import (
    Fold,
    FoldResult,
    WalkForwardResult,
)
from lstmforecast.walkforward.engine import _safe_float as safe_float

pytestmark = pytest.mark.unit


def test_walk_forward_config_defaults_and_effective_purge() -> None:
    cfg = WalkForwardConfig()
    assert cfg.look_back == 60
    # purge is clamped UP to look_back so no window straddles a boundary.
    assert cfg.effective_purge == 60
    cfg2 = WalkForwardConfig(look_back=60, purge=80)
    assert cfg2.effective_purge == 80
    d = cfg.to_dict()
    assert d["look_back"] == 60 and d["anchored"] is True


def test_walk_forward_config_validation() -> None:
    with pytest.raises(ValidationError):
        WalkForwardConfig(train_size=0)
    with pytest.raises(ValidationError):
        WalkForwardConfig(purge=-1)


def test_lstm_config_validation_and_to_dict() -> None:
    cfg = LstmConfig(architecture="attention", units=8, look_back=30)
    d = cfg.to_dict()
    assert d["architecture"] == "attention" and d["units"] == 8
    with pytest.raises(ValidationError):
        LstmConfig(units=0)
    with pytest.raises(ValidationError):
        LstmConfig(dropout=1.5)
    with pytest.raises(ValidationError):
        LstmConfig(learning_rate=0.0)


def test_feature_spec_validation_and_to_dict() -> None:
    spec = FeatureSpec()
    d = spec.to_dict()
    assert "momentum_windows" in d
    with pytest.raises(ValidationError):
        FeatureSpec(momentum_windows=(0,))


def test_fold_and_results_to_dict() -> None:
    fold = Fold(train=(0, 100), val=(160, 220), test=(280, 340))
    assert fold.to_dict() == {"train": [0, 100], "val": [160, 220], "test": [280, 340]}

    dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
    fr = FoldResult(
        fold=fold,
        dates=dates,
        y_true=np.array([0.01, -0.02]),
        y_pred_model=np.array([0.0, 0.0]),
        y_pred_naive=np.array([0.0, 0.0]),
    )
    frd = fr.to_dict()
    assert frd["y_true"] == [0.01, -0.02]
    assert frd["fold"]["test"] == [280, 340]

    wfr = WalkForwardResult(
        dates=dates,
        y_true=np.array([0.01, -0.02]),
        y_pred_model=np.array([0.0, 0.0]),
        y_pred_naive=np.array([0.0, 0.0]),
        n_folds=1,
        n_trials=12,
    )
    wfrd = wfr.to_dict()
    assert wfrd["n_trials"] == 12 and wfrd["n_folds"] == 1


def test_forecast_metrics_to_dict() -> None:
    m = ForecastMetrics(
        rmse_return=0.01,
        mae_return=0.008,
        mase_vs_persistence=1.02,
        directional_accuracy=0.49,
        directional_pvalue=0.7,
        dm_statistic=0.3,
        dm_pvalue=0.76,
        n_obs=500,
    )
    d = m.to_dict()
    assert d["mase_vs_persistence"] == 1.02 and d["n_obs"] == 500


def test_safe_float_helper() -> None:
    assert safe_float(1.5) == 1.5
    assert safe_float(float("nan")) is None
    assert safe_float(float("inf")) is None
    assert safe_float("not-a-number") is None


def test_norm_sf_helper() -> None:
    # Survival function: SF(0) = 0.5, SF(+inf) -> 0, symmetric.
    assert norm_sf(0.0) == pytest.approx(0.5)
    assert norm_sf(10.0) < 1e-6
    assert norm_sf(-1.0) == pytest.approx(1.0 - norm_sf(1.0))


def test_onnx_forecaster_is_import_pure_and_path_resolves() -> None:
    # Constructing the forecaster and resolving the artifact path must NOT import
    # onnxruntime (load() is lazy) - this stays on the import-pure serve path.
    forecaster = OnnxForecaster()
    assert forecaster.artifact_path.name == "lstm_forecast.onnx"
    assert default_artifact_path().parent.name == "artifacts"

    # The import-purity claim is process-global, so a prior in-session serve test
    # that legitimately loaded onnxruntime would pollute ``sys.modules`` here.
    # Assert it in a FRESH interpreter instead: constructing the forecaster +
    # resolving the path imports no inference engine.
    import subprocess
    import sys

    code = (
        "import sys; "
        "from lstmforecast.models.onnx_runtime import OnnxForecaster, default_artifact_path; "
        "f = OnnxForecaster(); "
        "assert f.artifact_path.name == 'lstm_forecast.onnx'; "
        "assert default_artifact_path().parent.name == 'artifacts'; "
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported by construction'; "
        "print('IMPORT_PURE_OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "IMPORT_PURE_OK" in proc.stdout
