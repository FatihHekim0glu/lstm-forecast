"""Unit tests for the implemented ``features`` group.

Covers the no-lookahead feature kernels (momentum, volatility, RSI, lagged
return), the shape/validation contracts of :func:`engineer_features`, and the
exact alignment + scaler semantics of :mod:`lstmforecast.features.sequences`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast import (
    InsufficientDataError,
    ValidationError,
)
from lstmforecast.features.engineer import FeatureSpec, engineer_features
from lstmforecast.features.sequences import (
    create_sequences,
    fit_scaler,
    scale_sequences,
)

pytestmark = pytest.mark.unit


def _ramp_prices(n: int, *, start: str = "2020-01-01") -> pd.Series:
    """A strictly-increasing geometric price ramp (deterministic, gap-free)."""
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(100.0 * (1.01 ** np.arange(n)), index=idx, name="close")


# --------------------------------------------------------------------------- #
# FeatureSpec                                                                  #
# --------------------------------------------------------------------------- #
def test_feature_spec_max_window() -> None:
    spec = FeatureSpec(momentum_windows=(5,), vol_windows=(30,), rsi_window=14)
    assert spec.max_window == 30


def test_feature_spec_requires_a_feature_group() -> None:
    with pytest.raises(ValidationError):
        FeatureSpec(momentum_windows=(), vol_windows=(), include_lagged_return=False)


def test_feature_spec_rejects_non_positive_windows() -> None:
    with pytest.raises(ValidationError):
        FeatureSpec(vol_windows=(0,))
    with pytest.raises(ValidationError):
        FeatureSpec(rsi_window=0)


# --------------------------------------------------------------------------- #
# engineer_features                                                           #
# --------------------------------------------------------------------------- #
def test_engineer_features_columns_and_density() -> None:
    spec = FeatureSpec(momentum_windows=(5, 10), vol_windows=(10,), rsi_window=14)
    feats = engineer_features(_ramp_prices(120), spec)
    assert list(feats.columns) == ["mom_5", "mom_10", "vol_10", "rsi_14", "ret_lag1"]
    assert feats.dtypes.eq("float64").all()
    assert not bool(feats.isna().to_numpy().any())  # dense after warm-up + shift


def test_engineer_features_no_close_level_column() -> None:
    """De-leak guard: no column carries the raw price level / its scale."""
    feats = engineer_features(_ramp_prices(120))
    assert "close" not in feats.columns
    # The return/momentum/volatility columns are differences/ratios, so they live
    # on a tiny scale nowhere near the ~100 price level (RSI is separately bounded
    # to [0, 100] by construction and excluded here).
    scale_free = feats.drop(columns=[c for c in feats.columns if c.startswith("rsi_")])
    assert float(scale_free.abs().to_numpy().max()) < 50.0
    rsi = feats[[c for c in feats.columns if c.startswith("rsi_")]].to_numpy()
    assert bool((rsi >= 0.0).all()) and bool((rsi <= 100.0).all())


def test_engineer_features_lagged_return_matches_reference() -> None:
    """ret_lag1 at date t equals the log-return realized at t-1 (strictly past)."""
    prices = _ramp_prices(40)
    spec = FeatureSpec(momentum_windows=(), vol_windows=(), rsi_window=2)
    feats = engineer_features(prices, spec)
    price_arr = prices.to_numpy()
    log_ret = np.concatenate([[np.nan], np.diff(np.log(price_arr))])
    first_feat_pos = len(prices) - len(feats)
    # Feature row at offset k from the first surviving date holds r at price-pos
    # (first_feat_pos + k - 1): one extra .shift(1) lag beyond the same-bar return.
    for k in range(10):
        pos = first_feat_pos + k
        assert feats["ret_lag1"].iloc[k] == pytest.approx(float(log_ret[pos - 1]))


def test_engineer_features_momentum_is_sum_of_past_returns() -> None:
    prices = _ramp_prices(60)
    spec = FeatureSpec(momentum_windows=(5,), vol_windows=(), rsi_window=2)
    feats = engineer_features(prices, spec)
    price_arr = prices.to_numpy()
    log_ret = np.concatenate([[np.nan], np.diff(np.log(price_arr))])
    first_feat_pos = len(prices) - len(feats)
    k = 20
    pos = first_feat_pos + k
    # mom_5 at t is the 5-bar cumulative return ending at t-1 (shift(1) lag).
    expected = float(np.nansum(log_ret[pos - 5 : pos]))
    assert feats["mom_5"].iloc[k] == pytest.approx(expected)


def test_engineer_features_can_omit_lagged_return() -> None:
    spec = FeatureSpec(momentum_windows=(5,), vol_windows=(), include_lagged_return=False)
    feats = engineer_features(_ramp_prices(60), spec)
    assert "ret_lag1" not in feats.columns
    assert list(feats.columns) == ["mom_5", "rsi_14"]


def test_engineer_features_rsi_bounded() -> None:
    spec = FeatureSpec(momentum_windows=(), vol_windows=(), rsi_window=14)
    feats = engineer_features(_ramp_prices(120), spec)
    rsi = feats["rsi_14"].to_numpy()
    assert bool((rsi >= 0.0).all()) and bool((rsi <= 100.0).all())


def test_engineer_features_rejects_non_positive_prices() -> None:
    prices = _ramp_prices(30)
    prices.iloc[5] = -1.0
    with pytest.raises(ValidationError):
        engineer_features(prices)


def test_engineer_features_rejects_non_monotonic_index() -> None:
    prices = _ramp_prices(30)
    prices = prices.iloc[::-1]  # descending in time
    with pytest.raises(ValidationError):
        engineer_features(prices)


def test_engineer_features_raises_when_too_short() -> None:
    spec = FeatureSpec(momentum_windows=(20,), vol_windows=(20,), rsi_window=14)
    with pytest.raises(ValidationError):
        engineer_features(_ramp_prices(5), spec)


def test_engineer_features_handles_gaps_without_ffill() -> None:
    """A NaN price bar must not be forward-filled into a spurious zero return."""
    prices = _ramp_prices(60)
    prices.iloc[10] = np.nan
    # ensure_series rejects NaN by default, so the de-leak contract surfaces it.
    with pytest.raises(ValidationError):
        engineer_features(prices)


# --------------------------------------------------------------------------- #
# create_sequences                                                            #
# --------------------------------------------------------------------------- #
def test_create_sequences_shapes_and_index() -> None:
    idx = pd.date_range("2020-01-01", periods=12, freq="B")
    feats = pd.DataFrame({"a": np.arange(12.0), "b": np.arange(12.0)}, index=idx)
    target = pd.Series(np.arange(12.0), index=idx)
    x, y, ix = create_sequences(feats, target, look_back=4)
    assert x.shape == (8, 4, 2)
    assert y.shape == (8,)
    assert list(ix) == list(idx[4:])


def test_create_sequences_window_excludes_label_row() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    feats = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
    target = pd.Series(np.arange(100.0, 110.0), index=idx)
    x, y, _ix = create_sequences(feats, target, look_back=3)
    # First window is rows 0,1,2; label is target row 3.
    np.testing.assert_array_equal(x[0, :, 0], np.array([0.0, 1.0, 2.0]))
    assert y[0] == 103.0


def test_create_sequences_validation() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    feats = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
    target = pd.Series(np.arange(10.0), index=idx)
    with pytest.raises(ValidationError):
        create_sequences(feats, target, look_back=0)
    with pytest.raises(ValidationError):
        create_sequences(feats.to_numpy(), target, look_back=3)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        create_sequences(feats, target.to_numpy(), look_back=3)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        create_sequences(pd.DataFrame(index=idx), target, look_back=3)


def test_create_sequences_insufficient_rows() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    feats = pd.DataFrame({"a": np.arange(4.0)}, index=idx)
    target = pd.Series(np.arange(4.0), index=idx)
    with pytest.raises(InsufficientDataError):
        create_sequences(feats, target, look_back=5)


# --------------------------------------------------------------------------- #
# fit_scaler / scale_sequences                                                #
# --------------------------------------------------------------------------- #
def test_fit_scaler_2d_and_3d_agree() -> None:
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(6, 5, 3))
    mean_3d, std_3d = fit_scaler(seq)
    mean_2d, std_2d = fit_scaler(seq.reshape(-1, 3))
    np.testing.assert_allclose(mean_3d, mean_2d)
    np.testing.assert_allclose(std_3d, std_2d)
    assert mean_3d.shape == (3,) and std_3d.shape == (3,)


def test_fit_scaler_floors_constant_column() -> None:
    arr = np.ones((10, 2))
    arr[:, 1] = np.arange(10.0)
    _mean, std = fit_scaler(arr)
    assert std[0] == 1.0  # constant column floored to 1, not 0
    assert std[1] > 0.0


def test_fit_scaler_validation() -> None:
    with pytest.raises(ValidationError):
        fit_scaler(np.empty((0, 3)))
    with pytest.raises(ValidationError):
        fit_scaler(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValidationError):
        fit_scaler(np.array([[1.0, np.nan]]))


def test_scale_sequences_roundtrip_standardizes_to_unit_scale() -> None:
    rng = np.random.default_rng(2)
    seq = rng.normal(loc=3.0, scale=2.0, size=(50, 10, 4))
    mean, std = fit_scaler(seq)
    scaled = scale_sequences(seq, mean=mean, std=std)
    flat = scaled.reshape(-1, 4)
    np.testing.assert_allclose(flat.mean(axis=0), np.zeros(4), atol=1e-9)
    np.testing.assert_allclose(flat.std(axis=0), np.ones(4), atol=1e-9)


def test_scale_sequences_validation() -> None:
    seq = np.zeros((3, 4, 2))
    with pytest.raises(ValidationError):
        scale_sequences(seq, mean=np.zeros(3), std=np.ones(2))  # mean wrong shape
    with pytest.raises(ValidationError):
        scale_sequences(seq, mean=np.zeros(2), std=np.zeros(2))  # std not positive
    with pytest.raises(ValidationError):
        scale_sequences(np.zeros((3, 4)), mean=np.zeros(4), std=np.ones(4))  # not 3-D
