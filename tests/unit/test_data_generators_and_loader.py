"""Unit tests for the data generators, log-return transform, and CSV loader.

Covers the pieces of :mod:`lstmforecast.data` that back the seeded conftest
fixtures (``trend_plus_noise``, ``pure_noise``) and the ``train --data <csv>``
retrain path, plus the no-lookahead ``to_log_returns`` transform. The headline
properties exercised here:

- generator determinism (same seed -> byte-identical output; different seed ->
  different output);
- random-walk / pure-noise returns are mean-zero and serially unpredictable
  (lag-1 autocorrelation ~ 0) — the statistical basis of the honest NULL;
- ``trend_plus_noise`` trends in PRICE space (the deceptive price-level R² trap)
  while its returns stay near-unpredictable;
- ``to_log_returns`` differences raw observed prices without forward-filling;
- ``load_prices`` round-trips a ``date,close`` CSV and rejects malformed files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast import ValidationError
from lstmforecast.data import (
    load_prices,
    pure_noise_returns,
    to_log_returns,
    trend_plus_noise_prices,
)

pytestmark = pytest.mark.unit


def _lag1_autocorr(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


# --------------------------------------------------------------------------- #
# trend_plus_noise_prices                                                      #
# --------------------------------------------------------------------------- #
def test_trend_plus_noise_shape_name_and_positivity() -> None:
    prices = trend_plus_noise_prices(n_obs=500, seed=7)
    assert isinstance(prices, pd.Series)
    assert len(prices) == 500
    assert prices.name == "close"
    assert (prices > 0).all()
    assert prices.index.is_monotonic_increasing
    assert prices.dtype == np.float64


def test_trend_plus_noise_first_obs_anchored_at_s0() -> None:
    prices = trend_plus_noise_prices(n_obs=10, seed=7, s0=123.0)
    assert prices.iloc[0] == pytest.approx(123.0)


def test_trend_plus_noise_is_reproducible() -> None:
    a = trend_plus_noise_prices(n_obs=300, seed=7)
    b = trend_plus_noise_prices(n_obs=300, seed=7)
    pd.testing.assert_series_equal(a, b)
    c = trend_plus_noise_prices(n_obs=300, seed=8)
    assert not np.array_equal(a.to_numpy(), c.to_numpy())


def test_trend_plus_noise_trends_in_price_but_returns_unpredictable() -> None:
    # Strong positive drift over a long horizon: the PRICE level trends up (this
    # is exactly why a price-level R² is deceptive), yet the next-day RETURN is
    # mu + white noise and stays serially unpredictable (lag-1 autocorr ~ 0).
    prices = trend_plus_noise_prices(n_obs=20000, seed=7, mu=0.0003, sigma=0.005)
    assert float(prices.iloc[-1]) > 2.0 * float(prices.iloc[0])

    log_ret = np.diff(np.log(prices.to_numpy()))
    assert float(log_ret.mean()) == pytest.approx(0.0003, abs=1e-4)  # ~ mu
    assert abs(_lag1_autocorr(log_ret)) < 0.05


def test_trend_plus_noise_zero_drift_matches_driftless_mean() -> None:
    prices = trend_plus_noise_prices(n_obs=5000, seed=7, mu=0.0, sigma=0.01)
    log_ret = np.diff(np.log(prices.to_numpy()))
    assert abs(float(log_ret.mean())) < 0.001  # no drift -> mean return ~ 0


def test_trend_plus_noise_validation() -> None:
    with pytest.raises(ValidationError):
        trend_plus_noise_prices(n_obs=1)
    with pytest.raises(ValidationError):
        trend_plus_noise_prices(n_obs=10, s0=0.0)
    with pytest.raises(ValidationError):
        trend_plus_noise_prices(n_obs=10, sigma=0.0)


# --------------------------------------------------------------------------- #
# pure_noise_returns                                                           #
# --------------------------------------------------------------------------- #
def test_pure_noise_returns_shape_name_and_stats() -> None:
    returns = pure_noise_returns(n_obs=5000, seed=7, sigma=0.01)
    assert isinstance(returns, pd.Series)
    assert len(returns) == 5000
    assert returns.name == "return"
    assert returns.dtype == np.float64
    # i.i.d. zero-mean Gaussian: mean ~ 0, std ~ sigma, lag-1 autocorr ~ 0.
    assert abs(float(returns.mean())) < 0.001
    assert float(returns.std()) == pytest.approx(0.01, abs=2e-3)
    assert abs(_lag1_autocorr(returns.to_numpy())) < 0.05


def test_pure_noise_returns_single_obs_is_allowed() -> None:
    out = pure_noise_returns(n_obs=1, seed=7)
    assert len(out) == 1


def test_pure_noise_returns_is_reproducible() -> None:
    a = pure_noise_returns(n_obs=300, seed=7)
    b = pure_noise_returns(n_obs=300, seed=7)
    pd.testing.assert_series_equal(a, b)
    c = pure_noise_returns(n_obs=300, seed=8)
    assert not np.array_equal(a.to_numpy(), c.to_numpy())


def test_pure_noise_returns_validation() -> None:
    with pytest.raises(ValidationError):
        pure_noise_returns(n_obs=0)
    with pytest.raises(ValidationError):
        pure_noise_returns(n_obs=10, sigma=0.0)


# --------------------------------------------------------------------------- #
# to_log_returns                                                              #
# --------------------------------------------------------------------------- #
def test_to_log_returns_matches_log_diff_reference() -> None:
    from lstmforecast.data import random_walk_prices

    prices = random_walk_prices(n_obs=64, seed=3)
    out = to_log_returns(prices)
    ref = np.log(prices).diff().iloc[1:]
    assert out.name == "return"
    assert len(out) == len(prices) - 1
    np.testing.assert_allclose(out.to_numpy(), ref.to_numpy())


def test_to_log_returns_drops_only_the_leading_nan() -> None:
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, 101.0, 99.0, 100.0], index=idx, name="close")
    out = to_log_returns(prices)
    # First row dropped; remaining indices preserved one-to-one.
    assert list(out.index) == list(idx[1:])
    assert out.iloc[0] == pytest.approx(np.log(101.0 / 100.0))


def test_to_log_returns_does_not_forward_fill_across_gaps() -> None:
    # A NaN price gap must NOT be silently bridged into a spurious zero return;
    # the log-diff straddling the gap is NaN, never 0.0.
    idx = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, np.nan, 102.0, 103.0], index=idx, name="close")
    out = to_log_returns(prices)
    assert np.isnan(out.iloc[0])  # 100 -> NaN
    assert np.isnan(out.iloc[1])  # NaN -> 102
    assert out.iloc[2] == pytest.approx(np.log(103.0 / 102.0))


def test_to_log_returns_rejects_non_positive_prices() -> None:
    idx = pd.date_range("2021-01-01", periods=3, freq="B")
    with pytest.raises(ValidationError):
        to_log_returns(pd.Series([100.0, 0.0, 101.0], index=idx))
    with pytest.raises(ValidationError):
        to_log_returns(pd.Series([100.0, -5.0, 101.0], index=idx))


def test_to_log_returns_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        to_log_returns(pd.Series([], dtype="float64"))


# --------------------------------------------------------------------------- #
# load_prices                                                                  #
# --------------------------------------------------------------------------- #
def _write_csv(tmp_path: object, text: str, name: str = "px.csv") -> str:
    import os

    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_load_prices_reads_date_close_csv(tmp_path: object) -> None:
    path = _write_csv(
        tmp_path,
        "date,close\n2021-01-04,100.0\n2021-01-05,101.5\n2021-01-06,99.0\n",
    )
    series, source = load_prices(path)
    assert source == "csv"
    assert series.name == "close"
    assert series.dtype == np.float64
    assert list(series.to_numpy()) == [100.0, 101.5, 99.0]
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index[0] == pd.Timestamp("2021-01-04")


def test_load_prices_sorts_unordered_rows_ascending(tmp_path: object) -> None:
    path = _write_csv(
        tmp_path,
        "date,close\n2021-01-06,99.0\n2021-01-04,100.0\n2021-01-05,101.5\n",
    )
    series, _ = load_prices(path)
    assert series.index.is_monotonic_increasing
    assert list(series.to_numpy()) == [100.0, 101.5, 99.0]


def test_load_prices_is_case_insensitive_in_headers(tmp_path: object) -> None:
    path = _write_csv(tmp_path, "Date,Close\n2021-01-04,100.0\n2021-01-05,101.0\n")
    series, source = load_prices(path)
    assert source == "csv"
    assert len(series) == 2


def test_load_prices_falls_back_to_first_column_as_date(tmp_path: object) -> None:
    # No literal 'date' header: the first column is treated as the date index.
    path = _write_csv(tmp_path, "dt,close\n2021-01-04,100.0\n2021-01-05,101.0\n")
    series, _ = load_prices(path)
    assert series.index[0] == pd.Timestamp("2021-01-04")


def test_load_prices_accepts_pathlib_path(tmp_path: object) -> None:
    from pathlib import Path

    raw = _write_csv(tmp_path, "date,close\n2021-01-04,100.0\n2021-01-05,101.0\n")
    series, source = load_prices(Path(raw))
    assert source == "csv"
    assert len(series) == 2


def test_load_prices_missing_file_raises() -> None:
    with pytest.raises(ValidationError):
        load_prices("/no/such/file/definitely_missing.csv")


def test_load_prices_unparseable_csv_raises(tmp_path: object) -> None:
    # A ragged file with an unterminated quote makes pandas' parser raise; the
    # loader must surface that as a ValidationError, never an opaque ParserError.
    path = _write_csv(
        tmp_path,
        'date,close\n2021-01-04,100.0,extra,fields\n"unterminated,1\n',
    )
    with pytest.raises(ValidationError):
        load_prices(path)


def test_load_prices_missing_close_column_raises(tmp_path: object) -> None:
    path = _write_csv(tmp_path, "date,open\n2021-01-04,100.0\n2021-01-05,101.0\n")
    with pytest.raises(ValidationError):
        load_prices(path)


def test_load_prices_empty_file_raises(tmp_path: object) -> None:
    path = _write_csv(tmp_path, "date,close\n")
    with pytest.raises(ValidationError):
        load_prices(path)


def test_load_prices_non_positive_close_raises(tmp_path: object) -> None:
    path = _write_csv(tmp_path, "date,close\n2021-01-04,100.0\n2021-01-05,-1.0\n")
    with pytest.raises(ValidationError):
        load_prices(path)


def test_load_prices_missing_close_value_raises(tmp_path: object) -> None:
    path = _write_csv(tmp_path, "date,close\n2021-01-04,100.0\n2021-01-05,\n")
    with pytest.raises(ValidationError):
        load_prices(path)


def test_load_prices_duplicate_dates_raise(tmp_path: object) -> None:
    path = _write_csv(
        tmp_path,
        "date,close\n2021-01-04,100.0\n2021-01-04,101.0\n",
    )
    with pytest.raises(ValidationError):
        load_prices(path)
