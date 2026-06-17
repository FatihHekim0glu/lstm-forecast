"""Unit tests for the implemented data generator and persistence baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast import ValidationError, persistence_returns, random_walk_prices
from lstmforecast.models.baselines import PersistenceForecaster

pytestmark = pytest.mark.unit


def test_random_walk_prices_shape_and_positivity() -> None:
    prices = random_walk_prices(n_obs=500, seed=7)
    assert isinstance(prices, pd.Series)
    assert len(prices) == 500
    assert prices.name == "close"
    assert (prices > 0).all()
    assert prices.index.is_monotonic_increasing


def test_random_walk_prices_reproducible() -> None:
    a = random_walk_prices(n_obs=300, seed=7)
    b = random_walk_prices(n_obs=300, seed=7)
    pd.testing.assert_series_equal(a, b)
    c = random_walk_prices(n_obs=300, seed=8)
    assert not np.array_equal(a.to_numpy(), c.to_numpy())


def test_random_walk_first_obs_anchored_at_s0() -> None:
    prices = random_walk_prices(n_obs=10, seed=7, s0=123.0)
    assert prices.iloc[0] == pytest.approx(123.0)


def test_random_walk_returns_are_mean_zero_unpredictable() -> None:
    prices = random_walk_prices(n_obs=5000, seed=7, sigma=0.01)
    log_ret = np.diff(np.log(prices.to_numpy()))
    # Driftless: mean log-return is ~0; lag-1 autocorrelation is ~0 (white noise).
    assert abs(float(log_ret.mean())) < 0.001
    ac1 = float(np.corrcoef(log_ret[:-1], log_ret[1:])[0, 1])
    assert abs(ac1) < 0.1


def test_random_walk_prices_validation() -> None:
    with pytest.raises(ValidationError):
        random_walk_prices(n_obs=1)
    with pytest.raises(ValidationError):
        random_walk_prices(n_obs=10, s0=0.0)
    with pytest.raises(ValidationError):
        random_walk_prices(n_obs=10, sigma=0.0)


def test_persistence_returns_are_zeros() -> None:
    out = persistence_returns(5)
    assert np.array_equal(out, np.zeros(5))
    assert out.dtype == np.float64


def test_persistence_returns_validation() -> None:
    with pytest.raises(ValidationError):
        persistence_returns(-1)


def test_persistence_forecaster_is_constructible() -> None:
    # The stateless baseline is constructible even before fit/predict are filled in.
    assert isinstance(PersistenceForecaster(), PersistenceForecaster)
