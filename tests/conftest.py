"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`lstmforecast._rng.make_rng` /
the seeded data generators) and returns pandas objects, so tests across the suite
share identical synthetic data with known structure:

- ``random_walk`` — a driftless geometric random-walk price series. On this data
  the next-day return is unpredictable, so a leakage-free LSTM CANNOT beat
  persistence — the honest NULL holds by construction (the anti-leakage guard).
- ``trend_plus_noise`` — a random walk PLUS a constant drift. The price LEVEL
  trends (which is exactly why a price-level R² is deceptive — the debunked
  trap), yet the next-day RETURN stays near-unpredictable.
- ``pure_noise`` — an i.i.d. zero-mean Gaussian RETURN series; the null on which
  persistence (``r_hat = 0``) is provably the error-minimizing point forecast.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast._rng import make_rng

#: Master seed shared by the fixtures for byte-identical synthetic data.
SEED = 20260617


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(SEED)


@pytest.fixture
def random_walk() -> pd.Series:
    """A driftless geometric random-walk price series, length 2000.

    The shipped-model data distribution: ``ln P_t = ln P_{t-1} + N(0, sigma^2)``,
    so the next-day log-return is white noise and unpredictable. A leakage-free
    LSTM must NOT beat persistence here (the integration guard).
    """
    from lstmforecast.data import random_walk_prices

    return random_walk_prices(n_obs=2000, seed=SEED, s0=100.0, sigma=0.01)


@pytest.fixture
def trend_plus_noise() -> pd.Series:
    """A random walk plus a constant per-step drift, length 2000.

    The price level trends (deceptive price-level R²), while the next-day return
    is ``mu + white noise`` and so remains near-unpredictable in return space.
    Implemented by ``lstmforecast.data.trend_plus_noise_prices``.
    """
    from lstmforecast.data import trend_plus_noise_prices

    return trend_plus_noise_prices(n_obs=2000, seed=SEED, s0=100.0, mu=0.0003, sigma=0.01)


@pytest.fixture
def pure_noise() -> pd.Series:
    """An i.i.d. zero-mean Gaussian RETURN series, length 2000.

    The pure-noise null on which persistence (``r_hat = 0``) is provably the
    error-minimizing forecast. Implemented by
    ``lstmforecast.data.pure_noise_returns``.
    """
    from lstmforecast.data import pure_noise_returns

    return pure_noise_returns(n_obs=2000, seed=SEED, sigma=0.01)
