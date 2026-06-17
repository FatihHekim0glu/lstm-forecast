"""Unit tests for the reused DSR/PSR layer and the fixed-bps cost model."""

from __future__ import annotations

import pytest

from lstmforecast import (
    ValidationError,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from lstmforecast.walkforward.costs import FixedBpsCost

pytestmark = pytest.mark.unit


def test_psr_in_unit_interval() -> None:
    psr = probabilistic_sharpe_ratio(0.1, n_obs=250)
    assert 0.0 <= psr <= 1.0
    # A higher observed Sharpe should not lower the PSR.
    assert probabilistic_sharpe_ratio(0.2, n_obs=250) >= psr


def test_psr_validates_n_obs() -> None:
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(0.1, n_obs=1)


def test_dsr_monotonic_in_n_trials() -> None:
    kwargs = {"n_obs": 500, "variance_of_trial_sharpes": 0.04}
    dsr_few = deflated_sharpe_ratio(0.15, n_trials=2, **kwargs)
    dsr_many = deflated_sharpe_ratio(0.15, n_trials=200, **kwargs)
    # More trials => higher multiplicity benchmark => non-increasing DSR.
    assert dsr_many <= dsr_few
    assert 0.0 <= dsr_many <= 1.0


def test_dsr_single_trial_reduces_to_psr_vs_zero() -> None:
    dsr = deflated_sharpe_ratio(0.15, n_obs=500, n_trials=1, variance_of_trial_sharpes=0.04)
    psr = probabilistic_sharpe_ratio(0.15, n_obs=500, benchmark_sharpe=0.0)
    assert dsr == pytest.approx(psr)


def test_dsr_validation() -> None:
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=500, n_trials=0, variance_of_trial_sharpes=0.04)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=500, n_trials=5, variance_of_trial_sharpes=-1.0)


def test_fixed_bps_cost() -> None:
    cost = FixedBpsCost(bps=10.0)
    assert cost.cost(0.0) == 0.0
    assert cost.cost(1.0) == pytest.approx(10.0 / 10_000.0)
    assert cost.cost(0.5) == pytest.approx(5.0 / 10_000.0)


def test_fixed_bps_cost_validation() -> None:
    with pytest.raises(ValidationError):
        FixedBpsCost(bps=-1.0)
    with pytest.raises(ValidationError):
        FixedBpsCost(bps=10.0).cost(-0.1)
