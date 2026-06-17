"""Return-space forecast-metric correctness (the honest-evaluation layer).

Covers the compute kernels in :mod:`lstmforecast.evaluation.metrics`:

- ``rmse`` / ``mae`` against hand-computed values and a NumPy reference;
- ``mase_vs_persistence`` — the scaled error whose ``>= 1`` reading is the NULL;
- ``directional_accuracy`` + exact binomial p-value (cross-checked to SciPy);
- ``hac_standard_error`` (Newey-West/Bartlett) and ``diebold_mariano`` on
  fixtures, including the random-walk honest-null behaviour;
- ``forecast_metrics`` assembly and its deliberate ABSENCE of any price-level R².

These complement the DSR n_trials guard (``test_dsr_and_costs``) and the pure
verdict truth table (``test_verdict_truth_table``); a cross-cutting honest-null
check ties the metric bundle to the verdict here too.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest

from lstmforecast import (
    ValidationError,
    deflated_sharpe_ratio,
    derive_verdict,
    random_walk_prices,
)
from lstmforecast._typing import FloatArray
from lstmforecast.evaluation.metrics import (
    ForecastMetrics,
    diebold_mariano,
    directional_accuracy,
    forecast_metrics,
    hac_standard_error,
    mae,
    mase_vs_persistence,
    rmse,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# RMSE / MAE                                                                   #
# --------------------------------------------------------------------------- #
def test_rmse_mae_known_values() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0])
    # errors = [0, 1, 2, 3]; MAE = 6/4 = 1.5; RMSE = sqrt(14/4) = sqrt(3.5).
    assert mae(y_true, y_pred) == pytest.approx(1.5)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(3.5))


def test_rmse_mae_zero_on_perfect_forecast() -> None:
    y = np.array([0.01, -0.02, 0.0, 0.03])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0


def test_rmse_matches_numpy_reference(rng: np.random.Generator) -> None:
    y_true = rng.normal(0.0, 0.01, size=500)
    y_pred = rng.normal(0.0, 0.01, size=500)
    ref = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert rmse(y_true, y_pred) == pytest.approx(ref)


@pytest.mark.parametrize("fn", [rmse, mae])
def test_error_metrics_validate_shape(fn: Callable[[FloatArray, FloatArray], float]) -> None:
    with pytest.raises(ValidationError):
        fn(np.array([]), np.array([]))
    with pytest.raises(ValidationError):
        fn(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValidationError):
        fn(np.array([1.0, np.nan]), np.array([1.0, 2.0]))  # non-finite y_true
    with pytest.raises(ValidationError):
        fn(np.array([1.0, 2.0]), np.array([1.0, np.inf]))  # non-finite y_pred


def test_naive_baseline_validation() -> None:
    y_true = np.array([0.01, -0.02, 0.015])
    model = np.array([0.0, 0.0, 0.0])
    # Length-mismatched explicit naive vector.
    with pytest.raises(ValidationError):
        mase_vs_persistence(y_true, model, np.zeros(2))
    # Non-finite explicit naive vector.
    with pytest.raises(ValidationError):
        diebold_mariano(y_true, model, np.array([0.0, np.nan, 0.0]))


# --------------------------------------------------------------------------- #
# MASE vs persistence                                                         #
# --------------------------------------------------------------------------- #
def test_mase_equals_one_when_model_is_persistence() -> None:
    # If the model forecasts r_hat = 0 it IS persistence => MASE == 1 exactly.
    y_true = np.array([0.01, -0.02, 0.015, -0.005])
    zeros = np.zeros_like(y_true)
    assert mase_vs_persistence(y_true, zeros) == pytest.approx(1.0)


def test_mase_below_one_for_a_better_model() -> None:
    y_true = np.array([0.02, -0.02, 0.02, -0.02])
    # A model that halves every error beats persistence => MASE = 0.5.
    better = y_true * 0.5
    assert mase_vs_persistence(y_true, better) == pytest.approx(0.5)


def test_mase_above_one_for_a_worse_model() -> None:
    y_true = np.array([0.02, -0.02, 0.02, -0.02])
    # A model that doubles the move (overshoots) is worse than r_hat = 0.
    worse = y_true * -1.0  # predicts the exact opposite move
    assert mase_vs_persistence(y_true, worse) > 1.0


def test_mase_default_naive_is_zeros() -> None:
    y_true = np.array([0.01, -0.02, 0.015])
    model = np.array([0.005, -0.01, 0.0])
    explicit = mase_vs_persistence(y_true, model, np.zeros_like(y_true))
    default = mase_vs_persistence(y_true, model)
    assert explicit == pytest.approx(default)


def test_mase_raises_on_degenerate_baseline() -> None:
    # All-zero realized returns => persistence MAE is zero => undefined MASE.
    y_true = np.zeros(8)
    with pytest.raises(ValidationError):
        mase_vs_persistence(y_true, np.ones(8))


# --------------------------------------------------------------------------- #
# Directional accuracy + exact binomial test                                  #
# --------------------------------------------------------------------------- #
def test_directional_accuracy_perfect_and_pvalue_small() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.01, -0.02])
    acc, pval = directional_accuracy(y_true, y_true)  # same sign everywhere
    assert acc == pytest.approx(1.0)
    assert pval < 0.05  # 8/8 hits is significant at 5%


def test_directional_accuracy_half_is_insignificant() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.01, -0.02, -0.03, 0.04])  # 2 of 4 correct
    acc, pval = directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(0.5)
    assert pval == pytest.approx(1.0)  # exact symmetric two-sided p-value


def test_directional_accuracy_excludes_zero_realized_direction() -> None:
    # A realized return of exactly zero has no direction to predict and is not
    # a scoreable trial; only the three non-zero moves count (all hit).
    y_true = np.array([0.0, 0.01, -0.02, 0.03])
    y_pred = np.array([0.5, 0.01, -0.02, 0.03])
    acc, _ = directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(1.0)


def test_directional_binomial_matches_scipy_reference() -> None:
    # 60 of 100 sign hits: cross-check the exact two-sided p-value to SciPy.
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=100)
    y_true = signs * 0.01
    flip = np.ones(100)
    flip[:40] = -1.0  # flip 40 -> 60 correct
    y_pred = y_true * flip
    acc, pval = directional_accuracy(y_true, y_pred)
    assert acc == pytest.approx(0.60)
    pytest.importorskip("scipy")
    from scipy.stats import binomtest

    ref = binomtest(60, 100, 0.5, alternative="two-sided").pvalue
    assert pval == pytest.approx(ref, abs=1e-9)


def test_directional_accuracy_all_zero_realized_raises() -> None:
    with pytest.raises(ValidationError):
        directional_accuracy(np.zeros(5), np.ones(5))


# --------------------------------------------------------------------------- #
# HAC standard error (Newey-West / Bartlett)                                   #
# --------------------------------------------------------------------------- #
def test_hac_iid_reduces_to_plain_se() -> None:
    # With lag=0 the Bartlett sum has no cross terms: omega == gamma0, so the
    # HAC SE equals the population-style SD/sqrt(T).
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=400)
    centred = x - x.mean()
    gamma0 = float(np.dot(centred, centred) / x.size)
    assert hac_standard_error(x, lag=0) == pytest.approx(math.sqrt(gamma0 / x.size))


def test_hac_matches_independent_newey_west_formula() -> None:
    # Cross-check against a fully independent re-derivation of the Bartlett-
    # weighted long-run variance on an AR(1) series (so the lag terms bite).
    rng = np.random.default_rng(2)
    n = 500
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + eps[i]
    lag = 5
    ours = hac_standard_error(x, lag=lag)

    c = x - x.mean()
    autocov = [
        float(np.sum(c[h:] * c[:-h]) / n) if h else float(np.sum(c * c) / n) for h in range(lag + 1)
    ]
    omega = autocov[0] + 2.0 * sum((1.0 - h / (lag + 1.0)) * autocov[h] for h in range(1, lag + 1))
    ref = math.sqrt(max(omega, 0.0) / n)
    assert ours == pytest.approx(ref, rel=1e-12)


def test_hac_validation() -> None:
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([1.0]))  # < 2 finite obs
    with pytest.raises(ValidationError):
        hac_standard_error(np.array([1.0, 2.0, 3.0]), lag=-1)


# --------------------------------------------------------------------------- #
# Diebold-Mariano                                                             #
# --------------------------------------------------------------------------- #
def test_dm_zero_when_model_equals_persistence() -> None:
    # Model == persistence => loss differential is identically zero => DM stat 0,
    # p-value 1 (no detectable difference in predictive accuracy).
    rng = np.random.default_rng(3)
    y_true = rng.normal(0.0, 0.01, size=300)
    zeros = np.zeros_like(y_true)
    stat, pval = diebold_mariano(y_true, zeros, zeros)
    assert stat == pytest.approx(0.0)
    assert pval == pytest.approx(1.0)


def test_dm_negative_and_significant_when_model_clearly_better() -> None:
    # Construct a model that is genuinely closer to y_true than r_hat = 0 at
    # every step: d_t = e_model^2 - e_naive^2 < 0 always => DM stat strongly
    # negative, p-value tiny.
    rng = np.random.default_rng(4)
    y_true = rng.normal(0.0, 0.02, size=400)
    model = y_true * 0.5  # halves every error
    stat, pval = diebold_mariano(y_true, model)
    assert stat < 0.0
    assert pval < 0.05


def test_dm_insignificant_on_random_walk() -> None:
    # The headline honest-null: on a genuine random walk a persistence-like LSTM
    # cannot beat r_hat = 0, so the DM test against the random walk is
    # insignificant. We use a leakage-free, no-skill forecast (persistence plus
    # an independent jitter) and assert the difference is NOT significant.
    prices = random_walk_prices(n_obs=2000, seed=20260617, s0=100.0, sigma=0.01).to_numpy()
    returns = np.diff(np.log(prices))
    rng = np.random.default_rng(5)
    # A no-skill forecast: tiny noise uncorrelated with the future return -> on
    # average no better than r_hat = 0.
    noise_model = rng.normal(0.0, returns.std() * 1e-3, size=returns.size)
    _, pval = diebold_mariano(returns, noise_model)
    assert pval >= 0.05


def test_dm_validation() -> None:
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.01]), np.array([0.0]))  # < 2 obs


# --------------------------------------------------------------------------- #
# forecast_metrics assembly + NO price-level R²                               #
# --------------------------------------------------------------------------- #
def test_forecast_metrics_bundle_shape() -> None:
    rng = np.random.default_rng(6)
    y_true = rng.normal(0.0, 0.01, size=250)
    model = y_true * 0.3 + rng.normal(0.0, 0.005, size=250)
    m = forecast_metrics(y_true, model)
    assert isinstance(m, ForecastMetrics)
    assert m.n_obs == 250
    assert m.rmse_return >= 0.0 and m.mae_return >= 0.0
    assert 0.0 <= m.directional_accuracy <= 1.0
    assert 0.0 <= m.directional_pvalue <= 1.0
    assert 0.0 <= m.dm_pvalue <= 1.0


def test_forecast_metrics_components_agree_with_individual_calls() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(0.0, 0.01, size=200)
    model = rng.normal(0.0, 0.01, size=200)
    m = forecast_metrics(y_true, model)
    assert m.rmse_return == pytest.approx(rmse(y_true, model))
    assert m.mae_return == pytest.approx(mae(y_true, model))
    assert m.mase_vs_persistence == pytest.approx(mase_vs_persistence(y_true, model))


def test_forecast_metrics_reports_no_price_level_r2() -> None:
    """The metric bundle must NOT carry any price-level R² (the debunked trap).

    The honest-null discipline forbids reporting a price-level R²: the unit-root
    trend inflates it into a meaningless artifact. Assert no field, attribute, or
    serialized key references R²/r_squared anywhere in the bundle.
    """
    rng = np.random.default_rng(8)
    y_true = rng.normal(0.0, 0.01, size=120)
    m = forecast_metrics(y_true, y_true * 0.4)

    field_names = {f.name for f in dataclasses.fields(m)}
    serialized_keys = set(m.to_dict())
    banned_substrings = ("r2", "r_squared", "rsquared", "r_2", "price_level", "level_r")
    for name in field_names | serialized_keys:
        lowered = name.lower()
        assert all(bad not in lowered for bad in banned_substrings), (
            f"forecast metrics must not expose a price-level R²-like field: {name!r}"
        )
    # And the module namespace exposes no r-squared helper either.
    import lstmforecast.evaluation.metrics as metrics_mod

    public = {n for n in dir(metrics_mod) if not n.startswith("_")}
    assert not any("r2" in n.lower() or "r_squared" in n.lower() for n in public)


# --------------------------------------------------------------------------- #
# Cross-cutting honest-null: metrics -> verdict on random-walk data           #
# --------------------------------------------------------------------------- #
def test_random_walk_metrics_drive_a_false_verdict() -> None:
    """End-to-end on random-walk data: the derived verdict is FALSE.

    Feeding a no-skill (persistence) forecast through the real metric bundle and
    then through the pure verdict yields ``beats_naive = False`` — the honest
    NULL holding by construction on random-walk data.
    """
    prices = random_walk_prices(n_obs=2000, seed=20260617, s0=100.0, sigma=0.01).to_numpy()
    returns = np.diff(np.log(prices))
    persistence = np.zeros_like(returns)  # r_hat = 0, the floor to beat
    m = forecast_metrics(returns, persistence)
    # MASE is exactly 1 for the persistence forecast; the verdict must be False.
    assert m.mase_vs_persistence == pytest.approx(1.0)
    verdict = derive_verdict(
        mase=m.mase_vs_persistence,
        dm_pvalue=m.dm_pvalue,
        directional_accuracy=m.directional_accuracy,
    )
    assert verdict.beats_naive is False


def test_dsr_n_trials_guard_uses_full_grid() -> None:
    """DSR must penalize the FULL HPO grid: more trials => non-increasing DSR.

    Guards against the data-snooping footgun of passing ``n_trials = 1`` (the
    selected config) instead of the full configuration count actually scored.
    """
    dsr_one = deflated_sharpe_ratio(0.2, n_obs=500, n_trials=1, variance_of_trial_sharpes=0.04)
    dsr_grid = deflated_sharpe_ratio(  # e.g. 2x4x3x2 = 48 configs scored
        0.2, n_obs=500, n_trials=48, variance_of_trial_sharpes=0.04
    )
    assert dsr_grid <= dsr_one
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.2, n_obs=500, n_trials=0, variance_of_trial_sharpes=0.04)
