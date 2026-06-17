"""Return-space forecast metrics and significance tests (NO price-level R²).

Everything here lives in RETURN space, where the honest comparison happens:

- :func:`rmse` / :func:`mae` — out-of-sample error of the model's return forecast;
- :func:`mase_vs_persistence` — Mean Absolute Scaled Error against the persistence
  baseline; ``MASE >= 1`` means the model does NOT beat the naive random walk;
- :func:`directional_accuracy` — sign-hit rate, with a binomial test vs. 0.5;
- :func:`diebold_mariano` — the Diebold-Mariano (1995) test of equal predictive
  accuracy against the random walk, using a Newey-West HAC long-run variance.

DEBUNKED TRAP (documented once, never computed as a metric): a price-LEVEL R²
looks deceptively high because the integrated/trended price level is dominated by
its own lag — that is a unit-root artifact, NOT forecasting skill. We therefore
NEVER report a price-level R². All skill is judged in return space.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lstmforecast._typing import FloatArray

# quantcore-candidate: HAC long-run variance mirrors
# pairs-trading:evaluation/hac.py (Newey-West, Bartlett, Andrews lag).


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """Immutable bundle of return-space out-of-sample forecast metrics.

    Attributes
    ----------
    rmse_return:
        Root-mean-squared error of the model's next-day return forecast.
    mae_return:
        Mean absolute error of the model's next-day return forecast.
    mase_vs_persistence:
        MAE scaled by the persistence baseline's MAE. ``>= 1`` => no improvement.
    directional_accuracy:
        Fraction of next-day return signs correctly predicted.
    directional_pvalue:
        Binomial-test p-value for ``directional_accuracy > 0.5``.
    dm_statistic:
        The Diebold-Mariano statistic (model vs. random walk).
    dm_pvalue:
        Two-sided p-value of the Diebold-Mariano test.
    n_obs:
        Number of out-of-sample forecasts evaluated.
    """

    rmse_return: float
    mae_return: float
    mase_vs_persistence: float
    directional_accuracy: float
    directional_pvalue: float
    dm_statistic: float
    dm_pvalue: float
    n_obs: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of these metrics."""
        return asdict(self)


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the root-mean-squared error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns (same length).

    Returns
    -------
    float
        ``sqrt(mean((y_true - y_pred)**2))``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return the mean absolute error of ``y_pred`` against ``y_true``.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns (same length).

    Returns
    -------
    float
        ``mean(|y_true - y_pred|)``.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def mase_vs_persistence(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> float:
    r"""Mean Absolute Scaled Error of the model relative to persistence.

    Returns ``MAE(model) / MAE(persistence)`` where the persistence forecast is
    ``r_hat = 0`` (the random walk). A value ``>= 1`` means the model does NOT
    beat the naive baseline in return space — the expected, honest outcome on
    random-walk data.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to an all-zeros vector.

    Returns
    -------
    float
        The MASE ratio.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or the baseline MAE is zero.
    """
    raise NotImplementedError


def directional_accuracy(y_true: FloatArray, y_pred: FloatArray) -> tuple[float, float]:
    """Return the sign-hit rate and a binomial-test p-value vs. 0.5.

    Counts observations where ``sign(y_pred) == sign(y_true)`` (zeros handled
    consistently) and tests the hit rate against the no-skill rate 0.5 with a
    two-sided binomial test.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred:
        Forecast next-day returns.

    Returns
    -------
    tuple[float, float]
        ``(accuracy, binomial_pvalue)``.

    Raises
    ------
    ValidationError
        If inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def hac_standard_error(series: FloatArray, *, lag: int | None = None) -> float:
    """Newey-West HAC standard error of the sample mean of ``series``.

    Uses Bartlett weights; ``lag=None`` selects the Andrews (1991) automatic
    truncation ``ceil(4 * (T/100)**(2/9))``. Used to build the Diebold-Mariano
    statistic's denominator from the loss-differential series.

    Parameters
    ----------
    series:
        A 1-D series (e.g. the DM loss differential).
    lag:
        Bartlett lag truncation; ``None`` => Andrews rule.

    Returns
    -------
    float
        ``sqrt(omega_hat / T)``, the HAC standard error of the mean.

    Raises
    ------
    ValidationError
        If ``series`` has fewer than two finite observations or ``lag < 0``.
    """
    raise NotImplementedError


def diebold_mariano(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
    *,
    lag: int | None = None,
) -> tuple[float, float]:
    r"""Diebold-Mariano (1995) test of equal predictive accuracy vs. the random walk.

    With per-observation squared-error losses ``e_model^2`` and ``e_naive^2``, the
    loss differential ``d_t = e_model_t^2 - e_naive_t^2`` has mean ``d_bar``; the
    DM statistic is ``d_bar / HAC_SE(d)``, asymptotically standard normal under
    the null of equal accuracy. A NEGATIVE statistic with a small p-value means
    the model beats persistence; a p-value ``>= alpha`` means the difference is
    insignificant (the honest NULL on random-walk data).

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros.
    lag:
        HAC Bartlett lag; ``None`` => Andrews rule.

    Returns
    -------
    tuple[float, float]
        ``(dm_statistic, two_sided_pvalue)``.

    Raises
    ------
    ValidationError
        If inputs are empty/mismatched or the loss-differential variance is zero.
    """
    raise NotImplementedError


def forecast_metrics(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> ForecastMetrics:
    """Compute the full return-space metric bundle in one call.

    Assembles RMSE, MAE, MASE-vs-persistence, directional accuracy + binomial
    p-value, and the Diebold-Mariano statistic/p-value into a frozen
    :class:`ForecastMetrics`. Deliberately omits any price-level R² (the debunked
    trap).

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros.

    Returns
    -------
    ForecastMetrics
        The frozen metric bundle.

    Raises
    ------
    ValidationError
        If inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def _norm_sf(x: float) -> float:
    """Standard-normal survival function ``1 - Phi(x)`` via the error function."""
    import math

    return 0.5 * math.erfc(x / math.sqrt(2.0))
