"""Anchored/expanding walk-forward engine with purge + embargo (THE de-leak core).

This module enforces, per fold, every leakage guard the original repo lacked:

1. **Per-fold feature recompute** — features are engineered separately inside each
   train/val/test slice so an indicator warm-up never straddles a boundary.
2. **Per-fold scaler fit on TRAIN ONLY** — the standardizer's mean/std are fitted
   on the train slice exclusively, then APPLIED (never re-fitted) to val and test.
   This is the headline fix for the full-series-scaler leakage bug.
3. **Purge (>= look_back)** — a gap of at least ``look_back`` bars is removed at
   every train/val/test boundary so no ``look_back``-length sequence window can
   span the split.
4. **Embargo** — a further gap after each test block, so adjacent folds cannot
   share information through overlapping windows.

The engine is generic over the model (it takes a ``model_factory`` callable), so
the persistence baseline and the LSTM run through the SAME splits — the only fair
way to compare them.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lstmforecast._typing import FloatArray

# quantcore-candidate: purge/embargo mirror pairs-trading:evaluation/_purge.py,
# generalized here to the sequence-window (look_back) setting.

#: A model factory builds a fresh, untrained forecaster for one fold. It receives
#: the fold's hyperparameter mapping and returns an object exposing
#: ``fit(X, y)`` and ``predict(X) -> ndarray`` (both the persistence baseline and
#: the LSTM satisfy this).
ModelFactory = Callable[[dict[str, Any]], Any]


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Immutable configuration for the walk-forward split.

    Attributes
    ----------
    look_back:
        Sequence window length; also the MINIMUM purge size (the engine clamps
        ``purge`` up to ``look_back``).
    train_size:
        Number of bars in each train slice (the initial size when ``anchored``).
    val_size:
        Number of bars in each validation slice (HPO scoring only).
    test_size:
        Number of bars in each out-of-sample test slice.
    step:
        Number of bars to advance the window between folds.
    purge:
        Boundary gap in bars; clamped up to ``look_back`` so no window straddles a
        split.
    embargo:
        Extra gap in bars after each test block.
    anchored:
        If ``True``, the train slice expands from a fixed start (anchored); else it
        rolls with a fixed length.
    """

    look_back: int = 60
    train_size: int = 750
    val_size: int = 125
    test_size: int = 125
    step: int = 125
    purge: int = 60
    embargo: int = 5
    anchored: bool = True

    def __post_init__(self) -> None:
        """Validate sizes and clamp ``purge`` up to ``look_back``.

        Raises
        ------
        ValidationError
            If any size/step is ``< 1`` or any gap is negative.
        """
        from lstmforecast._exceptions import ValidationError

        sizes = (self.look_back, self.train_size, self.val_size, self.test_size, self.step)
        if min(sizes) < 1:
            raise ValidationError(f"WalkForwardConfig: sizes/step must be >= 1, got {self!r}.")
        if self.purge < 0 or self.embargo < 0:
            raise ValidationError(
                f"WalkForwardConfig: purge/embargo must be >= 0, got "
                f"purge={self.purge}, embargo={self.embargo}."
            )

    @property
    def effective_purge(self) -> int:
        """Return the purge actually applied: ``max(purge, look_back)``."""
        return max(self.purge, self.look_back)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Fold:
    """Index boundaries (half-open ``[start, stop)``) for one walk-forward fold.

    Attributes
    ----------
    train:
        ``(start, stop)`` row positions of the train slice.
    val:
        ``(start, stop)`` row positions of the validation slice (post-purge).
    test:
        ``(start, stop)`` row positions of the test slice (post-purge/embargo).
    """

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold's boundaries."""
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Immutable per-fold inference output (OOS predictions vs. realized returns).

    Attributes
    ----------
    fold:
        The :class:`Fold` boundaries this result came from.
    dates:
        The target dates of the OOS test predictions.
    y_true:
        Realized next-day returns over the test slice.
    y_pred_model:
        The model's OOS next-day return forecasts.
    y_pred_naive:
        The persistence baseline's OOS forecasts (all zeros), aligned to
        ``y_true``.
    """

    fold: Fold
    dates: pd.Index
    y_true: FloatArray
    y_pred_model: FloatArray
    y_pred_naive: FloatArray

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fold result."""
        return {
            "fold": self.fold.to_dict(),
            "dates": [str(d) for d in self.dates],
            "y_true": [_safe_float(v) for v in self.y_true],
            "y_pred_model": [_safe_float(v) for v in self.y_pred_model],
            "y_pred_naive": [_safe_float(v) for v in self.y_pred_naive],
        }


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable aggregate of all walk-forward folds (stacked OOS series).

    Attributes
    ----------
    dates:
        Concatenated OOS target dates across folds.
    y_true:
        Concatenated realized next-day returns.
    y_pred_model:
        Concatenated model forecasts.
    y_pred_naive:
        Concatenated persistence forecasts.
    n_folds:
        Number of folds produced.
    n_trials:
        FULL count of HPO configurations scored on validation (feeds the DSR's
        ``n_trials`` — the honest multiplicity count).
    """

    dates: pd.Index
    y_true: FloatArray
    y_pred_model: FloatArray
    y_pred_naive: FloatArray
    n_folds: int
    n_trials: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the aggregate result."""
        return {
            "dates": [str(d) for d in self.dates],
            "y_true": [_safe_float(v) for v in self.y_true],
            "y_pred_model": [_safe_float(v) for v in self.y_pred_model],
            "y_pred_naive": [_safe_float(v) for v in self.y_pred_naive],
            "n_folds": int(self.n_folds),
            "n_trials": int(self.n_trials),
            "meta": dict(self.meta),
        }


def make_folds(n_obs: int, config: WalkForwardConfig) -> list[Fold]:
    """Compute the train/val/test boundaries for every walk-forward fold.

    NO-LOOKAHEAD GUARANTEE: between train and val, and between val and test, a gap
    of ``config.effective_purge`` (``>= look_back``) bars is removed so no
    sequence window spans a boundary; ``config.embargo`` bars are dropped after
    each test block. With ``anchored=True`` the train slice starts at row 0 and
    grows; otherwise it rolls with length ``train_size``.

    Parameters
    ----------
    n_obs:
        Total number of feature/target rows available.
    config:
        The walk-forward configuration.

    Returns
    -------
    list[Fold]
        Ordered folds; empty if ``n_obs`` is too small for even one fold.

    Raises
    ------
    InsufficientDataError
        If not a single fold fits in ``n_obs`` rows.
    """
    raise NotImplementedError


def run_walk_forward(
    prices: pd.Series,
    model_factory: ModelFactory,
    config: WalkForwardConfig,
    *,
    hpo_grid: list[dict[str, Any]] | None = None,
) -> WalkForwardResult:
    """Run a fully leakage-guarded walk-forward over a price series.

    For each fold produced by :func:`make_folds`:

    1. recompute log-returns and engineer features INSIDE the fold;
    2. fit the standardizer on the TRAIN slice ONLY and apply it to val/test;
    3. build sequences (``look_back``) per slice;
    4. (if ``hpo_grid``) score each config on the VAL slice, select the best, and
       count every scored config toward ``n_trials``;
    5. fit the selected model on TRAIN and predict the TEST slice;
    6. record OOS ``(dates, y_true, y_pred_model, y_pred_naive)``.

    The persistence baseline runs through the SAME folds so the comparison is
    apples-to-apples. The result's ``n_trials`` is the honest multiplicity count
    fed to the Deflated Sharpe.

    Parameters
    ----------
    prices:
        A strictly-positive, time-indexed price series.
    model_factory:
        Callable building a fresh forecaster from a hyperparameter mapping.
    config:
        The walk-forward configuration.
    hpo_grid:
        Optional list of hyperparameter mappings to score on the validation slice
        (its length sets the per-fold trial count). ``None`` trains a single
        default config.

    Returns
    -------
    WalkForwardResult
        The stacked OOS predictions and the honest ``n_trials`` count.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    InsufficientDataError
        If the series is too short for one fold.
    """
    raise NotImplementedError
