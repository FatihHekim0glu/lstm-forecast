"""Per-fold technical feature engineering on strictly-past bars.

THE DE-LEAK (part 1): every feature here is a function of bars STRICTLY EARLIER
than the bar it is attached to. Concretely:

- price changes use ``close.pct_change(fill_method=None)`` (no ffill-then-diff);
- every indicator is lagged one bar with ``.shift(1)`` so the feature row aligned
  to the target ``r_{t+1}`` is computed from information available at the close of
  day ``t`` and earlier — NEVER the same-bar close level (which would leak the
  label's scale);
- rolling / EWM windows have a finite warm-up that is dropped, and (in the
  walk-forward engine) are recomputed per fold so a warm-up never straddles a
  train/test boundary.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# quantcore-candidate: lagged technical features mirror factorlab feature kernels.


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable specification of the technical-feature set to engineer.

    Attributes
    ----------
    momentum_windows:
        Look-back windows (in bars) for lagged past-return momentum features.
    vol_windows:
        Look-back windows for rolling realized-volatility features.
    rsi_window:
        Window for the lagged Relative Strength Index.
    include_lagged_return:
        Whether to include the one-bar-lagged return ``r_{t}`` itself as a
        feature for predicting ``r_{t+1}`` (still strictly past).
    """

    momentum_windows: tuple[int, ...] = (5, 10, 20)
    vol_windows: tuple[int, ...] = (10, 20)
    rsi_window: int = 14
    include_lagged_return: bool = True
    feature_names: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        """Validate that every window is a positive integer.

        Raises
        ------
        ValidationError
            If any window is non-positive.
        """
        from lstmforecast._exceptions import ValidationError

        windows = (*self.momentum_windows, *self.vol_windows, self.rsi_window)
        if any(int(w) < 1 for w in windows):
            raise ValidationError(f"FeatureSpec: all windows must be >= 1, got {windows!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return asdict(self)


def engineer_features(prices: pd.Series, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Build a no-lookahead technical-feature matrix from a price series.

    Every returned column is lagged so that the feature row indexed at date ``t``
    uses only bars at or before ``t`` and is therefore safe to pair with the
    target ``r_{t+1}``. Rows with any NaN from indicator warm-up are dropped, so
    the returned frame is dense and ready to scale.

    DE-LEAK GUARANTEE: there is NO same-bar close-level feature; price-derived
    features are differences/ratios (returns, momentum, volatility, RSI), never
    the raw level, so the feature scale cannot encode the label's scale.

    Parameters
    ----------
    prices:
        A strictly-positive, time-indexed price series (sorted ascending).
    spec:
        The feature specification; defaults to :class:`FeatureSpec` defaults.

    Returns
    -------
    pandas.DataFrame
        A dense, float64 feature matrix indexed by date, columns named after the
        engineered features.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or too short for the requested windows.
    """
    raise NotImplementedError
