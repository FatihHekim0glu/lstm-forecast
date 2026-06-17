"""Supervised-sequence construction for the LSTM input tensor.

THE DE-LEAK (part 2): :func:`create_sequences` slices a feature matrix into
overlapping ``look_back``-length windows, each paired with the NEXT-day target
return ``r_{t+1}``. The window for a sample ends at bar ``t`` and the label is the
return realized at ``t+1``, so no sample ever contains its own label. In the
walk-forward engine a ``look_back``-sized purge ensures no window straddles a
train/test boundary.

Importing this module has no side effects.
"""

from __future__ import annotations

import pandas as pd

from lstmforecast._typing import FloatArray, SequenceTensor


def create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    look_back: int = 60,
) -> tuple[SequenceTensor, FloatArray, pd.Index]:
    r"""Slice ``features``/``target`` into ``(X, y, index)`` supervised sequences.

    For each end-bar position ``t`` with ``t >= look_back - 1`` and a defined
    next-day target, emits one sample:

    - ``X[i] = features.iloc[t - look_back + 1 : t + 1].to_numpy()`` — the trailing
      ``look_back`` feature rows up to and INCLUDING bar ``t``;
    - ``y[i] = target.loc[<date of r_{t+1}>]`` — the NEXT-day return.

    The returned ``index`` labels each sample by the date of its target
    ``r_{t+1}``, so callers can align predictions to dates and apply the
    walk-forward purge/embargo by date.

    NO-LEAK INVARIANT: ``X[i]`` never contains ``y[i]`` nor any bar later than the
    window end, and ``features`` and ``target`` must be index-aligned by the
    caller so that ``target`` at row ``t`` is the return earned at ``t`` (the
    window ending at ``t`` predicts the target one row later).

    Parameters
    ----------
    features:
        A dense, no-lookahead feature matrix indexed by date.
    target:
        The next-day log-return series, index-aligned to ``features``.
    look_back:
        The sequence window length in bars (``>= 1``).

    Returns
    -------
    tuple[SequenceTensor, FloatArray, pandas.Index]
        ``X`` of shape ``(n_samples, look_back, n_features)``, ``y`` of shape
        ``(n_samples,)``, and the target-date index of length ``n_samples``.

    Raises
    ------
    ValidationError
        If ``look_back < 1`` or ``features`` and ``target`` are misaligned.
    InsufficientDataError
        If there are fewer than ``look_back + 1`` aligned rows (no sample can be
        formed).
    """
    raise NotImplementedError


def scale_sequences(
    sequences: SequenceTensor,
    *,
    mean: FloatArray,
    std: FloatArray,
) -> SequenceTensor:
    r"""Standardize a 3-D sequence tensor with a TRAIN-fold-fitted mean/std.

    THE DE-LEAK (part 3): the ``mean``/``std`` MUST have been fitted on the TRAIN
    fold only (see :mod:`lstmforecast.walkforward.engine`) and are merely APPLIED
    here to train/val/test windows. This function never estimates statistics from
    ``sequences`` itself, so it cannot leak val/test scale into training.

    Parameters
    ----------
    sequences:
        A ``(n_samples, look_back, n_features)`` tensor.
    mean, std:
        Per-feature standardization statistics of shape ``(n_features,)``, fitted
        on the train fold. ``std`` entries are floored away from zero by the
        caller.

    Returns
    -------
    SequenceTensor
        The standardized tensor ``(sequences - mean) / std`` (same shape).

    Raises
    ------
    ValidationError
        If shapes are incompatible or any ``std`` entry is non-positive.
    """
    raise NotImplementedError


def fit_scaler(train_features: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Fit per-feature standardization statistics on TRAIN-fold features ONLY.

    Returns ``(mean, std)`` over the flattened ``(n_train_samples * look_back,
    n_features)`` train rows, with ``std`` floored at a small positive epsilon to
    guard constant columns. This is the SINGLE place scaler statistics are
    estimated, and it is called with train-fold data exclusively — the headline
    fix for the original repo's full-series-scaler leakage bug.

    Parameters
    ----------
    train_features:
        Train-fold feature rows, 2-D ``(n_rows, n_features)`` or a 3-D sequence
        tensor that will be reshaped to 2-D before fitting.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The per-feature ``mean`` and (epsilon-floored) ``std``, each shaped
        ``(n_features,)``.

    Raises
    ------
    ValidationError
        If ``train_features`` is empty or not 2-/3-dimensional.
    """
    raise NotImplementedError


__all__ = ["create_sequences", "fit_scaler", "scale_sequences"]
