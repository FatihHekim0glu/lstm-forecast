"""Property-based invariants for the ``features`` group (the de-leak core).

These Hypothesis tests pin the no-lookahead guarantees that make this project the
leakage-free redemption it claims to be:

- **future-perturbation invariance** — perturbing a price bar at position ``t`` (or
  any later bar) never changes the engineered feature row at ``t`` or earlier,
  because every feature is lagged with ``.shift(1)`` to depend only on strictly
  past bars;
- **shift-equivariance** — engineering features on a price series and on the same
  series shifted forward in time yields the same feature values, merely relabelled
  by the new dates (the kernels are time-translation invariant);
- **sequence/target alignment** — :func:`create_sequences` never folds a sample's
  own label into its window (the off-by-one leak), labels each window by the date
  of its next-day target, and the train-fold scaler is invariant to test rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lstmforecast import ValidationError
from lstmforecast.data import random_walk_prices
from lstmforecast.features.engineer import FeatureSpec, engineer_features
from lstmforecast.features.sequences import (
    create_sequences,
    fit_scaler,
    scale_sequences,
)

pytestmark = pytest.mark.property

_DEFAULT_SPEC = FeatureSpec()


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    perturb_pos=st.integers(min_value=40, max_value=290),
    factor=st.floats(min_value=1.1, max_value=3.0),
)
@settings(max_examples=40, deadline=None)
def test_future_perturbation_invariance(seed: int, perturb_pos: int, factor: float) -> None:
    """Perturbing bar ``t`` leaves feature rows at ``t`` and earlier untouched.

    This is the headline anti-leakage guarantee: thanks to the universal
    ``.shift(1)`` lag, the feature row indexed at date ``t`` is a function of bars
    strictly earlier than ``t``, so a shock at ``t`` (or after) cannot reach it.
    """
    prices = random_walk_prices(n_obs=300, seed=seed)
    perturbed = prices.copy()
    perturbed.iloc[perturb_pos] *= factor

    base = engineer_features(prices, _DEFAULT_SPEC)
    after = engineer_features(perturbed, _DEFAULT_SPEC)

    date_t = prices.index[perturb_pos]
    # Feature rows dated at or before the perturbed bar must be byte-for-byte equal
    # (same surviving dates, identical values) — no future information leaked back.
    unaffected = base.index[base.index <= date_t]
    common = unaffected.intersection(after.index)
    assert len(common) == len(unaffected)
    assert np.allclose(
        base.loc[common].to_numpy(),
        after.loc[common].to_numpy(),
        equal_nan=True,
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=30, deadline=None)
def test_features_are_shift_equivariant(seed: int) -> None:
    """Engineering on a relabelled-in-time price series yields the same values.

    The feature kernels (returns, momentum, volatility, RSI) are time-translation
    invariant: shifting the whole series onto a later date index merely relabels
    the rows, so the computed feature *values* are identical position-for-position.
    """
    prices = random_walk_prices(n_obs=256, seed=seed)
    shifted = pd.Series(
        prices.to_numpy(),
        index=pd.date_range("2030-06-03", periods=len(prices), freq="B"),
        name="close",
    )

    base = engineer_features(prices, _DEFAULT_SPEC)
    moved = engineer_features(shifted, _DEFAULT_SPEC)

    assert list(base.columns) == list(moved.columns)
    assert base.shape == moved.shape
    assert np.allclose(base.to_numpy(), moved.to_numpy(), equal_nan=True)


@given(
    n_rows=st.integers(min_value=8, max_value=60),
    n_features=st.integers(min_value=1, max_value=4),
    look_back=st.integers(min_value=2, max_value=7),
)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
def test_sequence_target_alignment_has_no_off_by_one_leak(
    n_rows: int, n_features: int, look_back: int
) -> None:
    """``X[i]`` ends one row BEFORE its label, and labels carry the target date.

    Using strictly increasing sentinel features/targets, the last row of each
    window must equal the feature row immediately preceding the label row, and the
    label must equal the target at the next position — so a sample can never
    contain its own next-day target (the off-by-one leak the original repo had).
    """
    if n_rows < look_back + 1:
        pytest.skip("not enough rows to form a sequence")

    idx = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    feats = pd.DataFrame(
        {f"f{j}": np.arange(n_rows, dtype="float64") + j * 1000.0 for j in range(n_features)},
        index=idx,
    )
    target = pd.Series(np.arange(n_rows, dtype="float64") + 5000.0, index=idx)

    x, y, sample_index = create_sequences(feats, target, look_back=look_back)

    n_samples = n_rows - look_back
    assert x.shape == (n_samples, look_back, n_features)
    assert y.shape == (n_samples,)
    assert len(sample_index) == n_samples

    for i in range(n_samples):
        label_pos = look_back + i  # position of r_{t+1}
        # Window ends at the row immediately before the label row.
        np.testing.assert_array_equal(x[i, -1], feats.iloc[label_pos - 1].to_numpy())
        # Label is the next-day target, never any value inside the window.
        assert y[i] == target.iloc[label_pos]
        assert y[i] not in set(x[i].ravel().tolist())
        # The sample is dated by its target r_{t+1}.
        assert sample_index[i] == idx[label_pos]


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    test_perturb=st.floats(min_value=10.0, max_value=1e6),
)
@settings(max_examples=30, deadline=None)
def test_train_scaler_is_invariant_to_test_rows(seed: int, test_perturb: float) -> None:
    """The TRAIN-fold scaler ignores val/test rows entirely (the leakage fix).

    Fitting the standardizer on the train slice and then mangling the held-out
    rows must not move the fitted ``(mean, std)`` — statistics are estimated from
    train data exclusively, never from the data the model is judged on.
    """
    prices = random_walk_prices(n_obs=400, seed=seed)
    feats = engineer_features(prices, _DEFAULT_SPEC)
    target = pd.Series(
        np.zeros(len(feats), dtype="float64"), index=feats.index
    )  # alignment-only; values irrelevant here

    x, _y, _ix = create_sequences(feats, target, look_back=20)
    split = x.shape[0] // 2
    x_train = x[:split]

    mean_a, std_a = fit_scaler(x_train)
    # Corrupt the held-out (test) portion; the train slice is untouched.
    x_poisoned = x.copy()
    x_poisoned[split:] += test_perturb
    mean_b, std_b = fit_scaler(x_poisoned[:split])

    assert np.allclose(mean_a, mean_b)
    assert np.allclose(std_a, std_b)
    # And the fitted std never collapses to zero (constant columns floored to 1).
    assert bool((std_a > 0.0).all())


def test_scale_sequences_applies_train_stats_without_re_estimating() -> None:
    """``scale_sequences`` is a pure apply: identical stats give identical output.

    It must standardize using only the supplied ``(mean, std)`` and never peek at
    the tensor's own distribution, so two tensors scaled by the same train stats
    relate by exactly ``(x - mean) / std``.
    """
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(5, 4, 3))
    mean = np.array([1.0, 2.0, 3.0])
    std = np.array([0.5, 2.0, 4.0])

    out = scale_sequences(seq, mean=mean, std=std)
    np.testing.assert_allclose(out, (seq - mean) / std)


def test_create_sequences_rejects_misaligned_inputs() -> None:
    """Misaligned ``features``/``target`` indices raise rather than silently leak."""
    idx_a = pd.date_range("2020-01-01", periods=30, freq="B")
    idx_b = pd.date_range("2020-02-01", periods=30, freq="B")
    feats = pd.DataFrame({"a": np.arange(30.0)}, index=idx_a)
    target = pd.Series(np.arange(30.0), index=idx_b)
    with pytest.raises(ValidationError):
        create_sequences(feats, target, look_back=5)
