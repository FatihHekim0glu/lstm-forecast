"""Property-based invariants for the implemented pure functions.

These Hypothesis tests pin invariants that the stubbed compute layer will have to
preserve once filled in. They currently exercise the implemented pieces: the
seeded random-walk generator, the DSR monotonicity, and the verdict's honest
gate.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lstmforecast import (
    deflated_sharpe_ratio,
    derive_verdict,
    random_walk_prices,
)

pytestmark = pytest.mark.property


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_random_walk_is_strictly_positive_and_reproducible(seed: int) -> None:
    a = random_walk_prices(n_obs=128, seed=seed)
    b = random_walk_prices(n_obs=128, seed=seed)
    assert (a > 0).all()
    assert np.array_equal(a.to_numpy(), b.to_numpy())


@given(
    sharpe=st.floats(min_value=0.0, max_value=0.5),
    n1=st.integers(min_value=1, max_value=50),
    extra=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=40, deadline=None)
def test_dsr_non_increasing_in_n_trials(sharpe: float, n1: int, extra: int) -> None:
    kwargs = {"n_obs": 400, "variance_of_trial_sharpes": 0.04}
    dsr_few = deflated_sharpe_ratio(sharpe, n_trials=n1, **kwargs)
    dsr_more = deflated_sharpe_ratio(sharpe, n_trials=n1 + extra, **kwargs)
    assert dsr_more <= dsr_few + 1e-12


@given(
    mase=st.floats(min_value=1.0, max_value=3.0),
    dm_pvalue=st.floats(min_value=0.0, max_value=1.0),
    da=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=40, deadline=None)
def test_mase_ge_one_never_beats_naive(mase: float, dm_pvalue: float, da: float) -> None:
    # No matter how favourable DM / directional accuracy look, MASE >= 1 forbids
    # a positive verdict — the honest-null gate.
    assert derive_verdict(mase=mase, dm_pvalue=dm_pvalue, directional_accuracy=da).beats_naive is (
        False
    )
