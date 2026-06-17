"""Regression: the pure ``beats_naive`` verdict truth table (the honest-null guard).

``derive_verdict`` must read ``beats_naive=True`` ONLY when all three lines of
evidence agree: ``MASE < 1`` AND Diebold-Mariano significant AND directional
accuracy robustly above 0.5. Any one failing => ``False``. This locks the
headline honesty so it cannot regress into a narrated "the LSTM beats the market".
"""

from __future__ import annotations

import pytest

from lstmforecast import ValidationError, Verdict, derive_verdict

pytestmark = pytest.mark.regression


def test_all_three_pass_yields_beats_naive() -> None:
    result = derive_verdict(mase=0.9, dm_pvalue=0.01, directional_accuracy=0.6)
    assert result.beats_naive is True
    assert result.verdict is Verdict.LSTM_BEATS_NAIVE


@pytest.mark.parametrize(
    ("mase", "dm_pvalue", "directional_accuracy"),
    [
        (1.0, 0.01, 0.6),  # MASE not below 1 (no return-space improvement)
        (1.5, 0.01, 0.6),  # MASE well above 1
        (0.9, 0.05, 0.6),  # DM exactly at alpha -> insignificant
        (0.9, 0.20, 0.6),  # DM clearly insignificant
        (0.9, 0.01, 0.5),  # directional accuracy not strictly above 0.5
        (0.9, 0.01, 0.4),  # directional accuracy below chance
    ],
)
def test_any_failing_condition_yields_no_difference(
    mase: float, dm_pvalue: float, directional_accuracy: float
) -> None:
    result = derive_verdict(
        mase=mase, dm_pvalue=dm_pvalue, directional_accuracy=directional_accuracy
    )
    assert result.beats_naive is False
    assert result.verdict is Verdict.NO_SIGNIFICANT_DIFFERENCE
    assert result.rationale  # a non-empty reason is always produced


def test_random_walk_typical_outputs_are_null() -> None:
    # On random-walk data MASE ~ 1, DM insignificant, directional ~ 0.5.
    result = derive_verdict(mase=1.0, dm_pvalue=0.5, directional_accuracy=0.5)
    assert result.beats_naive is False


def test_verdict_to_dict_is_json_safe() -> None:
    d = derive_verdict(mase=1.0, dm_pvalue=0.5, directional_accuracy=0.5).to_dict()
    assert d["verdict"] == "no_significant_difference"
    assert d["beats_naive"] is False
    assert isinstance(d["rationale"], str)


def test_verdict_input_validation() -> None:
    with pytest.raises(ValidationError):
        derive_verdict(mase=float("nan"), dm_pvalue=0.1, directional_accuracy=0.5)
    with pytest.raises(ValidationError):
        derive_verdict(mase=0.9, dm_pvalue=1.5, directional_accuracy=0.5)
    with pytest.raises(ValidationError):
        derive_verdict(mase=0.9, dm_pvalue=0.1, directional_accuracy=1.5)
