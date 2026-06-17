"""Regression + property tests for the walk-forward engine (THE de-leak core).

These tests own the leakage-guard contract of
:mod:`lstmforecast.walkforward.engine`:

- ``make_folds`` no-lookahead golden test: a ``>= look_back`` purge sits at every
  train/val and val/test boundary, so no ``look_back``-length sequence window can
  straddle a split; OOS test slices never overlap and advance monotonically.
- ``run_walk_forward`` records ``n_trials`` = the number of DISTINCT HPO configs,
  fits the standardizer on the TRAIN slice ONLY (perturbing test-fold rows leaves
  the fitted scaler AND the train-fold predictions unchanged), and produces an
  identical OOS test index regardless of model.

The feature / sequence / log-return helpers are owned by other build groups and
are stubbed here behind deterministic fakes (monkeypatched at the names the engine
imports lazily). That keeps these tests independent of those groups while still
exercising the engine's real fold arithmetic, scaler-fit-on-train discipline, and
HPO bookkeeping.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lstmforecast import InsufficientDataError, WalkForwardConfig
from lstmforecast.walkforward import engine as wf

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------- #
# make_folds — the no-lookahead golden contract                               #
# --------------------------------------------------------------------------- #

_GOLDEN_CFG = WalkForwardConfig(
    look_back=60,
    train_size=200,
    val_size=50,
    test_size=50,
    step=50,
    purge=60,
    embargo=5,
    anchored=True,
)


def test_make_folds_golden_first_fold_boundaries() -> None:
    folds = wf.make_folds(1000, _GOLDEN_CFG)
    assert folds, "expected at least one fold for 1000 obs"
    first = folds[0]
    # train [0,200) -> +60 purge -> val [260,310) -> +60 purge -> test [370,420)
    assert first.train == (0, 200)
    assert first.val == (260, 310)
    assert first.test == (370, 420)


def test_make_folds_purge_gap_never_lets_a_window_straddle_a_boundary() -> None:
    folds = wf.make_folds(1500, _GOLDEN_CFG)
    purge = _GOLDEN_CFG.effective_purge
    assert purge >= _GOLDEN_CFG.look_back
    for f in folds:
        # A >= look_back gap between consecutive slices means a look_back-length
        # window ending at the first row of the later slice cannot reach the
        # earlier slice — no window straddles the boundary.
        assert f.val[0] - f.train[1] >= _GOLDEN_CFG.look_back
        assert f.test[0] - f.val[1] >= _GOLDEN_CFG.look_back


def test_make_folds_test_slices_are_ordered_and_non_overlapping() -> None:
    folds = wf.make_folds(1500, _GOLDEN_CFG)
    prev_stop = -1
    for f in folds:
        start, stop = f.test
        assert start < stop
        assert start >= prev_stop  # monotone, non-overlapping OOS coverage
        prev_stop = stop


def test_make_folds_embargo_gap_separates_adjacent_test_blocks() -> None:
    # The embargo forces a strictly positive gap between one fold's test block and
    # the next fold's region, so adjacent folds cannot overlap windows.
    folds = wf.make_folds(1500, _GOLDEN_CFG)
    assert len(folds) >= 2
    for prev, nxt in itertools.pairwise(folds):
        assert nxt.test[0] - prev.test[1] >= _GOLDEN_CFG.embargo


def test_make_folds_anchored_expands_rolling_slides() -> None:
    anchored = wf.make_folds(1200, _GOLDEN_CFG)
    assert all(f.train[0] == 0 for f in anchored)  # anchored: fixed start
    assert anchored[0].train[1] < anchored[1].train[1]  # expanding length

    rolling_cfg = WalkForwardConfig(
        look_back=60,
        train_size=200,
        val_size=50,
        test_size=50,
        step=50,
        purge=60,
        embargo=5,
        anchored=False,
    )
    rolling = wf.make_folds(1200, rolling_cfg)
    # rolling: fixed length, sliding start; the per-fold advance = step + embargo.
    advance = rolling_cfg.step + rolling_cfg.embargo
    assert rolling[0].train == (0, 200)
    assert rolling[1].train[0] == advance
    assert all(f.train[1] - f.train[0] == 200 for f in rolling)


def test_make_folds_exact_minimum_and_too_few() -> None:
    purge = _GOLDEN_CFG.effective_purge
    need = _GOLDEN_CFG.train_size + 2 * purge + _GOLDEN_CFG.val_size + _GOLDEN_CFG.test_size
    assert len(wf.make_folds(need, _GOLDEN_CFG)) == 1
    with pytest.raises(InsufficientDataError):
        wf.make_folds(need - 1, _GOLDEN_CFG)


def test_make_folds_rejects_overlapping_test_slices() -> None:
    """FOLD-OVERLAP GUARD: ``step + embargo < test_size`` must be rejected.

    If consecutive test slices advanced by fewer rows than ``test_size`` they would
    overlap and double-count OOS observations. ``make_folds`` asserts
    ``step + embargo >= test_size`` and raises before producing any folds.
    """
    from lstmforecast import ValidationError

    overlapping = WalkForwardConfig(
        look_back=60,
        train_size=200,
        val_size=50,
        test_size=50,
        step=40,  # step + embargo = 40 + 5 = 45 < test_size (50) => OVERLAP
        purge=60,
        embargo=5,
        anchored=True,
    )
    with pytest.raises(ValidationError, match="do not overlap"):
        wf.make_folds(1500, overlapping)

    # The exact boundary (step + embargo == test_size) is permitted (no overlap).
    boundary = WalkForwardConfig(
        look_back=60,
        train_size=200,
        val_size=50,
        test_size=50,
        step=45,  # 45 + 5 == 50 == test_size => abutting, not overlapping
        purge=60,
        embargo=5,
        anchored=True,
    )
    folds = wf.make_folds(1500, boundary)
    assert len(folds) >= 2
    # Abutting (gap exactly == embargo) but never overlapping.
    for prev, nxt in itertools.pairwise(folds):
        assert nxt.test[0] >= prev.test[1]


# --------------------------------------------------------------------------- #
# Deterministic fakes for the other groups' lazily-imported helpers           #
# --------------------------------------------------------------------------- #


def _fake_engineer_features(prices: pd.Series, spec: Any = None) -> pd.DataFrame:
    """Two strictly-past (lagged) features from a price sub-series."""
    r = np.log(prices).diff()
    feat = pd.DataFrame(
        {
            "lag_ret": r.shift(1),
            "lag_ret2": r.shift(2),
        },
        index=prices.index,
    )
    return feat.dropna()


def _fake_to_log_returns(prices: pd.Series) -> pd.Series:
    out: pd.Series = np.log(prices).diff().dropna()
    return out


def _fake_create_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    look_back: int = 60,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Trailing-window sequences; sample i ends at row i+look_back-1."""
    f = features.to_numpy(dtype="float64")
    y = target.to_numpy(dtype="float64")
    n = f.shape[0]
    if n < look_back:
        return (
            np.empty((0, look_back, f.shape[1]), dtype="float64"),
            np.empty((0,), dtype="float64"),
            pd.Index([]),
        )
    xs = np.stack([f[i : i + look_back] for i in range(n - look_back + 1)])
    ys = y[look_back - 1 :]
    idx = features.index[look_back - 1 :]
    return xs.astype("float64"), ys.astype("float64"), idx


def _fake_fit_scaler(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(train_features, dtype="float64").reshape(-1, train_features.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype("float64"), std.astype("float64")


def _fake_scale_sequences(
    sequences: np.ndarray, *, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    scaled = (np.asarray(sequences, dtype="float64") - mean) / std
    return np.asarray(scaled.astype("float64"))


@pytest.fixture
def patched_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the helpers the engine imports lazily inside ``run_walk_forward``."""
    monkeypatch.setattr("lstmforecast.features.engineer.engineer_features", _fake_engineer_features)
    monkeypatch.setattr("lstmforecast.data.to_log_returns", _fake_to_log_returns)
    monkeypatch.setattr("lstmforecast.features.sequences.create_sequences", _fake_create_sequences)
    monkeypatch.setattr("lstmforecast.features.sequences.fit_scaler", _fake_fit_scaler)
    monkeypatch.setattr("lstmforecast.features.sequences.scale_sequences", _fake_scale_sequences)


class _MeanShiftModel:
    """A toy model that learns the train-mean target and records the scaler.

    ``fit`` stores the per-feature mean of the (already-scaled) train inputs so a
    test can read back EXACTLY what statistics the engine fed the model. ``predict``
    returns the learned train-mean target for every test sample — a deterministic
    function of TRAIN data only, so its test predictions must be invariant to any
    change in the test rows.
    """

    seen_train_mean: np.ndarray | None = None

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self._yhat = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> _MeanShiftModel:
        arr = np.asarray(x, dtype="float64")
        if arr.size:
            type(self).seen_train_mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        self._yhat = float(np.mean(y)) if y.size else 0.0
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        n = int(np.asarray(x).shape[0])
        return np.full(n, self._yhat, dtype="float64")


def _prices(n: int = 1200, seed: int = 11) -> pd.Series:
    gen = np.random.default_rng(seed)
    shocks = gen.normal(0.0, 0.01, size=n)
    shocks[0] = 0.0
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(100.0 * np.exp(np.cumsum(shocks)), index=idx, name="close")


# --------------------------------------------------------------------------- #
# run_walk_forward — n_trials, identical OOS index, scaler-on-train-only       #
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("patched_engine")
def test_run_records_n_trials_as_distinct_config_count() -> None:
    prices = _prices()
    grid = [{"units": 8}, {"units": 16}, {"units": 32}]
    res = wf.run_walk_forward(prices, lambda p: _MeanShiftModel(p), _GOLDEN_CFG, hpo_grid=grid)
    assert res.n_trials == len(grid)  # #DISTINCT configs, not configs * folds
    assert res.n_folds >= 1
    # Each fold scores every grid config on val => total scored = folds * grid.
    assert res.meta["n_trials_scored"] == res.n_folds * len(grid)


@pytest.mark.usefixtures("patched_engine")
def test_run_default_grid_is_single_trial() -> None:
    prices = _prices()
    res = wf.run_walk_forward(prices, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)
    assert res.n_trials == 1


@pytest.mark.usefixtures("patched_engine")
def test_run_oos_index_is_identical_across_models() -> None:
    prices = _prices()
    naive = wf.run_walk_forward(prices, lambda p: wf_persistence(p), _GOLDEN_CFG)
    model = wf.run_walk_forward(prices, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)
    # Same folds => same purged/embargoed OOS dates regardless of the forecaster.
    assert list(naive.dates) == list(model.dates)
    assert np.array_equal(naive.y_true, model.y_true)
    # The baseline column is always all-zero persistence.
    assert np.allclose(model.y_pred_naive, 0.0)


def wf_persistence(params: dict[str, Any]) -> Any:
    from lstmforecast.models.baselines import PersistenceForecaster

    # PersistenceForecaster.fit/predict are stubbed in another group; wrap a
    # local zero-forecaster so this engine test is self-contained.
    class _Zero:
        def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(int(np.asarray(x).shape[0]), dtype="float64")

    _ = PersistenceForecaster  # imported only to assert availability
    return _Zero()


@pytest.mark.usefixtures("patched_engine")
def test_scaler_fit_on_train_only_future_perturbation_invariance() -> None:
    """Perturbing TEST-fold rows leaves the fitted scaler + train preds unchanged.

    This is THE de-leak guarantee: the standardizer's mean/std come from the train
    slice ONLY, so corrupting later (test) bars cannot change what the model was
    trained on, nor the train-mean it learned, nor — for a train-only forecaster —
    its test predictions.
    """
    prices = _prices()

    # Run once on the clean series, capturing the scaler stats fed to the model.
    _MeanShiftModel.seen_train_mean = None
    clean = wf.run_walk_forward(prices, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)
    clean_scaler_mean = _MeanShiftModel.seen_train_mean
    assert clean_scaler_mean is not None

    # Perturb ONLY rows in the final fold's test slice (the future), leaving every
    # train row untouched. Use a per-row random multiplicative shock so the
    # WITHIN-slice returns genuinely change (a constant scale would cancel in the
    # log-difference and be a no-op).
    folds = wf.make_folds(int(prices.shape[0]), _GOLDEN_CFG)
    t_start, t_stop = folds[-1].test
    gen = np.random.default_rng(999)
    perturbed = prices.copy()
    shock = gen.uniform(0.8, 1.2, size=t_stop - t_start)
    perturbed.iloc[t_start:t_stop] = perturbed.iloc[t_start:t_stop].to_numpy() * shock

    # Sanity: the perturbation actually moves the realized OOS test returns (so
    # this invariance test has teeth — a leak WOULD be observable).
    _MeanShiftModel.seen_train_mean = None
    dirty = wf.run_walk_forward(perturbed, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)
    dirty_scaler_mean = _MeanShiftModel.seen_train_mean
    assert dirty_scaler_mean is not None
    assert not np.allclose(clean.y_true, dirty.y_true)

    # The LAST fit the engine performed is the final fold's refit-on-train; its
    # train slice was untouched, so the scaler statistics are bit-identical.
    assert np.allclose(clean_scaler_mean, dirty_scaler_mean)

    # And the train-only model's TEST predictions are unchanged (they depend on
    # the train-mean target, never on the perturbed test inputs).
    assert np.allclose(clean.y_pred_model, dirty.y_pred_model)


@pytest.mark.usefixtures("patched_engine")
def test_run_rejects_non_positive_and_unsorted_prices() -> None:
    from lstmforecast import ValidationError

    bad = _prices(300)
    bad.iloc[5] = -1.0
    with pytest.raises(ValidationError):
        wf.run_walk_forward(bad, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)

    unsorted = _prices(600)
    unsorted = unsorted.iloc[::-1]  # descending time index
    with pytest.raises(ValidationError):
        wf.run_walk_forward(unsorted, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)


@pytest.mark.usefixtures("patched_engine")
def test_fold_sequences_empty_when_slice_shorter_than_look_back() -> None:
    """A slice that cannot host a single ``look_back`` window yields an empty fold.

    Documents the engine's requirement that ``test_size`` (and ``val_size``)
    exceed ``look_back``: with a too-short slice and no left context the per-fold
    sequence builder returns a well-formed empty ``(0, look_back, n_feat)`` tensor
    rather than raising, so an undersized fold degrades gracefully.
    """
    prices = _prices(400)
    look_back = 60
    # A 10-row slice at the very start (no left context available) -> no windows.
    x, y, idx = wf._fold_sequences(
        prices, (0, 10), look_back, _fake_engineer_features, _fake_create_sequences
    )
    assert x.shape == (0, look_back, max(x.shape[-1], 1))
    assert x.shape[0] == 0 and y.shape[0] == 0 and len(idx) == 0


@pytest.mark.usefixtures("patched_engine")
def test_run_result_to_dict_is_json_safe() -> None:
    prices = _prices()
    res = wf.run_walk_forward(prices, lambda p: _MeanShiftModel(p), _GOLDEN_CFG)
    d = res.to_dict()
    assert d["n_trials"] == 1
    assert isinstance(d["dates"], list)
    assert len(d["y_true"]) == len(d["y_pred_model"]) == len(d["y_pred_naive"])
