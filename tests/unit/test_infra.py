"""Unit tests for the reused infrastructure (rng, constants, manifest, validation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lstmforecast import (
    EPS,
    PERIODS_PER_YEAR,
    TRADING_DAYS,
    InsufficientDataError,
    LstmForecastError,
    RunManifest,
    ValidationError,
    config_hash,
    ensure_dataframe,
    ensure_monotonic_index,
    ensure_series,
    make_rng,
    spawn_substreams,
    validate_min_obs,
)
from lstmforecast._exceptions import ArtifactError

pytestmark = pytest.mark.unit


def test_constants_are_sane() -> None:
    assert PERIODS_PER_YEAR == 252
    assert TRADING_DAYS == PERIODS_PER_YEAR
    assert 0 < EPS < 1e-6


def test_make_rng_is_reproducible() -> None:
    a = make_rng(7).standard_normal(16)
    b = make_rng(7).standard_normal(16)
    assert np.array_equal(a, b)


def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        make_rng(-1)


def test_spawn_substreams_independent_and_reproducible() -> None:
    s1 = spawn_substreams(7, 3)
    s2 = spawn_substreams(7, 3)
    assert len(s1) == 3
    draws1 = [g.standard_normal(4) for g in s1]
    draws2 = [g.standard_normal(4) for g in s2]
    for d1, d2 in zip(draws1, draws2, strict=True):
        assert np.array_equal(d1, d2)
    # Distinct substreams differ.
    assert not np.array_equal(draws1[0], draws1[1])


def test_spawn_substreams_validates() -> None:
    with pytest.raises(ValueError):
        spawn_substreams(-1, 2)
    with pytest.raises(ValueError):
        spawn_substreams(7, -1)


def test_config_hash_is_order_invariant() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_manifest_capture_and_to_dict() -> None:
    manifest = RunManifest.capture({"k": "v"}, seed=7)
    d = manifest.to_dict()
    assert d["seed"] == 7
    assert isinstance(d["config_hash"], str) and len(d["config_hash"]) == 32
    assert isinstance(d["dirty"], bool)


def test_ensure_series_coerces_and_rejects_nan() -> None:
    s = ensure_series([1.0, 2.0, 3.0], name="x")
    assert isinstance(s, pd.Series)
    assert s.dtype == "float64"
    with pytest.raises(ValidationError):
        ensure_series([1.0, np.nan], name="x")


def test_ensure_series_accepts_series_and_ndarray() -> None:
    src = pd.Series([1.0, 2.0])
    out = ensure_series(src, name="x")
    assert out is not src  # a defensive copy is returned
    assert ensure_series(np.array([1.0, 2.0]), name="x").tolist() == [1.0, 2.0]
    # allow_nan path
    assert ensure_series([1.0, np.nan], name="x", allow_nan=True).isna().any()


def test_ensure_series_rejects_bad_shapes() -> None:
    with pytest.raises(ValidationError):
        ensure_series(np.ones((2, 2)), name="x")  # 2-D ndarray
    with pytest.raises(ValidationError):
        ensure_series([], name="x")  # empty


def test_ensure_dataframe_coerces_and_rejects_nan() -> None:
    df = ensure_dataframe(np.ones((4, 2)), name="m")
    assert df.shape == (4, 2)
    with pytest.raises(ValidationError):
        ensure_dataframe(np.array([[1.0, np.nan]]), name="m")


def test_ensure_dataframe_shapes_and_columns() -> None:
    df = ensure_dataframe(np.ones((3, 2)), name="m", columns=["a", "b"])
    assert list(df.columns) == ["a", "b"]
    with pytest.raises(ValidationError):
        ensure_dataframe(np.ones((2, 2, 2)), name="m")  # 3-D ndarray
    with pytest.raises(ValidationError):
        ensure_dataframe(pd.DataFrame(), name="m")  # zero rows/cols
    # dict-coercible + allow_nan path
    out = ensure_dataframe({"a": [1.0, np.nan]}, name="m", allow_nan=True)
    assert out.shape == (2, 1)


def test_ensure_monotonic_index() -> None:
    good = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert ensure_monotonic_index(good, name="p") is good
    bad = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-02", "2020-01-01"]))
    with pytest.raises(ValidationError):
        ensure_monotonic_index(bad, name="p")


def test_validate_min_obs() -> None:
    df = pd.DataFrame(np.ones((3, 1)))
    validate_min_obs(df, 3)
    with pytest.raises(InsufficientDataError):
        validate_min_obs(df, 4)


def test_exception_hierarchy() -> None:
    assert issubclass(ValidationError, LstmForecastError)
    assert issubclass(InsufficientDataError, ValidationError)
    assert issubclass(ArtifactError, LstmForecastError)
