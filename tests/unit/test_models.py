"""Unit tests for the models layer: persistence, the Keras LSTM, and ONNX serve.

The persistence baseline is pure numpy and always runs. The Keras LSTM build /
forward-pass tests require the ``[train]`` extra (TensorFlow) and are marked
``slow`` + skipped when TF is unavailable. The ONNX serve test builds a tiny
fixture graph and runs it through :class:`OnnxForecaster`; it is skipped when
onnxruntime (and the ``onnx`` builder) are unavailable. The whole module must
import without pulling in TensorFlow or onnxruntime (import purity).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lstmforecast._exceptions import ArtifactError, ValidationError
from lstmforecast.models.baselines import PersistenceForecaster
from lstmforecast.models.lstm import LstmConfig, build_model, export_onnx, train_model
from lstmforecast.models.onnx_runtime import (
    DEFAULT_ARTIFACT_NAME,
    OnnxForecaster,
    default_artifact_path,
)

pytestmark = pytest.mark.unit

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_TF2ONNX = importlib.util.find_spec("tf2onnx") is not None


# --------------------------------------------------------------------------- #
# Persistence baseline (always runs - pure numpy)                             #
# --------------------------------------------------------------------------- #
def test_persistence_fit_returns_self() -> None:
    model = PersistenceForecaster()
    x = np.zeros((4, 60, 1))
    y = np.zeros(4)
    assert model.fit(x, y) is model


def test_persistence_predict_is_zeros_one_per_sequence() -> None:
    model = PersistenceForecaster()
    x = np.random.default_rng(0).standard_normal((7, 60, 3))
    out = model.predict(x)
    assert out.shape == (7,)
    assert out.dtype == np.float64
    assert np.array_equal(out, np.zeros(7))


def test_persistence_predict_ignores_feature_content() -> None:
    # The forecast is r_hat = 0 regardless of what the windows contain.
    model = PersistenceForecaster()
    a = model.predict(np.ones((5, 60, 2)))
    b = model.predict(np.full((5, 60, 2), -123.4))
    assert np.array_equal(a, b)


def test_persistence_predict_requires_3d() -> None:
    model = PersistenceForecaster()
    with pytest.raises(ValidationError):
        model.predict(np.zeros((10, 60)))
    with pytest.raises(ValidationError):
        model.predict(np.zeros(10))


# --------------------------------------------------------------------------- #
# LstmConfig validation (no TF needed - __post_init__ is pure)                #
# --------------------------------------------------------------------------- #
def test_lstm_config_defaults_and_to_dict() -> None:
    cfg = LstmConfig()
    assert cfg.architecture == "vanilla"
    assert cfg.look_back == 60
    d = cfg.to_dict()
    assert d["architecture"] == "vanilla"
    assert d["units"] == cfg.units


@pytest.mark.parametrize(
    "kwargs",
    [
        {"units": 0},
        {"look_back": 0},
        {"n_features": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"learning_rate": 0.0},
    ],
)
def test_lstm_config_rejects_out_of_range(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LstmConfig(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Keras LSTM build + forward pass (slow; requires the [train] extra)          #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not _HAS_TF, reason="LSTM build requires the [train] extra (tensorflow).")
@pytest.mark.parametrize("architecture", ["vanilla", "attention"])
def test_build_model_forward_shape(architecture: str) -> None:
    cfg = LstmConfig(
        architecture=architecture,  # type: ignore[arg-type]
        units=4,
        look_back=8,
        n_features=2,
        epochs=1,
    )
    model = build_model(cfg)
    x = (
        np.random.default_rng(1)
        .standard_normal((5, cfg.look_back, cfg.n_features))
        .astype("float32")
    )
    out = np.asarray(model.predict(x, verbose=0))
    # One scalar next-day return per input window.
    assert out.shape == (5, 1)
    assert np.isfinite(out).all()


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_TF, reason="LSTM training requires the [train] extra (tensorflow).")
def test_train_model_runs_and_predicts() -> None:
    cfg = LstmConfig(units=4, look_back=8, n_features=1, epochs=1, batch_size=4)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, cfg.look_back, cfg.n_features)).astype("float32")
    y = rng.standard_normal(20).astype("float32")
    model = train_model(cfg, x, y, x_val=x[:5], y_val=y[:5])
    out = np.asarray(model.predict(x, verbose=0))
    assert out.shape == (20, 1)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_TF, reason="train_model validation path needs a Keras model.")
def test_train_model_rejects_mismatched_shapes() -> None:
    cfg = LstmConfig(units=4, look_back=8, n_features=1, epochs=1)
    x = np.zeros((10, 8, 1), dtype="float32")
    with pytest.raises(ValidationError):
        train_model(cfg, x, np.zeros(9, dtype="float32"))
    with pytest.raises(ValidationError):
        train_model(cfg, np.zeros((10, 8), dtype="float32"), np.zeros(10, dtype="float32"))
    with pytest.raises(ValidationError):
        train_model(cfg, np.zeros((10, 7, 1), dtype="float32"), np.zeros(10, dtype="float32"))


# --------------------------------------------------------------------------- #
# ONNX serve path                                                             #
# --------------------------------------------------------------------------- #
def test_default_artifact_path_points_into_package() -> None:
    p = default_artifact_path()
    assert p.name == DEFAULT_ARTIFACT_NAME
    assert p.parent.name == "artifacts"


def test_onnx_forecaster_records_custom_path() -> None:
    f = OnnxForecaster("/tmp/does_not_exist_xyz.onnx")
    assert f.artifact_path.name == "does_not_exist_xyz.onnx"
    assert f._session is None  # constructing must not create a session


def test_onnx_load_missing_artifact_raises_artifact_error() -> None:
    f = OnnxForecaster("/tmp/definitely_missing_lstm_artifact.onnx")
    with pytest.raises(ArtifactError):
        f.load()


def _make_tiny_identity_onnx(path: str, look_back: int, n_features: int) -> None:
    """Build a minimal ONNX graph: (N, look_back, n_features) -> (N, 1) reduce-mean.

    Uses the ``onnx`` builder API only (no TensorFlow) so the serve path can be
    exercised without the heavy [train] extra.
    """
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info(
        "sequence", TensorProto.FLOAT, [None, look_back, n_features]
    )
    out = helper.make_tensor_value_info("return_hat", TensorProto.FLOAT, [None, 1])
    # Mean over time and feature axes -> a single scalar per window.
    reduce_node = helper.make_node(
        "ReduceMean",
        inputs=["sequence"],
        outputs=["pooled"],
        axes=[1, 2],
        keepdims=1,
    )
    reshape_shape = helper.make_tensor("shape", TensorProto.INT64, dims=[2], vals=[-1, 1])
    reshape_node = helper.make_node("Reshape", inputs=["pooled", "shape"], outputs=["return_hat"])
    graph = helper.make_graph(
        [reduce_node, reshape_node],
        "tiny_lstm_fixture",
        [inp],
        [out],
        initializer=[reshape_shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


@pytest.mark.skipif(
    not (_HAS_ORT and _HAS_ONNX),
    reason="ONNX serve test requires onnxruntime ([serve]) and the onnx builder.",
)
def test_onnx_forecaster_loads_and_runs_tiny_fixture(tmp_path: object) -> None:
    look_back, n_features = 6, 2
    artifact = tmp_path / "tiny.onnx"  # type: ignore[operator]
    _make_tiny_identity_onnx(str(artifact), look_back, n_features)

    f = OnnxForecaster(str(artifact))
    x = np.random.default_rng(3).standard_normal((4, look_back, n_features))
    out = f.predict(x)

    assert out.shape == (4,)
    assert out.dtype == np.float64
    # Idempotent load: a second predict reuses the session and matches.
    out2 = f.predict(x)
    assert np.allclose(out, out2)


@pytest.mark.skipif(
    not (_HAS_ORT and _HAS_ONNX),
    reason="ONNX serve test requires onnxruntime ([serve]) and the onnx builder.",
)
def test_onnx_forecaster_rejects_non_3d_input(tmp_path: object) -> None:
    artifact = tmp_path / "tiny.onnx"  # type: ignore[operator]
    _make_tiny_identity_onnx(str(artifact), 6, 2)
    f = OnnxForecaster(str(artifact))
    with pytest.raises(ArtifactError):
        f.predict(np.zeros((4, 6)))


# --------------------------------------------------------------------------- #
# ONNX-vs-Keras parity (slow; needs both [train] and onnxruntime)             #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(
    not (_HAS_TF and _HAS_TF2ONNX and _HAS_ORT),
    reason="parity needs [train] (tensorflow + tf2onnx) and onnxruntime.",
)
def test_export_onnx_matches_keras_within_tolerance(tmp_path: object) -> None:
    cfg = LstmConfig(units=4, look_back=8, n_features=2, epochs=1)
    model = build_model(cfg)
    x = (
        np.random.default_rng(4)
        .standard_normal((6, cfg.look_back, cfg.n_features))
        .astype("float32")
    )
    keras_out = np.asarray(model.predict(x, verbose=0)).reshape(-1)

    artifact = tmp_path / "parity.onnx"  # type: ignore[operator]
    written = export_onnx(model, str(artifact), config=cfg)
    assert written == str(artifact)

    onnx_out = OnnxForecaster(str(artifact)).predict(x.astype("float64"))
    np.testing.assert_allclose(onnx_out, keras_out, atol=1e-5)
