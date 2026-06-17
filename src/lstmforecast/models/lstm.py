"""Keras LSTM (and LSTM+Attention) builders — the TRAIN-ONLY path.

TensorFlow / Keras is the heaviest dependency in the project and is NEVER on the
import path of ``lstmforecast``: it is imported LAZILY inside the functions here,
which run only during offline training (the ``[train]`` extra). The container and
the FastAPI router never import this module — they serve via
:mod:`lstmforecast.models.onnx_runtime`.

Models are intentionally small (few units, few epochs): the deliverable is the
leakage-free methodology and the honest NULL, not a high-capacity forecaster.

Importing this module has no side effects (no TensorFlow import at load time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from lstmforecast._typing import FloatArray, SequenceTensor

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from tensorflow import keras

#: Which architecture to build.
Architecture = Literal["vanilla", "attention"]


@dataclass(frozen=True, slots=True)
class LstmConfig:
    """Immutable hyperparameters for a small LSTM forecaster.

    Attributes
    ----------
    architecture:
        ``"vanilla"`` (a single LSTM layer) or ``"attention"`` (LSTM + a light
        additive self-attention head).
    units:
        Number of LSTM hidden units (kept small on purpose).
    look_back:
        Sequence window length (must match the tensor's time dimension).
    n_features:
        Number of input features per timestep.
    dropout:
        Dropout rate applied after the recurrent layer.
    learning_rate:
        Adam learning rate.
    epochs:
        Number of training epochs (kept small on purpose).
    batch_size:
        Mini-batch size.
    seed:
        Master seed for TF/Keras determinism.
    """

    architecture: Architecture = "vanilla"
    units: int = 16
    look_back: int = 60
    n_features: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 5
    batch_size: int = 32
    seed: int = 7

    def __post_init__(self) -> None:
        """Validate that sizes/rates are in range.

        Raises
        ------
        ValidationError
            If any of ``units``, ``look_back``, ``n_features``, ``epochs``,
            ``batch_size`` is ``< 1``, or ``dropout`` is outside ``[0, 1)``, or
            ``learning_rate <= 0``.
        """
        from lstmforecast._exceptions import ValidationError

        if min(self.units, self.look_back, self.n_features, self.epochs, self.batch_size) < 1:
            raise ValidationError(f"LstmConfig: sizes must be >= 1, got {self!r}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(f"LstmConfig: dropout must be in [0, 1), got {self.dropout}.")
        if self.learning_rate <= 0.0:
            raise ValidationError(
                f"LstmConfig: learning_rate must be > 0, got {self.learning_rate}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_model(config: LstmConfig) -> keras.Model:
    """Construct (but do not train) a small Keras LSTM forecaster.

    LAZY IMPORT: TensorFlow/Keras is imported INSIDE this function. The graph maps
    a ``(look_back, n_features)`` input to a single linear output (the next-day
    return), with a ``"vanilla"`` single-LSTM or ``"attention"`` LSTM+attention
    body per ``config.architecture``. Seeds are set for reproducibility.

    Parameters
    ----------
    config:
        The validated hyperparameter bundle.

    Returns
    -------
    keras.Model
        A compiled (Adam + MSE) but untrained Keras model.
    """
    raise NotImplementedError


def train_model(
    config: LstmConfig,
    x_train: SequenceTensor,
    y_train: FloatArray,
    *,
    x_val: SequenceTensor | None = None,
    y_val: FloatArray | None = None,
) -> keras.Model:
    """Build and fit a small LSTM on PRE-SCALED train sequences.

    LAZY IMPORT: TensorFlow/Keras is imported inside this function. The inputs
    MUST already be standardized with a TRAIN-fold-fitted scaler (see
    :mod:`lstmforecast.features.sequences`); this function never fits a scaler, so
    it cannot reintroduce scaler leakage.

    Parameters
    ----------
    config:
        The validated hyperparameter bundle.
    x_train, y_train:
        Pre-scaled train sequences and next-day return targets.
    x_val, y_val:
        Optional pre-scaled validation sequences/targets for early-stopping /
        HPO scoring (never used to fit the scaler).

    Returns
    -------
    keras.Model
        The trained Keras model.
    """
    raise NotImplementedError


def export_onnx(model: keras.Model, path: str, *, config: LstmConfig) -> str:
    """Export a trained Keras model to a tiny (<5MB) ONNX artifact via tf2onnx.

    LAZY IMPORT: tensorflow and tf2onnx are imported inside this function. The
    exported graph is the artifact committed under
    ``src/lstmforecast/artifacts/`` and served by onnxruntime — the container
    never imports TF. The ONNX forward pass must match the Keras output to 1e-5
    (the parity test).

    Parameters
    ----------
    model:
        The trained Keras model to export.
    path:
        Destination ``.onnx`` filepath.
    config:
        The config the model was built with (its ``look_back``/``n_features``
        fix the exported input signature).

    Returns
    -------
    str
        The path the ONNX artifact was written to.
    """
    raise NotImplementedError
