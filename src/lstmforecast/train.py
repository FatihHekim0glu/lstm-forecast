"""End-to-end training orchestration (the ``[train]`` path).

Wires the whole pipeline: load data (synthetic random walk by default, or a real
CSV) -> walk-forward with per-fold scaler + purge/embargo -> train a small LSTM
-> evaluate vs. persistence in return space -> export a tiny ONNX artifact and a
:class:`RunManifest`. TensorFlow/tf2onnx are imported LAZILY inside the model
layer, so importing this module never imports TF.

Importing this module has no side effects (no training, no TF import at load).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lstmforecast._manifest import RunManifest
from lstmforecast.evaluation.metrics import ForecastMetrics
from lstmforecast.evaluation.verdict import VerdictResult

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Immutable result of an end-to-end training run.

    Attributes
    ----------
    metrics:
        The return-space out-of-sample metric bundle.
    verdict:
        The derived ``beats_naive`` verdict (``False`` on random-walk data).
    artifact_path:
        Path to the exported ONNX artifact.
    manifest:
        The reproducibility manifest stamped on the run.
    n_effective_trials:
        The honest multiplicity count (size of the HPO grid) fed to the DSR.
    data_source:
        Where the prices came from (``"synthetic"`` or ``"csv"``).
    """

    metrics: ForecastMetrics
    verdict: VerdictResult
    artifact_path: str
    manifest: RunManifest
    n_effective_trials: int
    data_source: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the train result."""
        out: dict[str, Any] = {
            "metrics": self.metrics.to_dict(),
            "verdict": self.verdict.to_dict(),
            "artifact_path": str(self.artifact_path),
            "manifest": self.manifest.to_dict(),
            "n_effective_trials": int(self.n_effective_trials),
            "data_source": str(self.data_source),
            "meta": dict(self.meta),
        }
        return out


def train_pipeline(
    *,
    data_path: str | Path | None = None,
    n_obs: int = 2000,
    look_back: int = 60,
    seed: int = 7,
    artifact_path: str | Path | None = None,
    export: bool = True,
) -> TrainResult:
    """Run the full leakage-free training pipeline and (optionally) export ONNX.

    Default path (``data_path=None``) trains on a seeded synthetic random walk —
    the SHIPPED model — so the honest NULL holds by construction. Pass
    ``data_path`` to retrain on a real ``date,close`` CSV.

    Steps: build/load prices -> :func:`lstmforecast.walkforward.run_walk_forward`
    (per-fold scaler fit on TRAIN only, purge >= ``look_back``, embargo) -> stack
    OOS predictions -> :func:`lstmforecast.evaluation.forecast_metrics` ->
    :func:`lstmforecast.evaluation.derive_verdict` -> (if ``export``) fit a final
    small LSTM and export a <5MB ONNX artifact with a :class:`RunManifest`.

    LAZY IMPORT: TensorFlow/tf2onnx are imported only when ``export`` triggers the
    Keras train/export; the metrics/walk-forward path is TF-free.

    Parameters
    ----------
    data_path:
        Optional real ``date,close`` CSV; ``None`` => synthetic random walk.
    n_obs:
        Number of synthetic observations when ``data_path is None``.
    look_back:
        Sequence window length / minimum purge size.
    seed:
        Master RNG/TF seed.
    artifact_path:
        Destination for the exported ONNX; ``None`` => the shipped default path.
    export:
        Whether to fit + export the final ONNX artifact (skip for fast eval-only
        runs / tests).

    Returns
    -------
    TrainResult
        The metrics, verdict, artifact path, manifest, and honest trial count.
    """
    raise NotImplementedError
