"""Plotly figure builders (LAZY plotly; the ``viz`` extra).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so no Plotly object leaks across the API
boundary. Plotly is OPTIONAL (the ``viz`` extra) and imported LAZILY inside each
builder; importing this module has no side effects and does not require Plotly.

The two figures back the honest story: predicted-vs-actual next-day RETURNS (not
price levels) and the model-vs-persistence error bar chart.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lstmforecast._typing import FloatArray

# quantcore-candidate: mirrors hrp / pairs-trading plots.py ({data, layout} shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp | pd.Period):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def _as_plain_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Plotly graph-object (trace/layout) to a plain, JSON-safe ``dict``."""
    raw = obj.to_plotly_json() if hasattr(obj, "to_plotly_json") else dict(obj)
    jsonified: dict[str, Any] = _jsonify(raw)
    return jsonified


def forecast_vs_actual_figure(
    dates: pd.Index,
    y_true: FloatArray,
    y_pred_model: FloatArray,
) -> FigureDict:
    """Build the predicted-vs-actual next-day RETURN series figure.

    Two line traces over ``dates``: realized next-day returns and the model's
    forecast returns. NEVER plots price levels (the whole point is return-space
    honesty). Plotly is imported lazily inside this function.

    Parameters
    ----------
    dates:
        The out-of-sample target dates.
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecast returns (same length).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping rendering the two return series.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError


def error_vs_baseline_figure(
    y_true: FloatArray,
    y_pred_model: FloatArray,
    y_pred_naive: FloatArray | None = None,
) -> FigureDict:
    """Build the model-vs-persistence out-of-sample error bar chart.

    A grouped/!two-bar figure comparing the model's return-space error (RMSE/MAE)
    to the persistence baseline's — the visual of the honest NULL (the bars are
    essentially equal on random-walk data). Plotly is imported lazily.

    Parameters
    ----------
    y_true:
        Realized next-day returns.
    y_pred_model:
        The model's forecasts.
    y_pred_naive:
        The persistence forecasts; defaults to all zeros.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` bar-chart mapping.

    Raises
    ------
    ValidationError
        If the inputs are empty or length-mismatched.
    """
    raise NotImplementedError
