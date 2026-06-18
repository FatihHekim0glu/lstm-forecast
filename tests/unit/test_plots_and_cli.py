"""Unit tests for the Plotly figure builders and the Typer CLI.

Covers:

- ``lstmforecast.plots`` — the two figure builders that back the honest story:
  ``forecast_vs_actual_figure`` (predicted-vs-actual next-day RETURNS — never
  price levels) and ``error_vs_baseline_figure`` (model-vs-persistence error
  bars, equal on a random walk). Every builder must return a plain
  ``{"data", "layout"}`` mapping whose contents are JSON-serializable (no
  numpy/pandas/Plotly object leaks across the API boundary), and we assert real
  numerical structure (trace types, ISO dates, equal-bar NULL) rather than merely
  "it runs". Malformed inputs raise :class:`ValidationError` (mapped to 422).
- ``lstmforecast.cli`` — ``--help`` lists the three commands, and a tiny synthetic
  ``forecast`` / ``evaluate`` run executes offline WITHOUT TensorFlow (the
  persistence/ONNX serve path), exiting ``0`` and reporting the honest NULL
  (``beats_naive = False``).

All inputs are synthetic/seeded; nothing touches the network or imports
TensorFlow.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from lstmforecast import plots
from lstmforecast._exceptions import ValidationError

pytestmark = pytest.mark.unit


def _assert_figure_dict(fig: object) -> dict:
    """Assert ``fig`` is a ``{"data", "layout"}`` mapping with JSON-safe contents.

    Returns the figure so callers can make further structural assertions. The
    ``json.dumps`` round-trip is the load-bearing check: it fails loudly if any
    numpy scalar/array, pandas object, or Plotly graph-object leaked through.
    """
    assert isinstance(fig, dict)
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    encoded = json.dumps(fig)
    assert json.loads(encoded) == fig
    return fig


def _assert_all_finite(values: object) -> None:
    """Assert every numeric leaf in a trace's ``y`` list is finite."""
    arr = np.asarray(values, dtype="float64")
    assert np.all(np.isfinite(arr))


# --------------------------------------------------------------------------- #
# forecast_vs_actual_figure                                                   #
# --------------------------------------------------------------------------- #
def test_forecast_vs_actual_two_return_traces_with_iso_dates() -> None:
    """Two line traces over ISO-formatted dates: realized + model forecast returns."""
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    y_true = np.array([0.01, -0.02, 0.0, 0.005, -0.001])
    y_pred = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # persistence

    fig = _assert_figure_dict(plots.forecast_vs_actual_figure(dates, y_true, y_pred))

    assert len(fig["data"]) == 2
    assert {t["name"] for t in fig["data"]} == {"actual", "model forecast"}
    for trace in fig["data"]:
        assert trace["type"] == "scatter"
        assert trace["mode"] == "lines"
        _assert_all_finite(trace["y"])

    actual_trace = next(t for t in fig["data"] if t["name"] == "actual")
    np.testing.assert_allclose(np.asarray(actual_trace["y"]), y_true)
    # The shared x-axis carries ISO date strings (no Timestamp leaked through).
    assert actual_trace["x"][0] == dates[0].isoformat()
    assert all(isinstance(v, str) for v in actual_trace["x"])
    # The honest title and return-space y-axis (NOT price).
    assert "return" in fig["layout"]["title"]["text"].lower()
    assert fig["layout"]["yaxis"]["title"]["text"] == "log-return"


def test_forecast_vs_actual_accepts_period_index() -> None:
    """A PeriodIndex is serialized to ISO strings (no pandas object leaks)."""
    periods = pd.period_range("2021-01", periods=3, freq="M")
    y_true = np.array([0.01, -0.01, 0.02])
    y_pred = np.zeros(3)
    fig = _assert_figure_dict(plots.forecast_vs_actual_figure(periods, y_true, y_pred))
    xs = fig["data"][0]["x"]
    assert all(isinstance(v, str) for v in xs)


def test_forecast_vs_actual_series_length_mismatch_raises() -> None:
    """A true/pred series length mismatch raises before the date check."""
    dates = pd.date_range("2021-01-01", periods=3, freq="B")
    with pytest.raises(ValidationError):
        # y_true has 3 elements, y_pred only 2 -> the size guard fires.
        plots.forecast_vs_actual_figure(dates, np.array([0.01, 0.02, 0.03]), np.zeros(2))


def test_jsonify_handles_nested_and_scalar_types() -> None:
    """``_jsonify`` recurses through dicts and unwraps numpy scalars/Timestamps."""
    value = {
        "a": np.int64(3),
        "b": [np.float64(1.5), pd.Timestamp("2021-01-01")],
        "c": np.array([1.0, 2.0]),
    }
    out = plots._jsonify(value)
    # Round-trips to JSON: every leaf is a native Python type.
    json.dumps(out)
    assert out["a"] == 3 and isinstance(out["a"], int)
    assert out["b"][0] == 1.5
    assert out["b"][1] == "2021-01-01T00:00:00"
    assert out["c"] == [1.0, 2.0]


def test_forecast_vs_actual_date_misalignment_raises() -> None:
    """A date index that does not align with the series length raises."""
    dates = pd.date_range("2021-01-01", periods=2, freq="B")
    with pytest.raises(ValidationError):
        plots.forecast_vs_actual_figure(dates, np.array([0.01, 0.02, 0.03]), np.zeros(3))


def test_forecast_vs_actual_empty_raises() -> None:
    """Empty inputs raise ValidationError rather than producing an empty figure."""
    with pytest.raises(ValidationError):
        plots.forecast_vs_actual_figure(pd.Index([]), np.array([]), np.array([]))


def test_forecast_vs_actual_non_finite_raises() -> None:
    """A NaN/inf in the series is rejected (finite-only figure inputs)."""
    dates = pd.date_range("2021-01-01", periods=3, freq="B")
    with pytest.raises(ValidationError):
        plots.forecast_vs_actual_figure(dates, np.array([0.01, np.nan, 0.02]), np.zeros(3))


# --------------------------------------------------------------------------- #
# error_vs_baseline_figure                                                    #
# --------------------------------------------------------------------------- #
def test_error_vs_baseline_grouped_rmse_mae_bars() -> None:
    """A grouped two-series bar chart over RMSE and MAE for model vs. persistence."""
    y_true = np.array([0.01, -0.02, 0.0, 0.005])
    y_model = np.array([0.002, -0.001, 0.0, 0.004])

    fig = _assert_figure_dict(plots.error_vs_baseline_figure(y_true, y_model))

    assert fig["layout"]["barmode"] == "group"
    assert {t["name"] for t in fig["data"]} == {"LSTM model", "persistence (naive)"}
    for trace in fig["data"]:
        assert trace["type"] == "bar"
        assert trace["x"] == ["RMSE", "MAE"]
        _assert_all_finite(trace["y"])
        assert len(trace["y"]) == 2

    # Cross-check the model RMSE against an independent computation.
    model_trace = next(t for t in fig["data"] if t["name"] == "LSTM model")
    expected_rmse = float(np.sqrt(np.mean((y_true - y_model) ** 2)))
    assert model_trace["y"][0] == pytest.approx(expected_rmse)


def test_error_vs_baseline_equal_bars_on_persistence_model() -> None:
    """When the model IS persistence, the model and naive bars are identical.

    This is the visual of the honest NULL: a forecaster that matches the
    random-walk baseline has exactly the baseline's error.
    """
    y_true = np.array([0.01, -0.02, 0.03, -0.004, 0.0])
    y_model = np.zeros_like(y_true)  # r_hat = 0 == persistence

    fig = _assert_figure_dict(plots.error_vs_baseline_figure(y_true, y_model))
    model_trace = next(t for t in fig["data"] if t["name"] == "LSTM model")
    naive_trace = next(t for t in fig["data"] if t["name"] == "persistence (naive)")
    np.testing.assert_allclose(np.asarray(model_trace["y"]), np.asarray(naive_trace["y"]))


def test_error_vs_baseline_explicit_naive_series() -> None:
    """An explicit naive forecast series is honored over the zeros default."""
    y_true = np.array([0.01, -0.02, 0.03])
    y_model = np.array([0.005, -0.01, 0.02])
    y_naive = np.array([0.0, 0.0, 0.0])

    fig = _assert_figure_dict(plots.error_vs_baseline_figure(y_true, y_model, y_naive))
    naive_trace = next(t for t in fig["data"] if t["name"] == "persistence (naive)")
    expected_naive_mae = float(np.mean(np.abs(y_true - y_naive)))
    assert naive_trace["y"][1] == pytest.approx(expected_naive_mae)


def test_error_vs_baseline_length_mismatch_raises() -> None:
    """Mismatched model / naive lengths raise ValidationError."""
    y_true = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValidationError):
        plots.error_vs_baseline_figure(y_true, np.array([0.01, 0.02]))
    with pytest.raises(ValidationError):
        plots.error_vs_baseline_figure(y_true, y_true, np.array([0.0, 0.0]))


def test_error_vs_baseline_empty_raises() -> None:
    """Empty inputs raise ValidationError."""
    with pytest.raises(ValidationError):
        plots.error_vs_baseline_figure(np.array([]), np.array([]))


def test_plots_import_does_not_require_plotly() -> None:
    """Importing the plots module imports neither Plotly nor TensorFlow (lazy viz)."""
    code = (
        "import sys; import lstmforecast.plots; "
        "assert 'plotly' not in sys.modules, 'Plotly imported at module load'; "
        "assert 'tensorflow' not in sys.modules, 'TensorFlow imported at module load'; "
        "print('PLOTS_IMPORT_PURE_OK')"
    )
    import subprocess

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "PLOTS_IMPORT_PURE_OK" in proc.stdout


# --------------------------------------------------------------------------- #
# CLI: --help and the synthetic forecast / evaluate smoke runs                #
# --------------------------------------------------------------------------- #
def test_cli_help_lists_the_three_commands() -> None:
    """``--help`` builds the app lazily and lists train / forecast / evaluate."""
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("train", "forecast", "evaluate"):
        assert command in result.output


def test_cli_no_args_shows_help() -> None:
    """Invoking with no arguments prints help (no_args_is_help) and exits 2."""
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), [])
    assert result.exit_code == 2
    assert "forecast" in result.output


@pytest.mark.parametrize("command", ["train", "forecast", "evaluate"])
def test_cli_each_subcommand_help_exits_zero(command: str) -> None:
    """Each subcommand exposes a working ``--help`` listing the --data option.

    Force a wide, colourless terminal so rich does not truncate the option name
    (on a narrow CI terminal ``--data`` is rendered as ``--da…`` and the literal
    substring disappears). With a fixed wide width the assertion checks what it
    means to: that the ``--data`` option is present in the help.
    """
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(
        build_app(),
        [command, "--help"],
        env={"COLUMNS": "200", "NO_COLOR": "1"},
    )
    assert result.exit_code == 0, result.output
    assert "--data" in result.output


def test_cli_forecast_synthetic_smoke_run_no_tensorflow() -> None:
    """``forecast`` runs offline on synthetic data via the persistence/ONNX path.

    No committed ONNX artifact ships in the test environment, so it falls back to
    the persistence baseline — and crucially imports NO TensorFlow.
    """
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["forecast", "--n-obs", "200", "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "data source        : synthetic" in result.stdout
    # Persistence (or onnx) forecaster — never the TF train path.
    assert "forecaster         :" in result.stdout
    assert "tensorflow" not in sys.modules


def test_cli_evaluate_synthetic_reports_honest_null() -> None:
    """``evaluate`` reports the honest NULL on synthetic random-walk data.

    Persistence vs. persistence gives MASE == 1, so ``beats_naive`` is False and
    the verdict is ``no_significant_difference`` — the documented deliverable.
    """
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["evaluate", "--n-obs", "200", "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "beats naive        : False" in result.stdout
    assert "no_significant_difference" in result.stdout
    # The evaluation is in return space; a price-level R^2 is never reported.
    assert "R^2" not in result.stdout or "NO price-level R^2" in result.stdout


def test_cli_forecast_with_real_csv(tmp_path: object) -> None:
    """``forecast`` reads a real 'date,close' CSV via the retrain/serve loader."""
    from pathlib import Path

    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    assert isinstance(tmp_path, Path)
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    prices = pd.Series(np.linspace(100.0, 110.0, 120), index=dates, name="close")
    csv = tmp_path / "prices.csv"
    prices.rename_axis("date").reset_index().to_csv(csv, index=False)

    result = CliRunner().invoke(build_app(), ["forecast", "--data", str(csv)])
    assert result.exit_code == 0, result.output
    assert "data source        : csv" in result.stdout


def test_build_app_is_a_fresh_instance() -> None:
    """build_app returns a fresh Typer app each call (no shared mutable state)."""
    import typer

    from lstmforecast.cli import build_app

    app_a = build_app()
    app_b = build_app()
    assert isinstance(app_a, typer.Typer)
    assert app_a is not app_b


def test_cli_module_import_is_side_effect_free() -> None:
    """Importing lstmforecast.cli imports neither Typer nor TensorFlow."""
    code = (
        "import sys; import lstmforecast.cli; "
        "assert 'typer' not in sys.modules, 'Typer imported at module load'; "
        "assert 'tensorflow' not in sys.modules, 'TensorFlow imported at module load'; "
        "print('CLI_IMPORT_PURE_OK')"
    )
    import subprocess

    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "CLI_IMPORT_PURE_OK" in proc.stdout


# --------------------------------------------------------------------------- #
# CLI: train command (stubbed pipeline — no TensorFlow in these tests)        #
# --------------------------------------------------------------------------- #
def _fake_train_result() -> object:
    """A minimal stand-in for ``train.TrainResult`` to exercise the print path."""
    from dataclasses import dataclass

    @dataclass
    class _Metrics:
        rmse_return: float = 0.01
        mae_return: float = 0.008
        mase_vs_persistence: float = 1.0
        directional_accuracy: float = 0.49
        dm_pvalue: float = 0.8
        n_obs: int = 100

    @dataclass
    class _Verdict:
        beats_naive: bool = False

    @dataclass
    class _Result:
        metrics: _Metrics
        verdict: _Verdict
        n_effective_trials: int
        artifact_path: str
        data_source: str

    return _Result(
        metrics=_Metrics(),
        verdict=_Verdict(),
        n_effective_trials=4,
        artifact_path="/tmp/lstm_forecast.onnx",
        data_source="synthetic",
    )


def test_cli_train_happy_path_prints_honest_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """``train`` prints the honest summary (beats naive: False) on a stubbed run.

    The real pipeline (the TensorFlow ``[train]`` path) is owned elsewhere; here
    we stub ``train_pipeline`` so the CLI orchestration + summary printing is
    covered without importing TensorFlow.
    """
    from typer.testing import CliRunner

    import lstmforecast.train as train_module
    from lstmforecast.cli import build_app

    monkeypatch.setattr(train_module, "train_pipeline", lambda **_: _fake_train_result())

    result = CliRunner().invoke(build_app(), ["train", "--no-export", "--n-obs", "100"])
    assert result.exit_code == 0, result.output
    assert "lstm-forecast training run" in result.stdout
    assert "beats naive        : False" in result.stdout
    assert "tensorflow" not in sys.modules


def test_cli_train_library_error_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A library error inside the pipeline is caught and reported with exit code 1."""
    from typer.testing import CliRunner

    import lstmforecast.train as train_module
    from lstmforecast._exceptions import ValidationError
    from lstmforecast.cli import build_app

    def _boom(**_: object) -> object:
        raise ValidationError("bad config")

    monkeypatch.setattr(train_module, "train_pipeline", _boom)

    result = CliRunner().invoke(build_app(), ["train"])
    assert result.exit_code == 1, result.output
    assert "error: bad config" in result.stdout


def test_cli_forecast_uses_onnx_when_artifact_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """When the committed ONNX artifact exists, the forecaster reports ``onnx``.

    We point ``default_artifact_path`` at a real (empty) file so the existence
    check passes; the printed rows still use the persistence forecast, so no
    onnxruntime session is created.
    """
    from pathlib import Path

    from typer.testing import CliRunner

    import lstmforecast.models.onnx_runtime as onnx_module
    from lstmforecast.cli import build_app

    assert isinstance(tmp_path, Path)
    fake_artifact = tmp_path / "lstm_forecast.onnx"
    fake_artifact.write_bytes(b"")  # existence is all the CLI checks
    monkeypatch.setattr(onnx_module, "default_artifact_path", lambda: fake_artifact)

    result = CliRunner().invoke(build_app(), ["forecast", "--n-obs", "120"])
    assert result.exit_code == 0, result.output
    assert "forecaster         : onnx" in result.stdout


def test_cli_forecast_library_error_exits_one() -> None:
    """A bad CSV path surfaces as a library error and exit code 1."""
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["forecast", "--data", "/nonexistent/missing.csv"])
    assert result.exit_code == 1, result.output
    assert "error:" in result.stdout


def test_cli_evaluate_library_error_exits_one() -> None:
    """``evaluate`` on a missing CSV reports the error and exits 1."""
    from typer.testing import CliRunner

    from lstmforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["evaluate", "--data", "/nonexistent/missing.csv"])
    assert result.exit_code == 1, result.output
    assert "error:" in result.stdout


def test_cli_main_entrypoint_runs_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` builds the app and invokes it (here with --help, exit 0)."""
    from lstmforecast.cli import main

    monkeypatch.setattr(sys, "argv", ["lstm-forecast", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
