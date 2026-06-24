# lstm-forecast

**A leakage-free rebuild of the classic LSTM stock-price-prediction project, and
an honest report of the NULL result.**

This is the explicit redemption of the well-known (and badly leaky) "predict the
stock price with an LSTM" project. Done correctly, it predicts the next-day
**log-return** (not the price level), validates with a purged, embargoed,
per-fold-scaled walk-forward, and tests honestly against a random-walk /
persistence baseline.

## Honest headline (the deliverable)

> A properly leakage-free, walk-forward-validated LSTM (and LSTM+Attention) does
> **not** beat a random-walk / persistence baseline on out-of-sample return-space
> error (MASE ≥ 1) or directional accuracy after costs (Diebold-Mariano
> insignificant). This is the correct, literature-backed NULL, reported as the
> deliverable. **No profit claim. No price-level R².**

The shipped demo model is trained on **synthetic random-walk data** (there is no
API key or real market data in this repo). On a true random walk the next-day
return is unpredictable, so the honest NULL holds **by construction**, which is
exactly what makes the leakage-guard integration test meaningful.

### The actual numbers (synthetic random walk, `seed=7`)

Running the shipped pipeline on the default seeded synthetic series
(`lstm-forecast evaluate`, reproduced below) yields:

| Metric                          | Value   | Reads as                                              |
| ------------------------------- | ------- | ----------------------------------------------------- |
| Return-space RMSE               | 0.0095  | error of the next-day **return** forecast             |
| Return-space MAE                | 0.0075  | "                                                     |
| **MASE vs. persistence**        | **1.00**| ≥ 1 → **no** improvement over the random walk         |
| Directional accuracy            | 0.00    | the `r_hat = 0` forecast never claims a side; no sign skill |
| **Diebold-Mariano p-value**     | **1.00**| ≫ 0.05 → cannot reject "equal accuracy" vs. the RW    |
| `n_effective_trials`            | 4       | honest multiplicity (the HPO grid size) fed to the DSR|
| **`beats_naive`**               | **`false`** | the pure-derived verdict, the documented NULL     |

`MASE = 1.00` and `DM p = 1.00` are not rounding artifacts: on a random walk the
optimal next-day return forecast **is** persistence (`r_hat = 0`), so the model
and the baseline collapse to the same error and the verdict is `false` by
construction. That is the deliverable, not a disappointment.

## What makes it leakage-free (the whole point)

- **Target = next-day log-return** `r_{t+1} = ln(P_{t+1} / P_t)`, stationary,
  never the raw price level.
- **Per-fold scaler fit on TRAIN only**, persisted and applied to val/test. This
  is the headline fix for the original repo's full-series-scaler leakage bug.
- **Purge (≥ 60 = `look_back`) + embargo** at every walk-forward train/val/test
  boundary, so no sequence window straddles a split.
- **Per-fold feature recompute** on strictly-past bars (`.shift(1)` discipline;
  `pct_change(fill_method=None)`); no same-bar close-level feature that leaks the
  label's scale.

## The debunked trap: price-level R²

A price-**level** R² looks deceptively high because the trended/integrated price
is dominated by its own lag, a unit-root artifact, **not** forecasting skill.
This project therefore **never** reports a price-level R². All skill is judged in
return space (RMSE/MAE, MASE vs. persistence, directional accuracy, and a
Diebold-Mariano test vs. the random walk with HAC standard errors).

> **Design & decisions.** How the package is layered and how data flows through a
> single walk-forward fold is in [`docs/DESIGN.md`](docs/DESIGN.md); the contested
> choices (per-fold scaler, return target, ONNX serving, the honest null, and the
> banned price-level R²) are recorded as ADRs in
> [`docs/decisions/`](docs/decisions/).

## Serving: ONNX, never TensorFlow in the container

TensorFlow/Keras is **train-only** (the `[train]` extra). The trained LSTM is
exported to a tiny (<5 MB) **ONNX** artifact, committed inside the package
(`src/lstmforecast/artifacts/lstm_forecast.onnx`), and served with **onnxruntime**
(the `[serve]` extra). `import lstmforecast` imports no TensorFlow and no inference
engine, so the package is import-pure.

The export has two equivalent backends: the canonical Keras→ONNX path (`tf2onnx`,
when the `[train]` extra is installed) and a TensorFlow-free native builder
(`models.onnx_export`, via the `onnx` builder) that produces the same
`(N, look_back, n_features) → (N, 1)` LSTM graph so the shipped artifact is
reproducible without a GPU. Either artifact serves identically through onnxruntime.

The backend calls two serve entrypoints (onnxruntime only, no TF):

- `forecast_from_onnx(features)`: run the committed ONNX graph on a pre-scaled
  sequence tensor.
- `run_forecast(...)`: the high-level entrypoint: leakage-free walk-forward to
  return-space metrics vs. persistence to honest `beats_naive` verdict to a JSON-safe
  `summary` (`rmse_return`, `mae_return`, `mase_vs_persistence`,
  `directional_accuracy`, `dm_pvalue`, `beats_naive`, `n_effective_trials`,
  `data_source`) plus the two Plotly `{data, layout}` figures.

## Validation

Every numeric claim is pinned to an independent reference (oracle → tolerance →
test):

| What                          | Oracle / reference                          | Tolerance | Test                                                      |
| ----------------------------- | ------------------------------------------- | --------- | --------------------------------------------------------- |
| Served ONNX = trained Keras   | the Keras forward pass (`model.predict`)    | `1e-5`    | `tests/parity/test_onnx_keras_parity.py` (`slow`)         |
| Per-fold scaler de-leak       | future-perturbation invariance              | exact     | `tests/property/` (perturbing test rows ⇏ train preds)    |
| No-lookahead walk-forward     | golden split snapshot                       | exact     | `tests/regression/test_walkforward_engine.py`             |
| Honest NULL on a random walk  | persistence is the floor                    | n/a       | `tests/integration/` (LSTM does **not** beat persistence) |
| `beats_naive` truth table     | the pure decision rule                      | exact     | `tests/regression/test_verdict_truth_table.py`            |
| DSR multiplicity + kurtosis   | the Bailey-Lopez de Prado formula           | `1e-4`    | `tests/unit/test_dsr_and_costs.py`                        |

The `1e-5` parity test certifies that the artifact the container serves is the
same model the `[train]` path produced; it is marked `slow` (needs the `[train]`
extra to build the Keras reference) and is skipped in the lean default run.

## Install

```bash
uv venv
# Lean dev install (no TensorFlow; the [train] extra is heavy and only needed to
# retrain/export the ONNX artifact):
uv pip install -e ".[data,serve,viz,dev]"
```

Extras: `data` (numpy, pandas, scipy, statsmodels, pyarrow, diskcache) ·
`serve` (onnxruntime) · `train` (tensorflow, tf2onnx) · `viz` (plotly, kaleido) ·
`dev` (pytest, pytest-cov, hypothesis, ruff, mypy).

## Usage (CLI)

```bash
lstm-forecast train                 # train on synthetic random walk, export ONNX
lstm-forecast train --data PATH.csv # retrain on a real date,close CSV
lstm-forecast evaluate              # walk-forward metrics vs. persistence
lstm-forecast forecast              # serve a forecast from the committed ONNX model
```

## Reproduce

The committed result is fully reproducible without a GPU or any API key. The
shipped model trains on the seeded synthetic random walk **by default**:

```bash
uv venv && uv pip install -e ".[data,serve,viz,dev]"

# 1. The headline NULL, straight from the served pipeline (seed=7):
lstm-forecast evaluate              # MASE=1.00, DM p=1.00, beats_naive=false

# 2. Same numbers from Python (what the backend calls):
python -c "from lstmforecast.serve import run_forecast; \
print(run_forecast(seed=7).summary.to_dict())"

# 3. Retrain end-to-end on the synthetic random walk and re-export the ONNX
#    artifact (no TensorFlow needed; uses the native onnx builder):
lstm-forecast train                 # writes src/lstmforecast/artifacts/lstm_forecast.onnx
```

To run against **real** data, pass a `date,close` CSV. The leakage-free
validation and the honest verdict are identical, only the price series changes:

```bash
lstm-forecast train --data prices.csv      # retrains; requires the [train] extra for tf2onnx
lstm-forecast evaluate --data prices.csv    # walk-forward metrics on your series
```

The literature-backed conclusion (the LSTM does not beat persistence
out-of-sample in return space) is what the random-walk default demonstrates by
construction; real data does not change the methodology, only the inputs.

## Limitations

- **Synthetic-trained demo.** The shipped model is trained on a seeded synthetic
  random walk, reproducible and honest, but **not** a market forecaster. It
  exists to certify the leakage-free pipeline and the honest NULL, not to predict
  prices. Retraining on real data (`--data`) does not change the literature-backed
  conclusion.
- **Fixed-universe survivorship (named blind spot).** The demo runs on a single,
  fixed price series. A fixed universe of *survivors* is biased upward: tickers
  that delisted or were dropped are silently excluded, so any backtest on today's
  constituents overstates skill. This repo does **not** correct for it.
  - **PIT upgrade path.** The honest fix is a **point-in-time (PIT)** universe:
    reconstruct the index/constituent membership *as it was known on each
    rebalance date* (with delisted names retained until they leave), and feed that
    time-varying universe into the same walk-forward engine. The walk-forward,
    purge/embargo, and per-fold-scaler machinery are universe-agnostic, so adding
    a PIT data provider is additive, it does not touch the de-leak core. The
    concrete entry point is a `sp500_universe.py` PIT provider (reconstructing
    S&P 500 membership-as-of-date, delisted tickers retained) that yields the
    rebalance-dated constituent set the engine consumes, a drop-in upstream of the
    existing price loader, leaving `run_walk_forward` and the metrics untouched.
- **Small models, on purpose.** The LSTM and LSTM+Attention are deliberately tiny
  (few units, few epochs). The point is the methodology and the honest null, not a
  top-of-the-leaderboard model; a bigger model on a random walk still cannot beat
  persistence.

## References

- **Efficient markets / random walk.** Fama, E.F. (1970). *Efficient Capital
  Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2).
  Prices behave close to a random walk, so the next-day return is approximately
  unpredictable and persistence (`r_hat = 0`) is the baseline to beat.
- **Unit-root / non-skill of level prediction.** A price level is integrated of
  order one (a unit root); regressing `P_{t+1}` on `P_t` yields a near-1.0 R²
  that reflects the trend, not forecasting skill, the debunked trap this project
  refuses to report. (Standard unit-root treatment, e.g. Hamilton, *Time Series
  Analysis*, 1994, Ch. 15 to 17.)
- **Diebold, F.X. & Mariano, R.S. (1995).** *Comparing Predictive Accuracy.*
  Journal of Business & Economic Statistics, 13(3). The equal-predictive-accuracy
  test (with a Newey-West HAC long-run variance) used here to compare the model
  against the random walk.
- **Bailey, D.H. & López de Prado, M. (2014).** *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* The
  Journal of Portfolio Management, 40(5). The deflation by the true `n_trials`
  (the HPO-grid size) and the full kurtosis term used in the DSR.
- **López de Prado, M. (2018).** *Advances in Financial Machine Learning*, Ch. 7.
  Purge-and-embargo cross-validation, here re-derived for the 60-day sequence
  window (see [`docs/decisions/0001-per-fold-scaler-deleak.md`](docs/decisions/0001-per-fold-scaler-deleak.md)).

## License

MIT, see [LICENSE](LICENSE).
