# lstm-forecast

**A leakage-free rebuild of the classic LSTM stock-price-prediction project — and
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
> insignificant). This is the correct, literature-backed NULL — reported as the
> deliverable. **No profit claim. No price-level R².**

The shipped demo model is trained on **synthetic random-walk data** (there is no
API key or real market data in this repo). On a true random walk the next-day
return is unpredictable, so the honest NULL holds **by construction** — which is
exactly what makes the leakage-guard integration test meaningful.

## What makes it leakage-free (the whole point)

- **Target = next-day log-return** `r_{t+1} = ln(P_{t+1} / P_t)` — stationary,
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
is dominated by its own lag — a unit-root artifact, **not** forecasting skill.
This project therefore **never** reports a price-level R². All skill is judged in
return space (RMSE/MAE, MASE vs. persistence, directional accuracy, and a
Diebold-Mariano test vs. the random walk with HAC standard errors).

## Serving: ONNX, never TensorFlow in the container

TensorFlow/Keras is **train-only** (the `[train]` extra). The trained LSTM is
exported to a tiny (<5 MB) **ONNX** artifact (`tf2onnx`), committed inside the
package, and served with **onnxruntime** (the `[serve]` extra). `import
lstmforecast` imports no TensorFlow and no inference engine — the package is
import-pure.

## Install

```bash
uv venv
# Lean dev install (no TensorFlow — the [train] extra is heavy and only needed to
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

## Limitations

- The shipped model is **synthetic-random-walk-trained** — reproducible and
  honest, but not a market forecaster. Retraining on real data does not change
  the literature-backed conclusion.
- A fixed-universe / single-series setup is subject to survivorship and
  selection effects that this demo does not attempt to correct.

## References

- Efficient-market and unit-root behaviour of prices (random walk; non-skill of
  level prediction).
- Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy.* JBES.
- Bailey, D.H. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* JPM.

## License

MIT — see [LICENSE](LICENSE).
