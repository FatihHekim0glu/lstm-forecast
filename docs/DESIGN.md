# Design

This document explains how `lstm-forecast` is put together: the layering, the
data flow through a single walk-forward fold, the leakage invariants the compute
core guarantees, and the testing strategy that keeps the honest NULL honest. For
*why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

This package is the explicit **redemption** of the well-known, badly leaky
"predict the stock price with an LSTM" project. The entire deliverable is doing
it *without* leakage and reporting the honest null: a properly validated LSTM
does **not** beat a random-walk / persistence baseline out-of-sample.

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging
  TensorFlow, an inference engine, or network dependencies along at import time.
- A walk-forward that fixes the original repo's headline bug: a **per-fold scaler
  fit on TRAIN only**, with **purge (≥ `look_back`) + embargo** at every boundary
  and per-fold feature recompute.
- A verdict that is a **pure function** of the inference and is *mechanically*
  prevented from over-claiming.
- A serve path that runs the trained model via **ONNX / onnxruntime only** — the
  container never imports TensorFlow.

**Non-goals**

- Beating persistence. The honest finding is that, validated correctly, the LSTM
  does not — on a random walk it cannot, by construction.
- A great model. The LSTM and LSTM+Attention are deliberately tiny; the point is
  the methodology and the null, not state of the art.
- A live trading system, or a real-data product. The shipped model trains on a
  seeded synthetic random walk (no API key, no market data in the repo).

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/lstmforecast/` has **zero import-time side effects** (no TensorFlow, no
onnxruntime, no I/O, no RNG draw at import), guarded by a subprocess
import-purity test.

```
        cli.py (Typer)        plots.py (lazy Plotly)        serve.py (backend entry)
             |                       |                            |
   ┌─────────┴───────────────────────┴────────────────────────────┘
   │                          train.py
   │   (synthetic/real -> walk-forward -> small LSTM -> evaluate -> export ONNX)
   ├──────────────────────────────────────────────────────────────────
   │                        evaluation/
   │            metrics.py · verdict.py · dsr.py
   │   (return-space RMSE/MAE, MASE, directional+binomial, Diebold-Mariano+HAC;
   │    pure beats_naive deriver; Deflated/Probabilistic Sharpe + true n_trials)
   ├──────────────────────────────────────────────────────────────────
   │                       walkforward/
   │                 engine.py · costs.py
   │   (anchored/expanding folds · per-fold scaler on TRAIN only · purge+embargo
   │    · per-fold feature recompute · per-side bps cost)
   ├──────────────────────────────────────────────────────────────────
   │            features/                          models/
   │   engineer.py · sequences.py        baselines.py · lstm.py (TRAIN-only,
   │   (.shift(1), pct_change,           lazy TF) · onnx_runtime.py (SERVE,
   │    create_sequences(look_back))     onnxruntime) · onnx_export.py (builder)
   ├──────────────────────────────────────────────────────────────────
   │   data.py                          foundation (no internal deps)
   │   (seeded random-walk generator,   _validation · _constants · _typing
   │    date,close CSV loader)           _exceptions · _manifest · _rng
   └──────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

Reused from `hrp-portfolio` (renamed `hrp` → `lstmforecast`):

- `_constants.py` — single source of truth for shared constants.
- `_validation.py` — input guards (shape, finiteness, sufficient observations).
- `_typing.py` / `_exceptions.py` — shared aliases (`FloatArray`,
  `SequenceTensor`) and the exception taxonomy (`LstmForecastError` base,
  `ArtifactError`, `ValidationError`, `InsufficientDataError`).
- `_manifest.py` / `_rng.py` — `RunManifest` (BLAKE2b config-hash) plus seeded
  PCG64 substreams. The same seed yields a byte-identical run.

### `data.py`

The seeded geometric random-walk price generator (`random_walk_prices`) and the
`date,close` CSV loader (`load_prices`). On a true random walk the next-day
return is unpredictable, so the honest null holds by construction — this is the
data-generating process that makes the anti-leakage integration test meaningful
([ADR-0004](decisions/0004-honest-null-vs-persistence.md)).

### `features/`

`engineer.py` computes technical features on strictly-past bars (`.shift(1)`
discipline; `pct_change(fill_method=None)`), with **no same-bar close-level
feature** that would leak the label's scale. `sequences.py` turns the feature
frame into the `create_sequences(look_back=60)` input tensor
`(N, look_back, n_features)`. The target is the **next-day log-return**, never the
price level ([ADR-0002](decisions/0002-return-target-not-price.md)).

### `models/`

- `baselines.py` — the persistence / random-walk forecaster (`r_hat = 0`); the
  floor every model must clear.
- `lstm.py` — the Keras LSTM and LSTM+Attention `build_model()`. TensorFlow is
  imported **lazily** and only on the `[train]` path; it is never reachable from a
  plain `import lstmforecast`.
- `onnx_runtime.py` — loads and runs the committed ONNX artifact via onnxruntime
  (the SERVE path; no TF) ([ADR-0003](decisions/0003-onnx-serve-no-tf.md)).
- `onnx_export.py` — two equivalent export backends: the canonical Keras→ONNX
  `tf2onnx` path (real-data retrain) and a TensorFlow-free native `onnx` builder
  that produces the same `(N, look_back, n_features) → (N, 1)` graph so the
  shipped artifact is reproducible without a GPU.

### `walkforward/`

`engine.py` is the de-leak core: anchored/expanding folds, **per-fold scaler fit
on TRAIN only** (persisted, applied to val/test), **purge (≥ `look_back`) +
embargo** at every boundary, and per-fold feature recompute so a warm-up never
straddles a split ([ADR-0001](decisions/0001-per-fold-scaler-deleak.md)). It is
generic over the model (a `model_factory` callable), so the persistence baseline
and the LSTM run through the **same** folds — the only fair comparison.
`costs.py` applies a per-side bps cost to the toy return-trading strategy.

### `evaluation/`

- `metrics.py` — return-space RMSE/MAE, MASE vs. persistence, directional
  accuracy with a two-sided binomial test, and the Diebold-Mariano (1995) test
  vs. the random walk with a Newey–West HAC long-run variance. **No price-level
  R²** ([ADR-0005](decisions/0005-no-price-level-r2.md)).
- `verdict.py` — the **pure** `derive_verdict`: `beats_naive` is `True` only when
  `MASE < 1` AND the DM test is significant AND directional accuracy is robustly
  above 0.5. Any failure → `NO_SIGNIFICANT_DIFFERENCE`, `beats_naive = False`.
- `dsr.py` — Deflated / Probabilistic Sharpe with the full kurtosis term and the
  true `n_trials` (= HPO-grid size).

## Data flow through one walk-forward fold

```
train slice (prices) ──► per-fold feature recompute (.shift(1), pct_change)
                            │
                            ├─► fit scaler on TRAIN ONLY  (mean/std persisted)
                            │
                            ▼
                  create_sequences(look_back=60) ─► X_train, y_train (next-day return)
                            │
        ┌───────────────────┴────── small HPO grid on the VAL slice only ──────────┐
        │                          (records n_trials = grid size)                  │
        ▼                                                                          │
  fit candidate model(s)                                                           │
        │                                                                          │
        ▼  purge (≥ look_back) + embargo at the boundary; scaler APPLIED (not refit)│
  TEST slice ──► scale with the TRAIN-fitted scaler ─► predict next-day RETURN  ───┘
        │
        ▼  (stack OOS forecasts across all folds; same folds for the baseline)
  return-space RMSE/MAE · MASE vs. persistence · directional acc + binomial ·
  Diebold-Mariano vs. random walk (HAC) · DSR with true n_trials
        │
        ▼
  verdict.derive_verdict(mase, dm_pvalue, directional)  ──►  beats_naive (pure)
```

The headline comparison is **LSTM vs. persistence**, run through identical folds.
On the synthetic random walk the model and the baseline collapse to the same
error, so `MASE = 1.00`, `DM p = 1.00`, and `beats_naive = false` — the
documented NULL.

## Key invariants

The compute core guarantees, and tests enforce:

1. **Return target.** The label is the next-day log-return; the price level is
   never a feature or a metric.
2. **Train-only scaler.** The standardizer's mean/std are fitted on the train
   slice exclusively, then applied (never refit) to val/test.
3. **No-lookahead (future-perturbation invariance).** Perturbing test-fold rows
   leaves the fitted scaler and the train-fold predictions unchanged.
4. **Window containment.** No `look_back`-length sequence spans a split (purge ≥
   `look_back`); an embargo follows each test block.
5. **Fair race.** The baseline and the LSTM run through the same folds via the
   shared `model_factory` interface.
6. **No price-level R².** The metric does not exist anywhere in the codebase.
7. **DSR multiplicity.** The Deflated Sharpe uses the full kurtosis term and the
   true `n_trials` (HPO-grid size).
8. **Verdict safety.** `derive_verdict` cannot emit `beats_naive = True` unless
   all three conditions hold (truth-table unit-tested).
9. **Parity.** The served ONNX forward pass matches the trained Keras output to
   `1e-5`.
10. **Determinism.** Same `RunManifest` seed → byte-identical outputs.
11. **Import purity.** Importing any `src/lstmforecast` module triggers no TF, no
    onnxruntime, no I/O, no network, no RNG draw (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`),
with seeded fixtures in `conftest.py` (`random_walk`, `trend_plus_noise`,
`pure_noise`):

- **`unit/`** — isolated kernels: data generator, baselines, features/sequences,
  metrics, DSR + costs, the verdict truth table, infra.
- **`property/`** (Hypothesis) — future-perturbation invariance (the de-leak),
  shift-equivariance of features, persistence-is-the-floor on pure noise.
- **`parity/`** — the ONNX-vs-Keras forward pass to `1e-5` (`slow`; needs the
  `[train]` extra to build the Keras reference).
- **`regression/`** — the no-lookahead golden test on the walk-forward engine,
  the DSR `n_trials` + kurtosis term.
- **`integration/`** — the headline anti-leakage guard: end-to-end on a synthetic
  random walk the LSTM does **not** beat persistence (`MASE ≥ 1`, DM
  insignificant); the import-purity subprocess test.

Coverage gate `fail_under = 85`; ruff + strict mypy clean. The `[train]` / TF
path is marked `slow` and the suite runs without a GPU.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`lstm-forecast[serve]` (onnxruntime, **not** TensorFlow) under
`api/lib/lstm_forecast/`, byte-for-byte including `artifacts/*.onnx`, and exposes
`POST /tools/lstm-forecast/run`. The router imports **no TensorFlow**; a
module-level `_SESSION = None` loads the ONNX session lazily on first call.
Scalars go through `_safe_float`; figures serialize via
`json.loads(pio.to_json(fig, validate=False))`. The frontend surfaces the
pure-derived verdict as a prominent **"Beats naive baseline: NO"** badge — the
first thing a visitor reads — alongside the honest caption "Predicts returns, not
prices; no price-level R²; does not beat a random walk."
