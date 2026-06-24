# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- End-to-end `train.train_pipeline`: synthetic/real → leakage-free walk-forward
  (per-fold scaler on TRAIN only, ≥`look_back` purge + embargo) → return-space
  metrics → honest `beats_naive` verdict → ONNX artifact + `RunManifest`. The
  honest multiplicity count (`n_effective_trials`) equals the HPO grid size.
- Serve entrypoints the backend calls (onnxruntime only, NEVER TensorFlow):
  `serve.forecast_from_onnx(features)` and `serve.run_forecast(...)` returning a
  JSON-safe `summary` (`rmse_return`, `mae_return`, `mase_vs_persistence`,
  `directional_accuracy`, `dm_pvalue`, `beats_naive`, `n_effective_trials`,
  `data_source`) plus the two Plotly `{data, layout}` figures.
- `models/onnx_export.build_native_lstm_onnx`: a TensorFlow-free builder that
  produces the same `(N, look_back, n_features) → (N, 1)` LSTM-shaped ONNX graph
  via the `onnx` builder, so the shipped artifact is reproducible without a GPU
  and the canonical `tf2onnx` path stays the real-data retrain route.
- Committed, synthetic-random-walk-trained ONNX artifact
  (`src/lstmforecast/artifacts/lstm_forecast.onnx`, <5 MB) served via onnxruntime.

### Changed

- Activated the headline anti-leakage integration test: on a synthetic random
  walk the leakage-free pipeline does NOT beat persistence (`MASE ≥ 1`,
  Diebold-Mariano insignificant, `beats_naive = False`).

### Documentation

- README finalized with the honest NULL headline, the **actual** synthetic
  metrics (`seed=7`: `MASE = 1.00`, `DM p = 1.00`, directional `0.00`,
  `beats_naive = false`), a Validation table (ONNX-vs-Keras `1e-5`; oracle →
  tolerance → test), a Reproduce block (synthetic by default, `--data` for real),
  the debunked price-level-R² note (stated once), Limitations naming the
  fixed-universe survivorship blind spot with a point-in-time (PIT) upgrade path,
  and references (efficient-market/unit-root; Diebold-Mariano 1995;
  Bailey-Lopez de Prado DSR; AFML purge/embargo).
- Added `docs/DESIGN.md` (layering, single-fold data flow, leakage invariants,
  testing strategy) and Architecture Decision Records under `docs/decisions/`:
  per-fold-scaler de-leak (0001), return-target-not-price (0002),
  ONNX-serve-no-TF (0003), honest-null-vs-persistence (0004), and
  no-price-level-R² (0005).
- Added `CITATION.cff` (with the efficient-market, Diebold-Mariano, and
  Bailey-Lopez de Prado references).

## [0.1.0] - 2026-06-17

### Added

- Initial package scaffold (src-layout, import name `lstmforecast`,
  import-pure with `py.typed`).
- Core infra reused from `hrp-portfolio` (renamed `hrp` → `lstmforecast`):
  `_constants`, `_typing`, `_exceptions` (`LstmForecastError` base +
  `ArtifactError`), `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64 generator + substream spawning).
- Reused honest-statistics layer: `evaluation/dsr.py` (PSR/DSR with the full
  kurtosis term + true `n_trials`) and `walkforward/costs.py` (`FixedBpsCost`).
- Stub signatures with full contracts for the new modules: `data` (synthetic
  random-walk generator + CSV loader), `features/{engineer,sequences}`,
  `models/{baselines,lstm,onnx_runtime}`, `walkforward/engine`,
  `evaluation/{metrics,verdict}`, `train`, `plots`, and `cli`.
- Implemented now (load-bearing for the honest NULL): the seeded random-walk
  price generator, the persistence-forecast helper, and the pure `derive_verdict`
  (`beats_naive` is `False` unless `MASE < 1` and Diebold-Mariano is significant
  and directional accuracy is robustly above 0.5).
- Curated top-level `__init__.py` re-exporting the public API with NO TensorFlow
  / onnxruntime imported at module load.
- Seeded test fixtures (`random_walk`, `trend_plus_noise`, `pure_noise`) and the
  partitioned `tests/` tree (unit/parity/property/regression/integration).

[Unreleased]: https://github.com/FatihHekim0glu/lstm-forecast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/lstm-forecast/releases/tag/v0.1.0
