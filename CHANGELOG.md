# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
