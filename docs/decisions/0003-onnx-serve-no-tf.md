# ADR-0003: Serve via ONNX / onnxruntime, TensorFlow is train-only

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** lstm-forecast maintainers
- **Related:** [DESIGN.md](../DESIGN.md) (layering & import purity)

## Context

The model is an LSTM (and LSTM+Attention) built and trained with Keras /
TensorFlow. But the hosted tool runs in a small, shared API container alongside
the other portfolio tools. TensorFlow is a heavy dependency (hundreds of MB,
slow cold start, large memory footprint) and importing it at request time, or at
package import time, would be unacceptable for a lean inference service. The
package must also be **import-pure**: `import lstmforecast` must not pull in
TensorFlow, an inference engine, or any I/O.

We need a way to train with Keras but serve without it.

## Decision

**Training and serving use different engines, split across optional extras:**

- **Train (`[train]` extra, offline, heavy):** build/fit the LSTM with Keras/TF,
  then export the trained graph to a tiny (< 5 MB) **ONNX** artifact and commit it
  inside the package (`src/lstmforecast/artifacts/lstm_forecast.onnx`).
- **Serve (`[serve]` extra, container, lean):** load and run that committed ONNX
  artifact with **onnxruntime** (numpy + onnxruntime only). The container **never
  imports TensorFlow.**

Two equivalent export backends exist in `models/onnx_export.py`:

1. the canonical Keras→ONNX `tf2onnx` path (the real-data retrain route), and
2. a **TensorFlow-free native `onnx` builder** that emits the same
   `(N, look_back, n_features) → (N, 1)` LSTM-shaped graph, so the shipped
   artifact is reproducible without a GPU or TensorFlow.

Either artifact serves identically through onnxruntime. A **parity test** asserts
the exported ONNX forward pass matches the trained Keras output to **`1e-5`**, so
the thing the container serves is the model the `[train]` path produced.

Import purity is enforced: onnxruntime is imported **lazily** (inside the model
layer, on first call), TensorFlow is never reachable from a plain import, and a
subprocess test verifies no import-time side effects.

## Consequences

- **Positive.** The serve container is tiny and fast; no TensorFlow at all.
- **Positive.** The committed artifact is reproducible on CPU (native builder) and
  certified equivalent to Keras (`1e-5` parity).
- **Positive.** `import lstmforecast` stays side-effect-free and vendorable.
- **Cost.** Two export backends to maintain, and an ONNX/Keras parity test that
  needs the `[train]` extra (so it is marked `slow` and skipped in the lean run).
- **Risk addressed.** "TensorFlow leaks into the inference container / the package
  is not import-pure": both are structurally prevented.
