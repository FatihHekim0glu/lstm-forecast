# ADR-0001: Per-fold scaler fit on TRAIN only (the de-leak)

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** lstm-forecast maintainers
- **Related:** [ADR-0002](0002-return-target-not-price.md) (return target),
  [ADR-0004](0004-honest-null-vs-persistence.md) (the honest null)

## Context

The original "predict the stock price with an LSTM" project — the repo this one
redeems — has a textbook leakage bug: it fits the feature **scaler on the whole
series** (train + validation + test together) *before* splitting. The test set's
mean and variance therefore bleed into the standardization the model sees at
train time. The model has effectively peeked at the future scale of the data, and
its out-of-sample error is optimistically biased. Combined with a price-level
target (see [ADR-0002](0002-return-target-not-price.md)), this is what produces
the deceptively pretty "the LSTM tracks the price!" charts that don't survive
honest validation.

This is the single most important defect to fix. If it is fixed correctly, the
honest null follows almost automatically; if it is not, leakage re-enters and the
model appears to "beat" persistence for the wrong reasons.

## Decision

Scaling and feature engineering are **recomputed per fold**, and the scaler is
**fitted on the TRAIN slice only**, then *applied* (never re-fitted) to the
validation and test slices. Concretely, in `walkforward/engine.py`, for each
anchored/expanding fold:

1. **Per-fold feature recompute.** Features are engineered separately inside each
   train/val/test slice, on strictly-past bars (`.shift(1)`,
   `pct_change(fill_method=None)`), so an indicator warm-up never straddles a
   split boundary.
2. **Train-only scaler.** The standardizer's mean/std are estimated on the train
   slice exclusively, persisted, and then applied to val and test.
3. **Purge (≥ `look_back` = 60).** A gap of at least `look_back` bars is removed
   at every train/val/test boundary, so no `look_back`-length sequence window can
   span a split.
4. **Embargo.** A further gap follows each test block, so adjacent folds cannot
   share information through overlapping windows.

The no-lookahead property is enforced by a **future-perturbation invariance**
property test: perturbing the test-fold rows must leave the fitted scaler and the
train-fold predictions byte-identical. If it doesn't, leakage has re-entered and
the test fails.

## Consequences

- **Positive.** The headline bug of the source repo is removed and *provably
  absent* (the invariance test is the proof).
- **Positive.** The same folds are reused for the baseline and the LSTM, so the
  comparison is fair (see [ADR-0004](0004-honest-null-vs-persistence.md)).
- **Cost.** Per-fold recompute is more expensive than scaling once globally, and
  the purge/embargo gaps reduce usable observations. Both are accepted as the
  price of correctness.
- **Risk addressed.** "Full-series scaler leakage" — the defect that makes the
  original project a cautionary tale — cannot recur silently; the property test
  guards it.
