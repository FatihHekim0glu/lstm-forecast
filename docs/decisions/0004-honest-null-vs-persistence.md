# ADR-0004: The deliverable is the honest NULL vs. persistence

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** lstm-forecast maintainers
- **Related:** [ADR-0001](0001-per-fold-scaler-deleak.md) (de-leak),
  [ADR-0002](0002-return-target-not-price.md) (return target)

## Context

The temptation in an "LSTM stock predictor" project is to produce a positive
result by any means, and the usual means is leakage. The honest, literature-
backed finding is the opposite: prices are close to a random walk (efficient
markets), so a properly validated model **does not** beat naive persistence
out-of-sample in return space. We decided up front that the **null is the
deliverable**, not a failure to be papered over.

To make that null *demonstrable and reproducible*, rather than merely asserted,
the shipped pipeline trains and evaluates on a **synthetic random walk**.

## Decision

1. **Baseline = persistence / random walk** (`r_hat = 0`). This is the floor every
   model must clear, run through the *same* walk-forward folds as the LSTM (via
   the shared `model_factory` interface) so the comparison is fair.
2. **Synthetic random-walk data by default.** Tests and the shipped model use the
   seeded `data.random_walk_prices` generator (no API key, no market data). On a
   true random walk the next-day return is unpredictable, so the optimal forecast
   *is* persistence and the LSTM **cannot** beat it, so the null holds **by
   construction.**
3. **A pure verdict.** `beats_naive` is derived by `evaluation/verdict.py` and is
   `True` only if **all three** hold: `MASE < 1`, the Diebold-Mariano test is
   significant (`p < 0.05`), and directional accuracy is robustly above 0.5.
   Otherwise the verdict is `NO_SIGNIFICANT_DIFFERENCE` and `beats_naive = False`.
4. **Honest multiplicity.** The Deflated/Probabilistic Sharpe is deflated by the
   *true* `n_trials` = the HPO-grid size, with the full kurtosis term.
5. **The anti-leakage integration test (the headline guard).** End-to-end on a
   synthetic random walk, the LSTM must **not** beat persistence (`MASE ≥ ~1`, DM
   insignificant). If it ever does, leakage has re-entered, and the test
   **fails.** The actual shipped numbers (`seed=7`): `MASE = 1.00`,
   `DM p = 1.00`, directional `0.00`, `beats_naive = false`.

## Consequences

- **Positive.** The headline is *mechanically* honest: the verdict is derived
  from the inference, not narrated, and cannot read `True` while the evidence is
  absent.
- **Positive.** The null is reproducible by anyone on CPU with no credentials.
- **Positive.** The integration test doubles as a leakage tripwire for the whole
  pipeline.
- **Cost.** The shipped model is not a market forecaster; a `--data` path retrains
  on real series, but the methodology, and the literature-backed conclusion,
  are unchanged.
- **Risk addressed.** "Quietly claiming the LSTM beats the market" is impossible
  here: it would require `beats_naive` to flip to `True`, which the verdict logic
  and the random-walk guard jointly forbid.
