# ADR-0005: Never report a price-level R² (the debunked trap)

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** lstm-forecast maintainers
- **Related:** [ADR-0002](0002-return-target-not-price.md) (return target)

## Context

The single most misleading number in the genre of "LSTM stock predictor" projects
is the **price-level R²**: the coefficient of determination of predicted vs.
actual *price levels*. It is routinely reported as 0.95+ and presented as proof
the model "works."

It is a trap. A price series has a **unit root** (it is integrated of order one),
so it is dominated by its own lag. Predicting `P_{t+1} ≈ P_t` (i.e. doing
*nothing*) already explains almost all of the level's variance, because that
variance *is* the trend. A high level R² therefore certifies that the series
trends, not that the model forecasts. Pair it with the scaler-leakage bug
([ADR-0001](0001-per-fold-scaler-deleak.md)) and you get a beautiful, completely
hollow result.

## Decision

The price-level R² is **banned**. It is not computed, not stored, and not
reported anywhere in the codebase, the API response, the frontend, or the docs.
All skill is judged in **return space** ([ADR-0002](0002-return-target-not-price.md)):

- return-space RMSE / MAE,
- **MASE vs. persistence** (`≥ 1` ⇒ no improvement over the random walk),
- directional accuracy with a two-sided binomial test,
- the **Diebold-Mariano** (1995) test vs. the random walk, with a Newey-West HAC
  long-run variance.

This ADR exists so the trap is documented **once, explicitly**, as a debunked
metric, and so the absence of a level R² is a deliberate, defensible choice
rather than an oversight.

## Consequences

- **Positive.** The headline metrics cannot be inflated by the unit-root artifact;
  every reported number reflects actual return-space accuracy.
- **Positive.** Readers who expect the familiar "R² = 0.97" chart get an explicit
  explanation of why it is meaningless instead.
- **Cost.** The project's headline numbers look modest (MASE = 1.00) compared to a
  level-R² display. That honesty is the deliverable.
- **Risk addressed.** "Reporting a unit-root-inflated R² as forecasting skill",
  the defining mistake of the original repo, is structurally excluded.
