# ADR-0002: Forecast the next-day log-return, never the price level

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** lstm-forecast maintainers
- **Related:** [ADR-0001](0001-per-fold-scaler-deleak.md) (de-leak),
  [ADR-0005](0005-no-price-level-r2.md) (the banned price-level R²)

## Context

The original project predicts the **price level** `P_{t+1}` and reports a high R²
against the realized price. This looks impressive and is almost entirely an
artifact. A price series has a **unit root** (it is integrated of order one): its
best one-step predictor is its own last value, so any model that learns "tomorrow
≈ today" scores a near-1.0 level R². That number measures the trend, not
forecasting skill. Worse, training a regressor on a non-stationary target lets it
exploit the level's slow drift, which interacts badly with the scaler-leakage bug
([ADR-0001](0001-per-fold-scaler-deleak.md)) to manufacture apparent skill.

The honest, stationary quantity to forecast is the **return**.

## Decision

The target is the **next-day log-return**

```
r_{t+1} = ln(P_{t+1} / P_t)
```

never the raw price level. Consequences that follow from this choice:

- **No target-in-features.** Indicators are computed on strictly-past bars only;
  there is no same-bar close-level feature that leaks the label's scale.
- **Persistence is the floor.** The natural baseline becomes `r_hat = 0` (a
  random-walk price), which is exactly the persistence forecaster every model is
  measured against ([ADR-0004](0004-honest-null-vs-persistence.md)).
- **Skill is judged in return space.** All metrics — RMSE/MAE, MASE vs.
  persistence, directional accuracy, Diebold-Mariano vs. the random walk — operate
  on returns. The price-level R² is *banned*
  ([ADR-0005](0005-no-price-level-r2.md)).

## Consequences

- **Positive.** The forecasting problem is now stationary and honest: a model
  that beats persistence in return space genuinely has skill.
- **Positive.** The setup is directly comparable to the efficient-markets /
  random-walk literature, where return predictability is the live question.
- **Cost.** Return forecasting is *hard* — and on a random walk it is impossible.
  The result is the documented null rather than a flashy chart. That is the point.
- **Risk addressed.** "Unit-root-inflated level prediction masquerading as skill"
  is eliminated at the target definition, before any model is trained.
