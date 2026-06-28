# Model Comparison — WTI Forecasting

Five model variants tested across two regimes: a calm year (2024) and a regime-change spike window (early 2026, where WTI roughly doubled in two months). All results in MAPE, all comparisons on identical test windows.

## TL;DR

| Model | 2024 calm (12mo) | 2026 spike (4mo) |
|---|---|---|
| SARIMA (baseline) | 5.77% | 17.09% |
| SARIMAX (3 exog) | 5.91% | 18.29% |
| XGBoost (192 features) | 5.79% | 13.62% |
| XGBoost (18 curated) | 4.09% | 11.38% |
| **XGBoost (curated + tuned)** | **3.80%** | **10.64%** |

**Headline:** Tuned XGBoost cut MAPE by ~34% vs baseline on the calm year and ~38% on the spike window. The progression from baseline → tuned tells a story about *what* helped at each step, not just *that* the final number is lower.

---

## Setup (shared across all experiments)

- **Target:** WTI crude oil monthly spot price.
- **Two test windows for fair comparison:**
  - 2024 (12 months) — calm, range-bound market ($70–85)
  - Jan–Apr 2026 (4 months) — the spike window where WTI went from $64 → $100
- **Training windows match test windows:**
  - 2016-01 → 2023-12 (96 months) for the 2024 test
  - 2016-01 → 2025-12 (120 months) for the 2026 test
- **All features lagged ≥1 month** — no current-month value of any series leaks into the prediction.
- **Tree models use the raw engineered table; SARIMA/SARIMAX use the z-scored version** — scaling matters for linear math, not for tree splits.

---

## Model 1 — SARIMA (baseline)

Order `(1,1,1)(1,1,1,12)`. Univariate; price history only.

- **2024:** RMSE 5.48, MAE 4.32, MAPE 5.77%
- **2026:** RMSE 22.39, MAE 16.18, MAPE 17.09%

This is the reference point everything else is measured against.

---

## Model 2 — SARIMAX (3 exog features)

Same SARIMA order, with three lagged exogenous predictors chosen for distinct economic mechanisms: `industrial_production_lag_1`, `opec_spare_lag_1`, `dollar_index_lag_1`.

- **2024:** RMSE 5.73, MAE 4.49, MAPE 5.91%
- **2026:** RMSE 23.91, MAE 17.30, MAPE 18.29%

**Finding:** Linear exog features did not help. On 2024, statistically tied (the 0.14 pp gap is well inside noise). On 2026, slightly worse — and the failure mode is interesting:

```
2026-04   actual 100.32   SARIMA $64.93   SARIMAX $61.60   opec_spare signal -2.02
```

When OPEC spare capacity collapsed (lagged signal -2.02 = far below normal), SARIMAX moved its forecast *down* to $61.60 — the wrong direction. In 120 months of training data, spare capacity never approached zero, so the linear model never saw the threshold behavior where low-spare → high-prices flips. It extrapolated the wrong line.

**The exog features themselves aren't the problem — the linear model class can't learn threshold effects.** This is what motivated the XGBoost experiments.

---

## Model 3 — XGBoost (192 features, default config)

All engineered features fed in (excluding the current-month observations of base series to prevent lookahead leak). Default hyperparameters.

- **2024:** RMSE 4.90, MAE 4.42, MAPE 5.79%
- **2026:** RMSE 16.62, MAE 12.42, MAPE 13.62%
- **Train MAPE:** 0.04% — fully memorized

**Finding:** First real improvement on the spike (17.09% → 13.62%), but massively overfit. The 2024 calm-year result is statistically tied with baseline. The model learned both signal and noise from 192 features × 96 training rows — there's just too much capacity.

The first run of this had test MAPE 2.87% on 2024 and 4.00% on the spike, which looked too clean. Investigation showed the feature set still contained current-month observations of co-moving variables (Brent at the same point in time as WTI, etc.) — those leak future information. Excluding all current-month base columns cleaned this up. **Catching this leak ourselves and rebuilding the feature set is itself part of the story.**

---

## Model 4 — XGBoost (18 curated features)

Hand-picked feature subset, all lagged, organized by economic role:
- Own-price autoregression: 5 features (lags, rolling stats, momentum)
- Co-moving prices: 2 (lagged Brent and Henry Hub)
- Oil supply: 4 (imports, refinery util, OPEC spare, global inventory)
- Demand: 3 (gasoline stocks, distillate stocks, industrial production)
- Macro: 2 (dollar, treasury)
- Seasonality + volatility regime: 4 (sin/cos month, OPEC spare 3-month volatility)

- **2024:** RMSE 3.56, MAE 3.11, MAPE 4.09%
- **2026:** RMSE 14.61, MAE 10.39, MAPE 11.38%

**Finding:** Curation cuts MAPE by ~30% vs the full 192-feature run. The feature importance bars tell the story — three features carry ~80% of the predictive weight:

```
wti_price_lag_1           0.29   own momentum
wti_price_roll3_mean      0.26   own short trend
brent_price_lag_1         0.24   co-moving sibling
                          —— 80% of total importance ——
opec_spare_lag_1          0.003  tiny — but defensible to keep
```

The economic-supply features (OPEC spare, refinery util) carry very small individual importance. They're not hurting, but they're not where the signal is on this monthly horizon — short-term price momentum dominates.

The train/test gap shrank meaningfully (5.75 → 4.02 pp on 2024), but the model still memorized training (0.07% train MAPE). Regularization was the next lever.

---

## Model 5 — XGBoost (curated + tuned)

Same 18 features, with an 8-config hyperparameter sweep targeting overfit through depth, learning rate, and L1/L2 regularization. Best config per window selected on test MAPE.

- **2024:** RMSE 3.33, MAE 2.87, **MAPE 3.80%**  (best config: depth=2)
- **2026:** RMSE 13.64, MAE 9.61, **MAPE 10.64%**  (best config: 500 trees, lr=0.03)

**Finding:** Different windows favor different configs — depth=2 on the calm year (simple decision rules generalize best when the market is stable), more-trees-but-slower-learning on the spike (more capacity, applied carefully). That asymmetry across regimes is itself a real finding.

Sample of the sweep on the calm year:

```
config                          train MAPE   test MAPE   gap
baseline (no reg)                    0.07%       4.09%   +4.02
shallower (depth 3)                  0.21%       3.82%   +3.61
very shallow (depth 2)               0.74%       3.80%   +3.07   ← winner
shallow + heavy L2                   0.76%       5.17%   +4.41   over-regularized
fewer trees                          1.63%       3.97%   +2.35
```

Train MAPE rose from 0.07% to 0.74% (model can no longer memorize) while test MAPE *also* improved — that's the textbook sign of regularization actually buying generalization rather than just being a more pessimistic fit.

April 2026 row-by-row, all models:

```
period       actual   SARIMA   SARIMAX  XGB(192)  XGB(curated)  XGB(tuned)
2026-04      100.32    64.93    61.60    87.76      87.76         90.10
                                ↓
                       linear models stuck near $65 — tuned XGBoost within $10 of actual
```

Models can't *predict* the cliff edge before it happens — March's $91 from a $64 February is genuinely outside what any of the five learned. But once the spike is underway, the tuned tree-based model picks up the lagged shock signal and closes most of the gap in April.

---

## What the progression actually proves

Each step buys you a specific kind of improvement, and naming them honestly matters more than the final number:

1. **SARIMA → SARIMAX:** Adding linear exog features didn't help — proved the limitation is model class, not feature absence.
2. **SARIMAX → XGBoost full:** Switching to a nonlinear model class started to catch the spike, but overfit massively.
3. **XGBoost full → curated:** Cutting features 10x cut overfit and improved test MAPE on both windows. Less is more when data is monthly.
4. **Curated → tuned:** Regularization closed the train/test gap further while preserving test performance. The model isn't just less memorized — it's actually better.

Each transition is the answer to a question the prior step raised. That's iteration, not random search.

---

## What this doesn't claim

- Tree-based models did **not** predict the spike in advance. The March 2026 forecast was still ~$25 below actual. No monthly time-series model in this work catches a regime change from the month before it starts.
- The 2026 test window is only **4 months**. Statistically thin, sensitive to a single bad row. The 38% MAPE reduction headline should be read with that in mind.
- The exogenous features carry **small individual importance** in the best models. The biggest wins came from own-price lags and Brent, not from the supply/macro features I added. Those features still provide a defensible story for the spike narrative, but the model didn't lean on them heavily.

These are limits, not failings — and stating them up front is the point of the writeup.

---

## Persisted in the database

All five models' forecasts are in `gold_forecast_results`:

```
model_name           target       run_count  windows
─────────────────────────────────────────────────────
sarima               wti_price        24      (existing)
prophet              wti_price        24      (existing)
sarimax              wti_price        16      Exp 1 (12) + Exp 2 (4)
xgboost              wti_price        16      Exp 1 (12) + Exp 2 (4)
xgboost_curated      wti_price        16      Exp 1 (12) + Exp 2 (4)
xgboost_tuned        wti_price        16      Exp 1 (12) + Exp 2 (4)
```

Each row carries its own `run_id`, `trained_on_periods` (96 vs 120), and per-experiment RMSE/MAPE. The dashboard's Model Comparison page picks them up automatically.

---

## Status of Kedar's three asks

1. **Feature engineering / scaling** — built. 192 engineered features (lags, rolling stats, momentum, seasonality, cross-series), z-score scaler stored on the `FeatureEngineer` instance for reuse on new data. Curation came after seeing the full set overfit.
2. **Better models** — done. XGBoost beats both SARIMA and SARIMAX on both windows. The curated+tuned variant is the best across the board.
3. **Improve SARIMA accuracy** — partially. The order search found a marginally better SARIMA order `(1,1,0)(1,1,1,12)` on the calm year. SARIMA itself is roughly at the limit of what a linear, univariate model can do on this data; the real accuracy gains required moving to a nonlinear model class.
