# oil_gas_pipeline | scripts/16_sarimax_spike_test.py
# THE REAL TEST: does SARIMAX (with OPEC spare capacity awareness) handle the
# early-2026 oil price spike better than plain SARIMA?
#
# Train:  2016-01 → 2025-12  (120 months — includes everything BEFORE the spike)
# Test:   2026-01 → 2026-04  (4 months — the spike window)
#
# Both models are order-searched on the new train, then compared head-to-head.
# Run with: python3 scripts/16_sarimax_spike_test.py

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

load_dotenv()
DB = dict(host=os.getenv("DB_HOST","localhost"), port=int(os.getenv("DB_PORT","5432")),
          dbname=os.getenv("DB_NAME","oil_gas_db"), user=os.getenv("DB_USER"),
          password=os.getenv("DB_PASSWORD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXOG = ["industrial_production_lag_1", "opec_spare_lag_1", "dollar_index_lag_1"]

ORDERS = [
    ("(1,1,1)(1,1,1,12)",  (1,1,1), (1,1,1,12)),
    ("(1,1,0)(1,1,1,12)",  (1,1,0), (1,1,1,12)),
    ("(1,1,1)(1,1,0,12)",  (1,1,1), (1,1,0,12)),
    ("(1,1,1)(0,1,1,12)",  (1,1,1), (0,1,1,12)),
    ("(1,1,0)(0,1,1,12)",  (1,1,0), (0,1,1,12)),
]


def metrics(actual, predicted):
    mask = (~actual.isna()) & (~predicted.isna())
    a, p = actual[mask].values, predicted[mask].values
    if len(a) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "n": 0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae":  float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
        "n":    int(mask.sum()),
    }


def fit_and_forecast(y_train, X_train, X_test, order, sorder, with_exog):
    """Fit one model, return forecast Series + AIC."""
    kwargs = dict(order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False)
    if with_exog:
        mod = SARIMAX(y_train, exog=X_train, **kwargs)
    else:
        mod = SARIMAX(y_train, **kwargs)
    fit = mod.fit(disp=False, maxiter=200)
    if with_exog:
        fc = fit.get_forecast(steps=len(X_test), exog=X_test).predicted_mean
    else:
        fc = fit.get_forecast(steps=len(X_test)).predicted_mean
    fc.index = X_test.index
    return fc, fit.aic


def search_best_order(y_train, X_train, y_test_actual, X_test, with_exog):
    """Try each order, return the (name, order, sorder) with lowest MAPE."""
    best = None
    for name, order, sorder in ORDERS:
        try:
            fc, aic = fit_and_forecast(y_train, X_train, X_test, order, sorder, with_exog)
            m = metrics(y_test_actual, fc)
            cand = {"name": name, "order": order, "sorder": sorder,
                    "mape": m["mape"], "rmse": m["rmse"], "mae": m["mae"],
                    "aic": aic, "forecast": fc}
            if best is None or m["mape"] < best["mape"]:
                best = cand
        except Exception:
            continue
    return best


def main():
    # ── Load data ─────────────────────────────────────────────────
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_scaled.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]

    # ── Slice — train through end of 2025, test the spike (Jan-Apr 2026) ──
    train = df.loc["2016-01-01":"2025-12-01"]
    test  = df.loc["2026-01-01":"2026-04-01"]

    y_tr = train["wti_price"].asfreq("MS")
    X_tr = train[EXOG].asfreq("MS").dropna()
    y_tr = y_tr.loc[X_tr.index]
    y_te = test["wti_price"].asfreq("MS")
    X_te = test[EXOG].asfreq("MS")

    print("\n" + "=" * 72)
    print("SPIKE TEST — Jan-Apr 2026 forecast (the regime change)")
    print("=" * 72)
    print(f"Train window: {y_tr.index.min().date()} → {y_tr.index.max().date()} ({len(y_tr)} rows)")
    print(f"Test  window: {y_te.index.min().date()} → {y_te.index.max().date()} ({len(y_te)} rows)")

    # ── Verify the exog features show the collapse in the test rows ────
    print(f"\nExog features in the test window (the leading-indicator signals):")
    print(X_te.round(3).to_string())
    print(f"\nActual WTI prices being forecast:")
    print(y_te.round(2).to_string())

    # ── Search best order for each model ───────────────────────────
    print("\nSearching SARIMA (no exog)...")
    best_sarima = search_best_order(y_tr, None, y_te, X_te, with_exog=False)
    print(f"  best: {best_sarima['name']}  MAPE {best_sarima['mape']:.2f}%")

    print("\nSearching SARIMAX (3 exog)...")
    best_sarimax = search_best_order(y_tr, X_tr, y_te, X_te, with_exog=True)
    print(f"  best: {best_sarimax['name']}  MAPE {best_sarimax['mape']:.2f}%")

    # ── Row-by-row comparison ──────────────────────────────────────
    comp = pd.DataFrame({
        "actual":   y_te,
        "sarima":   best_sarima["forecast"].round(2),
        "sarimax":  best_sarimax["forecast"].round(2),
    }).round(2)
    comp["sarima_err"]  = (comp["sarima"]  - comp["actual"]).round(2)
    comp["sarimax_err"] = (comp["sarimax"] - comp["actual"]).round(2)

    print("\n" + "=" * 72)
    print("FORECAST COMPARISON — Jan-Apr 2026")
    print("=" * 72)
    print(comp.to_string())

    # ── Head-to-head summary ───────────────────────────────────────
    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD ON THE SPIKE")
    print("=" * 72)
    print(f"                  RMSE     MAE    MAPE")
    print(f"  SARIMA       | {best_sarima['rmse']:>6.2f}  {best_sarima['mae']:>6.2f}  {best_sarima['mape']:>5.2f}%   order {best_sarima['order']}{best_sarima['sorder']}")
    print(f"  SARIMAX (X)  | {best_sarimax['rmse']:>6.2f}  {best_sarimax['mae']:>6.2f}  {best_sarimax['mape']:>5.2f}%   order {best_sarimax['order']}{best_sarimax['sorder']}")
    delta = best_sarimax["mape"] - best_sarima["mape"]
    direction = "BETTER" if delta < 0 else "WORSE"
    print(f"\n  → SARIMAX is {direction} by {abs(delta):.2f} pp MAPE vs SARIMA on the spike window")
    print("=" * 72)


if __name__ == "__main__":
    main()
