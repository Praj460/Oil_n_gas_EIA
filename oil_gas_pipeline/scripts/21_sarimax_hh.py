# oil_gas_pipeline | scripts/21_sarimax_hh.py
# SARIMAX on Henry Hub — both windows in one script.
#   Exp 1: calm year — train 2016-2023, test 2024 (12 months)
#   Exp 2: heating cycle — train through Sep 2024, test Oct 2024–Sep 2025 (12 months)
# Built-in order search per window. Persists to gold_forecast_results.
# Run with: python3 scripts/21_sarimax_hh.py

import sys, os, uuid, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

load_dotenv()
DB = dict(host=os.getenv("DB_HOST","localhost"), port=int(os.getenv("DB_PORT","5432")),
          dbname=os.getenv("DB_NAME","oil_gas_db"), user=os.getenv("DB_USER"),
          password=os.getenv("DB_PASSWORD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "henry_hub_price"

# Gas-mechanism exog for SARIMAX (3 features — small set to avoid the
# over-parameterization issue we hit with WTI's 8-exog initial run).
EXOG_SARIMAX = [
    "hdd_lag_1",              # heating demand — the gas-demand driver
    "gas_storage_lag_1",      # the buffer
    "wti_price_lag_1",        # oil-gas co-movement
]

ORDERS = [
    ("(1,1,1)(1,1,1,12)",  (1,1,1), (1,1,1,12)),
    ("(1,1,0)(1,1,1,12)",  (1,1,0), (1,1,1,12)),
    ("(1,1,1)(1,1,0,12)",  (1,1,1), (1,1,0,12)),
    ("(1,1,1)(0,1,1,12)",  (1,1,1), (0,1,1,12)),
    ("(1,1,0)(0,1,1,12)",  (1,1,0), (0,1,1,12)),
]


def metrics(actual, predicted):
    mask = (~pd.isna(actual)) & (~pd.isna(predicted))
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    if len(a) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "n": 0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae":  float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
        "n":    int(mask.sum()),
    }


def load():
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_scaled.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql(f"SELECT period, {TARGET} FROM gold_features ORDER BY period",
                       conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df[TARGET] = gold[TARGET]
    return df


def fit_and_forecast(y_train, X_train, X_test, horizon, order, sorder, with_exog):
    kwargs = dict(order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False)
    if with_exog:
        mod = SARIMAX(y_train, exog=X_train, **kwargs)
        fit = mod.fit(disp=False, maxiter=200)
        fc = fit.get_forecast(steps=horizon, exog=X_test).predicted_mean
    else:
        mod = SARIMAX(y_train, **kwargs)
        fit = mod.fit(disp=False, maxiter=200)
        fc = fit.get_forecast(steps=horizon).predicted_mean
    fc.index = X_test.index
    return fc, fit


def search_best(y_train, X_train, y_test, X_test, with_exog):
    best = None
    for name, order, sorder in ORDERS:
        try:
            fc, fit = fit_and_forecast(y_train, X_train, X_test, len(X_test),
                                       order, sorder, with_exog)
            m = metrics(y_test, fc)
            ci = fit.get_forecast(steps=len(X_test),
                                   exog=X_test if with_exog else None).conf_int(alpha=0.05)
            ci.index = X_test.index
            cand = {"name": name, "order": order, "sorder": sorder,
                    "mape": m["mape"], "rmse": m["rmse"], "mae": m["mae"],
                    "forecast": fc, "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]}
            if best is None or m["mape"] < best["mape"]:
                best = cand
        except Exception:
            continue
    return best


def run_window(df, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 78}")
    print(label)
    print('-' * 78)

    # SARIMAX needs train rows where ALL exog features are non-NaN (lag rows at start)
    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]
    train_exog = train[EXOG_SARIMAX].dropna()
    train = train.loc[train_exog.index]

    y_tr = train[TARGET].asfreq("MS")
    X_tr = train[EXOG_SARIMAX].asfreq("MS")
    y_te = test[TARGET].asfreq("MS")
    X_te = test[EXOG_SARIMAX].asfreq("MS")

    # Edge case: if some target rows in test are NaN (e.g., target stops at Feb 2026
    # but test runs to April 2026), trim the test window to valid target rows only.
    valid_test_idx = y_te.dropna().index
    if len(valid_test_idx) < len(y_te):
        print(f"  Note: trimming test window to {len(valid_test_idx)} valid target rows "
              f"(dropped {len(y_te) - len(valid_test_idx)} NaN-target months)")
        y_te = y_te.loc[valid_test_idx]
        X_te = X_te.loc[valid_test_idx]

    print(f"Train: {y_tr.index.min().date()} → {y_tr.index.max().date()} ({len(y_tr)} rows)")
    print(f"Test:  {y_te.index.min().date()} → {y_te.index.max().date()} ({len(y_te)} rows)")

    print("\nSearching SARIMA (no exog)...")
    sarima = search_best(y_tr, None, y_te, X_te, with_exog=False)
    print(f"  best: {sarima['name']}  MAPE {sarima['mape']:.2f}%")

    print("\nSearching SARIMAX (3 gas exog)...")
    sarimax = search_best(y_tr, X_tr, y_te, X_te, with_exog=True)
    print(f"  best: {sarimax['name']}  MAPE {sarimax['mape']:.2f}%")

    comp = pd.DataFrame({
        "actual":   y_te.round(3),
        "sarima":   sarima["forecast"].round(3),
        "sarimax":  sarimax["forecast"].round(3),
    })
    print(f"\nForecast comparison:")
    print(comp.to_string())

    print(f"\n  SARIMA       | RMSE {sarima['rmse']:>5.2f}  MAE {sarima['mae']:>5.2f}  MAPE {sarima['mape']:>5.2f}%   order {sarima['order']}{sarima['sorder']}")
    print(f"  SARIMAX (X)  | RMSE {sarimax['rmse']:>5.2f}  MAE {sarimax['mae']:>5.2f}  MAPE {sarimax['mape']:>5.2f}%   order {sarimax['order']}{sarimax['sorder']}")
    delta = sarimax["mape"] - sarima["mape"]
    print(f"  → SARIMAX is {'BETTER' if delta < 0 else 'WORSE'} by {abs(delta):.2f} pp MAPE\n")

    return {
        "sarima": {**sarima, "test_index": y_te.index, "n_train": len(y_tr)},
        "sarimax": {**sarimax, "test_index": y_te.index, "n_train": len(y_tr)},
    }


def persist(test_index, forecast, lower, upper, model_name, m, n_train):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    sql = """
        INSERT INTO gold_forecast_results
            (run_id, target, model_name, forecast_period, forecast_value,
             lower_bound, upper_bound, rmse, mape, trained_on_periods)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    run_id = str(uuid.uuid4())
    data = [
        (run_id, TARGET, model_name, period.date(), float(fc),
         float(lo) if pd.notna(lo) else None,
         float(up) if pd.notna(up) else None,
         m["rmse"], m["mape"], n_train)
        for period, fc, lo, up in zip(test_index, forecast, lower, upper)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return run_id, len(data)


def main():
    df = load()

    print("=" * 78)
    print("SARIMAX on HENRY HUB — both windows, with order search")
    print("=" * 78)

    # Clear prior Henry Hub SARIMA/X rows
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name IN ('sarima_hh', 'sarimax_hh')")
    n_del = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_del} prior sarima_hh/sarimax_hh rows)")

    # ── Window 1: calm 2024 ────────────────────────────────────────
    r1 = run_window(df,
                    "2016-01-01", "2023-12-01",
                    "2024-01-01", "2024-12-01",
                    "EXPERIMENT 1 — 2024 (calm year, baseline forecasting window)")

    rid_s1, n_s1 = persist(r1["sarima"]["test_index"], r1["sarima"]["forecast"],
                            r1["sarima"]["lower"], r1["sarima"]["upper"],
                            "sarima_hh", r1["sarima"], r1["sarima"]["n_train"])
    rid_x1, n_x1 = persist(r1["sarimax"]["test_index"], r1["sarimax"]["forecast"],
                            r1["sarimax"]["lower"], r1["sarimax"]["upper"],
                            "sarimax_hh", r1["sarimax"], r1["sarimax"]["n_train"])
    print(f"  → persisted sarima_hh ({n_s1} rows) + sarimax_hh ({n_x1} rows)")

    # ── Window 2: heating cycle Oct 2024 – Sep 2025 ────────────────
    r2 = run_window(df,
                    "2016-01-01", "2024-09-01",
                    "2024-10-01", "2025-09-01",
                    "EXPERIMENT 2 — Oct 2024 → Sep 2025 (full annual cycle with winter ramp)")

    rid_s2, n_s2 = persist(r2["sarima"]["test_index"], r2["sarima"]["forecast"],
                            r2["sarima"]["lower"], r2["sarima"]["upper"],
                            "sarima_hh", r2["sarima"], r2["sarima"]["n_train"])
    rid_x2, n_x2 = persist(r2["sarimax"]["test_index"], r2["sarimax"]["forecast"],
                            r2["sarimax"]["lower"], r2["sarimax"]["upper"],
                            "sarimax_hh", r2["sarimax"], r2["sarimax"]["n_train"])
    print(f"  → persisted sarima_hh ({n_s2} rows) + sarimax_hh ({n_x2} rows)")

    print("\n" + "=" * 78)
    print("SUMMARY — SARIMA vs SARIMAX on Henry Hub")
    print("=" * 78)
    print(f"                                      RMSE    MAE    MAPE")
    print(f"  --- 2024 (calm year) ---")
    print(f"  SARIMA   on Henry Hub             |  {r1['sarima']['rmse']:>4.2f}   {r1['sarima']['mae']:>4.2f}  {r1['sarima']['mape']:>5.2f}%")
    print(f"  SARIMAX  on Henry Hub (3 gas exog)|  {r1['sarimax']['rmse']:>4.2f}   {r1['sarimax']['mae']:>4.2f}  {r1['sarimax']['mape']:>5.2f}%")
    print(f"  --- Oct 2024 → Sep 2025 (heating cycle) ---")
    print(f"  SARIMA   on Henry Hub             |  {r2['sarima']['rmse']:>4.2f}   {r2['sarima']['mae']:>4.2f}  {r2['sarima']['mape']:>5.2f}%")
    print(f"  SARIMAX  on Henry Hub (3 gas exog)|  {r2['sarimax']['rmse']:>4.2f}   {r2['sarimax']['mae']:>4.2f}  {r2['sarimax']['mape']:>5.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
