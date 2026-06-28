# oil_gas_pipeline | scripts/17_persist_sarimax_results.py
# Saves the two SARIMAX experiment results to gold_forecast_results:
#   1. SARIMAX on 2024 (the 5.91% MAPE tie-with-baseline)
#   2. SARIMAX on Jan-Apr 2026 spike (the 18.29% miss)
# Both use the order-tuned config we identified, with 3 exog features.
#
# Run with: python3 scripts/17_persist_sarimax_results.py

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

EXOG = ["industrial_production_lag_1", "opec_spare_lag_1", "dollar_index_lag_1"]


def load_features():
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_scaled.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]
    return df


def fit_sarimax(y_train, X_train, X_test, order, sorder):
    """Fit SARIMAX, return (forecast_mean, lower, upper)."""
    mod = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False)
    fit = mod.fit(disp=False, maxiter=200)
    fo = fit.get_forecast(steps=len(X_test), exog=X_test)
    mean = fo.predicted_mean
    ci   = fo.conf_int(alpha=0.05)
    mean.index = X_test.index
    ci.index   = X_test.index
    return mean, ci.iloc[:, 0], ci.iloc[:, 1], len(y_train)


def metrics(actual, predicted):
    mask = (~actual.isna()) & (~predicted.isna())
    a, p = actual[mask].values, predicted[mask].values
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
    }


def persist(rows, target, model_name, run_id, m, train_periods):
    """Write forecast rows to gold_forecast_results."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    sql = """
        INSERT INTO gold_forecast_results
            (run_id, target, model_name, forecast_period, forecast_value,
             lower_bound, upper_bound, rmse, mape, trained_on_periods)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = [
        (run_id, target, model_name, period.date(),
         float(fc), float(lo), float(up),
         m["rmse"], m["mape"], train_periods)
        for period, fc, lo, up in rows
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def run_experiment(df, train_start, train_end, test_start, test_end,
                   order, sorder, label):
    """Fit on a window, forecast the test window, return everything needed to persist."""
    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]
    y_tr = train["wti_price"].asfreq("MS")
    X_tr = train[EXOG].asfreq("MS").dropna()
    y_tr = y_tr.loc[X_tr.index]
    y_te = test["wti_price"].asfreq("MS")
    X_te = test[EXOG].asfreq("MS")

    mean, lower, upper, n_train = fit_sarimax(y_tr, X_tr, X_te, order, sorder)
    m = metrics(y_te, mean)

    print(f"\n{label}")
    print(f"  Train: {y_tr.index.min().date()} → {y_tr.index.max().date()} ({n_train} rows)")
    print(f"  Test:  {y_te.index.min().date()} → {y_te.index.max().date()} ({len(y_te)} rows)")
    print(f"  Order: {order}{sorder}")
    print(f"  RMSE: {m['rmse']:.2f}   MAPE: {m['mape']:.2f}%")

    rows = list(zip(mean.index, mean.values, lower.values, upper.values))
    return rows, m, n_train


def main():
    df = load_features()

    print("=" * 60)
    print("PERSISTING SARIMAX EXPERIMENTS TO gold_forecast_results")
    print("=" * 60)

    # ── Clear prior SARIMAX rows so re-runs don't stack ─────────────
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'sarimax'")
    n_deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_deleted} prior SARIMAX rows)")

    # ── Experiment 1 — 2024 validation (calm year) ─────────────────
    rows1, m1, n1 = run_experiment(
        df,
        train_start="2016-01-01", train_end="2023-12-01",
        test_start="2024-01-01",  test_end="2024-12-01",
        order=(1,1,0), sorder=(1,1,1,12),
        label="EXPERIMENT 1 — 2024 validation (calm year)"
    )
    run_id_1 = str(uuid.uuid4())
    n_written_1 = persist(rows1, "wti_price", "sarimax", run_id_1, m1, n1)
    print(f"  → wrote {n_written_1} rows under run_id {run_id_1[:8]}…")

    # ── Experiment 2 — 2026 spike window ───────────────────────────
    rows2, m2, n2 = run_experiment(
        df,
        train_start="2016-01-01", train_end="2025-12-01",
        test_start="2026-01-01",  test_end="2026-04-01",
        order=(1,1,1), sorder=(1,1,1,12),
        label="EXPERIMENT 2 — Jan-Apr 2026 spike window"
    )
    run_id_2 = str(uuid.uuid4())
    n_written_2 = persist(rows2, "wti_price", "sarimax", run_id_2, m2, n2)
    print(f"  → wrote {n_written_2} rows under run_id {run_id_2[:8]}…")

    print("\n" + "=" * 60)
    print("SUMMARY (rows now in gold_forecast_results, model='sarimax')")
    print("=" * 60)
    print(f"  Exp 1 (2024 calm) :  MAPE {m1['mape']:.2f}%   {n_written_1} forecast months")
    print(f"  Exp 2 (2026 spike):  MAPE {m2['mape']:.2f}%   {n_written_2} forecast months")
    print("\nGlance at the new rows:")
    conn = psycopg2.connect(**DB)
    df_check = pd.read_sql("""
        SELECT model_name, target, forecast_period, forecast_value, mape, trained_on_periods
        FROM gold_forecast_results
        WHERE model_name = 'sarimax'
        ORDER BY trained_on_periods, forecast_period
    """, conn)
    conn.close()
    print(df_check.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
