# oil_gas_pipeline | scripts/23_xgboost_hh_strict.py
# Tests TWO variants of XGBoost on Henry Hub to quantify the rolling-feature gap:
#
#   VARIANT A (current):  uses pre-engineered rolling features from the full CSV.
#                         Roll features at test time use sequential ACTUAL prices.
#                         This is fine for single-step-ahead but not strict multi-step.
#
#   VARIANT B (strict):   rolling-origin one-step-ahead forecast. At each month t,
#                         features are recomputed using only data through t-1.
#                         No sequential actuals bleed into test-window features.
#                         This is the fair multi-step comparison against SARIMA.
#
# The gap between A and B tells us how much the result was real vs roll-feature bleed.
# Run with: python3 scripts/23_xgboost_hh_strict.py

import sys, os, uuid, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

load_dotenv()
DB = dict(host=os.getenv("DB_HOST","localhost"), port=int(os.getenv("DB_PORT","5432")),
          dbname=os.getenv("DB_NAME","oil_gas_db"), user=os.getenv("DB_USER"),
          password=os.getenv("DB_PASSWORD"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "henry_hub_price"

HH_CURATED = [
    "henry_hub_price_lag_1", "henry_hub_price_lag_3",
    "henry_hub_price_roll3_mean", "henry_hub_price_roll6_mean",
    "henry_hub_price_mom_pct",
    "hdd_lag_1", "hdd_roll3_mean",
    "cdd_lag_1",
    "gas_storage_lag_1", "storage_vs_12mo_avg",
    "gas_production_lag_1",
    "wti_price_lag_1",
    "industrial_production_lag_1",
    "month_sin", "month_cos",
]

LAGS   = [1, 3, 6, 12]
ROLLS  = [3, 6, 12]


def metrics(actual, predicted):
    mask = (~pd.isna(actual)) & (~pd.isna(predicted))
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    if len(a) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae":  float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
    }


def load_raw():
    """Load the RAW gold_features (not engineered CSV) for strict rolling computation."""
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM gold_features ORDER BY period", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period").sort_index()
    return df


def load_engineered():
    """Load pre-engineered CSV for the current (non-strict) variant."""
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_raw.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql(f"SELECT period, {TARGET} FROM gold_features ORDER BY period",
                       conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df[TARGET] = gold[TARGET]
    return df


def build_feature_row_strict(history_df, period):
    """
    Build ONE feature row for `period` using only data available up to period-1.
    This is what we'd actually have at forecast time in production.

    `history_df` = the raw gold_features slice through the month BEFORE `period`.
    """
    row = {}
    tgt = history_df[TARGET].dropna()

    # Own-price lags and rolling (from past prices only)
    for lag in LAGS:
        row[f"henry_hub_price_lag_{lag}"] = tgt.iloc[-lag] if len(tgt) >= lag else np.nan
    for w in ROLLS:
        row[f"henry_hub_price_roll{w}_mean"] = tgt.iloc[-w:].mean() if len(tgt) >= w else np.nan
    # mom_pct = (last - second_last) / second_last * 100
    row["henry_hub_price_mom_pct"] = (
        (tgt.iloc[-1] - tgt.iloc[-2]) / tgt.iloc[-2] * 100
        if len(tgt) >= 2 else np.nan
    )

    # Exogenous features — all safely lag_1 (value from month before period)
    prev = history_df.iloc[-1]   # last row in history = month before `period`
    row["hdd_lag_1"]              = prev.get("hdd", np.nan)
    row["hdd_roll3_mean"]         = history_df["hdd"].dropna().iloc[-3:].mean() if "hdd" in history_df.columns else np.nan
    row["cdd_lag_1"]              = prev.get("cdd", np.nan)
    row["gas_storage_lag_1"]      = prev.get("gas_storage", np.nan)
    row["gas_production_lag_1"]   = prev.get("gas_production", np.nan)
    row["wti_price_lag_1"]        = prev.get("wti_price", np.nan)
    row["industrial_production_lag_1"] = prev.get("industrial_production", np.nan)

    # storage_vs_12mo_avg = gas_storage - its own 12-month rolling mean
    gs = history_df["gas_storage"].dropna()
    if len(gs) >= 12:
        row["storage_vs_12mo_avg"] = gs.iloc[-1] - gs.iloc[-12:].mean()
    else:
        row["storage_vs_12mo_avg"] = np.nan

    # Seasonality — always known (it's just the calendar month)
    row["month_sin"] = np.sin(2 * np.pi * period.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * period.month / 12)

    return pd.Series(row)


def variant_a_current(df_eng, train_start, train_end, test_start, test_end):
    """Current approach — pre-engineered CSV, test rows use sequential actuals."""
    train = df_eng.loc[train_start:train_end].dropna(subset=HH_CURATED + [TARGET])
    test  = df_eng.loc[test_start:test_end].dropna(subset=HH_CURATED + [TARGET])
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(train[HH_CURATED], train[TARGET])
    preds = model.predict(test[HH_CURATED])
    return test[TARGET], preds


def variant_b_strict(df_raw, df_eng, train_start, train_end, test_start, test_end):
    """
    Strict rolling-origin forecast. Train once on training data.
    For each test month, rebuild features from scratch using only data through month-1.
    NO sequential actuals from the test window bleed into feature computation.
    """
    # Train on engineered training data (same as variant A)
    train_eng = df_eng.loc[train_start:train_end].dropna(subset=HH_CURATED + [TARGET])
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(train_eng[HH_CURATED], train_eng[TARGET])

    test_periods = pd.date_range(test_start, test_end, freq="MS")
    actuals, preds = [], []

    for period in test_periods:
        # History = everything up to (but NOT including) this period
        history = df_raw.loc[:period - pd.offsets.MonthBegin(1)]
        if history.empty:
            continue
        # Actual target for this period
        if period not in df_raw.index or pd.isna(df_raw.loc[period, TARGET]):
            continue
        actual = df_raw.loc[period, TARGET]

        # Build feature row using only past data
        feat_row = build_feature_row_strict(history, period)
        if feat_row[HH_CURATED].isna().any():
            continue   # skip rows we can't fully construct

        pred = model.predict(feat_row[HH_CURATED].values.reshape(1, -1))[0]
        actuals.append((period, actual))
        preds.append((period, pred))

    actual_series = pd.Series([v for _, v in actuals],
                               index=pd.DatetimeIndex([p for p, _ in actuals]))
    pred_series   = pd.Series([v for _, v in preds],
                               index=pd.DatetimeIndex([p for p, _ in preds]))
    return actual_series, pred_series.values


def persist_strict(test_index, predictions, m, n_train):
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
        (run_id, TARGET, "xgboost_hh_strict", period.date(), float(pred),
         None, None, m["rmse"], m["mape"], n_train)
        for period, pred in zip(test_index, predictions)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def run_comparison(df_raw, df_eng, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 78}")
    print(label)
    print('-' * 78)

    print("Running Variant A (current — pre-engineered features)...")
    y_a, pred_a = variant_a_current(df_eng, train_start, train_end, test_start, test_end)
    m_a = metrics(y_a.values, pred_a)
    print(f"  VARIANT A (current):  MAPE {m_a['mape']:.2f}%  RMSE {m_a['rmse']:.4f}")

    print("Running Variant B (strict rolling-origin)...")
    y_b, pred_b = variant_b_strict(df_raw, df_eng, train_start, train_end, test_start, test_end)
    m_b = metrics(y_b.values, pred_b)
    print(f"  VARIANT B (strict):   MAPE {m_b['mape']:.2f}%  RMSE {m_b['rmse']:.4f}")

    gap = m_b["mape"] - m_a["mape"]
    print(f"\n  Gap (B - A): {gap:+.2f} pp MAPE")
    if gap < 2:
        verdict = "✅ Gap is small — the current result is mostly real signal"
    elif gap < 6:
        verdict = "⚠️  Moderate gap — rolling-feature bleed explains some of the current result"
    else:
        verdict = "❌ Large gap — current result is significantly inflated by roll-feature bleed"
    print(f"  Verdict: {verdict}")

    # Row-by-row comparison
    comp = pd.DataFrame({
        "actual":     y_b.round(3),
        "variant_a":  np.round(pred_a[:len(pred_b)], 3),
        "variant_b":  np.round(pred_b, 3),
        "err_a":      np.round(pred_a[:len(pred_b)] - y_b.values, 3),
        "err_b":      np.round(pred_b - y_b.values, 3),
    })
    print(f"\nRow-by-row (A=current, B=strict):")
    print(comp.to_string())

    return y_b, pred_b, m_b, len(df_eng.loc[train_start:train_end].dropna(subset=[TARGET]))


def main():
    df_raw = load_raw()
    df_eng = load_engineered()

    print("=" * 78)
    print("XGBOOST on HENRY HUB — Variant A (current) vs Variant B (strict rolling-origin)")
    print("=" * 78)

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost_hh_strict'")
    n_del = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_del} prior xgboost_hh_strict rows)")

    # Window 1: calm 2024
    y1, p1, m1, n1 = run_comparison(
        df_raw, df_eng,
        "2016-01-01", "2023-12-01",
        "2024-01-01", "2024-12-01",
        "EXPERIMENT 1 — 2024 (calm year)",
    )
    n_w1 = persist_strict(y1.index, p1, m1, n1)
    print(f"  → persisted {n_w1} rows as xgboost_hh_strict")

    # Window 2: heating cycle
    y2, p2, m2, n2 = run_comparison(
        df_raw, df_eng,
        "2016-01-01", "2024-09-01",
        "2024-10-01", "2025-09-01",
        "EXPERIMENT 2 — Oct 2024 → Sep 2025 (heating cycle)",
    )
    n_w2 = persist_strict(y2.index, p2, m2, n2)
    print(f"  → persisted {n_w2} rows as xgboost_hh_strict")

    print("\n" + "=" * 78)
    print("FINAL HEAD-TO-HEAD — ALL HENRY HUB MODELS")
    print("=" * 78)
    conn = psycopg2.connect(**DB)
    cmp = pd.read_sql("""
        SELECT model_name, trained_on_periods,
               ROUND(MIN(rmse)::numeric, 4) AS rmse,
               ROUND(MIN(mape)::numeric, 2) AS mape
        FROM gold_forecast_results
        WHERE target = 'henry_hub_price'
        GROUP BY model_name, trained_on_periods
        ORDER BY trained_on_periods, mape
    """, conn)
    conn.close()
    print(cmp.to_string(index=False))
    print("=" * 78)
    print("\nKey: sarima_hh=baseline, sarimax_hh=linear+exog,")
    print("     xgboost_hh_curated=current(A), xgboost_hh_strict=strict(B)")


if __name__ == "__main__":
    main()
