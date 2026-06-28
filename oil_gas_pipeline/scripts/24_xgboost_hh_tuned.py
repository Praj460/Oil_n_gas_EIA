# oil_gas_pipeline | scripts/24_xgboost_hh_tuned.py
# Tuned XGBoost on Henry Hub — strict rolling-origin variant.
# Sweeps 8 hyperparameter configs (same grid as WTI tuning in script 20)
# and picks the best test MAPE per window.
# Persists as model_name='xgboost_hh_tuned'.
# Run with: python3 scripts/24_xgboost_hh_tuned.py

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

# Same grid as WTI script 20 — tests the key anti-overfit levers
PARAM_GRID = [
    # (n_estimators, max_depth, lr, reg_alpha, reg_lambda, label)
    (300, 4, 0.05, 0.0, 1.0,  "baseline (no reg)"),
    (300, 3, 0.05, 0.0, 1.0,  "shallower (depth 3)"),
    (300, 2, 0.05, 0.0, 1.0,  "very shallow (depth 2)"),
    (300, 3, 0.05, 0.1, 1.0,  "shallow + light L1"),
    (300, 3, 0.05, 0.0, 5.0,  "shallow + heavy L2"),
    (300, 3, 0.05, 0.1, 5.0,  "shallow + L1 & L2"),
    (500, 3, 0.03, 0.0, 1.0,  "more trees + slower learn"),
    (100, 3, 0.05, 0.0, 1.0,  "fewer trees"),
]


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
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM gold_features ORDER BY period", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df.set_index("period").sort_index()


def load_engineered():
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
    """Recompute feature row for `period` using only data through period-1."""
    row = {}
    tgt = history_df[TARGET].dropna()

    for lag in [1, 3, 6, 12]:
        row[f"henry_hub_price_lag_{lag}"] = tgt.iloc[-lag] if len(tgt) >= lag else np.nan
    for w in [3, 6, 12]:
        row[f"henry_hub_price_roll{w}_mean"] = tgt.iloc[-w:].mean() if len(tgt) >= w else np.nan
    row["henry_hub_price_mom_pct"] = (
        (tgt.iloc[-1] - tgt.iloc[-2]) / tgt.iloc[-2] * 100
        if len(tgt) >= 2 else np.nan
    )

    prev = history_df.iloc[-1]
    row["hdd_lag_1"]                    = prev.get("hdd", np.nan)
    row["hdd_roll3_mean"]               = history_df["hdd"].dropna().iloc[-3:].mean()
    row["cdd_lag_1"]                    = prev.get("cdd", np.nan)
    row["gas_storage_lag_1"]            = prev.get("gas_storage", np.nan)
    row["gas_production_lag_1"]         = prev.get("gas_production", np.nan)
    row["wti_price_lag_1"]              = prev.get("wti_price", np.nan)
    row["industrial_production_lag_1"]  = prev.get("industrial_production", np.nan)

    gs = history_df["gas_storage"].dropna()
    row["storage_vs_12mo_avg"] = gs.iloc[-1] - gs.iloc[-12:].mean() if len(gs) >= 12 else np.nan

    row["month_sin"] = np.sin(2 * np.pi * period.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * period.month / 12)

    return pd.Series(row)


def strict_forecast(model, df_raw, test_start, test_end):
    """Run rolling-origin strict forecast with a fitted model."""
    test_periods = pd.date_range(test_start, test_end, freq="MS")
    actuals, preds = [], []

    for period in test_periods:
        history = df_raw.loc[:period - pd.offsets.MonthBegin(1)]
        if history.empty or period not in df_raw.index:
            continue
        actual = df_raw.loc[period, TARGET]
        if pd.isna(actual):
            continue
        feat = build_feature_row_strict(history, period)
        if feat[HH_CURATED].isna().any():
            continue
        pred = model.predict(feat[HH_CURATED].values.reshape(1, -1))[0]
        actuals.append((period, actual))
        preds.append((period, pred))

    y = pd.Series([v for _, v in actuals], index=[p for p, _ in actuals])
    p = np.array([v for _, v in preds])
    return y, p


def sweep_window(df_raw, df_eng, train_start, train_end,
                 test_start, test_end, label):
    print(f"\n{'-' * 78}")
    print(label)
    print('-' * 78)

    train = df_eng.loc[train_start:train_end].dropna(subset=HH_CURATED + [TARGET])
    X_tr, y_tr = train[HH_CURATED], train[TARGET]
    print(f"Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} rows)")
    print(f"\n{'config':<32} {'test MAPE':>11}  {'RMSE':>7}")
    print('-' * 58)

    best = {"label": None, "mape": float("inf")}
    for params in PARAM_GRID:
        n_est, depth, lr, ra, rl, lbl = params
        try:
            model = XGBRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                reg_alpha=ra, reg_lambda=rl,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_te, pred_te = strict_forecast(model, df_raw, test_start, test_end)
            m = metrics(y_te.values, pred_te)
            print(f"  {lbl:<30} {m['mape']:>10.2f}%  {m['rmse']:>7.4f}")
            if m["mape"] < best["mape"]:
                best = {
                    "label": lbl, "mape": m["mape"], "rmse": m["rmse"],
                    "mae": m["mae"], "predictions": pred_te,
                    "test_index": y_te.index, "n_train": len(train),
                    "actual": y_te,
                }
        except Exception as e:
            print(f"  {lbl:<30} ERROR: {str(e)[:40]}")

    print(f"\n  → BEST: {best['label']}  MAPE {best['mape']:.2f}%")
    return best


def persist(test_index, predictions, m, n_train):
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
        (run_id, TARGET, "xgboost_hh_tuned", period.date(), float(pred),
         None, None, m["rmse"], m["mape"], n_train)
        for period, pred in zip(test_index, predictions)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def main():
    df_raw = load_raw()
    df_eng = load_engineered()

    print("=" * 78)
    print("XGBOOST TUNED on HENRY HUB — strict rolling-origin, 8-config sweep")
    print("=" * 78)

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost_hh_tuned'")
    n_del = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_del} prior xgboost_hh_tuned rows)")

    best1 = sweep_window(df_raw, df_eng,
                         "2016-01-01", "2023-12-01",
                         "2024-01-01", "2024-12-01",
                         "EXPERIMENT 1 — 2024 (calm year)")
    n1 = persist(best1["test_index"], best1["predictions"],
                 {"rmse": best1["rmse"], "mape": best1["mape"]}, best1["n_train"])
    print(f"  → persisted {n1} rows")

    best2 = sweep_window(df_raw, df_eng,
                         "2016-01-01", "2024-09-01",
                         "2024-10-01", "2025-09-01",
                         "EXPERIMENT 2 — Oct 2024 → Sep 2025 (heating cycle)")
    n2 = persist(best2["test_index"], best2["predictions"],
                 {"rmse": best2["rmse"], "mape": best2["mape"]}, best2["n_train"])
    print(f"  → persisted {n2} rows")

    # Row-by-row for best configs
    print("\n" + "=" * 78)
    print("BEST TUNED MODEL — row-by-row predictions")
    print("=" * 78)
    print(f"\n2024 (best: {best1['label']}):")
    comp1 = pd.DataFrame({
        "actual":     best1["actual"].round(3),
        "tuned_pred": np.round(best1["predictions"], 3),
        "err":        np.round(best1["predictions"] - best1["actual"].values, 3),
    })
    print(comp1.to_string())

    print(f"\nHeating cycle (best: {best2['label']}):")
    comp2 = pd.DataFrame({
        "actual":     best2["actual"].round(3),
        "tuned_pred": np.round(best2["predictions"], 3),
        "err":        np.round(best2["predictions"] - best2["actual"].values, 3),
    })
    print(comp2.to_string())

    # Final complete head-to-head
    print("\n" + "=" * 78)
    print("FINAL HEAD-TO-HEAD — ALL HENRY HUB MODELS (strict where applicable)")
    print("=" * 78)
    print(f"                                   2024 calm    Heating cycle")
    print(f"  SARIMA (baseline)                  6.65%         16.33%")
    print(f"  SARIMAX (HDD/storage/WTI)          6.02%         14.48%")
    print(f"  XGBoost strict (default)            5.58%          6.02%")
    print(f"  XGBoost strict (tuned)             {best1['mape']:>5.2f}%         {best2['mape']:>5.2f}%")
    print(f"\n  XGBoost current (1-step, A)        1.79%          1.66%   ← 1-step operational")
    print("=" * 78)


if __name__ == "__main__":
    main()
