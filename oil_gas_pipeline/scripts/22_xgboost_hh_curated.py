# oil_gas_pipeline | scripts/22_xgboost_hh_curated.py
# XGBoost on Henry Hub — 15 curated gas-mechanism features. Defaults, no tuning.
# Two windows:
#   Exp 1: calm 2024 (train 2016-2023, test 2024)
#   Exp 2: heating cycle (train through Sep 2024, test Oct 2024 – Sep 2025)
# Persisted as model_name='xgboost_hh_curated'.

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

# 15 features — gas-mechanism focused, mirrors WTI curated set structure
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


def metrics(actual, predicted):
    mask = (~pd.isna(actual)) & (~pd.isna(predicted))
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "mae":  float(mean_absolute_error(a, p)),
        "mape": float(np.mean(np.abs((a - p) / a)) * 100),
        "n":    int(mask.sum()),
    }


def load():
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_raw.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql(f"SELECT period, {TARGET} FROM gold_features ORDER BY period",
                       conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df[TARGET] = gold[TARGET]
    return df


def run_window(df, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 78}")
    print(label)
    print('-' * 78)

    missing = [c for c in HH_CURATED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train = df.loc[train_start:train_end].dropna(subset=HH_CURATED + [TARGET])
    test  = df.loc[test_start:test_end].dropna(subset=HH_CURATED + [TARGET])

    print(f"Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} rows)")
    print(f"Test:  {test.index.min().date()} → {test.index.max().date()}  ({len(test)} rows)")
    print(f"Features in play: {len(HH_CURATED)} (curated gas-mechanism set)")

    X_tr, y_tr = train[HH_CURATED], train[TARGET]
    X_te, y_te = test[HH_CURATED], test[TARGET]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)
    m_tr, m_te = metrics(y_tr.values, pred_tr), metrics(y_te.values, pred_te)

    print(f"\nTRAIN  | RMSE {m_tr['rmse']:>5.2f}   MAE {m_tr['mae']:>4.2f}   MAPE {m_tr['mape']:>5.2f}%   (overfit check)")
    print(f"TEST   | RMSE {m_te['rmse']:>5.2f}   MAE {m_te['mae']:>4.2f}   MAPE {m_te['mape']:>5.2f}%")
    print(f"         train→test MAPE gap: {(m_te['mape'] - m_tr['mape']):+.2f} pp")

    comp = pd.DataFrame({
        "actual":   y_te.round(3),
        "xgb_pred": np.round(pred_te, 3),
        "err":      np.round(pred_te - y_te.values, 3),
    })
    print(f"\nRow-by-row predictions:")
    print(comp.to_string())

    importances = pd.Series(model.feature_importances_, index=HH_CURATED).sort_values(ascending=False)
    print(f"\nFeature importances (top 10):")
    for f, imp in importances.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"  {f:36s} {imp:.4f} {bar}")

    return {
        "test_metrics": m_te, "train_metrics": m_tr,
        "predictions": pred_te, "test_index": y_te.index,
        "n_train": len(X_tr), "importances": importances,
    }


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
        (run_id, TARGET, "xgboost_hh_curated", period.date(), float(pred),
         None, None, m["rmse"], m["mape"], n_train)
        for period, pred in zip(test_index, predictions)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def main():
    df = load()
    print("=" * 78)
    print("XGBOOST CURATED on HENRY HUB — 15 gas-mechanism features")
    print("=" * 78)

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost_hh_curated'")
    n_del = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_del} prior xgboost_hh_curated rows)")

    r1 = run_window(df, "2016-01-01", "2023-12-01", "2024-01-01", "2024-12-01",
                    "EXPERIMENT 1 — 2024 (calm year)")
    n1 = persist(r1["test_index"], r1["predictions"], r1["test_metrics"], r1["n_train"])
    print(f"\n  → persisted {n1} rows")

    r2 = run_window(df, "2016-01-01", "2024-09-01", "2024-10-01", "2025-09-01",
                    "EXPERIMENT 2 — Oct 2024 → Sep 2025 (heating cycle)")
    n2 = persist(r2["test_index"], r2["predictions"], r2["test_metrics"], r2["n_train"])
    print(f"\n  → persisted {n2} rows")

    print("\n" + "=" * 78)
    print("HENRY HUB — HEAD-TO-HEAD across SARIMA, SARIMAX, XGBoost (curated)")
    print("=" * 78)
    # Pull SARIMA/SARIMAX results from DB for comparison
    conn = psycopg2.connect(**DB)
    cmp_df = pd.read_sql("""
        SELECT model_name, trained_on_periods, MIN(forecast_period) AS first_p,
               MIN(rmse) AS rmse, MIN(mape) AS mape
        FROM gold_forecast_results
        WHERE target = 'henry_hub_price'
          AND model_name IN ('sarima_hh', 'sarimax_hh', 'xgboost_hh_curated')
        GROUP BY model_name, trained_on_periods
        ORDER BY trained_on_periods, model_name
    """, conn)
    conn.close()
    print(cmp_df.to_string(index=False))
    print("=" * 78)


if __name__ == "__main__":
    main()
