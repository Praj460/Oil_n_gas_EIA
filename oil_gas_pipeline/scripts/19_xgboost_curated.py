# oil_gas_pipeline | scripts/19_xgboost_curated.py
# XGBoost with a CURATED ~18-feature subset chosen for economic story
# (vs 192 features in the prior run, which overfit hard).
# Same two test windows as before for fair comparison.
# Persisted as model_name='xgboost_curated'.

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
TARGET = "wti_price"

# Curated feature set — every column hand-picked for economic story, all lagged.
CURATED_FEATURES = [
    # Own-price history and momentum (most reliable autoregressive info)
    "wti_price_lag_1", "wti_price_lag_3",
    "wti_price_roll3_mean", "wti_price_roll6_mean",
    "wti_price_mom_pct",
    # Co-moving prices
    "brent_price_lag_1", "henry_hub_price_lag_1",
    # Oil supply
    "crude_imports_lag_1", "refinery_util_lag_1",
    "opec_spare_lag_1", "global_inv_lag_1",
    # Oil demand
    "gasoline_stocks_lag_1", "distillate_stocks_lag_1",
    "industrial_production_lag_1",
    # Macro / cost of carry
    "dollar_index_lag_1", "treasury_10y_lag_1",
    # Seasonality (the cyclical-encoded version, no jump from Dec to Jan)
    "month_sin", "month_cos",
    # Volatility regime hint — spare-capacity volatility jumped 25x in the spike
    "opec_spare_roll3_std",
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


def load_features():
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_raw.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period",
                       conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]
    return df


def run_experiment(df, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 72}")
    print(label)
    print('-' * 72)

    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]

    # Verify all curated features exist
    missing = [c for c in CURATED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing curated features: {missing}")

    # Drop training rows where any curated feature is NaN (early-lag rows)
    needed = CURATED_FEATURES + [TARGET]
    train_complete = train.dropna(subset=needed)
    print(f"Train: {train_complete.index.min().date()} → {train_complete.index.max().date()}  "
          f"({len(train_complete)}/{len(train)} usable rows)")
    print(f"Test:  {test.index.min().date()} → {test.index.max().date()}  ({len(test)} rows)")
    print(f"Features in play: {len(CURATED_FEATURES)} (curated)")

    X_train, y_train = train_complete[CURATED_FEATURES], train_complete[TARGET]
    X_test,  y_test  = test[CURATED_FEATURES],            test[TARGET]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    m_train = metrics(y_train.values, pred_train)
    m_test  = metrics(y_test.values,  pred_test)

    print(f"\nTRAIN  | RMSE {m_train['rmse']:>6.2f}   MAE {m_train['mae']:>5.2f}   MAPE {m_train['mape']:>5.2f}%   (overfit check)")
    print(f"TEST   | RMSE {m_test['rmse']:>6.2f}   MAE {m_test['mae']:>5.2f}   MAPE {m_test['mape']:>5.2f}%")
    print(f"         train→test MAPE gap: {(m_test['mape'] - m_train['mape']):+.2f} pp")

    comp = pd.DataFrame({
        "actual":   y_test.round(2),
        "xgb_pred": np.round(pred_test, 2),
        "xgb_err":  np.round(pred_test - y_test.values, 2),
    })
    print(f"\nRow-by-row predictions:")
    print(comp.to_string())

    importances = pd.Series(model.feature_importances_, index=CURATED_FEATURES).sort_values(ascending=False)
    print(f"\nFeature importances (all {len(CURATED_FEATURES)}, sorted):")
    for f, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"  {f:38s} {imp:.4f} {bar}")

    return {
        "test_metrics":  m_test,
        "train_metrics": m_train,
        "predictions":   pred_test,
        "test_index":    y_test.index,
        "n_train":       len(X_train),
    }


def persist(test_index, predictions, target, m, run_id, n_train):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    sql = """
        INSERT INTO gold_forecast_results
            (run_id, target, model_name, forecast_period, forecast_value,
             lower_bound, upper_bound, rmse, mape, trained_on_periods)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = [
        (run_id, target, "xgboost_curated", period.date(), float(pred),
         None, None, m["rmse"], m["mape"], n_train)
        for period, pred in zip(test_index, predictions)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def main():
    df = load_features()
    print("=" * 72)
    print("XGBOOST EXPERIMENTS — CURATED FEATURE SUBSET (18 features)")
    print("=" * 72)

    # Clear prior xgboost_curated rows
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost_curated'")
    n_deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_deleted} prior xgboost_curated rows)")

    r1 = run_experiment(df, "2016-01-01", "2023-12-01", "2024-01-01", "2024-12-01",
                        "EXPERIMENT 1 — 2024 validation (calm year)")
    rid1 = str(uuid.uuid4())
    n1 = persist(r1["test_index"], r1["predictions"], TARGET, r1["test_metrics"], rid1, r1["n_train"])
    print(f"\n  → wrote {n1} rows under run_id {rid1[:8]}…")

    r2 = run_experiment(df, "2016-01-01", "2025-12-01", "2026-01-01", "2026-04-01",
                        "EXPERIMENT 2 — Jan-Apr 2026 spike window")
    rid2 = str(uuid.uuid4())
    n2 = persist(r2["test_index"], r2["predictions"], TARGET, r2["test_metrics"], rid2, r2["n_train"])
    print(f"\n  → wrote {n2} rows under run_id {rid2[:8]}…")

    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD — ALL MODELS, BOTH WINDOWS")
    print("=" * 72)
    print(f"                                 RMSE    MAE    MAPE")
    print(f"  --- 2024 calm year (12-month) ---")
    print(f"  SARIMA   (baseline)         |   5.48   4.32   5.77%")
    print(f"  SARIMAX                     |   5.73   4.49   5.91%")
    print(f"  XGBoost (192 features)      |   4.90   4.42   5.79%")
    print(f"  XGBoost (18 curated)        | {r1['test_metrics']['rmse']:>6.2f}  {r1['test_metrics']['mae']:>5.2f}  {r1['test_metrics']['mape']:>5.2f}%")
    print(f"")
    print(f"  --- 2026 spike window (4-month) ---")
    print(f"  SARIMA                      |  22.39  16.18  17.09%")
    print(f"  SARIMAX                     |  23.91  17.30  18.29%")
    print(f"  XGBoost (192 features)      |  16.62  12.42  13.62%")
    print(f"  XGBoost (18 curated)        | {r2['test_metrics']['rmse']:>6.2f}  {r2['test_metrics']['mae']:>5.2f}  {r2['test_metrics']['mape']:>5.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
