# oil_gas_pipeline | scripts/18_xgboost_experiments.py
# XGBoost on the engineered features — same two experiments as SARIMAX:
#   1. 2024 calm year (train 2016-2023, test 2024)
#   2. 2026 spike window (train 2016-2025, test Jan-Apr 2026)
# Uses all 192 engineered features (target excluded). Default hyperparameters.
# Results saved to gold_forecast_results with model_name='xgboost'.
# Run with: python3 scripts/18_xgboost_experiments.py

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

# Columns that are NOT features:
# - period          (the index)
# - wti_price       (the target itself — leaking it would be cheating)
# - created_at      (housekeeping)
# Note: wti_price_lag_1, wti_price_lag_3 etc. ARE legitimate features
# (lagged target — same info SARIMA uses internally via AR terms).
NON_FEATURE_COLS = {
    "period", "wti_price", "created_at",
    # All other current-month base series — leak risk because they are
    # observed simultaneously with the target. Only their LAGGED versions
    # (col_lag_N, col_rollN_mean, col_mom_pct) are legitimate forecast features.
    "henry_hub_price", "brent_price", "oil_production",
    "crude_imports", "refinery_util", "gasoline_stocks", "distillate_stocks",
    "gas_storage", "gas_production",
    "hdd", "cdd",
    "opec_spare", "global_inv",
    "dollar_index", "industrial_production", "treasury_10y",
}


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
    """Load RAW engineered features (trees don't need scaling). Re-attach raw target."""
    df = pd.read_csv(f"{ROOT}/gold_features_engineered_raw.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run_experiment(df, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 72}")
    print(f"{label}")
    print('-' * 72)

    train = df.loc[train_start:train_end]
    test  = df.loc[test_start:test_end]

    feats = feature_cols(df)
    # Drop training rows where ANY feature is still NaN (early rows have unfilled lag_12, etc.)
    train_complete = train.dropna(subset=feats)
    print(f"Train: {train.index.min().date()} → {train.index.max().date()}  "
          f"({len(train_complete)}/{len(train)} usable rows after dropping NaN-rich early rows)")
    print(f"Test:  {test.index.min().date()} → {test.index.max().date()}  ({len(test)} rows)")
    print(f"Features in play: {len(feats)}")

    X_train = train_complete[feats]
    y_train = train_complete[TARGET]
    X_test  = test[feats]
    y_test  = test[TARGET]

    # Default XGBoost regressor
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict on both train (for overfit check) and test
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    m_train = metrics(y_train.values, pred_train)
    m_test  = metrics(y_test.values,  pred_test)

    print(f"\nTRAIN  | RMSE {m_train['rmse']:>6.2f}   MAE {m_train['mae']:>5.2f}   MAPE {m_train['mape']:>5.2f}%   (overfit check)")
    print(f"TEST   | RMSE {m_test['rmse']:>6.2f}   MAE {m_test['mae']:>5.2f}   MAPE {m_test['mape']:>5.2f}%")
    overfit_gap = m_test["mape"] - m_train["mape"]
    print(f"         train→test MAPE gap: {overfit_gap:+.2f} pp"
          + ("  (large = overfit warning)" if overfit_gap > 5 else ""))

    # Row-by-row prediction table
    comp = pd.DataFrame({
        "actual":     y_test.round(2),
        "xgb_pred":   np.round(pred_test, 2),
        "xgb_err":    np.round(pred_test - y_test.values, 2),
    })
    print(f"\nRow-by-row predictions:")
    print(comp.to_string())

    # Top 10 feature importances
    importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(10)
    print(f"\nTop 10 features by importance:")
    for f, imp in importances.items():
        print(f"  {f:38s} {imp:.4f}")

    return {
        "test_metrics": m_test,
        "train_metrics": m_train,
        "predictions": pred_test,
        "test_index": y_test.index,
        "n_train": len(X_train),
        "top_features": importances,
    }


def persist(test_index, predictions, target, m, run_id, n_train):
    """Save forecast rows to gold_forecast_results. XGBoost has no native CI; leave bounds NULL."""
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    sql = """
        INSERT INTO gold_forecast_results
            (run_id, target, model_name, forecast_period, forecast_value,
             lower_bound, upper_bound, rmse, mape, trained_on_periods)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data = [
        (run_id, target, "xgboost", period.date(), float(pred),
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
    print("XGBOOST EXPERIMENTS — same windows as SARIMAX, all 192 features")
    print("=" * 72)
    print(f"Features available: {len(feature_cols(df))} (target excluded)")

    # Clear prior XGBoost rows
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost'")
    n_deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_deleted} prior xgboost rows)")

    # Experiment 1 — calm year
    r1 = run_experiment(
        df,
        train_start="2016-01-01", train_end="2023-12-01",
        test_start="2024-01-01",  test_end="2024-12-01",
        label="EXPERIMENT 1 — 2024 validation (calm year)"
    )
    rid1 = str(uuid.uuid4())
    n1 = persist(r1["test_index"], r1["predictions"], TARGET, r1["test_metrics"], rid1, r1["n_train"])
    print(f"\n  → wrote {n1} rows under run_id {rid1[:8]}…")

    # Experiment 2 — spike window
    r2 = run_experiment(
        df,
        train_start="2016-01-01", train_end="2025-12-01",
        test_start="2026-01-01",  test_end="2026-04-01",
        label="EXPERIMENT 2 — Jan-Apr 2026 spike window"
    )
    rid2 = str(uuid.uuid4())
    n2 = persist(r2["test_index"], r2["predictions"], TARGET, r2["test_metrics"], rid2, r2["n_train"])
    print(f"\n  → wrote {n2} rows under run_id {rid2[:8]}…")

    # ── Head-to-head across all models tested ──────────────────────
    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD — ALL MODELS, BOTH WINDOWS")
    print("=" * 72)
    print(f"                         RMSE    MAE    MAPE")
    print(f"  --- 2024 calm year (12-month forecast) ---")
    print(f"  SARIMA   (baseline) |   5.48   4.32   5.77%")
    print(f"  SARIMAX             |   5.73   4.49   5.91%")
    print(f"  XGBoost             | {r1['test_metrics']['rmse']:>6.2f}  {r1['test_metrics']['mae']:>5.2f}  {r1['test_metrics']['mape']:>5.2f}%")
    print(f"")
    print(f"  --- 2026 spike window (4-month forecast) ---")
    print(f"  SARIMA              |  22.39  16.18  17.09%")
    print(f"  SARIMAX             |  23.91  17.30  18.29%")
    print(f"  XGBoost             | {r2['test_metrics']['rmse']:>6.2f}  {r2['test_metrics']['mae']:>5.2f}  {r2['test_metrics']['mape']:>5.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
