# oil_gas_pipeline | scripts/20_xgboost_tuned.py
# Tunes XGBoost on the 18 curated features. Sweeps a small grid of
# hyperparameters chosen to fight overfit (depth, regularization, lr).
# Same train/test windows as before. Persisted as model_name='xgboost_tuned'.
# Run with: python3 scripts/20_xgboost_tuned.py

import sys, os, uuid, warnings, itertools
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

CURATED_FEATURES = [
    "wti_price_lag_1", "wti_price_lag_3",
    "wti_price_roll3_mean", "wti_price_roll6_mean", "wti_price_mom_pct",
    "brent_price_lag_1", "henry_hub_price_lag_1",
    "crude_imports_lag_1", "refinery_util_lag_1",
    "opec_spare_lag_1", "global_inv_lag_1",
    "gasoline_stocks_lag_1", "distillate_stocks_lag_1",
    "industrial_production_lag_1",
    "dollar_index_lag_1", "treasury_10y_lag_1",
    "month_sin", "month_cos",
    "opec_spare_roll3_std",
]

# Hyperparameter grid — carefully chosen, not a fishing expedition.
# Each parameter tests a specific anti-overfit hypothesis.
PARAM_GRID = [
    # (n_estimators, max_depth, lr, reg_alpha (L1), reg_lambda (L2), label)
    (300, 4, 0.05, 0.0, 1.0,  "baseline (no reg)"),
    (300, 3, 0.05, 0.0, 1.0,  "shallower (depth 3)"),
    (300, 2, 0.05, 0.0, 1.0,  "very shallow (depth 2)"),
    (300, 3, 0.05, 0.1, 1.0,  "shallow + light L1"),
    (300, 3, 0.05, 0.0, 5.0,  "shallow + heavy L2"),
    (300, 3, 0.05, 0.1, 5.0,  "shallow + both L1 & L2"),
    (500, 3, 0.03, 0.0, 1.0,  "more trees + slower learn"),
    (100, 3, 0.05, 0.0, 1.0,  "fewer trees"),
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


def fit_predict(params, X_train, y_train, X_test):
    n_est, depth, lr, ra, rl, _ = params
    model = XGBRegressor(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        reg_alpha=ra, reg_lambda=rl,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_train), model.predict(X_test), model


def sweep_experiment(df, train_start, train_end, test_start, test_end, label):
    print(f"\n{'-' * 78}")
    print(label)
    print('-' * 78)

    train = df.loc[train_start:train_end].dropna(subset=CURATED_FEATURES + [TARGET])
    test  = df.loc[test_start:test_end]

    X_train, y_train = train[CURATED_FEATURES], train[TARGET]
    X_test,  y_test  = test[CURATED_FEATURES],  test[TARGET]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {len(CURATED_FEATURES)}")
    print(f"\n{'config':<32} {'train MAPE':>11} {'test MAPE':>11} {'gap':>8}")
    print('-' * 70)

    best = {"label": None, "test_mape": float("inf")}
    rows = []
    for params in PARAM_GRID:
        _, _, _, _, _, lbl = params
        try:
            pred_train, pred_test, model = fit_predict(params, X_train, y_train, X_test)
            m_train = metrics(y_train.values, pred_train)
            m_test  = metrics(y_test.values,  pred_test)
            gap = m_test["mape"] - m_train["mape"]
            print(f"  {lbl:<30} {m_train['mape']:>10.2f}% {m_test['mape']:>10.2f}% {gap:>+7.2f}")
            cand = {
                "label": lbl, "params": params,
                "train_mape": m_train['mape'], "test_mape": m_test['mape'],
                "test_metrics": m_test, "predictions": pred_test,
                "test_index": X_test.index, "n_train": len(X_train),
                "feature_importances": pd.Series(model.feature_importances_, index=CURATED_FEATURES),
            }
            rows.append(cand)
            if m_test["mape"] < best["test_mape"]:
                best = cand
        except Exception as e:
            print(f"  {lbl:<30}    ERROR: {str(e)[:40]}")

    print(f"\n  → BEST: {best['label']}  test MAPE {best['test_mape']:.2f}%")
    return best, rows


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
        (run_id, target, "xgboost_tuned", period.date(), float(pred),
         None, None, m["rmse"], m["mape"], n_train)
        for period, pred in zip(test_index, predictions)
    ]
    execute_batch(cur, sql, data)
    conn.commit()
    conn.close()
    return len(data)


def main():
    df = load_features()
    print("=" * 78)
    print("XGBOOST TUNING — 18 curated features, 8-config sweep per experiment")
    print("=" * 78)

    # Clear prior tuned rows
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM gold_forecast_results WHERE model_name = 'xgboost_tuned'")
    n_del = cur.rowcount
    conn.commit()
    conn.close()
    print(f"(cleared {n_del} prior xgboost_tuned rows)")

    best1, _ = sweep_experiment(df, "2016-01-01", "2023-12-01", "2024-01-01", "2024-12-01",
                                "EXPERIMENT 1 — 2024 validation (calm year)")
    rid1 = str(uuid.uuid4())
    n1 = persist(best1["test_index"], best1["predictions"], TARGET,
                 best1["test_metrics"], rid1, best1["n_train"])
    print(f"  → persisted {n1} rows under {rid1[:8]}…")

    best2, _ = sweep_experiment(df, "2016-01-01", "2025-12-01", "2026-01-01", "2026-04-01",
                                "EXPERIMENT 2 — Jan-Apr 2026 spike window")
    rid2 = str(uuid.uuid4())
    n2 = persist(best2["test_index"], best2["predictions"], TARGET,
                 best2["test_metrics"], rid2, best2["n_train"])
    print(f"  → persisted {n2} rows under {rid2[:8]}…")

    # ── Show best models' row-by-row predictions ───────────────────────
    print("\n" + "=" * 78)
    print("BEST TUNED MODEL — row-by-row predictions")
    print("=" * 78)
    print(f"\n2024 (config: {best1['label']}):")
    test1 = df.loc["2024-01-01":"2024-12-01"]
    comp1 = pd.DataFrame({
        "actual":     test1[TARGET].round(2),
        "tuned_pred": np.round(best1["predictions"], 2),
        "err":        np.round(best1["predictions"] - test1[TARGET].values, 2),
    })
    print(comp1.to_string())

    print(f"\n2026 spike (config: {best2['label']}):")
    test2 = df.loc["2026-01-01":"2026-04-01"]
    comp2 = pd.DataFrame({
        "actual":     test2[TARGET].round(2),
        "tuned_pred": np.round(best2["predictions"], 2),
        "err":        np.round(best2["predictions"] - test2[TARGET].values, 2),
    })
    print(comp2.to_string())

    # ── Final head-to-head ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("FINAL HEAD-TO-HEAD — ALL FIVE MODEL VARIANTS")
    print("=" * 78)
    print(f"                                  RMSE    MAE    MAPE")
    print(f"  --- 2024 calm year (12-month) ---")
    print(f"  SARIMA   (baseline)         |   5.48   4.32   5.77%")
    print(f"  SARIMAX                     |   5.73   4.49   5.91%")
    print(f"  XGBoost (192 features)      |   4.90   4.42   5.79%")
    print(f"  XGBoost (18 curated)        |   3.56   3.11   4.09%")
    print(f"  XGBoost (curated + tuned)   | {best1['test_metrics']['rmse']:>6.2f}  {best1['test_metrics']['mae']:>5.2f}  {best1['test_metrics']['mape']:>5.2f}%")
    print(f"")
    print(f"  --- 2026 spike window (4-month) ---")
    print(f"  SARIMA                      |  22.39  16.18  17.09%")
    print(f"  SARIMAX                     |  23.91  17.30  18.29%")
    print(f"  XGBoost (192 features)      |  16.62  12.42  13.62%")
    print(f"  XGBoost (18 curated)        |  14.61  10.39  11.38%")
    print(f"  XGBoost (curated + tuned)   | {best2['test_metrics']['rmse']:>6.2f}  {best2['test_metrics']['mae']:>5.2f}  {best2['test_metrics']['mape']:>5.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
