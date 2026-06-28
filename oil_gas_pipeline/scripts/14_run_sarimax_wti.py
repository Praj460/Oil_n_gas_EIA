# oil_gas_pipeline | scripts/14_run_sarimax_wti.py
# Step 14 — Run SARIMAX on WTI and compare against the SARIMA baseline.
# Same train/test window as the baseline (2016-2023 / 2024) for fair comparison.
# Run with: python3 scripts/14_run_sarimax_wti.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="sarimax_wti.log")

import logging
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.forecast_model import ForecastModel

logger = logging.getLogger(__name__)
load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exogenous features for WTI SARIMAX — economic story:
#   supply:    crude_imports_lag_1, refinery_util_lag_1
#   downstream:gasoline_stocks_lag_1
#   fragility: opec_spare_lag_1, global_inv_lag_1
#   macro:     dollar_index_lag_1, industrial_production_lag_1, treasury_10y_lag_1



EXOG_COLS = [
    "industrial_production_lag_1",   # the one statistically significant
    "opec_spare_lag_1",              # your supply-fragility story
    "dollar_index_lag_1",            # macro currency, sign correct
]

TARGET = "wti_price"

# Same window as the SARIMA baseline → fair head-to-head
TRAIN_START = "2016-01-01"
TRAIN_END   = "2023-12-01"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-01"


def metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    mask = (~actual.isna()) & (~predicted.isna())
    a = actual[mask].values
    p = predicted[mask].values
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    mae  = float(mean_absolute_error(a, p))
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape, "n": int(mask.sum())}


def main():
    print("\n" + "=" * 60)
    print("STEP 14 — SARIMAX on WTI (vs SARIMA baseline)")
    print("=" * 60)

    # ── Load engineered scaled features ───────────────────────────
    # SARIMAX needs scaled inputs (linear math sensitive to magnitudes)
    path = os.path.join(PROJECT_ROOT, "gold_features_engineered_scaled.csv")
    df = pd.read_csv(path, parse_dates=["period"])
    df = df.set_index("period").sort_index()
    print(f"Loaded scaled engineered features: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── ⚠️ IMPORTANT: target should NOT be scaled (we want real prices back out) ──
    # Re-load the raw (unscaled) target from gold_features so RMSE is in $.
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn)
    conn.close()
    gold["period"] = pd.to_datetime(gold["period"])
    gold = gold.set_index("period").sort_index()
    # Replace the scaled target with the raw one
    df[TARGET] = gold[TARGET]

    # ── Slice train and test ──────────────────────────────────────
    train_full = df.loc[TRAIN_START:TRAIN_END]
    test_full  = df.loc[TEST_START:TEST_END]
    print(f"\nTrain window: {train_full.index.min().date()} → {train_full.index.max().date()}  ({len(train_full)} rows)")
    print(f"Test window:  {test_full.index.min().date()}  → {test_full.index.max().date()}   ({len(test_full)} rows)")

    # Verify exog columns exist
    missing = [c for c in EXOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing exog columns: {missing}")

    train_y = train_full[TARGET].asfreq("MS")
    train_X = train_full[EXOG_COLS].asfreq("MS")
    test_y  = test_full[TARGET].asfreq("MS")
    test_X  = test_full[EXOG_COLS].asfreq("MS")

    # Drop any rows where exog features still have NaN (shouldn't happen post-engineering, but safety)
    before = len(train_X)
    valid = train_X.dropna().index
    train_X = train_X.loc[valid]
    train_y = train_y.loc[valid]
    if len(train_X) < before:
        print(f"  Dropped {before - len(train_X)} train rows with NaN exog")

    print(f"\nExog features ({len(EXOG_COLS)}):")
    for c in EXOG_COLS:
        print(f"   • {c}")

    # ── Fit SARIMAX ───────────────────────────────────────────────
    print(f"\nFitting SARIMAX...")
    model = ForecastModel()
    model.fit_sarimax(train_y, train_X)

    # ── Forecast ──────────────────────────────────────────────────
    result = model.predict_sarimax(exog_test=test_X, horizon=len(test_X))
    result.target = TARGET
    fc = result.forecast_df.set_index("period")["forecast"]

    # ── Compare against actuals ───────────────────────────────────
    fc.index = pd.to_datetime(fc.index)
    aligned = pd.concat([test_y.rename("actual"), fc.rename("predicted")], axis=1).dropna()
    print(f"\nForecast vs actual (2024):")
    print(aligned.round(2).to_string())

    m_sarimax = metrics(aligned["actual"], aligned["predicted"])

    # ── Baseline (your reported SARIMA result) ────────────────────
    BASELINE_SARIMA = {"rmse": 5.4752, "mae": 4.3228, "mape": 5.7677}

    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD ON 2024 WTI")
    print("=" * 60)
    print(f"                       RMSE      MAE     MAPE")
    print(f"  SARIMA (baseline) | {BASELINE_SARIMA['rmse']:>6.2f}   {BASELINE_SARIMA['mae']:>6.2f}   {BASELINE_SARIMA['mape']:>5.2f}%")
    print(f"  SARIMAX           | {m_sarimax['rmse']:>6.2f}   {m_sarimax['mae']:>6.2f}   {m_sarimax['mape']:>5.2f}%")
    delta = m_sarimax["mape"] - BASELINE_SARIMA["mape"]
    direction = "improved by" if delta < 0 else "worsened by"
    print(f"\n  → MAPE {direction} {abs(delta):.2f} percentage points")
    print("=" * 60)


if __name__ == "__main__":
    main()
