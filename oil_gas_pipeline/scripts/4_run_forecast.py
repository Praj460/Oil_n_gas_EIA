# oil_gas_pipeline | scripts/4_run_forecast.py
# Step 4 — Train SARIMA and Prophet on gold data
# Generates 12-month ahead forecasts for WTI and Henry Hub
# Saves results to gold_forecast_results table
# Run with: python3 scripts/4_run_forecast.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="forecast.log")

import logging
import pandas as pd
import psycopg2

from models.preprocessor import Preprocessor
from models.forecast_model import ForecastModel
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

TARGETS = ["wti_price", "henry_hub_price"]
MODELS  = ["prophet", "sarima"]

def main():
    print("\n" + "=" * 60)
    print("STEP 4 — FORECASTING")
    print("=" * 60)

    # ── Load gold data ────────────────────────────────────────────
    conn = psycopg2.connect(**DB)
    df   = pd.read_sql("SELECT * FROM gold_energy_prices ORDER BY period ASC", conn)
    conn.close()

    print(f"Gold rows loaded: {len(df)}")
    print(f"Date range: {df['period'].min()} → {df['period'].max()}\n")

    db = DatabaseManager()

    # ── Clear existing forecasts ──────────────────────────────────
    conn2 = psycopg2.connect(**DB)
    conn2.cursor().execute("TRUNCATE gold_forecast_results")
    conn2.commit()
    conn2.close()
    print("Cleared existing forecast results\n")

    total_saved = 0

    for target in TARGETS:
        for model_name in MODELS:
            print(f"--- {model_name.upper()} | {target} ---")

            try:
                # Preprocess
                prep        = Preprocessor()
                clean       = prep.fit_transform(df, target_col=target)
                train, test = prep.train_test_split(clean, target)

                print(f"  Train: {len(train)} rows | Test: {len(test)} rows")

                # Fit and forecast
                fm          = ForecastModel()
                result      = fm.fit_predict(train, target, model=model_name)
                result.target = target

                print(f"  Forecast horizon: {len(result.forecast_df)} months")
                print(f"  First forecast : ${result.forecast_df['forecast'].iloc[0]:.2f}")
                print(f"  Last forecast  : ${result.forecast_df['forecast'].iloc[-1]:.2f}")

                # Save to DB
                db_df         = result.to_db_df()
                db_df["rmse"] = None
                db_df["mape"] = None
                rows          = db.write_forecast_results(db_df)
                total_saved  += rows
                print(f"  ✅ Saved {rows} rows\n")

            except Exception as e:
                print(f"  ❌ Failed: {e}\n")
                logger.error(f"Forecast failed | model={model_name} | target={target} | {e}")

    print("=" * 60)
    print("FORECAST SUMMARY")
    print("=" * 60)
    print(f"✅ Total forecast rows saved: {total_saved}")
    print(f"   {len(TARGETS)} targets × {len(MODELS)} models × 12 months = {len(TARGETS)*len(MODELS)*12} expected")
    print("=" * 60)

if __name__ == "__main__":
    main()
