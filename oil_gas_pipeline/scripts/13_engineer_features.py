# oil_gas_pipeline | scripts/13_engineer_features.py
# Step 13 — Build the engineered feature table for SARIMAX modeling
# Loads gold_features, runs FeatureEngineer, writes results.
# Run with: python3 scripts/13_engineer_features.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="feature_engineer.log")

import logging
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from models.feature_engineer import FeatureEngineer

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


def main():
    print("\n" + "=" * 60)
    print("STEP 13 — ENGINEER FEATURES")
    print("=" * 60)

    # ── Load gold_features ────────────────────────────────────────
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM gold_features ORDER BY period", conn)
    conn.close()
    print(f"Loaded gold_features: {df.shape[0]} rows × {df.shape[1]} cols")

    # Drop created_at — engineer doesn't need it
    if "created_at" in df.columns:
        df = df.drop(columns=["created_at"])

    # ── Engineer features ─────────────────────────────────────────
    fe = FeatureEngineer()
    raw, scaled = fe.transform(df)

    print(f"\nEngineered raw : {raw.shape[0]} rows × {raw.shape[1]} cols")
    print(f"Engineered scl : {scaled.shape[0]} rows × {scaled.shape[1]} cols")

    # ── Null report on engineered (lags + rolls create NaNs at the start) ─
    print("\nFirst-few-row null pattern (expected — lags need history):")
    print(f"  Row 0 (Jan 2015) nulls: {raw.iloc[0].isna().sum()} of {len(raw.columns)}")
    print(f"  Row 12 (Jan 2016) nulls: {raw.iloc[12].isna().sum()} of {len(raw.columns)}")
    print(f"  Row 24 (Jan 2017) nulls: {raw.iloc[24].isna().sum()} of {len(raw.columns)}")
    print(f"  Row -1 (latest) nulls : {raw.iloc[-1].isna().sum()} of {len(raw.columns)}")

    # First fully-usable row = row where all 12-month-lag features exist (~row 12)
    # We'll keep all rows in storage; the modeling step decides where to start training.

    # ── Write to disk: raw + scaled CSVs ──────────────────────────
    raw_path    = os.path.join(PROJECT_ROOT, "gold_features_engineered_raw.csv")
    scaled_path = os.path.join(PROJECT_ROOT, "gold_features_engineered_scaled.csv")
    raw.to_csv(raw_path,    index=False)
    scaled.to_csv(scaled_path, index=False)
    print(f"\n✅ Wrote {raw_path}")
    print(f"✅ Wrote {scaled_path}")

    # ── Show a sample feature group so you can see what was built ────
    print("\nSample of engineered columns for 'opec_spare':")
    sample_cols = ["period"] + [c for c in raw.columns if c.startswith("opec_spare")]
    print(raw[sample_cols].tail(3).to_string(index=False))

    print("\n" + "=" * 60)
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"Base features:        {17}")
    print(f"Engineered features:  {raw.shape[1] - 17 - 1}  (added on top)")
    print(f"Total columns:        {raw.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
