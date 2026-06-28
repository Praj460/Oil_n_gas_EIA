# oil_gas_pipeline | scripts/11_build_features.py
# Step 11 — Build the wide gold_features table for SARIMAX modeling
# Joins all 17 series (targets + exogenous predictors) into one row per month.
# Spine = petroleum date range (the WTI target); exogenous gaps left as NULL.
# Run with: python3 scripts/11_build_features.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="features.log")

import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# Each single-series bronze table → its output column name.
# (period normalized to month-start so all formats align.)
SINGLE_SERIES = {
    "bronze_crude_imports":          "crude_imports",
    "bronze_refinery_utilization":   "refinery_util",
    "bronze_gasoline_stocks":        "gasoline_stocks",
    "bronze_distillate_stocks":      "distillate_stocks",
    "bronze_heating_degree_days":    "hdd",
    "bronze_cooling_degree_days":    "cdd",
    "bronze_opec_spare_capacity":    "opec_spare",
    "bronze_global_oil_inventory":   "global_inv",
    "bronze_dollar_index":           "dollar_index",
    "bronze_industrial_production":  "industrial_production",
    "bronze_treasury_10y":           "treasury_10y",
}


def read_single(conn, table, colname):
    """Read a single-series bronze table → DataFrame[period, colname]."""
    df = pd.read_sql(f"SELECT period, value FROM {table}", conn)
    df["period"] = pd.to_datetime(df["period"]).dt.to_period("M").dt.to_timestamp()
    df = (df.groupby("period", as_index=False)["value"].mean()
            .rename(columns={"value": colname}))
    return df


def main():
    print("\n" + "=" * 60)
    print("STEP 11 — BUILD GOLD FEATURES TABLE")
    print("=" * 60)

    conn = psycopg2.connect(**DB)

    # ── Petroleum (the spine — defines the date range) ────────────
    pet = pd.read_sql("SELECT * FROM bronze_petroleum", conn)
    pet["period"] = pd.to_datetime(pet["period"]).dt.to_period("M").dt.to_timestamp()
    pet_pivot = pet.pivot_table(index="period", columns="series_id",
                                values="value", aggfunc="mean").reset_index()
    pet_pivot.columns.name = None
    pet_pivot = pet_pivot.rename(columns={
        "PET.RWTC.M":     "wti_price",
        "PET.RBRTE.M":    "brent_price",
        "PET.MCRFPUS2.M": "oil_production",
    })

    # ── Natural gas ───────────────────────────────────────────────
    gas = pd.read_sql("SELECT * FROM bronze_natural_gas", conn)
    gas["period"] = pd.to_datetime(gas["period"]).dt.to_period("M").dt.to_timestamp()
    gas_pivot = gas.pivot_table(index="period", columns="series_id",
                                values="value", aggfunc="mean").reset_index()
    gas_pivot.columns.name = None
    gas_pivot = gas_pivot.rename(columns={
        "NG.RNGWHHD.M":              "henry_hub_price",
        "NG.NW2_EPG0_SWO_R48_BCF.M": "gas_storage",
        "NG.N9010US2.M":             "gas_production",
    })

    # ── Start from petroleum spine, join gas ──────────────────────
    feat = pd.merge(pet_pivot, gas_pivot, on="period", how="left")

    # ── Join each single-series exogenous table ───────────────────
    for table, colname in SINGLE_SERIES.items():
        s = read_single(conn, table, colname)
        feat = pd.merge(feat, s, on="period", how="left")
        logger.info(f"joined {table} → {colname}")

    conn.close()

    feat = feat.sort_values("period").reset_index(drop=True)

    # ── Column order ──────────────────────────────────────────────
    cols = [
        "period",
        # targets
        "wti_price", "henry_hub_price",
        # petroleum
        "brent_price", "oil_production",
        "crude_imports", "refinery_util", "gasoline_stocks", "distillate_stocks",
        # natural gas
        "gas_storage", "gas_production",
        # weather
        "hdd", "cdd",
        # supply fragility
        "opec_spare", "global_inv",
        # macro
        "dollar_index", "industrial_production", "treasury_10y",
    ]
    feat = feat[cols]

    print(f"\nFeature table rows : {len(feat)}")
    print(f"Date range         : {feat['period'].min().date()} → {feat['period'].max().date()}")
    print(f"Columns ({len(feat.columns)})      : {list(feat.columns)}")

    # ── Null report — how complete is each column? ────────────────
    print("\nNull counts per column (out of {} rows):".format(len(feat)))
    for c in feat.columns:
        n = feat[c].isna().sum()
        flag = "  ← has gaps" if n else ""
        print(f"   {c:24s} {n:4d}{flag}")

    # ── Create table (drop + recreate so schema always matches) ───
    conn2 = psycopg2.connect(**DB)
    cur = conn2.cursor()
    cur.execute("DROP TABLE IF EXISTS gold_features")
    cur.execute("""
        CREATE TABLE gold_features (
            period               DATE PRIMARY KEY,
            wti_price            NUMERIC(12,4),
            henry_hub_price      NUMERIC(12,4),
            brent_price          NUMERIC(12,4),
            oil_production       NUMERIC(14,2),
            crude_imports        NUMERIC(14,2),
            refinery_util        NUMERIC(10,4),
            gasoline_stocks      NUMERIC(14,2),
            distillate_stocks    NUMERIC(14,2),
            gas_storage          NUMERIC(14,2),
            gas_production       NUMERIC(14,2),
            hdd                  NUMERIC(10,2),
            cdd                  NUMERIC(10,2),
            opec_spare           NUMERIC(10,4),
            global_inv           NUMERIC(12,4),
            dollar_index         NUMERIC(10,4),
            industrial_production NUMERIC(10,4),
            treasury_10y         NUMERIC(8,4),
            created_at           TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn2.commit()

    insert_cols = [c for c in cols]  # same order, minus created_at (defaulted)
    placeholders = ",".join(["%s"] * len(insert_cols))
    insert_query = f"INSERT INTO gold_features ({','.join(insert_cols)}) VALUES ({placeholders})"

    def val(x):
        return float(x) if pd.notna(x) else None

    data = [tuple(val(getattr(row, c)) if c != "period" else getattr(row, c)
                  for c in insert_cols)
            for row in feat.itertuples()]

    execute_batch(cur, insert_query, data)
    conn2.commit()
    conn2.close()

    print("\n" + "=" * 60)
    print("FEATURE TABLE SUMMARY")
    print("=" * 60)
    print(f"✅ Inserted {len(data)} rows into gold_features")
    print(f"   {len(insert_cols)-1} feature columns + period")
    print(f"   WTI range: ${feat['wti_price'].min():.2f} — ${feat['wti_price'].max():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
