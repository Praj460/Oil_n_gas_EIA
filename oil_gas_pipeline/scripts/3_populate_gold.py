# oil_gas_pipeline | scripts/3_populate_gold.py
# Step 3 — Transform bronze data into gold_energy_prices table
# Pivots raw EIA series into one wide row per month
# Adds derived features: price spread, MoM change, oil/gas ratio
# Run with: python3 scripts/3_populate_gold.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="transform.log")

import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

logger = logging.getLogger(__name__)

load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

def main():
    print("\n" + "=" * 60)
    print("STEP 3 — POPULATE GOLD TABLE")
    print("=" * 60)

    conn = psycopg2.connect(**DB)

    # ── Read bronze tables ────────────────────────────────────────
    pet = pd.read_sql("SELECT * FROM bronze_petroleum", conn)
    gas = pd.read_sql("SELECT * FROM bronze_natural_gas", conn)
    conn.close()

    print(f"Bronze petroleum rows : {len(pet)}")
    print(f"Bronze natural gas rows: {len(gas)}")

    # ── Pivot petroleum — one row per month ───────────────────────
    pet["period"] = pd.to_datetime(pet["period"]).dt.to_period("M").dt.to_timestamp()
    pet_pivot = pet.pivot_table(
        index="period", columns="series_id", values="value", aggfunc="mean"
    ).reset_index()
    pet_pivot.columns.name = None
    pet_pivot = pet_pivot.rename(columns={
        "PET.RWTC.M":     "wti_price",
        "PET.RBRTE.M":    "brent_price",
        "PET.MCRFPUS2.M": "us_oil_production",
    })

    # ── Pivot natural gas — one row per month ─────────────────────
    gas["period"] = pd.to_datetime(gas["period"]).dt.to_period("M").dt.to_timestamp()
    gas_pivot = gas.pivot_table(
        index="period", columns="series_id", values="value", aggfunc="mean"
    ).reset_index()
    gas_pivot.columns.name = None
    gas_pivot = gas_pivot.rename(columns={
        "NG.RNGWHHD.M":               "henry_hub_price",
        "NG.NW2_EPG0_SWO_R48_BCF.M":  "us_gas_storage_bcf",
        "NG.N9010US2.M":               "us_gas_production",
    })

    # ── Join petroleum + gas on period ────────────────────────────
    gold = pd.merge(pet_pivot, gas_pivot, on="period", how="outer").sort_values("period")

    # ── Add derived features ──────────────────────────────────────
    gold["price_spread"]   = (gold["brent_price"] - gold["wti_price"]).round(4)
    gold["wti_mom_change"] = gold["wti_price"].pct_change() * 100
    gold["gas_mom_change"] = gold["henry_hub_price"].pct_change() * 100
    gold["oil_gas_ratio"]  = (gold["wti_price"] / gold["henry_hub_price"]).round(4)

    print(f"\nGold table rows to insert: {len(gold)}")
    print(f"Date range: {gold['period'].min().date()} → {gold['period'].max().date()}")
    print(f"Columns: {list(gold.columns)}\n")

    # ── Write to gold_energy_prices ───────────────────────────────
    conn2 = psycopg2.connect(**DB)
    cursor = conn2.cursor()
    cursor.execute("TRUNCATE gold_energy_prices")

    insert_query = """
        INSERT INTO gold_energy_prices
            (period, wti_price, brent_price, price_spread, us_oil_production,
             wti_mom_change, henry_hub_price, us_gas_storage_bcf,
             us_gas_production, gas_mom_change, oil_gas_ratio)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    data = [
        (
            row.period,
            float(row.wti_price)          if pd.notna(row.wti_price)          else None,
            float(row.brent_price)         if pd.notna(row.brent_price)        else None,
            float(row.price_spread)        if pd.notna(row.price_spread)       else None,
            float(row.us_oil_production)   if pd.notna(row.us_oil_production)  else None,
            float(row.wti_mom_change)      if pd.notna(row.wti_mom_change)     else None,
            float(row.henry_hub_price)     if pd.notna(row.henry_hub_price)    else None,
            float(row.us_gas_storage_bcf)  if pd.notna(row.us_gas_storage_bcf) else None,
            float(row.us_gas_production)   if pd.notna(row.us_gas_production)  else None,
            float(row.gas_mom_change)      if pd.notna(row.gas_mom_change)     else None,
            float(row.oil_gas_ratio)       if pd.notna(row.oil_gas_ratio)      else None,
        )
        for row in gold.itertuples()
    ]

    execute_batch(cursor, insert_query, data)
    conn2.commit()
    conn2.close()

    print("=" * 60)
    print("TRANSFORM SUMMARY")
    print("=" * 60)
    print(f"✅ Inserted {len(data)} rows into gold_energy_prices")
    print(f"   WTI price range    : ${gold['wti_price'].min():.2f} — ${gold['wti_price'].max():.2f}")
    print(f"   Henry Hub range    : ${gold['henry_hub_price'].min():.2f} — ${gold['henry_hub_price'].max():.2f}")
    print(f"   Oil/Gas ratio range: {gold['oil_gas_ratio'].min():.1f}x — {gold['oil_gas_ratio'].max():.1f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
