# oil_gas_pipeline | scripts/10_ingest_supply_signals.py
# Ingests OPEC spare capacity + global oil inventory into bronze tables.
# Source: EIA STEO route. Filters to actuals only (period <= current month),
# since STEO carries ~18 months of forward projections.
# Usage: PYTHONPATH=$(pwd) python3 scripts/10_ingest_supply_signals.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime

from config.config import config
from ingestion.eia_client import EIAClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ingest_supply_signals")

SERIES = [
    ("fetch_opec_spare_capacity",  "bronze_opec_spare_capacity"),
    ("fetch_global_oil_inventory", "bronze_global_oil_inventory"),
]

START = "2015-01"


def write_rows(conn, table, df):
    if df.empty:
        log.warning(f"{table}: nothing to write (empty DataFrame)")
        return 0
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["series_id"], r["series_name"], r["period"].strftime("%Y-%m"),
            float(r["value"]), r["unit"], json.dumps(r["raw_response"]),
        ))
    sql = f"""
        INSERT INTO {table}
            (series_id, series_name, period, value, unit, raw_response)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=200)
    conn.commit()
    return len(rows)


def main():
    client = EIAClient()
    conn = psycopg2.connect(**config.db.psycopg2_params)

    today = datetime.today()
    cutoff = pd.Timestamp(year=today.year, month=today.month, day=1)
    log.info(f"Actuals-only cutoff: keeping periods <= {cutoff.date()}")

    total = 0
    for method_name, table in SERIES:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
        conn.commit()

        fetch = getattr(client, method_name)
        df = fetch(start=START)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        before = len(df)
        df = df[df["period"] <= cutoff]
        dropped = before - len(df)

        n = write_rows(conn, table, df)
        total += n
        log.info(f"{table:32s} ← {n:4d} rows  (dropped {dropped} projection months)")

    conn.close()
    log.info(f"Done. {total} supply-signal rows ingested (actuals only).")


if __name__ == "__main__":
    main()
