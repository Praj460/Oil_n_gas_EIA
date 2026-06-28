# oil_gas_pipeline | scripts/6_ingest_exogenous.py
# Ingests the 4 new exogenous EIA series into their bronze tables.
# Self-contained: fetches via EIAClient, writes via psycopg2 directly.
# Usage: PYTHONPATH=$(pwd) python3 scripts/6_ingest_exogenous.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import psycopg2
from psycopg2.extras import execute_batch

from config.config import config
from ingestion.eia_client import EIAClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ingest_exogenous")

# Map each fetch method to its destination table
SERIES = [
    ("fetch_crude_imports",        "bronze_crude_imports"),
    ("fetch_refinery_utilization", "bronze_refinery_utilization"),
    ("fetch_gasoline_stocks",      "bronze_gasoline_stocks"),
    ("fetch_distillate_stocks",    "bronze_distillate_stocks"),
]

START = "2015-01"


def write_rows(conn, table, df):
    """Insert a parsed DataFrame into the given bronze table."""
    if df.empty:
        log.warning(f"{table}: nothing to write (empty DataFrame)")
        return 0

    # Convert period (datetime) back to 'YYYY-MM' string to match varchar schema
    rows = []
    for _, r in df.iterrows():
        rows.append((
            r["series_id"],
            r["series_name"],
            r["period"].strftime("%Y-%m"),     # varchar period, matches bronze_petroleum
            float(r["value"]),
            r["unit"],
            json.dumps(r["raw_response"]),      # jsonb
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

    total = 0
    for method_name, table in SERIES:
        # Clear existing rows first so re-runs don't stack duplicates
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
        conn.commit()

        fetch = getattr(client, method_name)
        df = fetch(start=START)
        n = write_rows(conn, table, df)
        total += n
        log.info(f"{table:32s} ← {n:4d} rows")

    conn.close()
    log.info(f"Done. {total} exogenous rows ingested across {len(SERIES)} tables.")


if __name__ == "__main__":
    main()
