# oil_gas_pipeline | scripts/26_backfill_pipeline_runs.py
# Logs pipeline_runs entries for the ingestion that's already happened but
# wasn't recorded (FRED, STEO, exogenous, supply signals, feature builds).
# Counts actual rows in each table so the logged numbers are real, not made up.
# Run with: python3 scripts/26_backfill_pipeline_runs.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="backfill_runs.log")

import logging
import psycopg2
from dotenv import load_dotenv
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)
load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

# Run name → list of tables it populated. Row count is summed across them.
RUN_DEFINITIONS = {
    "exogenous_petroleum_ingestion": [
        "bronze_crude_imports", "bronze_refinery_utilization",
        "bronze_gasoline_stocks", "bronze_distillate_stocks",
    ],
    "fred_macro_ingestion": [
        "bronze_dollar_index", "bronze_industrial_production", "bronze_treasury_10y",
    ],
    "steo_weather_ingestion": [
        "bronze_heating_degree_days", "bronze_cooling_degree_days",
    ],
    "supply_signals_ingestion": [
        "bronze_opec_spare_capacity", "bronze_global_oil_inventory",
    ],
    "feature_engineering_build": [
        "gold_features",
    ],
}


def count_rows(conn, table):
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def main():
    print("\n" + "=" * 60)
    print("BACKFILL PIPELINE RUN HISTORY")
    print("=" * 60)

    conn = psycopg2.connect(**DB)
    db = DatabaseManager()

    # Avoid double-logging: check which run_names already exist
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT run_name FROM pipeline_runs")
    existing = {r[0] for r in cur.fetchall()}

    logged = 0
    for run_name, tables in RUN_DEFINITIONS.items():
        if run_name in existing:
            print(f"  ⏭  {run_name} already logged — skipping")
            continue

        total_rows = sum(count_rows(conn, t) for t in tables)
        db.log_pipeline_run(
            run_name=run_name,
            status="success",
            rows_ingested=total_rows,
            rows_failed=0,
            error_message=None,
        )
        print(f"  ✅ {run_name:34s} {total_rows:5d} rows  ({len(tables)} tables)")
        logged += 1

    conn.close()

    print("\n" + "=" * 60)
    print(f"Logged {logged} new pipeline runs.")
    print("Existing runs were left untouched.")
    print("=" * 60)


if __name__ == "__main__":
    main()
