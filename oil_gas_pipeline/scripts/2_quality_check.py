# oil_gas_pipeline | scripts/2_quality_check.py
# Step 2 — Run Great Expectations quality checks on bronze tables
# Logs results to data_quality_results table
# Run with: python3 scripts/2_quality_check.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="quality.log")

import logging
import pandas as pd
import psycopg2

from database.db_manager import DatabaseManager
from great_expectations.petroleum_suite import PetroleumSuite
from great_expectations.gas_suite import GasSuite

logger = logging.getLogger(__name__)

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

def main():
    print("\n" + "=" * 60)
    print("STEP 2 — DATA QUALITY CHECKS")
    print("=" * 60)

    conn = psycopg2.connect(**DB)
    pet_df = pd.read_sql("SELECT * FROM bronze_petroleum", conn)
    gas_df = pd.read_sql("SELECT * FROM bronze_natural_gas", conn)
    conn.close()

    print(f"Loaded {len(pet_df)} petroleum rows")
    print(f"Loaded {len(gas_df)} natural gas rows\n")

    db = DatabaseManager()

    # ── Petroleum suite ───────────────────────────────────────────
    pet_report = PetroleumSuite().run(pet_df)
    print(pet_report.summary())

    db.log_data_quality(
        suite_name="petroleum_suite",
        table_name="bronze_petroleum",
        total_expectations=pet_report.total,
        passed=pet_report.passed,
        failed=pet_report.failed,
    )

    # ── Gas suite ─────────────────────────────────────────────────
    gas_report = GasSuite().run(gas_df)
    print(gas_report.summary())

    db.log_data_quality(
        suite_name="gas_suite",
        table_name="bronze_natural_gas",
        total_expectations=gas_report.total,
        passed=gas_report.passed,
        failed=gas_report.failed,
    )

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    print("QUALITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Petroleum : {pet_report.passed}/{pet_report.total} passed ({pet_report.success_rate:.1f}%) {'✅' if pet_report.is_passing else '❌'}")
    print(f"Gas       : {gas_report.passed}/{gas_report.total} passed ({gas_report.success_rate:.1f}%) {'✅' if gas_report.is_passing else '❌'}")
    print("Results saved to data_quality_results table")
    print("=" * 60)

if __name__ == "__main__":
    main()
