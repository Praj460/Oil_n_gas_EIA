# oil_gas_pipeline | scripts/25_quality_check_all.py
# Runs data quality checks on ALL 13 bronze tables (not just petroleum + gas).
# Logs each table's results to data_quality_results via DatabaseManager,
# using the same schema as the original 2_quality_check.py.
# Run with: python3 scripts/25_quality_check_all.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logging
setup_logging(log_filename="quality_all.log")

import logging
import pandas as pd
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

# Each table: suite name, and the expected (min, max) sane range for its value.
# Range is generous — these checks catch corruption/parsing errors, not normal variation.
TABLE_CHECKS = {
    "bronze_crude_imports":          ("crude_imports_suite",      0,      50_000),
    "bronze_refinery_utilization":   ("refinery_util_suite",      0,      100),     # percent
    "bronze_gasoline_stocks":        ("gasoline_stocks_suite",    0,      500_000),
    "bronze_distillate_stocks":      ("distillate_stocks_suite",  0,      500_000),
    "bronze_heating_degree_days":    ("hdd_suite",                0,      2_000),
    "bronze_cooling_degree_days":    ("cdd_suite",                0,      2_000),
    "bronze_opec_spare_capacity":    ("opec_spare_suite",        -1,      15),       # mbd
    "bronze_global_oil_inventory":   ("global_inv_suite",         0,      5_000),
    "bronze_dollar_index":           ("dollar_index_suite",       50,     200),
    "bronze_industrial_production":  ("industrial_prod_suite",    50,     150),      # index
    "bronze_treasury_10y":           ("treasury_10y_suite",      -2,      20),       # percent
}


def run_checks(df, table, vmin, vmax):
    """
    Runs a standard battery of quality checks on a bronze table DataFrame.
    Returns (total_checks, passed, failed, details list).
    """
    checks = []

    # 1. Table is not empty
    checks.append(("table_not_empty", len(df) > 0))

    # 2. Has a 'value' column
    has_value = "value" in df.columns
    checks.append(("has_value_column", has_value))

    # 3. Has a 'period' column
    has_period = "period" in df.columns
    checks.append(("has_period_column", has_period))

    if has_value:
        vals = pd.to_numeric(df["value"], errors="coerce")
        # 4. No nulls in value
        checks.append(("value_no_nulls", vals.notna().all()))
        # 5. All values within sane range
        in_range = vals.dropna().between(vmin, vmax).all()
        checks.append(("value_in_range", bool(in_range)))
        # 6. Values are not all identical (would signal a parsing/constant bug)
        checks.append(("value_has_variance", vals.dropna().nunique() > 1))

    if has_period:
        # 7. No nulls in period
        checks.append(("period_no_nulls", df["period"].notna().all()))
        # 8. No duplicate periods (per single-series tables)
        checks.append(("period_no_duplicates", not df["period"].duplicated().any()))

    # 9. Has a source tag (the provider column we added)
    if "source" in df.columns:
        checks.append(("source_tag_present", df["source"].notna().all()))

    total  = len(checks)
    passed = sum(1 for _, ok in checks if ok)
    failed = total - passed
    return total, passed, failed, checks


def main():
    print("\n" + "=" * 64)
    print("DATA QUALITY CHECKS — ALL 13 BRONZE TABLES")
    print("=" * 64)

    conn = psycopg2.connect(**DB)
    db = DatabaseManager()

    results = []
    for table, (suite, vmin, vmax) in TABLE_CHECKS.items():
        try:
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
        except Exception as e:
            print(f"  ⚠️  {table}: could not read ({str(e)[:40]})")
            continue

        total, passed, failed, checks = run_checks(df, table, vmin, vmax)

        db.log_data_quality(
            suite_name=suite,
            table_name=table,
            total_expectations=total,
            passed=passed,
            failed=failed,
        )

        rate = (passed / total * 100) if total else 0
        flag = "✅" if failed == 0 else "❌"
        print(f"  {flag} {suite:26s} {passed}/{total} passed ({rate:.0f}%)  [{table}]")

        # Show which checks failed, if any
        for name, ok in checks:
            if not ok:
                print(f"        ✗ failed: {name}")

        results.append((suite, passed, total, failed))

    conn.close()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    total_checks = sum(t for _, _, t, _ in results)
    total_passed = sum(p for _, p, _, _ in results)
    n_suites     = len(results)
    n_clean      = sum(1 for _, _, _, f in results if f == 0)
    print(f"  Tables checked:     {n_suites}")
    print(f"  Suites fully clean: {n_clean}/{n_suites}")
    print(f"  Total checks:       {total_passed}/{total_checks} passed "
          f"({total_passed/total_checks*100:.1f}%)")
    print(f"\n  Note: original petroleum_suite + gas_suite are run separately by")
    print(f"  scripts/2_quality_check.py. Run that too for full coverage of all 13 tables.")
    print("=" * 64)


if __name__ == "__main__":
    main()
