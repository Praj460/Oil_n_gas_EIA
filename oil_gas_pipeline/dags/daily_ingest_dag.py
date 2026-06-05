# oil_gas_pipeline | dags/daily_ingest_dag.py
# Airflow DAG — runs every day at 6am
# Pulls fresh EIA petroleum + natural gas data → bronze tables
# Then runs Great Expectations quality checks
# If quality gate passes → triggers transform_dag

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

from config.logging_config import setup_logging
setup_logging(log_filename="daily_ingest.log")

import logging
logger = logging.getLogger(__name__)

# ── Default DAG arguments ─────────────────────────────────────────────────────

default_args = {
    "owner":            "oil_gas_pipeline",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── Task functions ────────────────────────────────────────────────────────────

def ingest_petroleum(**context):
    """
    Task 1 — Pull petroleum data from EIA API and write to bronze_petroleum.
    Runs daily — only pulls the latest available month to avoid re-ingesting
    historical data on every run. On first run, pulls from 2000-01.
    """
    from ingestion.ingester import DataIngester

    # Check if this is the first run (no data in bronze yet)
    from database.db_manager import DatabaseManager
    import pandas as pd

    db       = DatabaseManager()
    existing = pd.read_sql("SELECT MAX(period) as max_period FROM bronze_petroleum", db.engine)
    max_period = existing["max_period"].iloc[0]

    if max_period is None:
        start = "2000-01"
        logger.info("First run detected — pulling full historical petroleum data from 2000-01")
    else:
        # Pull from 2 months before max to catch any late-arriving data
        start = (max_period - timedelta(days=60)).strftime("%Y-%m")
        logger.info(f"Incremental run — pulling petroleum data from {start}")

    ingester = DataIngester()
    results  = ingester.ingest_petroleum(start=start)

    # Push results to XCom so downstream tasks can read them
    context["ti"].xcom_push(key="petroleum_results", value=results)
    logger.info(f"Petroleum ingestion complete | results={results}")


def ingest_natural_gas(**context):
    """
    Task 2 — Pull natural gas data from EIA API and write to bronze_natural_gas.
    Same incremental logic as petroleum ingestion.
    """
    from ingestion.ingester import DataIngester
    from database.db_manager import DatabaseManager
    import pandas as pd

    db         = DatabaseManager()
    existing   = pd.read_sql("SELECT MAX(period) as max_period FROM bronze_natural_gas", db.engine)
    max_period = existing["max_period"].iloc[0]

    if max_period is None:
        start = "2000-01"
        logger.info("First run — pulling full historical natural gas data from 2000-01")
    else:
        start = (max_period - timedelta(days=60)).strftime("%Y-%m")
        logger.info(f"Incremental run — pulling natural gas data from {start}")

    ingester = DataIngester()
    results  = ingester.ingest_natural_gas(start=start)

    context["ti"].xcom_push(key="gas_results", value=results)
    logger.info(f"Natural gas ingestion complete | results={results}")


def run_petroleum_quality_check(**context):
    """
    Task 3 — Run PetroleumSuite Great Expectations checks on bronze_petroleum.
    Logs results to data_quality_results table.
    """
    from database.db_manager import DatabaseManager
    from great_expectations.petroleum_suite import PetroleumSuite
    import pandas as pd

    db  = DatabaseManager()
    df  = pd.read_sql(
        "SELECT * FROM bronze_petroleum ORDER BY ingested_at DESC LIMIT 5000",
        db.engine
    )

    suite  = PetroleumSuite()
    report = suite.run(df)

    # Log results to database
    db.log_data_quality(
        suite_name=report.suite_name,
        table_name="bronze_petroleum",
        total_expectations=report.total,
        passed=report.passed,
        failed=report.failed,
    )

    logger.info(report.summary())

    # Push success rate to XCom for the quality gate branch
    context["ti"].xcom_push(
        key="petroleum_quality_rate",
        value=report.success_rate,
    )

    if not report.is_passing:
        raise ValueError(
            f"Petroleum quality check failed | "
            f"passed={report.passed}/{report.total} | "
            f"rate={report.success_rate:.1f}%"
        )


def run_gas_quality_check(**context):
    """
    Task 4 — Run GasSuite Great Expectations checks on bronze_natural_gas.
    Logs results to data_quality_results table.
    """
    from database.db_manager import DatabaseManager
    from great_expectations.gas_suite import GasSuite
    import pandas as pd

    db  = DatabaseManager()
    df  = pd.read_sql(
        "SELECT * FROM bronze_natural_gas ORDER BY ingested_at DESC LIMIT 5000",
        db.engine
    )

    suite  = GasSuite()
    report = suite.run(df)

    db.log_data_quality(
        suite_name=report.suite_name,
        table_name="bronze_natural_gas",
        total_expectations=report.total,
        passed=report.passed,
        failed=report.failed,
    )

    logger.info(report.summary())

    context["ti"].xcom_push(
        key="gas_quality_rate",
        value=report.success_rate,
    )

    if not report.is_passing:
        raise ValueError(
            f"Gas quality check failed | "
            f"passed={report.passed}/{report.total} | "
            f"rate={report.success_rate:.1f}%"
        )


def check_quality_gate(**context):
    """
    Task 5 — Branch task. Checks if both quality suites passed.
    If both pass → proceed to dbt transform.
    If either fails → go to quality_gate_failed (stops pipeline).
    """
    ti = context["ti"]
    petroleum_rate = ti.xcom_pull(key="petroleum_quality_rate", task_ids="petroleum_quality_check")
    gas_rate       = ti.xcom_pull(key="gas_quality_rate",       task_ids="gas_quality_check")

    min_rate = 80.0   # from great_expectations.yml quality_gate config

    if petroleum_rate >= min_rate and gas_rate >= min_rate:
        logger.info(
            f"Quality gate PASSED | "
            f"petroleum={petroleum_rate:.1f}% | gas={gas_rate:.1f}%"
        )
        return "quality_gate_passed"
    else:
        logger.warning(
            f"Quality gate FAILED | "
            f"petroleum={petroleum_rate:.1f}% | gas={gas_rate:.1f}% | "
            f"minimum required={min_rate}%"
        )
        return "quality_gate_failed"


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="daily_ingest_dag",
    description="Daily EIA data ingestion + Great Expectations quality checks",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * *",   # every day at 6:00 AM
    catchup=False,                    # don't backfill missed runs
    max_active_runs=1,                # only one run at a time
    tags=["ingestion", "eia", "daily"],
) as dag:

    # ── Task 1: Ingest petroleum ──────────────────────────────────────────────
    t_ingest_petroleum = PythonOperator(
        task_id="ingest_petroleum",
        python_callable=ingest_petroleum,
    )

    # ── Task 2: Ingest natural gas ────────────────────────────────────────────
    t_ingest_gas = PythonOperator(
        task_id="ingest_natural_gas",
        python_callable=ingest_natural_gas,
    )

    # ── Task 3: Petroleum quality check ──────────────────────────────────────
    t_petroleum_qc = PythonOperator(
        task_id="petroleum_quality_check",
        python_callable=run_petroleum_quality_check,
    )

    # ── Task 4: Gas quality check ─────────────────────────────────────────────
    t_gas_qc = PythonOperator(
        task_id="gas_quality_check",
        python_callable=run_gas_quality_check,
    )

    # ── Task 5: Quality gate branch ───────────────────────────────────────────
    t_quality_gate = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=check_quality_gate,
    )

    # ── Task 6a: Gate passed — trigger transform ──────────────────────────────
    t_gate_passed = EmptyOperator(
        task_id="quality_gate_passed",
    )

    # ── Task 6b: Gate failed — stop pipeline ─────────────────────────────────
    t_gate_failed = EmptyOperator(
        task_id="quality_gate_failed",
    )

    # ── Task 7: Done ──────────────────────────────────────────────────────────
    t_done = EmptyOperator(
        task_id="ingestion_complete",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    # Both ingestion tasks run in parallel
    # Quality checks run after their respective ingestion tasks
    # Gate checks both quality results before allowing transform

    [t_ingest_petroleum, t_ingest_gas]
    t_ingest_petroleum >> t_petroleum_qc
    t_ingest_gas       >> t_gas_qc
    [t_petroleum_qc, t_gas_qc] >> t_quality_gate
    t_quality_gate >> [t_gate_passed, t_gate_failed]
    [t_gate_passed, t_gate_failed] >> t_done