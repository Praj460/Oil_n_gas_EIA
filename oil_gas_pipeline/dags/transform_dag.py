# oil_gas_pipeline | dags/transform_dag.py
# Airflow DAG — runs after daily_ingest_dag quality gate passes
# Executes dbt models: bronze → silver (staging) → gold (mart)
# Schedule: daily at 7am — 1 hour after ingestion

import sys
sys.path.insert(0, '/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

from config.logging_config import setup_logging
setup_logging(log_filename="transform.log")

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)



# ── Default DAG arguments ─────────────────────────────────────────────────────

default_args = {
    "owner":            "oil_gas_pipeline",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
}

# Project root and dbt directory
PROJECT_ROOT = Path(__file__).parent.parent
DBT_DIR      = PROJECT_ROOT / "dbt"

# ── Task functions ────────────────────────────────────────────────────────────

def check_bronze_data(**context):
    """
    Task 1 — Pre-flight check before running dbt.
    Verifies that bronze tables have data before wasting time on transforms.
    Fails fast if ingestion didn't populate any rows.
    """
    from database.db_manager import DatabaseManager
    import pandas as pd

    db = DatabaseManager()

    petroleum_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM bronze_petroleum", db.engine
    ).iloc[0]["cnt"]

    gas_count = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM bronze_natural_gas", db.engine
    ).iloc[0]["cnt"]

    logger.info(f"Bronze row counts | petroleum={petroleum_count} | gas={gas_count}")

    if petroleum_count == 0:
        raise ValueError("bronze_petroleum is empty — skipping dbt run")
    if gas_count == 0:
        raise ValueError("bronze_natural_gas is empty — skipping dbt run")

    context["ti"].xcom_push(key="petroleum_count", value=int(petroleum_count))
    context["ti"].xcom_push(key="gas_count",       value=int(gas_count))
    logger.info("Pre-flight check passed — bronze tables have data")


def verify_silver_output(**context):
    """
    Task 4 — Verifies dbt staging models populated the silver views.
    Checks that silver_petroleum and silver_natural_gas have data.
    """
    from database.db_manager import DatabaseManager
    import pandas as pd

    db = DatabaseManager()

    try:
        pet_count = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM silver_petroleum", db.engine
        ).iloc[0]["cnt"]

        gas_count = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM silver_natural_gas", db.engine
        ).iloc[0]["cnt"]

        logger.info(f"Silver row counts | petroleum={pet_count} | gas={gas_count}")

        if pet_count == 0:
            raise ValueError("silver_petroleum view is empty after dbt run")
        if gas_count == 0:
            raise ValueError("silver_natural_gas view is empty after dbt run")

        logger.info("Silver layer verification passed")

    except Exception as e:
        logger.error(f"Silver verification failed | {e}")
        raise


def verify_gold_output(**context):
    """
    Task 6 — Verifies dbt mart model populated the gold table.
    Checks that gold_energy_prices has data and logs row count.
    """
    from database.db_manager import DatabaseManager
    import pandas as pd

    db = DatabaseManager()

    try:
        gold_count = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM gold_energy_prices", db.engine
        ).iloc[0]["cnt"]

        latest = pd.read_sql(
            "SELECT MAX(period) as latest FROM gold_energy_prices", db.engine
        ).iloc[0]["latest"]

        logger.info(f"Gold table | rows={gold_count} | latest_period={latest}")

        if gold_count == 0:
            raise ValueError("gold_energy_prices table is empty after dbt run")

        context["ti"].xcom_push(key="gold_row_count", value=int(gold_count))
        logger.info("Gold layer verification passed")

    except Exception as e:
        logger.error(f"Gold verification failed | {e}")
        raise


def log_transform_run(**context):
    """
    Task 7 — Logs the completed transform run to pipeline_runs table.
    """
    from database.db_manager import DatabaseManager

    ti         = context["ti"]
    gold_count = ti.xcom_pull(key="gold_row_count", task_ids="verify_gold")

    db = DatabaseManager()
    db.log_pipeline_run(
        run_name="dbt_transform",
        status="success",
        rows_ingested=gold_count or 0,
    )
    logger.info(f"Transform run logged | gold_rows={gold_count}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="transform_dag",
    description="dbt transformations: bronze → silver (staging) → gold (mart)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 7 * * *",   # every day at 7:00 AM (1hr after ingestion)
    catchup=False,
    max_active_runs=1,
    tags=["dbt", "transform", "daily"],
) as dag:

    # ── Task 1: Pre-flight check ──────────────────────────────────────────────
    t_preflight = PythonOperator(
        task_id="check_bronze_data",
        python_callable=check_bronze_data,
    )

    # ── Task 2: dbt run staging models (bronze → silver) ─────────────────────
    t_dbt_staging = BashOperator(
        task_id="dbt_run_staging",
        bash_command=f"cd {DBT_DIR} && dbt run --select staging --profiles-dir ~/.dbt",
        env={**os.environ, "DBT_PROFILES_DIR": str(Path.home() / ".dbt")},
    )

    # ── Task 3: dbt test staging models ──────────────────────────────────────
    t_dbt_test_staging = BashOperator(
        task_id="dbt_test_staging",
        bash_command=f"cd {DBT_DIR} && dbt test --select staging --profiles-dir ~/.dbt",
        env={**os.environ},
    )

    # ── Task 4: Verify silver output ──────────────────────────────────────────
    t_verify_silver = PythonOperator(
        task_id="verify_silver",
        python_callable=verify_silver_output,
    )

    # ── Task 5: dbt run mart model (silver → gold) ────────────────────────────
    t_dbt_mart = BashOperator(
        task_id="dbt_run_mart",
        bash_command=f"cd {DBT_DIR} && dbt run --select marts --profiles-dir ~/.dbt",
        env={**os.environ},
    )

    # ── Task 6: Verify gold output ────────────────────────────────────────────
    t_verify_gold = PythonOperator(
        task_id="verify_gold",
        python_callable=verify_gold_output,
    )

    # ── Task 7: Log the run ───────────────────────────────────────────────────
    t_log_run = PythonOperator(
        task_id="log_transform_run",
        python_callable=log_transform_run,
    )

    # ── Task 8: Done ──────────────────────────────────────────────────────────
    t_done = EmptyOperator(task_id="transform_complete")

    # ── Dependencies ──────────────────────────────────────────────────────────
    (
        t_preflight
        >> t_dbt_staging
        >> t_dbt_test_staging
        >> t_verify_silver
        >> t_dbt_mart
        >> t_verify_gold
        >> t_log_run
        >> t_done
    )