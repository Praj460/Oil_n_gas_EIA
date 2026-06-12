# oil_gas_pipeline | dags/forecast_dag.py
# Airflow DAG — runs every Sunday at 8am
# Reads gold_energy_prices → preprocesses → fits SARIMA + Prophet
# → saves forecast results to gold_forecast_results table

import sys
sys.path.insert(0, '/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline')


from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from config.logging_config import setup_logging
setup_logging(log_filename="forecast.log")

import logging
logger = logging.getLogger(__name__)


# ── Default DAG arguments ─────────────────────────────────────────────────────

default_args = {
    "owner":            "oil_gas_pipeline",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=15),
}

# Targets to forecast — one run per target per model
FORECAST_TARGETS = ["wti_price", "henry_hub_price"]

# ── Task functions ────────────────────────────────────────────────────────────

def load_and_preprocess(**context):
    """
    Task 1 — Loads gold_energy_prices and preprocesses it for forecasting.
    Saves preprocessed DataFrames to XCom for downstream forecast tasks.
    """
    from database.db_manager import DatabaseManager
    from models.preprocessor import Preprocessor

    db  = DatabaseManager()
    df  = db.read_gold_energy_prices()

    if df.empty:
        raise ValueError(
            "gold_energy_prices is empty — run transform_dag first"
        )

    logger.info(f"Loaded gold data | rows={len(df)} | cols={list(df.columns)}")

    prep_results = {}

    for target in FORECAST_TARGETS:
        if target not in df.columns:
            logger.warning(f"Target {target} not found in gold table — skipping")
            continue

        prep   = Preprocessor()
        clean  = prep.fit_transform(
            df,
            target_col=target,
            feature_cols=[c for c in df.columns if c != target and c != "period"],
        )

        summary = prep.summary(clean)
        logger.info(f"Preprocessed {target} | {summary}")

        # Store as JSON-serializable dict for XCom
        prep_results[target] = {
            "rows":      len(clean),
            "date_range": summary["date_range"],
        }

        # Save preprocessed DataFrame to parquet for forecast tasks to read
        from config.config import config
        out_path = config.paths.data_processed / f"{target}_preprocessed.parquet"
        clean.to_parquet(out_path)
        logger.info(f"Saved preprocessed data | path={out_path}")

    context["ti"].xcom_push(key="prep_results", value=prep_results)


def run_forecast(target: str, model_name: str):
    """
    Factory function — returns a task function for a specific target + model.
    This pattern lets us dynamically create tasks for each combination.

    Args:
        target:     e.g. "wti_price"
        model_name: "sarima" or "prophet"
    """
    def _forecast(**context):
        import pandas as pd
        from config.config import config
        from models.forecast_model import ForecastModel
        from models.evaluator import Evaluator
        from models.preprocessor import Preprocessor
        from database.db_manager import DatabaseManager

        logger.info(f"Running {model_name} forecast for {target}")

        # Load preprocessed data
        data_path = config.paths.data_processed / f"{target}_preprocessed.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {data_path} — "
                f"did load_and_preprocess task run successfully?"
            )

        df = pd.read_parquet(data_path)

        # Train/test split — 80% train, 20% test
        prep        = Preprocessor()
        prep.target_col = target
        train, test = prep.train_test_split(df, target, test_size=0.2)

        # Fit and forecast
        fm     = ForecastModel()
        result = fm.fit_predict(train, target, model=model_name)
        result.target = target

        # Evaluate on test set using fitted values
        evaluator = Evaluator()

        if result.fitted_values is not None:
            # Align fitted values with test set periods
            test_fitted = result.fitted_values.reindex(test.index)
            metrics     = evaluator.evaluate(
                actual=test[target],
                predicted=test_fitted,
                model_name=model_name,
                target=target,
            )
            logger.info(f"Backtest metrics | {metrics.summary()}")

            # Add metrics to forecast result for DB storage
            result.forecast_df["rmse"] = metrics.rmse
            result.forecast_df["mape"] = metrics.mape
        else:
            result.forecast_df["rmse"] = None
            result.forecast_df["mape"] = None

        # Save model to disk
        fm.save(model_name=model_name, target_col=target)

        # Write forecast results to gold layer
        db     = DatabaseManager()
        db_df  = result.to_db_df()
        rows   = db.write_forecast_results(db_df)

        logger.info(
            f"Forecast complete | model={model_name} | target={target} | "
            f"horizon={len(result.forecast_df)} | rows_saved={rows}"
        )

        context["ti"].xcom_push(
            key=f"{model_name}_{target}_rows",
            value=rows,
        )

    _forecast.__name__ = f"forecast_{model_name}_{target}"
    return _forecast


def log_forecast_run(**context):
    """
    Final task — logs the completed weekly forecast run to pipeline_runs.
    """
    from database.db_manager import DatabaseManager

    ti         = context["ti"]
    total_rows = 0

    for target in FORECAST_TARGETS:
        for model_name in ["sarima", "prophet"]:
            rows = ti.xcom_pull(
                key=f"{model_name}_{target}_rows",
                task_ids=f"forecast_{model_name}_{target}",
            )
            total_rows += rows or 0

    db = DatabaseManager()
    db.log_pipeline_run(
        run_name="weekly_forecast",
        status="success",
        rows_ingested=total_rows,
    )
    logger.info(f"Weekly forecast run logged | total_forecast_rows={total_rows}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="forecast_dag",
    description="Weekly SARIMA + Prophet forecasting for WTI and Henry Hub prices",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 8 * * 0",   # every Sunday at 8:00 AM
    catchup=False,
    max_active_runs=1,
    tags=["forecast", "sarima", "prophet", "weekly"],
) as dag:

    # ── Task 1: Load and preprocess ───────────────────────────────────────────
    t_preprocess = PythonOperator(
        task_id="load_and_preprocess",
        python_callable=load_and_preprocess,
    )

    # ── Tasks 2-5: Forecast each target with each model ───────────────────────
    # Dynamically creates 4 tasks:
    #   forecast_sarima_wti_price
    #   forecast_prophet_wti_price
    #   forecast_sarima_henry_hub_price
    #   forecast_prophet_henry_hub_price

    forecast_tasks = []

    for target in FORECAST_TARGETS:
        for model_name in ["sarima", "prophet"]:
            task = PythonOperator(
                task_id=f"forecast_{model_name}_{target}",
                python_callable=run_forecast(target, model_name),
            )
            forecast_tasks.append(task)
            # All forecast tasks run after preprocessing
            t_preprocess >> task

    # ── Task 6: Log the run ───────────────────────────────────────────────────
    t_log = PythonOperator(
        task_id="log_forecast_run",
        python_callable=log_forecast_run,
    )

    # ── Task 7: Done ──────────────────────────────────────────────────────────
    t_done = EmptyOperator(task_id="forecast_complete")

    # All forecast tasks must complete before logging
    forecast_tasks >> t_log >> t_done