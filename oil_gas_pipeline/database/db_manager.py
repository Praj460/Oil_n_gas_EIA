# oil_gas_pipeline | database/db_manager.py
# DatabaseManager class — single place for all PostgreSQL reads and writes
# Fixed for SQLAlchemy 2.x compatibility — uses psycopg2 directly for all reads
# Usage: from database.db_manager import DatabaseManager

import logging
import psycopg2
import psycopg2.extras
import pandas as pd
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from config.config import config

logger = logging.getLogger(__name__)


class DatabaseManager:

    def __init__(self):
        self.db_config = config.db
        logger.info(f"DatabaseManager initialized | host={self.db_config.host} | db={self.db_config.name}")

    @property
    def engine(self):
        """SQLAlchemy engine — kept for backward compatibility."""
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = create_engine(
            f"postgresql+psycopg2://{self.db_config.user}:{quote_plus(self.db_config.password)}"
            f"@{self.db_config.host}:{self.db_config.port}/{self.db_config.name}"
)
        return self._engine
    
    def _get_conn(self):
        """Returns a raw psycopg2 connection."""
        return psycopg2.connect(
            host=self.db_config.host,
            port=self.db_config.port,
            dbname=self.db_config.name,
            user=self.db_config.user,
            password=self.db_config.password,
        )

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self._get_conn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error — rolled back | {e}")
            raise
        finally:
            if conn:
                conn.close()

    def test_connection(self) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                logger.info("Database connection test passed")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed | {e}")
            return False

    def _read_sql(self, query: str) -> pd.DataFrame:
        """Runs a SELECT and returns DataFrame using psycopg2 directly."""
        conn = self._get_conn()
        try:
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()

    # ── Bronze writes ─────────────────────────────────────────────────────────

    def write_bronze_petroleum(self, df: pd.DataFrame) -> int:
        required = {"series_id", "period", "value"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        rows = 0
        with self.get_connection() as conn:
            cur = conn.cursor()
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO bronze_petroleum
                        (series_id, series_name, period, value, unit, raw_response)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (row.get("series_id"), row.get("series_name"), row.get("period"),
                      row.get("value"), row.get("unit"),
                      psycopg2.extras.Json(row.get("raw_response", {}))))
                rows += 1
        logger.info(f"Bronze petroleum — inserted {rows} rows")
        return rows

    def write_bronze_natural_gas(self, df: pd.DataFrame) -> int:
        required = {"series_id", "period", "value"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        rows = 0
        with self.get_connection() as conn:
            cur = conn.cursor()
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO bronze_natural_gas
                        (series_id, series_name, period, value, unit, raw_response)
                    VALUES (%s,%s,%s,%s,%s,%s)
                """, (row.get("series_id"), row.get("series_name"), row.get("period"),
                      row.get("value"), row.get("unit"),
                      psycopg2.extras.Json(row.get("raw_response", {}))))
                rows += 1
        logger.info(f"Bronze natural gas — inserted {rows} rows")
        return rows

    def write_bronze_well_production(self, df: pd.DataFrame) -> int:
        with self.get_connection() as conn:
            cur = conn.cursor()
            data = [(row.get("well_id"), row.get("state"), row.get("production_date"),
                     row.get("oil_bbl"), row.get("gas_mcf"), row.get("water_bbl"))
                    for _, row in df.iterrows()]
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO bronze_well_production
                    (well_id, state, production_date, oil_bbl, gas_mcf, water_bbl)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, data, page_size=500)
        logger.info(f"Bronze well production — inserted {len(data)} rows")
        return len(data)

    # ── Silver reads ──────────────────────────────────────────────────────────

    def read_silver_petroleum(self, start_date=None, end_date=None) -> pd.DataFrame:
        q = "SELECT * FROM silver_petroleum WHERE 1=1"
        if start_date: q += f" AND period >= '{start_date}'"
        if end_date:   q += f" AND period <= '{end_date}'"
        return self._read_sql(q + " ORDER BY period ASC")

    def read_silver_natural_gas(self, start_date=None, end_date=None) -> pd.DataFrame:
        q = "SELECT * FROM silver_natural_gas WHERE 1=1"
        if start_date: q += f" AND period >= '{start_date}'"
        if end_date:   q += f" AND period <= '{end_date}'"
        return self._read_sql(q + " ORDER BY period ASC")

    # ── Gold reads and writes ─────────────────────────────────────────────────

    def read_gold_energy_prices(self) -> pd.DataFrame:
        df = self._read_sql("SELECT * FROM gold_energy_prices ORDER BY period ASC")
        logger.info(f"Read gold energy prices — {len(df)} rows")
        return df

    def write_forecast_results(self, df: pd.DataFrame) -> int:
        required = {"target", "model_name", "forecast_period", "forecast_value"}
        if missing := required - set(df.columns):
            raise ValueError(f"Missing columns: {missing}")
        with self.get_connection() as conn:
            cur  = conn.cursor()
            data = [(row.get("target"), row.get("model_name"), row.get("forecast_period"),
                     row.get("forecast_value"), row.get("lower_bound"), row.get("upper_bound"),
                     row.get("rmse"), row.get("mape"), row.get("trained_on_periods"))
                    for _, row in df.iterrows()]
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO gold_forecast_results
                    (target, model_name, forecast_period, forecast_value,
                     lower_bound, upper_bound, rmse, mape, trained_on_periods)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, data)
        logger.info(f"Gold forecast results — inserted {len(data)} rows")
        return len(data)

    def read_forecast_results(self, target=None, model_name=None) -> pd.DataFrame:
        q = "SELECT * FROM gold_forecast_results WHERE 1=1"
        if target:     q += f" AND target = '{target}'"
        if model_name: q += f" AND model_name = '{model_name}'"
        df = self._read_sql(q + " ORDER BY forecast_period ASC")
        logger.info(f"Read forecast results — {len(df)} rows")
        return df

    # ── Pipeline logging ──────────────────────────────────────────────────────

    def log_pipeline_run(self, run_name, status, rows_ingested=0,
                         rows_failed=0, error_message=None, started_at=None) -> str:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO pipeline_runs
                    (run_name, status, rows_ingested, rows_failed,
                     error_message, started_at, finished_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """, (run_name, status, rows_ingested, rows_failed, error_message,
                  started_at or datetime.now(timezone.utc), datetime.now(timezone.utc)))
            run_id = str(cur.fetchone()[0])
        logger.info(f"Pipeline run logged | run={run_name} | status={status} | id={run_id}")
        return run_id

    def log_data_quality(self, suite_name, table_name, total_expectations, passed, failed):
        rate = round((passed / total_expectations) * 100, 2) if total_expectations > 0 else 0.0
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO data_quality_results
                    (suite_name, table_name, total_expectations, passed, failed, success_rate)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (suite_name, table_name, total_expectations, passed, failed, rate))
        logger.info(f"Data quality logged | suite={suite_name} | passed={passed}/{total_expectations}")

    def read_pipeline_runs(self, limit=50) -> pd.DataFrame:
        return self._read_sql(f"SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT {limit}")

    def read_data_quality_results(self, limit=50) -> pd.DataFrame:
        return self._read_sql(f"SELECT * FROM data_quality_results ORDER BY run_at DESC LIMIT {limit}")