# oil_gas_pipeline | ingestion/ingester.py
# DataIngester class — orchestrates EIAClient + KaggleLoader
# Validates schema, writes to bronze PostgreSQL tables, logs pipeline runs
# Usage: from ingestion.ingester import DataIngester

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ingestion.eia_client import EIAClient
from ingestion.kaggle_loader import KaggleLoader
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class DataIngester:
    """
    Orchestrates the full ingestion layer.

    Responsibilities:
    - Calls EIAClient to fetch petroleum and natural gas data
    - Calls KaggleLoader to load well production CSV
    - Validates that DataFrames are not empty and have required columns
    - Writes all data to bronze PostgreSQL tables via DatabaseManager
    - Logs every run (success or failure) to pipeline_runs table

    This is the class called by the Airflow daily_ingest_dag.
    """

    def __init__(self):
        self.eia_client  = EIAClient()
        self.kaggle      = KaggleLoader()
        self.db          = DatabaseManager()
        logger.info("DataIngester initialized")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame, required_cols: set, name: str) -> bool:
        """
        Validates a DataFrame before writing to the database.

        Checks:
        - DataFrame is not None
        - DataFrame is not empty
        - All required columns are present

        Args:
            df:            DataFrame to validate
            required_cols: set of column names that must exist
            name:          label for logging (e.g. "WTI price")

        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            logger.error(f"Validation failed — {name}: DataFrame is empty")
            return False

        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Validation failed — {name}: missing columns {missing}")
            return False

        logger.debug(f"Validation passed — {name}: {len(df)} rows")
        return True

    # ── Petroleum ingestion ───────────────────────────────────────────────────

    def ingest_petroleum(self, start: str = "2000-01") -> dict:
        """
        Fetches all petroleum series from EIA API and writes to bronze_petroleum.

        Args:
            start: start period for historical pull e.g. "2000-01"

        Returns:
            Dict with rows_inserted and rows_failed counts per series
        """
        logger.info("Starting petroleum ingestion...")
        started_at = datetime.now(timezone.utc)
        results    = {"wti": 0, "brent": 0, "production": 0, "failed": []}

        # Fetch all petroleum series
        petroleum_data = self.eia_client.fetch_all_petroleum(start=start)

        required = {"series_id", "period", "value"}

        # WTI prices
        if self._validate(petroleum_data.get("wti"), required, "WTI price"):
            try:
                rows = self.db.write_bronze_petroleum(petroleum_data["wti"])
                results["wti"] = rows
            except Exception as e:
                logger.error(f"Failed to write WTI data | {e}")
                results["failed"].append("wti")
        else:
            results["failed"].append("wti")

        # Brent prices
        if self._validate(petroleum_data.get("brent"), required, "Brent price"):
            try:
                rows = self.db.write_bronze_petroleum(petroleum_data["brent"])
                results["brent"] = rows
            except Exception as e:
                logger.error(f"Failed to write Brent data | {e}")
                results["failed"].append("brent")
        else:
            results["failed"].append("brent")

        # US production
        if self._validate(petroleum_data.get("production"), required, "US oil production"):
            try:
                rows = self.db.write_bronze_petroleum(petroleum_data["production"])
                results["production"] = rows
            except Exception as e:
                logger.error(f"Failed to write US oil production | {e}")
                results["failed"].append("production")
        else:
            results["failed"].append("production")

        total_inserted = results["wti"] + results["brent"] + results["production"]
        total_failed   = len(results["failed"])
        status         = "success" if total_failed == 0 else "failed"

        self.db.log_pipeline_run(
            run_name="petroleum_ingestion",
            status=status,
            rows_ingested=total_inserted,
            rows_failed=total_failed,
            error_message=str(results["failed"]) if results["failed"] else None,
            started_at=started_at,
        )

        logger.info(
            f"Petroleum ingestion complete | "
            f"inserted={total_inserted} | failed_series={results['failed']}"
        )
        return results

    # ── Natural gas ingestion ─────────────────────────────────────────────────

    def ingest_natural_gas(self, start: str = "2000-01") -> dict:
        """
        Fetches all natural gas series from EIA API and writes to bronze_natural_gas.

        Args:
            start: start period for historical pull e.g. "2000-01"

        Returns:
            Dict with rows_inserted and rows_failed counts per series
        """
        logger.info("Starting natural gas ingestion...")
        started_at = datetime.now(timezone.utc)
        results    = {"henry_hub": 0, "storage": 0, "production": 0, "failed": []}

        gas_data = self.eia_client.fetch_all_natural_gas(start=start)
        required = {"series_id", "period", "value"}

        # Henry Hub prices
        if self._validate(gas_data.get("henry_hub"), required, "Henry Hub price"):
            try:
                rows = self.db.write_bronze_natural_gas(gas_data["henry_hub"])
                results["henry_hub"] = rows
            except Exception as e:
                logger.error(f"Failed to write Henry Hub data | {e}")
                results["failed"].append("henry_hub")
        else:
            results["failed"].append("henry_hub")

        # US gas storage
        if self._validate(gas_data.get("storage"), required, "US gas storage"):
            try:
                rows = self.db.write_bronze_natural_gas(gas_data["storage"])
                results["storage"] = rows
            except Exception as e:
                logger.error(f"Failed to write US gas storage | {e}")
                results["failed"].append("storage")
        else:
            results["failed"].append("storage")

        # US gas production
        if self._validate(gas_data.get("production"), required, "US gas production"):
            try:
                rows = self.db.write_bronze_natural_gas(gas_data["production"])
                results["production"] = rows
            except Exception as e:
                logger.error(f"Failed to write US gas production | {e}")
                results["failed"].append("production")
        else:
            results["failed"].append("production")

        total_inserted = results["henry_hub"] + results["storage"] + results["production"]
        total_failed   = len(results["failed"])
        status         = "success" if total_failed == 0 else "failed"

        self.db.log_pipeline_run(
            run_name="natural_gas_ingestion",
            status=status,
            rows_ingested=total_inserted,
            rows_failed=total_failed,
            error_message=str(results["failed"]) if results["failed"] else None,
            started_at=started_at,
        )

        logger.info(
            f"Natural gas ingestion complete | "
            f"inserted={total_inserted} | failed_series={results['failed']}"
        )
        return results

    # ── Kaggle well production ingestion ──────────────────────────────────────



    # ── Run everything ────────────────────────────────────────────────────────

    def run_full_ingestion(
        self,
        start: str = "2000-01",
        include_kaggle: bool = True,
    ) -> dict:
        """
        Runs the complete ingestion pipeline — all sources in sequence.
        Called by the Airflow daily_ingest_dag.

        Args:
            start:          historical start period e.g. "2000-01"
            include_kaggle: set to False after first run to skip the
                            large CSV re-load on daily refreshes

        Returns:
            Summary dict with results from all three ingestion steps
        """
        logger.info("=" * 60)
        logger.info("FULL INGESTION PIPELINE STARTED")
        logger.info("=" * 60)

        summary = {}

        summary["petroleum"]    = self.ingest_petroleum(start=start)
        summary["natural_gas"]  = self.ingest_natural_gas(start=start)

        if include_kaggle:
            summary["well_production"] = self.ingest_well_production()
        else:
            logger.info("Skipping Kaggle well production (include_kaggle=False)")
            summary["well_production"] = {"status": "skipped"}

        logger.info("=" * 60)
        logger.info(f"FULL INGESTION COMPLETE | summary={summary}")
        logger.info("=" * 60)

        return summary