# oil_gas_pipeline | ingestion/kaggle_loader.py
# KaggleLoader class — reads historical well production CSV from data/raw/
# One-time historical load, not refreshed daily like the EIA API
# Usage: from ingestion.kaggle_loader import KaggleLoader

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from config.config import config

logger = logging.getLogger(__name__)


class KaggleLoader:
    """
    Loads historical well-level oil and gas production data from
    a locally stored Kaggle CSV file.

    Expected file location: data/raw/well_production.csv

    Expected CSV columns (flexible — maps common naming variants):
        - well identifier  : well_id, WellID, API, api_number
        - state            : state, State, STATE
        - date             : production_date, date, Date, ReportDate
        - oil production   : oil_bbl, OilBBL, Oil_BBL, oil_production
        - gas production   : gas_mcf, GasMCF, Gas_MCF, gas_production
        - water production : water_bbl, WaterBBL, Water_BBL (optional)
    """

    # Maps common Kaggle column name variants → our standard names
    COLUMN_MAP = {
        # well id variants
        "wellid":         "well_id",
        "well_id":        "well_id",
        "api":            "well_id",
        "api_number":     "well_id",
        "apinumber":      "well_id",

        # state variants
        "state":          "state",
        "statename":      "state",
        "state_name":     "state",

        # date variants
        "production_date": "production_date",
        "date":            "production_date",
        "reportdate":      "production_date",
        "report_date":     "production_date",
        "proddate":        "production_date",

        # oil variants
        "oil_bbl":         "oil_bbl",
        "oilbbl":          "oil_bbl",
        "oil_bbl_per_day": "oil_bbl",
        "oil_production":  "oil_bbl",
        "oilproduction":   "oil_bbl",
        "liquid":          "oil_bbl",

        # gas variants
        "gas_mcf":         "gas_mcf",
        "gasmcf":          "gas_mcf",
        "gas_production":  "gas_mcf",
        "gasproduction":   "gas_mcf",
        "gas":             "gas_mcf",

        # water variants
        "water_bbl":       "water_bbl",
        "waterbbl":        "water_bbl",
        "water_production": "water_bbl",
        "water":           "water_bbl",
    }

    def __init__(self, file_path: Optional[Path] = None):
        """
        Args:
            file_path: path to the CSV file.
                       Defaults to data/raw/well_production.csv
        """
        self.file_path = file_path or (config.paths.data_raw / "well_production.csv")
        logger.info(f"KaggleLoader initialized | file={self.file_path}")

    # ── Core load method ──────────────────────────────────────────────────────

    def load(self, chunksize: Optional[int] = None) -> pd.DataFrame:
        """
        Loads and cleans the well production CSV.

        Args:
            chunksize: if set, reads CSV in chunks (useful for very large files).
                       None reads the entire file at once.

        Returns:
            Clean DataFrame with standardized columns:
            [well_id, state, production_date, oil_bbl, gas_mcf, water_bbl]

        Raises:
            FileNotFoundError if the CSV doesn't exist at file_path
            ValueError if required columns are missing after mapping
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Kaggle CSV not found at: {self.file_path}\n"
                f"Download from https://www.kaggle.com/datasets/banlevan/oil-and-gas-production-data\n"
                f"and place it at data/raw/well_production.csv"
            )

        logger.info(f"Loading Kaggle CSV from {self.file_path}")

        if chunksize:
            chunks = []
            for chunk in pd.read_csv(self.file_path, chunksize=chunksize, low_memory=False):
                chunks.append(self._clean(chunk))
            df = pd.concat(chunks, ignore_index=True)
        else:
            raw = pd.read_csv(self.file_path, low_memory=False)
            df = self._clean(raw)

        logger.info(f"Kaggle CSV loaded | rows={len(df)}")
        return df

    # ── Cleaning pipeline ─────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all cleaning steps to a raw DataFrame chunk.

        Steps:
        1. Normalize column names (lowercase, strip whitespace)
        2. Map to standard column names using COLUMN_MAP
        3. Keep only the columns we need
        4. Parse production_date to datetime
        5. Cast numeric columns to float
        6. Drop rows missing both oil and gas values
        7. Clip negative values to zero
        8. Remove duplicate rows

        Args:
            df: raw DataFrame from pd.read_csv

        Returns:
            Cleaned DataFrame
        """
        # Step 1 — normalize column names
        df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

        # Step 2 — map to standard names
        df = df.rename(columns=self.COLUMN_MAP)

        # Step 3 — keep only standard columns that exist
        standard_cols = ["well_id", "state", "production_date", "oil_bbl", "gas_mcf", "water_bbl"]
        present_cols  = [c for c in standard_cols if c in df.columns]
        df = df[present_cols].copy()

        # Check required columns exist after mapping
        required = {"production_date"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing after mapping: {missing}\n"
                f"Available columns in CSV: {list(df.columns)}\n"
                f"Update COLUMN_MAP in KaggleLoader to match your CSV."
            )

        # Step 4 — parse dates
        df["production_date"] = pd.to_datetime(
            df["production_date"], errors="coerce", infer_datetime_format=True
        )

        # Step 5 — cast numeric columns
        for col in ["oil_bbl", "gas_mcf", "water_bbl"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 6 — drop rows where both oil and gas are null
        if "oil_bbl" in df.columns and "gas_mcf" in df.columns:
            before = len(df)
            df = df.dropna(subset=["oil_bbl", "gas_mcf"], how="all")
            dropped = before - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with null oil and gas values")

        # Also drop rows where production_date couldn't be parsed
        before = len(df)
        df = df.dropna(subset=["production_date"])
        date_dropped = before - len(df)
        if date_dropped > 0:
            logger.warning(f"Dropped {date_dropped} rows with unparseable dates")

        # Step 7 — clip negatives to zero
        for col in ["oil_bbl", "gas_mcf", "water_bbl"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Step 8 — remove duplicates
        before = len(df)
        df = df.drop_duplicates()
        dupes = before - len(df)
        if dupes > 0:
            logger.info(f"Removed {dupes} duplicate rows")

        df = df.reset_index(drop=True)
        return df

    # ── Summary stats ─────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> dict:
        """
        Returns a quick summary of the loaded dataset.
        Useful for logging and the Streamlit data quality page.

        Args:
            df: cleaned DataFrame from load()

        Returns:
            Dict with row count, date range, null counts, state count
        """
        summary = {
            "total_rows":    len(df),
            "date_range":    f"{df['production_date'].min()} → {df['production_date'].max()}" if "production_date" in df.columns else "N/A",
            "unique_wells":  df["well_id"].nunique() if "well_id" in df.columns else "N/A",
            "unique_states": df["state"].nunique() if "state" in df.columns else "N/A",
            "null_oil":      df["oil_bbl"].isna().sum() if "oil_bbl" in df.columns else "N/A",
            "null_gas":      df["gas_mcf"].isna().sum() if "gas_mcf" in df.columns else "N/A",
        }
        logger.info(f"Dataset summary: {summary}")
        return summary