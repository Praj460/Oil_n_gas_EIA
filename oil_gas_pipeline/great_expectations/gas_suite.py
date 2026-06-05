# oil_gas_pipeline | great_expectations/gas_suite.py
# Data quality checks for bronze_natural_gas table
# Validates price ranges, storage levels, nulls, and date integrity
# Usage: from great_expectations.gas_suite import GasSuite

import logging
import pandas as pd
from datetime import datetime
from typing import Optional

from great_expectations.petroleum_suite import ExpectationResult, SuiteReport

logger = logging.getLogger(__name__)


class GasSuite:
    """
    Data quality suite for natural gas price, storage, and production data.

    Mirrors PetroleumSuite structure but with gas-specific bounds and series IDs.

    Usage:
        suite  = GasSuite()
        report = suite.run(df)
        print(report.summary())
    """

    VALID_SERIES = {
        "NG.RNGWHHD.M",           # Henry Hub price
        "NG.NW2_EPG0_SWO_R48_BCF.M",  # US storage
        "NG.N9010US2.M",          # US production
    }

    # Henry Hub price has ranged from ~$1 to ~$20/MMBtu historically
    HENRY_HUB_MIN, HENRY_HUB_MAX = 0.0, 25.0

    # US gas storage in BCF — typically 1,000 to 4,000 BCF
    STORAGE_MIN, STORAGE_MAX = 0.0, 5000.0

    MIN_DATE = datetime(1993, 1, 1)   # Henry Hub data starts ~1993
    MAX_DATE = datetime.now()

    def __init__(self):
        self.results: list[ExpectationResult] = []

    # ── Individual expectations ───────────────────────────────────────────────

    def expect_no_null_series_id(self, df: pd.DataFrame) -> ExpectationResult:
        nulls = df["series_id"].isna().sum()
        return ExpectationResult(
            expectation="expect_no_null_series_id",
            passed=nulls == 0,
            column="series_id",
            details=f"{nulls} null series_id values" if nulls else "All series_id values present",
        )

    def expect_no_null_period(self, df: pd.DataFrame) -> ExpectationResult:
        nulls = df["period"].isna().sum()
        return ExpectationResult(
            expectation="expect_no_null_period",
            passed=nulls == 0,
            column="period",
            details=f"{nulls} null period values" if nulls else "All periods present",
        )

    def expect_no_null_value(self, df: pd.DataFrame) -> ExpectationResult:
        """Allows up to 5% nulls — EIA occasionally has gaps in gas data."""
        total    = len(df)
        nulls    = df["value"].isna().sum()
        null_pct = (nulls / total * 100) if total > 0 else 0
        threshold = 5.0
        return ExpectationResult(
            expectation="expect_no_null_value",
            passed=null_pct <= threshold,
            column="value",
            details=f"{nulls} null values ({null_pct:.1f}%) — threshold {threshold}%",
        )

    def expect_valid_series_ids(self, df: pd.DataFrame) -> ExpectationResult:
        unknown = set(df["series_id"].dropna().unique()) - self.VALID_SERIES
        return ExpectationResult(
            expectation="expect_valid_series_ids",
            passed=len(unknown) == 0,
            column="series_id",
            details=f"Unknown series IDs: {unknown}" if unknown else "All series IDs valid",
        )

    def expect_henry_hub_price_in_range(self, df: pd.DataFrame) -> ExpectationResult:
        """Henry Hub spot price must be between $0 and $25/MMBtu."""
        hh = df[df["series_id"] == "NG.RNGWHHD.M"]["value"].dropna()
        if hh.empty:
            return ExpectationResult(
                expectation="expect_henry_hub_price_in_range",
                passed=True,
                column="value",
                details="No Henry Hub rows found — skipped",
            )
        out_of_range = ((hh < self.HENRY_HUB_MIN) | (hh > self.HENRY_HUB_MAX)).sum()
        return ExpectationResult(
            expectation="expect_henry_hub_price_in_range",
            passed=out_of_range == 0,
            column="value",
            details=(
                f"{out_of_range} Henry Hub values outside [{self.HENRY_HUB_MIN}, {self.HENRY_HUB_MAX}]"
                if out_of_range else
                f"All Henry Hub prices in range | min={hh.min():.2f} max={hh.max():.2f}"
            ),
        )

    def expect_storage_in_range(self, df: pd.DataFrame) -> ExpectationResult:
        """US gas storage must be between 0 and 5,000 BCF."""
        stor_id = "NG.NW2_EPG0_SWO_R48_BCF.M"
        stor = df[df["series_id"] == stor_id]["value"].dropna()
        if stor.empty:
            return ExpectationResult(
                expectation="expect_storage_in_range",
                passed=True,
                column="value",
                details="No storage rows found — skipped",
            )
        out_of_range = ((stor < self.STORAGE_MIN) | (stor > self.STORAGE_MAX)).sum()
        return ExpectationResult(
            expectation="expect_storage_in_range",
            passed=out_of_range == 0,
            column="value",
            details=(
                f"{out_of_range} storage values outside [{self.STORAGE_MIN}, {self.STORAGE_MAX}] BCF"
                if out_of_range else
                f"All storage values in range | min={stor.min():.0f} max={stor.max():.0f} BCF"
            ),
        )

    def expect_no_future_periods(self, df: pd.DataFrame) -> ExpectationResult:
        periods = pd.to_datetime(df["period"], errors="coerce")
        future  = (periods > pd.Timestamp.now()).sum()
        return ExpectationResult(
            expectation="expect_no_future_periods",
            passed=future == 0,
            column="period",
            details=f"{future} future periods found" if future else "No future periods",
        )

    def expect_no_duplicate_rows(self, df: pd.DataFrame) -> ExpectationResult:
        dupes = df.duplicated(subset=["series_id", "period"]).sum()
        return ExpectationResult(
            expectation="expect_no_duplicate_rows",
            passed=dupes == 0,
            column="series_id, period",
            details=f"{dupes} duplicate rows" if dupes else "No duplicates",
        )

    def expect_minimum_row_count(self, df: pd.DataFrame, min_rows: int = 100) -> ExpectationResult:
        total = len(df)
        return ExpectationResult(
            expectation="expect_minimum_row_count",
            passed=total >= min_rows,
            column=None,
            details=f"{total} rows found — minimum required: {min_rows}",
        )

    def expect_period_not_before_min_date(self, df: pd.DataFrame) -> ExpectationResult:
        periods = pd.to_datetime(df["period"], errors="coerce").dropna()
        too_old = (periods < pd.Timestamp(self.MIN_DATE)).sum()
        return ExpectationResult(
            expectation="expect_period_not_before_min_date",
            passed=too_old == 0,
            column="period",
            details=f"{too_old} rows before {self.MIN_DATE.date()}" if too_old else f"All periods after {self.MIN_DATE.date()}",
        )

    def expect_henry_hub_not_negative(self, df: pd.DataFrame) -> ExpectationResult:
        """
        Henry Hub price must never be negative.
        Unlike WTI which briefly went negative in April 2020,
        Henry Hub has never gone negative on a monthly basis.
        """
        hh = df[df["series_id"] == "NG.RNGWHHD.M"]["value"].dropna()
        if hh.empty:
            return ExpectationResult(
                expectation="expect_henry_hub_not_negative",
                passed=True,
                column="value",
                details="No Henry Hub rows — skipped",
            )
        negatives = (hh < 0).sum()
        return ExpectationResult(
            expectation="expect_henry_hub_not_negative",
            passed=negatives == 0,
            column="value",
            details=f"{negatives} negative Henry Hub values" if negatives else "No negative Henry Hub prices",
        )

    # ── Run all expectations ──────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> SuiteReport:
        """
        Runs all expectations against the natural gas DataFrame.

        Args:
            df: DataFrame from bronze_natural_gas table

        Returns:
            SuiteReport with all results and summary stats
        """
        logger.info(f"Running GasSuite on {len(df)} rows...")

        self.results = [
            self.expect_no_null_series_id(df),
            self.expect_no_null_period(df),
            self.expect_no_null_value(df),
            self.expect_valid_series_ids(df),
            self.expect_henry_hub_price_in_range(df),
            self.expect_storage_in_range(df),
            self.expect_no_future_periods(df),
            self.expect_no_duplicate_rows(df),
            self.expect_minimum_row_count(df),
            self.expect_period_not_before_min_date(df),
            self.expect_henry_hub_not_negative(df),
        ]

        report = SuiteReport(suite_name="gas_suite", results=self.results)
        logger.info(
            f"GasSuite complete | "
            f"passed={report.passed}/{report.total} | "
            f"success_rate={report.success_rate:.1f}%"
        )
        return report