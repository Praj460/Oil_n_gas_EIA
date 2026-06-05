# oil_gas_pipeline | great_expectations/petroleum_suite.py
# Data quality checks for bronze_petroleum table
# Validates price ranges, nulls, date integrity, and schema
# Usage: from great_expectations.petroleum_suite import PetroleumSuite

import logging
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExpectationResult:
    """Result of a single data quality check."""
    expectation:  str
    passed:       bool
    column:       Optional[str]
    details:      str


class PetroleumSuite:
    """
    Data quality suite for petroleum price and production data.

    Runs expectations against a DataFrame loaded from bronze_petroleum.
    Each expectation is a method that returns an ExpectationResult.

    Designed to mirror how Great Expectations works — each check is:
    - Named clearly (what it checks)
    - Self-contained (no side effects)
    - Returns pass/fail + detail message

    Usage:
        suite  = PetroleumSuite()
        report = suite.run(df)
        print(report.summary())
    """

    # Valid EIA petroleum series IDs we expect to see
    VALID_SERIES = {
        "PET.RWTC.M",      # WTI spot price
        "PET.RBRTE.M",     # Brent spot price
        "PET.MCRFPUS2.M",  # US crude production
    }

    # Reasonable price bounds — crude oil has never gone above $200
    # or below $0 on a monthly average basis
    WTI_MIN,   WTI_MAX   = 0.0, 200.0
    BRENT_MIN, BRENT_MAX = 0.0, 200.0

    # Earliest and latest valid data periods
    MIN_DATE = datetime(1986, 1, 1)   # EIA petroleum data starts ~1986
    MAX_DATE = datetime.now()

    def __init__(self):
        self.results: list[ExpectationResult] = []

    # ── Individual expectations ───────────────────────────────────────────────

    def expect_no_null_series_id(self, df: pd.DataFrame) -> ExpectationResult:
        """series_id must never be null — it identifies what data we have."""
        nulls = df["series_id"].isna().sum()
        return ExpectationResult(
            expectation="expect_no_null_series_id",
            passed=nulls == 0,
            column="series_id",
            details=f"{nulls} null series_id values found" if nulls else "All series_id values present",
        )

    def expect_no_null_period(self, df: pd.DataFrame) -> ExpectationResult:
        """period must never be null — every row needs a timestamp."""
        nulls = df["period"].isna().sum()
        return ExpectationResult(
            expectation="expect_no_null_period",
            passed=nulls == 0,
            column="period",
            details=f"{nulls} null period values found" if nulls else "All periods present",
        )

    def expect_no_null_value(self, df: pd.DataFrame) -> ExpectationResult:
        """
        Value (price or production) should not be null.
        Allows up to 5% nulls — EIA occasionally has data gaps.
        """
        total = len(df)
        nulls = df["value"].isna().sum()
        null_pct = (nulls / total * 100) if total > 0 else 0
        threshold = 5.0
        return ExpectationResult(
            expectation="expect_no_null_value",
            passed=null_pct <= threshold,
            column="value",
            details=f"{nulls} null values ({null_pct:.1f}%) — threshold {threshold}%",
        )

    def expect_valid_series_ids(self, df: pd.DataFrame) -> ExpectationResult:
        """All series_id values must be in our known valid set."""
        unknown = set(df["series_id"].dropna().unique()) - self.VALID_SERIES
        return ExpectationResult(
            expectation="expect_valid_series_ids",
            passed=len(unknown) == 0,
            column="series_id",
            details=f"Unknown series IDs: {unknown}" if unknown else "All series IDs valid",
        )

    def expect_wti_price_in_range(self, df: pd.DataFrame) -> ExpectationResult:
        """WTI prices must be between $0 and $200 per barrel."""
        wti = df[df["series_id"] == "PET.RWTC.M"]["value"].dropna()
        if wti.empty:
            return ExpectationResult(
                expectation="expect_wti_price_in_range",
                passed=True,
                column="value",
                details="No WTI rows found — skipped",
            )
        out_of_range = ((wti < self.WTI_MIN) | (wti > self.WTI_MAX)).sum()
        return ExpectationResult(
            expectation="expect_wti_price_in_range",
            passed=out_of_range == 0,
            column="value",
            details=(
                f"{out_of_range} WTI values outside [{self.WTI_MIN}, {self.WTI_MAX}]"
                if out_of_range else
                f"All WTI prices in range | min={wti.min():.2f} max={wti.max():.2f}"
            ),
        )

    def expect_brent_price_in_range(self, df: pd.DataFrame) -> ExpectationResult:
        """Brent prices must be between $0 and $200 per barrel."""
        brent = df[df["series_id"] == "PET.RBRTE.M"]["value"].dropna()
        if brent.empty:
            return ExpectationResult(
                expectation="expect_brent_price_in_range",
                passed=True,
                column="value",
                details="No Brent rows found — skipped",
            )
        out_of_range = ((brent < self.BRENT_MIN) | (brent > self.BRENT_MAX)).sum()
        return ExpectationResult(
            expectation="expect_brent_price_in_range",
            passed=out_of_range == 0,
            column="value",
            details=(
                f"{out_of_range} Brent values outside [{self.BRENT_MIN}, {self.BRENT_MAX}]"
                if out_of_range else
                f"All Brent prices in range | min={brent.min():.2f} max={brent.max():.2f}"
            ),
        )

    def expect_no_future_periods(self, df: pd.DataFrame) -> ExpectationResult:
        """No period should be in the future — we only store historical data."""
        if "period" not in df.columns:
            return ExpectationResult(
                expectation="expect_no_future_periods",
                passed=False,
                column="period",
                details="Column 'period' not found",
            )
        periods   = pd.to_datetime(df["period"], errors="coerce")
        future    = (periods > pd.Timestamp.now()).sum()
        return ExpectationResult(
            expectation="expect_no_future_periods",
            passed=future == 0,
            column="period",
            details=f"{future} future periods found" if future else "No future periods",
        )

    def expect_no_duplicate_rows(self, df: pd.DataFrame) -> ExpectationResult:
        """No duplicate (series_id, period) combinations should exist."""
        if not {"series_id", "period"}.issubset(df.columns):
            return ExpectationResult(
                expectation="expect_no_duplicate_rows",
                passed=False,
                column="series_id, period",
                details="Required columns missing",
            )
        dupes = df.duplicated(subset=["series_id", "period"]).sum()
        return ExpectationResult(
            expectation="expect_no_duplicate_rows",
            passed=dupes == 0,
            column="series_id, period",
            details=f"{dupes} duplicate (series_id, period) rows" if dupes else "No duplicates",
        )

    def expect_minimum_row_count(self, df: pd.DataFrame, min_rows: int = 100) -> ExpectationResult:
        """
        Dataset must have at least min_rows rows.
        A very small dataset suggests the API pull failed partially.
        """
        total = len(df)
        return ExpectationResult(
            expectation="expect_minimum_row_count",
            passed=total >= min_rows,
            column=None,
            details=f"{total} rows found — minimum required: {min_rows}",
        )

    def expect_period_not_before_min_date(self, df: pd.DataFrame) -> ExpectationResult:
        """No period should be before 1986 — EIA data doesn't go further back."""
        periods = pd.to_datetime(df["period"], errors="coerce").dropna()
        too_old = (periods < pd.Timestamp(self.MIN_DATE)).sum()
        return ExpectationResult(
            expectation="expect_period_not_before_min_date",
            passed=too_old == 0,
            column="period",
            details=f"{too_old} rows before {self.MIN_DATE.date()}" if too_old else f"All periods after {self.MIN_DATE.date()}",
        )

    # ── Run all expectations ──────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> "SuiteReport":
        """
        Runs all expectations against the DataFrame.

        Args:
            df: DataFrame from bronze_petroleum table

        Returns:
            SuiteReport with all results and summary stats
        """
        logger.info(f"Running PetroleumSuite on {len(df)} rows...")

        self.results = [
            self.expect_no_null_series_id(df),
            self.expect_no_null_period(df),
            self.expect_no_null_value(df),
            self.expect_valid_series_ids(df),
            self.expect_wti_price_in_range(df),
            self.expect_brent_price_in_range(df),
            self.expect_no_future_periods(df),
            self.expect_no_duplicate_rows(df),
            self.expect_minimum_row_count(df),
            self.expect_period_not_before_min_date(df),
        ]

        report = SuiteReport(suite_name="petroleum_suite", results=self.results)
        logger.info(
            f"PetroleumSuite complete | "
            f"passed={report.passed}/{report.total} | "
            f"success_rate={report.success_rate:.1f}%"
        )
        return report


# ── Suite Report ──────────────────────────────────────────────────────────────

class SuiteReport:
    """Holds results from a suite run and provides summary methods."""

    def __init__(self, suite_name: str, results: list[ExpectationResult]):
        self.suite_name = suite_name
        self.results    = results
        self.total      = len(results)
        self.passed     = sum(1 for r in results if r.passed)
        self.failed     = self.total - self.passed
        self.success_rate = (self.passed / self.total * 100) if self.total > 0 else 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Suite : {self.suite_name}",
            f"  Total : {self.total}  Passed : {self.passed}  Failed : {self.failed}",
            f"  Rate  : {self.success_rate:.1f}%",
            f"{'='*55}",
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"  {icon}  {r.expectation}")
            lines.append(f"       {r.details}")
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Returns summary as a dict — used by DatabaseManager.log_data_quality()."""
        return {
            "suite_name":         self.suite_name,
            "total_expectations": self.total,
            "passed":             self.passed,
            "failed":             self.failed,
            "success_rate":       round(self.success_rate, 2),
        }

    @property
    def is_passing(self) -> bool:
        """Returns True if all expectations passed."""
        return self.failed == 0