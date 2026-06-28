# oil_gas_pipeline | ingestion/fred_client.py
# FREDClient class — pulls macro-economic series from the FRED API
# (Federal Reserve Economic Data, St. Louis Fed)
# Mirrors EIAClient's structure but handles FRED's different response format.
# Usage: from ingestion.fred_client import FREDClient

import time
import logging
import requests
import pandas as pd
from typing import Optional

from config.config import config

logger = logging.getLogger(__name__)


class FREDClient:
    """
    Client for the FRED API.

    Pulls macro-economic signals used as exogenous features for
    energy price forecasting:
    - Dollar index (DTWEXBGS)      — currency strength (oil is dollar-priced)
    - Industrial production (INDPRO) — real economic demand for energy
    - 10-year Treasury yield (DGS10) — financing/inventory-holding cost

    Returns DataFrames in the SAME shape as EIAClient so they drop
    straight into the existing bronze-write flow:
    [series_id, series_name, period, value, unit, raw_response]

    NOTE: FRED's JSON differs from EIA's. FRED returns:
        {"observations": [{"date": "2024-01-01", "value": "122.8"}, ...]}
    and uses the string "." for missing values. This client handles both.

    FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
    """

    def __init__(self):
        self.api_key     = config.fred.api_key
        self.base_url    = config.fred.base_url
        self.timeout     = config.fred.timeout
        self.max_retries = config.fred.max_retries
        self.retry_delay = config.fred.retry_delay
        self.series      = config.fred.series
        logger.info("FREDClient initialized")

    # ── Core request method (same retry pattern as EIAClient) ─────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Makes a GET request to the FRED API with retry logic.

        Args:
            endpoint: API path e.g. "/series/observations"
            params:   query parameters dict

        Returns:
            Parsed JSON response as a dict

        Raises:
            RuntimeError if all retries are exhausted
        """
        params["api_key"]    = self.api_key
        params["file_type"]  = "json"
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"GET {url} | attempt {attempt}/{self.max_retries}")
                response = requests.get(url, params=params, timeout=self.timeout)

                if response.status_code == 429:
                    wait = self.retry_delay * attempt
                    logger.warning(f"Rate limited by FRED API — waiting {wait}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # FRED returns errors with an "error_code" / "error_message"
                if "error_code" in data:
                    raise ValueError(
                        f"FRED API error {data['error_code']}: "
                        f"{data.get('error_message', 'unknown')}"
                    )

                return data

            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out | attempt {attempt}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error | attempt {attempt}")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error {e} | attempt {attempt}")

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"FRED API request failed after {self.max_retries} attempts | url={url}"
        )

    # ── Response parser (handles FRED's format) ───────────────────────────────

    def _parse_response(
        self,
        response: dict,
        series_id: str,
        series_name: str,
        unit: str,
    ) -> pd.DataFrame:
        """
        Parses a raw FRED API response into a clean DataFrame.

        FRED gives daily/monthly observations as:
            {"observations": [{"date": "2024-01-01", "value": "122.8"}, ...]}
        Missing values arrive as the string ".".

        Returns:
            DataFrame: [series_id, series_name, period, value, unit, raw_response]
            (Resampled to month-start to match the monthly EIA series.)
        """
        try:
            obs = response["observations"]
        except KeyError:
            logger.error(f"Unexpected FRED response structure for {series_id}")
            return pd.DataFrame()

        if not obs:
            logger.warning(f"No observations returned for {series_id}")
            return pd.DataFrame()

        records = []
        for row in obs:
            records.append({
                "series_id":    series_id,
                "series_name":  series_name,
                "period":       row.get("date"),       # "YYYY-MM-DD"
                "value":        row.get("value"),       # string, may be "."
                "unit":         unit,
                "raw_response": row,
            })

        df = pd.DataFrame(records)

        # FRED missing values are the string "." — turn into real NaN
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Parse the date
        df["period"] = pd.to_datetime(df["period"], errors="coerce")

        # Drop rows with no date or no value
        before = len(df)
        df = df.dropna(subset=["period", "value"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with null/'.' value for {series_id}")

        # Some FRED series (DGS10, DTWEXBGS) are DAILY. Resample to month-start
        # so they line up with the monthly EIA series for joining later.
        df = df.set_index("period").sort_index()
        monthly = (
            df[["value"]]
            .resample("MS")
            .mean()
            .reset_index()
        )

        # Re-attach the metadata columns lost during resample
        monthly["series_id"]    = series_id
        monthly["series_name"]  = series_name
        monthly["unit"]         = unit
        monthly["raw_response"] = monthly["period"].dt.strftime("%Y-%m-%d").map(
            lambda d: {"date": d, "resampled": "monthly_mean"}
        )

        # Re-order to match EIA shape
        monthly = monthly[
            ["series_id", "series_name", "period", "value", "unit", "raw_response"]
        ]
        monthly["value"] = monthly["value"].round(4)

        logger.info(f"Parsed {len(monthly)} monthly rows for {series_id}")
        return monthly

    # ── Generic fetch ─────────────────────────────────────────────────────────

    def _fetch_series(
        self,
        series_id: str,
        series_name: str,
        unit: str,
        start: Optional[str] = "2000-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetches one FRED series by ID and returns a monthly DataFrame."""
        params = {
            "series_id": series_id,
            "observation_start": start,
        }
        if end:
            params["observation_end"] = end

        response = self._get("/series/observations", params)
        return self._parse_response(response, series_id, series_name, unit)

    # ── Named fetch methods (mirror EIAClient style) ──────────────────────────

    def fetch_dollar_index(self, start: str = "2000-01-01") -> pd.DataFrame:
        """Trade-weighted US Dollar Index (broad). Currency strength signal."""
        return self._fetch_series(
            series_id="DTWEXBGS",
            series_name="Trade-Weighted US Dollar Index (Broad)",
            unit="Index 2006=100",
            start=start,
        )

    def fetch_industrial_production(self, start: str = "2000-01-01") -> pd.DataFrame:
        """US Industrial Production Index. Real economic demand signal."""
        return self._fetch_series(
            series_id="INDPRO",
            series_name="US Industrial Production Index",
            unit="Index 2017=100",
            start=start,
        )

    def fetch_treasury_10y(self, start: str = "2000-01-01") -> pd.DataFrame:
        """10-Year Treasury constant maturity yield. Financing-cost signal."""
        return self._fetch_series(
            series_id="DGS10",
            series_name="10-Year Treasury Yield",
            unit="Percent",
            start=start,
        )

    # ── Convenience — fetch all macro series at once ──────────────────────────

    def fetch_all_macro(self, start: str = "2000-01-01") -> dict[str, pd.DataFrame]:
        """
        Fetches all macro series in one call.

        Returns:
            Dict of DataFrames keyed by short name:
            {"dollar_index": df, "industrial_production": df, "treasury_10y": df}
        """
        logger.info("Fetching all FRED macro series...")
        results = {}

        for key, fetch in [
            ("dollar_index",          self.fetch_dollar_index),
            ("industrial_production", self.fetch_industrial_production),
            ("treasury_10y",          self.fetch_treasury_10y),
        ]:
            try:
                results[key] = fetch(start=start)
            except Exception as e:
                logger.error(f"Failed to fetch {key} | {e}")
                results[key] = pd.DataFrame()

        total = sum(len(df) for df in results.values())
        logger.info(f"FRED macro fetch complete | total rows={total}")
        return results
