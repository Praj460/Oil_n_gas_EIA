# oil_gas_pipeline | ingestion/eia_client.py
# EIAClient class — pulls petroleum and natural gas data from EIA OpenData API
# Handles retries, rate limiting, and response parsing
# Usage: from ingestion.eia_client import EIAClient

import time
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

from config.config import config

logger = logging.getLogger(__name__)


class EIAClient:
    """
    Client for the EIA OpenData API v2.

    Pulls:
    - Crude oil spot prices (WTI and Brent)
    - US crude oil production
    - Henry Hub natural gas prices
    - US natural gas storage and production

    All data is returned as a pandas DataFrame ready to be
    written to the bronze layer by DatabaseManager.

    EIA API docs: https://www.eia.gov/opendata/documentation.php
    """

    def __init__(self):
        self.api_key    = config.eia.api_key
        self.base_url   = config.eia.base_url
        self.timeout    = config.eia.timeout
        self.max_retries = config.eia.max_retries
        self.retry_delay = config.eia.retry_delay

        # Series IDs defined in config
        self.petroleum_series   = config.eia.petroleum_series
        self.natural_gas_series = config.eia.natural_gas_series

        logger.info("EIAClient initialized")

    # ── Core request method ───────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        """
        Makes a GET request to the EIA API with retry logic.

        Args:
            endpoint: API path e.g. "/petroleum/pri/spt/data/"
            params:   query parameters dict

        Returns:
            Parsed JSON response as a dict

        Raises:
            RuntimeError if all retries are exhausted
        """
        params["api_key"] = self.api_key
        url = f"{self.base_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"GET {url} | attempt {attempt}/{self.max_retries}")
                response = requests.get(url, params=params, timeout=self.timeout)

                # EIA returns 200 even for some errors — check response body too
                if response.status_code == 429:
                    wait = self.retry_delay * attempt
                    logger.warning(f"Rate limited by EIA API — waiting {wait}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # EIA wraps errors in the response body
                if "error" in data:
                    raise ValueError(f"EIA API error: {data['error']}")

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
            f"EIA API request failed after {self.max_retries} attempts | url={url}"
        )

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(
        self,
        response: dict,
        series_id: str,
        series_name: str,
        unit: str,
    ) -> pd.DataFrame:
        """
        Parses a raw EIA API response into a clean DataFrame.

        Args:
            response:    raw JSON dict from _get()
            series_id:   EIA series ID string
            series_name: human readable name
            unit:        unit of measurement

        Returns:
            DataFrame with columns:
            [series_id, series_name, period, value, unit, raw_response]
        """
        try:
            rows = response["response"]["data"]
        except KeyError:
            logger.error(f"Unexpected response structure for {series_id}")
            return pd.DataFrame()

        if not rows:
            logger.warning(f"No data returned for series {series_id}")
            return pd.DataFrame()

        records = []
        for row in rows:
            records.append({
                "series_id":   series_id,
                "series_name": series_name,
                "period":      row.get("period"),       # e.g. "2024-01"
                "value":       row.get("value"),
                "unit":        unit,
                "raw_response": row,                    # store full row as JSON
            })

        df = pd.DataFrame(records)

        # Convert period to datetime — EIA uses "YYYY-MM" for monthly data
        df["period"] = pd.to_datetime(df["period"], format="%Y-%m", errors="coerce")

        # Drop rows where period or value couldn't be parsed
        before = len(df)
        df = df.dropna(subset=["period", "value"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with null period/value for {series_id}")

        df = df.sort_values("period").reset_index(drop=True)
        logger.info(f"Parsed {len(df)} rows for {series_id}")
        return df

    # ── Petroleum data pulls ──────────────────────────────────────────────────

    def fetch_wti_price(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly WTI crude oil spot price.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12" (defaults to latest available)

        Returns:
            DataFrame with WTI price per month
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "RWTC",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/pri/spt/data/", params)
        return self._parse_response(
            response,
            series_id="PET.RWTC.M",
            series_name="WTI Crude Oil Spot Price",
            unit="Dollars per Barrel",
        )

    def fetch_brent_price(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly Brent crude oil spot price.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with Brent price per month
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "RBRTE",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/pri/spt/data/", params)
        return self._parse_response(
            response,
            series_id="PET.RBRTE.M",
            series_name="Brent Crude Oil Spot Price",
            unit="Dollars per Barrel",
        )

    def fetch_us_oil_production(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US crude oil production.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with US production per month in thousand barrels/day
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "MCRFPUS2",
            "facets[duoarea][]": "NUS",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/crd/crpdn/data/", params)
        return self._parse_response(
            response,
            series_id="PET.MCRFPUS2.M",
            series_name="US Crude Oil Production",
            unit="Thousand Barrels per Day",
        )

    def fetch_crude_imports(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US crude oil imports.

        More imports = more supply available = downward pressure on prices.
        A useful exogenous signal for forecasting WTI.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with US crude imports per month in thousand barrels/day
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "MCRIMUS2",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/move/imp/data/", params)
        return self._parse_response(
            response,
            series_id="PET.MCRIMUS2.M",
            series_name="US Crude Oil Imports",
            unit="Thousand Barrels per Day",
        )

    def fetch_refinery_utilization(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US refinery utilization (% of capacity in use).

        High utilization = strong crude demand from refiners = upward
        price pressure on WTI. A demand-side signal.

        Returns:
            DataFrame with refinery utilization per month (percent)
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "MOPUEUS2",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/pnp/unc/data/", params)
        return self._parse_response(
            response,
            series_id="PET.MOPUEUS2.M",
            series_name="US Refinery Utilization",
            unit="Percent",
        )

    def fetch_gasoline_stocks(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US total gasoline stocks (inventory).

        High stocks = oversupplied downstream = bearish for crude.
        Low stocks = tight market = bullish. A downstream demand signal.

        Returns:
            DataFrame with gasoline stocks per month in thousand barrels
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "MGFSTUS1",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/stoc/typ/data/", params)
        return self._parse_response(
            response,
            series_id="PET.MGFSTUS1.M",
            series_name="US Finished Motor Gasoline Stocks",
            unit="Thousand Barrels",
        )

    def fetch_distillate_stocks(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US distillate fuel oil stocks (diesel + heating oil).

        Distillate inventory reflects industrial + heating demand.
        Low winter stocks can spike both oil and gas prices.

        Returns:
            DataFrame with distillate stocks per month in thousand barrels
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "MDISTUS1",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/petroleum/stoc/typ/data/", params)
        return self._parse_response(
            response,
            series_id="PET.MDISTUS1.M",
            series_name="US Distillate Fuel Oil Stocks",
            unit="Thousand Barrels",
        )

    # ── Natural gas data pulls ────────────────────────────────────────────────
    def fetch_henry_hub_price(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly Henry Hub natural gas spot price.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with Henry Hub price per month in $/MMBtu
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[duoarea][]": "NUS",
            "facets[product][]": "EPG0",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/natural-gas/pri/fut/data/", params)
        return self._parse_response(
            response,
            series_id="NG.RNGWHHD.M",
            series_name="Henry Hub Natural Gas Spot Price",
            unit="Dollars per MMBtu",
        )

    def fetch_us_gas_storage(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US natural gas storage levels.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with US gas storage in billion cubic feet (BCF)
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[duoarea][]": "NUS",
            "facets[product][]": "EPG0",
            "facets[process][]": "SAO",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/natural-gas/stor/sum/data/", params)
        return self._parse_response(
            response,
            series_id="NG.NW2_EPG0_SWO_R48_BCF.M",
            series_name="US Natural Gas Storage",
            unit="Billion Cubic Feet",
            )
    def fetch_us_gas_production(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US natural gas production.

        Args:
            start: start period e.g. "2000-01"
            end:   end period e.g. "2024-12"

        Returns:
            DataFrame with US gas production in BCF/month
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[series][]": "N9010US2",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/natural-gas/prod/sum/data/", params)
        return self._parse_response(
            response,
            series_id="NG.N9010US2.M",
            series_name="US Natural Gas Production",
            unit="Billion Cubic Feet",
        )

    # ── Weather data (STEO route — different structure) ───────────────────────

    def fetch_heating_degree_days(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US population-weighted Heating Degree Days (HDD).

        Higher HDD = colder month = more heating demand = upward pressure
        on natural gas prices. The primary winter demand signal for gas.

        Note: comes from EIA's STEO route, which uses 'facets[seriesId][]'
        rather than the 'facets[series][]' used by the petroleum/gas routes.

        Returns:
            DataFrame with monthly US-average HDD
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[seriesId][]": "ZWHDPUS",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/steo/data/", params)
        return self._parse_response(
            response,
            series_id="STEO.ZWHDPUS.M",
            series_name="US Heating Degree Days",
            unit="Degree Days",
        )

    def fetch_cooling_degree_days(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly US population-weighted Cooling Degree Days (CDD).

        Higher CDD = hotter month = more air conditioning = more electricity
        demand = power plants burn more gas = upward pressure on gas prices.
        The primary summer demand signal for gas.

        Returns:
            DataFrame with monthly US-average CDD
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[seriesId][]": "ZWCDPUS",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/steo/data/", params)
        return self._parse_response(
            response,
            series_id="STEO.ZWCDPUS.M",
            series_name="US Cooling Degree Days",
            unit="Degree Days",
        )

    # ── Convenience method — fetch everything at once ─────────────────────────

    def fetch_all_petroleum(self, start: str = "2000-01") -> dict[str, pd.DataFrame]:
        """
        Fetches all petroleum series in one call.

        Returns:
            Dict of DataFrames keyed by series name:
            {
                "wti":        DataFrame,
                "brent":      DataFrame,
                "production": DataFrame,
            }
        """
        logger.info("Fetching all petroleum series from EIA API...")
        results = {}

        try:
            results["wti"] = self.fetch_wti_price(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch WTI price | {e}")
            results["wti"] = pd.DataFrame()

        try:
            results["brent"] = self.fetch_brent_price(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch Brent price | {e}")
            results["brent"] = pd.DataFrame()

        try:
            results["production"] = self.fetch_us_oil_production(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch US oil production | {e}")
            results["production"] = pd.DataFrame()

        total_rows = sum(len(df) for df in results.values())
        logger.info(f"Petroleum fetch complete | total rows={total_rows}")
        return results

    def fetch_all_natural_gas(self, start: str = "2000-01") -> dict[str, pd.DataFrame]:
        """
        Fetches all natural gas series in one call.

        Returns:
            Dict of DataFrames keyed by series name:
            {
                "henry_hub":  DataFrame,
                "storage":    DataFrame,
                "production": DataFrame,
            }
        """
        logger.info("Fetching all natural gas series from EIA API...")
        results = {}

        try:
            results["henry_hub"] = self.fetch_henry_hub_price(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch Henry Hub price | {e}")
            results["henry_hub"] = pd.DataFrame()

        try:
            results["storage"] = self.fetch_us_gas_storage(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch US gas storage | {e}")
            results["storage"] = pd.DataFrame()

        try:
            results["production"] = self.fetch_us_gas_production(start=start)
        except Exception as e:
            logger.error(f"Failed to fetch US gas production | {e}")
            results["production"] = pd.DataFrame()

        total_rows = sum(len(df) for df in results.values())
        logger.info(f"Natural gas fetch complete | total rows={total_rows}")
        return results
    def fetch_opec_spare_capacity(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly OPEC total spare crude oil production capacity.

        Spare capacity = how much extra oil OPEC could bring online quickly.
        LOW spare capacity = no cushion to absorb supply shocks = prices
        become highly sensitive to disruptions (the early-2026 spike regime).
        A key supply-side fragility signal. From EIA's STEO route.

        Returns:
            DataFrame with monthly OPEC spare capacity (million barrels/day)
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[seriesId][]": "COPS_OPEC",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/steo/data/", params)
        return self._parse_response(
            response,
            series_id="STEO.COPS_OPEC.M",
            series_name="OPEC Spare Crude Capacity",
            unit="Million Barrels per Day",
        )

    def fetch_global_oil_inventory(
        self,
        start: Optional[str] = "2000-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches monthly total global commercial crude + liquids inventory.

        Global end-of-period commercial inventories. LOW inventories = tight
        market with little buffer = upward price pressure and shock sensitivity.
        Complements the US-specific stock data with a worldwide view.
        From EIA's STEO route.

        Returns:
            DataFrame with monthly global commercial inventory (million barrels)
        """
        params = {
            "frequency":  "monthly",
            "data[0]":    "value",
            "facets[seriesId][]": "PASXPUS",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        response = self._get("/steo/data/", params)
        return self._parse_response(
            response,
            series_id="STEO.PASXPUS.M",
            series_name="Global Commercial Oil Inventory",
            unit="Million Barrels",
        )