# oil_gas_pipeline | tests/test_fred_client.py
# Unit tests for FREDClient — the second-provider macro data client
# Run with: python3 -m pytest tests/test_fred_client.py -v

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ingestion.fred_client import FREDClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fred_daily_response():
    """A small FRED daily-frequency response — the format DTWEXBGS and DGS10 return.
    Includes the '.' string FRED uses for missing values."""
    return {
        "observations": [
            {"date": "2024-01-01", "value": "100.5"},
            {"date": "2024-01-02", "value": "101.2"},
            {"date": "2024-01-03", "value": "."},          # missing — FRED's placeholder
            {"date": "2024-01-04", "value": "102.1"},
            {"date": "2024-01-05", "value": "101.8"},
            {"date": "2024-02-01", "value": "103.0"},
            {"date": "2024-02-02", "value": "103.5"},
            {"date": "2024-03-01", "value": "104.0"},
        ]
    }


@pytest.fixture
def mock_fred_monthly_response():
    """A FRED monthly-frequency response (INDPRO style — no daily noise)."""
    return {
        "observations": [
            {"date": "2024-01-01", "value": "100.0"},
            {"date": "2024-02-01", "value": "101.5"},
            {"date": "2024-03-01", "value": "102.3"},
            {"date": "2024-04-01", "value": "101.8"},
        ]
    }


# ── Response parser tests ─────────────────────────────────────────────────────

class TestFREDParseResponse:

    def test_parse_returns_dataframe(self, mock_fred_daily_response):
        """_parse_response should always return a DataFrame."""
        client = FREDClient()
        df = client._parse_response(
            mock_fred_daily_response,
            series_id="TEST.M",
            series_name="Test series",
            unit="Index",
        )
        assert isinstance(df, pd.DataFrame)

    def test_parse_has_required_columns(self, mock_fred_daily_response):
        """Output must have the same column set as EIA parsing for consistency."""
        client = FREDClient()
        df = client._parse_response(
            mock_fred_daily_response,
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        required = {"series_id", "series_name", "period", "value", "unit", "raw_response"}
        assert required.issubset(set(df.columns))

    def test_dot_values_are_dropped(self, mock_fred_daily_response):
        """FRED's '.' placeholder must be converted to NaN and dropped, NEVER
        treated as a numeric value. Same for the parser."""
        client = FREDClient()
        df = client._parse_response(
            mock_fred_daily_response,
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        # No value should be the literal string '.'
        assert not (df["value"].astype(str) == ".").any()
        # All values should be numeric
        assert pd.api.types.is_numeric_dtype(df["value"])

    def test_daily_resampled_to_month_start(self, mock_fred_daily_response):
        """Daily input must come out as month-start (MS) periods only."""
        client = FREDClient()
        df = client._parse_response(
            mock_fred_daily_response,
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        # All periods should be on the first of the month
        assert (df["period"].dt.day == 1).all()
        # Should have 3 distinct months (Jan, Feb, Mar 2024)
        assert len(df) == 3

    def test_monthly_mean_is_correct(self, mock_fred_daily_response):
        """For January: 4 valid values (100.5, 101.2, 102.1, 101.8) — the '.' was dropped.
        Mean should be the average of those four."""
        client = FREDClient()
        df = client._parse_response(
            mock_fred_daily_response,
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        jan_value = df[df["period"] == "2024-01-01"]["value"].iloc[0]
        expected = (100.5 + 101.2 + 102.1 + 101.8) / 4
        assert jan_value == pytest.approx(expected, rel=1e-3)

    def test_empty_observations_returns_empty_df(self):
        """No observations should return an empty DataFrame, not crash."""
        client = FREDClient()
        df = client._parse_response(
            {"observations": []},
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        assert df.empty

    def test_missing_observations_key_returns_empty(self):
        """If FRED returns a malformed payload (no 'observations' key), don't crash."""
        client = FREDClient()
        df = client._parse_response(
            {"error_code": 400},
            series_id="TEST.M",
            series_name="Test",
            unit="Index",
        )
        assert df.empty


# ── HTTP retry tests ──────────────────────────────────────────────────────────

class TestFREDClientGet:

    @patch("ingestion.fred_client.requests.get")
    def test_get_includes_api_key_and_file_type(self, mock_get, mock_fred_daily_response):
        """_get must inject api_key and file_type=json into the params."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_fred_daily_response
        mock_get.return_value = mock_response

        client = FREDClient()
        client._get("/series/observations", {"series_id": "TEST"})

        called_params = mock_get.call_args.kwargs["params"]
        assert "api_key" in called_params
        assert called_params["file_type"] == "json"

    @patch("ingestion.fred_client.requests.get")
    def test_get_retries_on_timeout(self, mock_get):
        """_get should retry on timeout and ultimately raise RuntimeError."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout

        client = FREDClient()
        client.max_retries = 2
        client.retry_delay = 0

        with pytest.raises(RuntimeError):
            client._get("/series/observations", {"series_id": "TEST"})

    @patch("ingestion.fred_client.requests.get")
    def test_get_raises_on_fred_error_code(self, mock_get):
        """FRED returns errors in the body with an 'error_code'. We must raise, not return."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "error_code": 400,
            "error_message": "Bad Request: invalid series",
        }
        mock_get.return_value = mock_response

        client = FREDClient()
        client.max_retries = 1
        client.retry_delay = 0
        with pytest.raises((ValueError, RuntimeError)):
            client._get("/series/observations", {"series_id": "BAD"})


# ── Public fetch methods ──────────────────────────────────────────────────────

class TestFREDFetchMethods:

    @patch("ingestion.fred_client.requests.get")
    def test_fetch_dollar_index_returns_dataframe(self, mock_get, mock_fred_daily_response):
        """fetch_dollar_index should call the API and return a parsed DataFrame."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_fred_daily_response
        mock_get.return_value = mock_response

        client = FREDClient()
        df = client.fetch_dollar_index(start="2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert mock_get.called

    @patch("ingestion.fred_client.requests.get")
    def test_fetch_industrial_production_returns_dataframe(self, mock_get, mock_fred_monthly_response):
        """fetch_industrial_production should return a parsed DataFrame from a monthly response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_fred_monthly_response
        mock_get.return_value = mock_response

        client = FREDClient()
        df = client.fetch_industrial_production(start="2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4   # 4 months in the fixture

    @patch("ingestion.fred_client.requests.get")
    def test_fetch_treasury_10y_returns_dataframe(self, mock_get, mock_fred_daily_response):
        """fetch_treasury_10y should work on a daily-style response (DGS10 is daily)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_fred_daily_response
        mock_get.return_value = mock_response

        client = FREDClient()
        df = client.fetch_treasury_10y(start="2024-01-01")
        assert isinstance(df, pd.DataFrame)
        # Daily data should still come out as monthly rows
        if not df.empty:
            assert (df["period"].dt.day == 1).all()

    @patch("ingestion.fred_client.requests.get")
    def test_fetch_all_macro_returns_three_series(self, mock_get, mock_fred_monthly_response):
        """fetch_all_macro should populate all three keyed series."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_fred_monthly_response
        mock_get.return_value = mock_response

        client = FREDClient()
        results = client.fetch_all_macro(start="2024-01-01")
        assert "dollar_index" in results
        assert "industrial_production" in results
        assert "treasury_10y" in results
        for key, df in results.items():
            assert isinstance(df, pd.DataFrame)
