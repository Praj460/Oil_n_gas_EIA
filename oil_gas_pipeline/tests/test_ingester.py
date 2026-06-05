# oil_gas_pipeline | tests/test_ingester.py
# Unit tests for EIAClient, KaggleLoader, and DataIngester
# Run with: python3 -m pytest tests/test_ingester.py -v

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ingestion.eia_client import EIAClient
from ingestion.kaggle_loader import KaggleLoader
from ingestion.ingester import DataIngester


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_eia_response():
    """Mocked EIA API response structure."""
    return {
        "response": {
            "data": [
                {"period": "2023-01", "value": 78.54, "series-description": "WTI"},
                {"period": "2023-02", "value": 76.23, "series-description": "WTI"},
                {"period": "2023-03", "value": 72.11, "series-description": "WTI"},
                {"period": "2023-04", "value": 79.40, "series-description": "WTI"},
                {"period": "2023-05", "value": 71.83, "series-description": "WTI"},
            ]
        }
    }


@pytest.fixture
def sample_petroleum_df():
    """Sample petroleum DataFrame matching bronze_petroleum schema."""
    return pd.DataFrame({
        "series_id":   ["PET.RWTC.M"] * 5,
        "series_name": ["WTI Crude Oil Spot Price"] * 5,
        "period":      pd.date_range("2023-01-01", periods=5, freq="MS"),
        "value":       [78.54, 76.23, 72.11, 79.40, 71.83],
        "unit":        ["Dollars per Barrel"] * 5,
        "raw_response": [{}] * 5,
    })


@pytest.fixture
def sample_well_df():
    """Sample well production DataFrame matching bronze_well_production schema."""
    return pd.DataFrame({
        "well_id":         ["W001", "W002", "W003"],
        "state":           ["Texas", "Texas", "Oklahoma"],
        "production_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
        "oil_bbl":         [1200.0, 980.0, 1450.0],
        "gas_mcf":         [3400.0, 2100.0, 4200.0],
        "water_bbl":       [200.0, 150.0, 310.0],
    })


# ── EIAClient tests ───────────────────────────────────────────────────────────

class TestEIAClient:

    def test_parse_response_returns_dataframe(self, sample_eia_response):
        """_parse_response should return a non-empty DataFrame."""
        client = EIAClient()
        df = client._parse_response(
            sample_eia_response,
            series_id="PET.RWTC.M",
            series_name="WTI Crude Oil Spot Price",
            unit="Dollars per Barrel",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_parse_response_has_required_columns(self, sample_eia_response):
        """Parsed DataFrame must have all required columns."""
        client = EIAClient()
        df = client._parse_response(
            sample_eia_response,
            series_id="PET.RWTC.M",
            series_name="WTI Crude Oil Spot Price",
            unit="Dollars per Barrel",
        )
        required = {"series_id", "series_name", "period", "value", "unit"}
        assert required.issubset(set(df.columns))

    def test_parse_response_period_is_datetime(self, sample_eia_response):
        """period column must be parsed as datetime."""
        client = EIAClient()
        df = client._parse_response(
            sample_eia_response,
            series_id="PET.RWTC.M",
            series_name="WTI",
            unit="Dollars per Barrel",
        )
        assert pd.api.types.is_datetime64_any_dtype(df["period"])

    def test_parse_response_sorted_by_period(self, sample_eia_response):
        """Rows must be sorted by period ascending."""
        client = EIAClient()
        df = client._parse_response(
            sample_eia_response,
            series_id="PET.RWTC.M",
            series_name="WTI",
            unit="Dollars per Barrel",
        )
        assert df["period"].is_monotonic_increasing

    def test_parse_empty_response_returns_empty_df(self):
        """Empty response data should return empty DataFrame."""
        client   = EIAClient()
        response = {"response": {"data": []}}
        df = client._parse_response(response, "PET.RWTC.M", "WTI", "USD")
        assert df.empty

    @patch("ingestion.eia_client.requests.get")
    def test_fetch_wti_price_calls_api(self, mock_get, sample_eia_response):
        """fetch_wti_price should make a GET request to the EIA API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_eia_response
        mock_get.return_value = mock_response

        client = EIAClient()
        df = client.fetch_wti_price(start="2023-01")

        assert mock_get.called
        assert isinstance(df, pd.DataFrame)

    @patch("ingestion.eia_client.requests.get")
    def test_fetch_retries_on_timeout(self, mock_get):
        """_get should retry on timeout and raise RuntimeError after max retries."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout

        client = EIAClient()
        client.max_retries = 2
        client.retry_delay = 0   # no sleep in tests

        with pytest.raises(RuntimeError):
            client._get("/petroleum/pri/spt/data/", {})


# ── KaggleLoader tests ────────────────────────────────────────────────────────

class TestKaggleLoader:

    def test_column_map_normalizes_names(self):
        """_clean should map variant column names to standard names."""
        loader = KaggleLoader()
        raw = pd.DataFrame({
            "WellID":          ["W001", "W002"],
            "State":           ["TX", "OK"],
            "ReportDate":      ["2022-01-01", "2022-02-01"],
            "OilBBL":          [1200, 980],
            "GasMCF":          [3400, 2100],
        })
        cleaned = loader._clean(raw)
        assert "well_id"         in cleaned.columns
        assert "state"           in cleaned.columns
        assert "production_date" in cleaned.columns
        assert "oil_bbl"         in cleaned.columns
        assert "gas_mcf"         in cleaned.columns

    def test_clean_drops_null_oil_and_gas(self):
        """Rows where both oil_bbl and gas_mcf are null should be dropped."""
        loader = KaggleLoader()
        raw = pd.DataFrame({
            "production_date": ["2022-01-01", "2022-02-01", "2022-03-01"],
            "oil_bbl":         [1200.0, None, None],
            "gas_mcf":         [3400.0, None, None],
        })
        cleaned = loader._clean(raw)
        assert len(cleaned) == 1

    def test_clean_clips_negative_values(self):
        """Negative production values should be clipped to zero."""
        loader = KaggleLoader()
        raw = pd.DataFrame({
            "production_date": ["2022-01-01"],
            "oil_bbl":         [-500.0],
            "gas_mcf":         [-100.0],
        })
        cleaned = loader._clean(raw)
        assert cleaned["oil_bbl"].iloc[0] == 0.0
        assert cleaned["gas_mcf"].iloc[0] == 0.0

    def test_load_raises_if_file_missing(self, tmp_path):
        """load() should raise FileNotFoundError if CSV doesn't exist."""
        loader = KaggleLoader(file_path=tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()


# ── DataIngester tests ────────────────────────────────────────────────────────

class TestDataIngester:

    def test_validate_passes_for_valid_df(self, sample_petroleum_df):
        """_validate should return True for a valid DataFrame."""
        ingester = DataIngester()
        result   = ingester._validate(
            sample_petroleum_df,
            required_cols={"series_id", "period", "value"},
            name="test",
        )
        assert result is True

    def test_validate_fails_for_empty_df(self):
        """_validate should return False for an empty DataFrame."""
        ingester = DataIngester()
        result   = ingester._validate(
            pd.DataFrame(),
            required_cols={"series_id", "period", "value"},
            name="test",
        )
        assert result is False

    def test_validate_fails_for_missing_columns(self, sample_petroleum_df):
        """_validate should return False when required columns are missing."""
        ingester = DataIngester()
        result   = ingester._validate(
            sample_petroleum_df,
            required_cols={"series_id", "period", "value", "nonexistent_col"},
            name="test",
        )
        assert result is False

    @patch("ingestion.ingester.EIAClient")
    @patch("ingestion.ingester.DatabaseManager")
    def test_ingest_petroleum_logs_pipeline_run(
        self, mock_db_cls, mock_eia_cls, sample_petroleum_df
    ):
        """ingest_petroleum should call log_pipeline_run exactly once."""
        mock_eia = MagicMock()
        mock_eia.fetch_all_petroleum.return_value = {
            "wti":        sample_petroleum_df,
            "brent":      sample_petroleum_df,
            "production": sample_petroleum_df,
        }
        mock_eia_cls.return_value = mock_eia

        mock_db = MagicMock()
        mock_db.write_bronze_petroleum.return_value = 5
        mock_db_cls.return_value = mock_db

        ingester = DataIngester()
        ingester.ingest_petroleum(start="2023-01")

        assert mock_db.log_pipeline_run.call_count == 1