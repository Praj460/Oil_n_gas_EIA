# oil_gas_pipeline | config/config.py
# Central configuration — loads all environment variables from .env
# Every other file imports from here — never hardcode secrets anywhere else
# Usage: from config.config import config

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ── Logging setup ────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class EIAConfig:
    """EIA OpenData API settings."""
    api_key: str
    base_url: str = "https://api.eia.gov/v2"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5  # seconds between retries

    # Series IDs for the data we want to pull
    petroleum_series: dict = field(default_factory=lambda: {
        "wti_spot":   "PET.RWTC.M",       # WTI Crude Oil spot price (monthly)
        "brent_spot": "PET.RBRTE.M",      # Brent Crude spot price (monthly)
        "us_production": "PET.MCRFPUS2.M" # US crude oil production (monthly)
    })

    natural_gas_series: dict = field(default_factory=lambda: {
        "henry_hub":    "NG.RNGWHHD.M",   # Henry Hub natural gas price (monthly)
        "us_storage":   "NG.NW2_EPG0_SWO_R48_BCF.W", # US nat gas storage (weekly)
        "us_production": "NG.N9010US2.M"  # US nat gas production (monthly)
    })


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings."""
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def url(self) -> str:
        """SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def psycopg2_params(self) -> dict:
        """Raw psycopg2 connection params."""
        return {
            "host":     self.host,
            "port":     self.port,
            "dbname":   self.name,
            "user":     self.user,
            "password": self.password,
        }


@dataclass
class ModelConfig:
    """Forecasting model parameters."""
    forecast_horizon: int = 12      # months ahead to forecast
    train_test_split: float = 0.8   # 80% train, 20% test

    # SARIMA order — (p, d, q)(P, D, Q, s)
    sarima_order: tuple = (1, 1, 1)
    sarima_seasonal_order: tuple = (1, 1, 1, 12)

    # Prophet settings
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "multiplicative"

    # Which model to use as primary ("sarima" or "prophet")
    primary_model: str = "prophet"


@dataclass
class PathConfig:
    """File system paths used across the project."""
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_raw(self) -> Path:
        p = self.root / "data" / "raw"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_processed(self) -> Path:
        p = self.root / "data" / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs(self) -> Path:
        p = self.root / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_saved(self) -> Path:
        p = self.root / "saved_models"
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class AppConfig:
    """Top-level config — groups all sub-configs together."""
    eia: EIAConfig
    db: DatabaseConfig
    model: ModelConfig
    paths: PathConfig
    env: str = "development"

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


# ── Factory function ──────────────────────────────────────────────────────────

def _load_config() -> AppConfig:
    """
    Build AppConfig from environment variables.
    Raises a clear error if any required variable is missing.
    """
    missing = []

    def require(key: str) -> str:
        val = os.getenv(key)
        if not val:
            missing.append(key)
        return val or ""

    eia_key = require("EIA_API_KEY")
    db_host  = require("DB_HOST")
    db_name  = require("DB_NAME")
    db_user  = require("DB_USER")
    db_pass  = require("DB_PASSWORD")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Make sure your .env file exists and contains all required keys.\n"
            f"Copy .env.example → .env and fill in the values."
        )

    cfg = AppConfig(
        eia=EIAConfig(api_key=eia_key),
        db=DatabaseConfig(
            host=db_host,
            port=int(os.getenv("DB_PORT", "5432")),
            name=db_name,
            user=db_user,
            password=db_pass,
        ),
        model=ModelConfig(),
        paths=PathConfig(),
        env=os.getenv("ENV", "development"),
    )

    logger.info(f"Config loaded | env={cfg.env} | db={cfg.db.host}/{cfg.db.name}")
    return cfg


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this in every other file:
#   from config.config import config
#   print(config.eia.api_key)

config: AppConfig = _load_config()