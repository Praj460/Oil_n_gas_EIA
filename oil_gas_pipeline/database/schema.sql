-- oil_gas_pipeline | database/schema.sql
-- Creates all PostgreSQL tables for the medallion architecture
-- Bronze (raw) → Silver (cleaned) → Gold (analysis-ready + forecasts)
-- Run with: psql -d oil_gas_db -f database/schema.sql

-- ─────────────────────────────────────────────────────────────────────────────
-- EXTENSIONS
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ─────────────────────────────────────────────────────────────────────────────
-- BRONZE LAYER — raw, untouched data exactly as it arrives from the API
-- Never update or delete rows here. Append only.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bronze_petroleum (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,               -- EIA series ID e.g. PET.RWTC.M
    series_name     VARCHAR(120),                       -- human readable name
    period          VARCHAR(20) NOT NULL,               -- e.g. "2024-01" (monthly)
    value           NUMERIC(12, 4),                     -- price or volume value
    unit            VARCHAR(40),                        -- e.g. "Dollars per Barrel"
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),          -- when we pulled it
    raw_response    JSONB                               -- full API response stored as-is
);

CREATE TABLE IF NOT EXISTS bronze_natural_gas (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,
    series_name     VARCHAR(120),
    period          VARCHAR(20) NOT NULL,
    value           NUMERIC(12, 4),
    unit            VARCHAR(40),
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_response    JSONB
);



-- Indexes on bronze tables for faster lookups by series and period
CREATE INDEX IF NOT EXISTS idx_bronze_pet_series  ON bronze_petroleum (series_id, period);
CREATE INDEX IF NOT EXISTS idx_bronze_gas_series  ON bronze_natural_gas (series_id, period);



-- ─────────────────────────────────────────────────────────────────────────────
-- SILVER LAYER — cleaned, validated, resampled time series
-- Nulls handled, outliers flagged, resampled to monthly frequency
-- Built by dbt models (stg_petroleum.sql, stg_natural_gas.sql)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS silver_petroleum (
    id                  UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    period              DATE        NOT NULL,            -- first day of the month
    wti_price           NUMERIC(10, 4),                  -- WTI spot price $/barrel
    brent_price         NUMERIC(10, 4),                  -- Brent spot price $/barrel
    us_production       NUMERIC(14, 2),                  -- US crude production (thousand barrels/day)
    price_spread        NUMERIC(10, 4),                  -- brent_price - wti_price
    wti_mom_change      NUMERIC(10, 4),                  -- month-over-month % change in WTI
    is_outlier          BOOLEAN     DEFAULT FALSE,        -- flagged by Great Expectations
    transformed_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS silver_natural_gas (
    id                  UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    period              DATE        NOT NULL,
    henry_hub_price     NUMERIC(10, 4),                  -- Henry Hub price $/MMBtu
    us_storage_bcf      NUMERIC(12, 2),                  -- storage in billion cubic feet
    us_production       NUMERIC(14, 2),                  -- production in BCF/month
    price_mom_change    NUMERIC(10, 4),                  -- month-over-month % change
    is_outlier          BOOLEAN     DEFAULT FALSE,
    transformed_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Unique constraint — one row per month per silver table
CREATE UNIQUE INDEX IF NOT EXISTS idx_silver_pet_period ON silver_petroleum (period);
CREATE UNIQUE INDEX IF NOT EXISTS idx_silver_gas_period ON silver_natural_gas (period);


-- ─────────────────────────────────────────────────────────────────────────────
-- GOLD LAYER — analysis-ready combined table + forecast results
-- Built by dbt model (mart_energy_prices.sql)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold_energy_prices (
    id                  UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    period              DATE        NOT NULL UNIQUE,
    -- petroleum
    wti_price           NUMERIC(10, 4),
    brent_price         NUMERIC(10, 4),
    price_spread        NUMERIC(10, 4),
    us_oil_production   NUMERIC(14, 2),
    wti_mom_change      NUMERIC(10, 4),
    -- natural gas
    henry_hub_price     NUMERIC(10, 4),
    us_gas_storage_bcf  NUMERIC(12, 2),
    us_gas_production   NUMERIC(14, 2),
    gas_mom_change      NUMERIC(10, 4),
    -- derived features for modeling
    oil_gas_ratio       NUMERIC(10, 4),                  -- wti / henry_hub
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold_forecast_results (
    id                  UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id              UUID        DEFAULT uuid_generate_v4(),  -- groups one forecast run
    target              VARCHAR(40) NOT NULL,             -- e.g. "wti_price", "henry_hub_price"
    model_name          VARCHAR(20) NOT NULL,             -- "sarima" or "prophet"
    forecast_period     DATE        NOT NULL,             -- the month being forecasted
    forecast_value      NUMERIC(10, 4) NOT NULL,
    lower_bound         NUMERIC(10, 4),                   -- 95% confidence interval lower
    upper_bound         NUMERIC(10, 4),                   -- 95% confidence interval upper
    -- evaluation metrics (filled after backtesting)
    rmse                NUMERIC(10, 4),
    mape                NUMERIC(10, 4),
    -- metadata
    trained_on_periods  INT,                              -- how many months of data used
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gold_energy_period    ON gold_energy_prices (period);
CREATE INDEX IF NOT EXISTS idx_gold_forecast_run     ON gold_forecast_results (run_id, target, model_name);
CREATE INDEX IF NOT EXISTS idx_gold_forecast_period  ON gold_forecast_results (forecast_period, target);


-- ─────────────────────────────────────────────────────────────────────────────
-- PIPELINE METADATA — tracks every run for monitoring and debugging
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_name        VARCHAR(80) NOT NULL,                -- e.g. "daily_ingest", "dbt_transform"
    status          VARCHAR(20) NOT NULL,                -- "success", "failed", "running"
    rows_ingested   INT         DEFAULT 0,
    rows_failed     INT         DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS data_quality_results (
    id                  UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    suite_name          VARCHAR(80) NOT NULL,             -- e.g. "petroleum_suite"
    table_name          VARCHAR(80) NOT NULL,
    total_expectations  INT,
    passed              INT,
    failed              INT,
    success_rate        NUMERIC(5, 2),                    -- percentage
    run_at              TIMESTAMPTZ DEFAULT NOW()
);


-- Exogenous predictor series — additional EIA signals used as
-- features for SARIMAX forecasting (imports, refinery util, stocks)

CREATE TABLE IF NOT EXISTS bronze_crude_imports (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,
    series_name     VARCHAR(120),
    period          VARCHAR(20) NOT NULL,
    value           NUMERIC(12, 4),
    unit            VARCHAR(40),
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_response    JSONB
);

CREATE TABLE IF NOT EXISTS bronze_refinery_utilization (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,
    series_name     VARCHAR(120),
    period          VARCHAR(20) NOT NULL,
    value           NUMERIC(12, 4),
    unit            VARCHAR(40),
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_response    JSONB
);

CREATE TABLE IF NOT EXISTS bronze_gasoline_stocks (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,
    series_name     VARCHAR(120),
    period          VARCHAR(20) NOT NULL,
    value           NUMERIC(12, 4),
    unit            VARCHAR(40),
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_response    JSONB
);

CREATE TABLE IF NOT EXISTS bronze_distillate_stocks (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    series_id       VARCHAR(60) NOT NULL,
    series_name     VARCHAR(120),
    period          VARCHAR(20) NOT NULL,
    value           NUMERIC(12, 4),
    unit            VARCHAR(40),
    source          VARCHAR(20) DEFAULT 'EIA_API',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_response    JSONB
);

CREATE INDEX IF NOT EXISTS idx_bronze_imports_series    ON bronze_crude_imports (series_id, period);
CREATE INDEX IF NOT EXISTS idx_bronze_refinery_series   ON bronze_refinery_utilization (series_id, period);
CREATE INDEX IF NOT EXISTS idx_bronze_gasoline_series   ON bronze_gasoline_stocks (series_id, period);
CREATE INDEX IF NOT EXISTS idx_bronze_distillate_series ON bronze_distillate_stocks (series_id, period);
-- ─────────────────────────────────────────────────────────────────────────────
-- QUICK VERIFICATION
-- After running this file, check all tables were created:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- ─────────────────────────────────────────────────────────────────────────────
