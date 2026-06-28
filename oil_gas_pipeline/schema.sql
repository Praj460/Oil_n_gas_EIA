--
-- PostgreSQL database dump
--

\restrict mxPpaI1femCLtB1mmudma5ZZWW6EaN62Rt7kBVvBdl5TZEXPUh6XQS0HoKRJJ12

-- Dumped from database version 15.18 (Homebrew)
-- Dumped by pg_dump version 15.18 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: bronze_cooling_degree_days; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_cooling_degree_days (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_STEO'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_cooling_degree_days OWNER TO prajwalanand;

--
-- Name: bronze_crude_imports; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_crude_imports (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_crude_imports OWNER TO prajwalanand;

--
-- Name: bronze_distillate_stocks; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_distillate_stocks (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_distillate_stocks OWNER TO prajwalanand;

--
-- Name: bronze_dollar_index; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_dollar_index (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'FRED_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_dollar_index OWNER TO prajwalanand;

--
-- Name: bronze_gasoline_stocks; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_gasoline_stocks (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_gasoline_stocks OWNER TO prajwalanand;

--
-- Name: bronze_global_oil_inventory; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_global_oil_inventory (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_STEO'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_global_oil_inventory OWNER TO prajwalanand;

--
-- Name: bronze_heating_degree_days; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_heating_degree_days (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_STEO'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_heating_degree_days OWNER TO prajwalanand;

--
-- Name: bronze_industrial_production; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_industrial_production (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'FRED_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_industrial_production OWNER TO prajwalanand;

--
-- Name: bronze_natural_gas; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_natural_gas (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_natural_gas OWNER TO prajwalanand;

--
-- Name: bronze_opec_spare_capacity; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_opec_spare_capacity (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_STEO'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_opec_spare_capacity OWNER TO prajwalanand;

--
-- Name: bronze_petroleum; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_petroleum (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_petroleum OWNER TO prajwalanand;

--
-- Name: bronze_refinery_utilization; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_refinery_utilization (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'EIA_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_refinery_utilization OWNER TO prajwalanand;

--
-- Name: bronze_treasury_10y; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.bronze_treasury_10y (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    series_id character varying(60) NOT NULL,
    series_name character varying(120),
    period character varying(20) NOT NULL,
    value numeric(12,4),
    unit character varying(40),
    source character varying(20) DEFAULT 'FRED_API'::character varying,
    ingested_at timestamp with time zone DEFAULT now(),
    raw_response jsonb
);


ALTER TABLE public.bronze_treasury_10y OWNER TO prajwalanand;

--
-- Name: data_quality_results; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.data_quality_results (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    suite_name character varying(80) NOT NULL,
    table_name character varying(80) NOT NULL,
    total_expectations integer,
    passed integer,
    failed integer,
    success_rate numeric(5,2),
    run_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.data_quality_results OWNER TO prajwalanand;

--
-- Name: gold_energy_prices; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.gold_energy_prices (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    period date NOT NULL,
    wti_price numeric(10,4),
    brent_price numeric(10,4),
    price_spread numeric(10,4),
    us_oil_production numeric(14,2),
    wti_mom_change numeric(10,4),
    henry_hub_price numeric(10,4),
    us_gas_storage_bcf numeric(12,2),
    us_gas_production numeric(14,2),
    gas_mom_change numeric(10,4),
    oil_gas_ratio numeric(10,4),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.gold_energy_prices OWNER TO prajwalanand;

--
-- Name: gold_features; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.gold_features (
    period date NOT NULL,
    wti_price numeric(12,4),
    henry_hub_price numeric(12,4),
    brent_price numeric(12,4),
    oil_production numeric(14,2),
    crude_imports numeric(14,2),
    refinery_util numeric(10,4),
    gasoline_stocks numeric(14,2),
    distillate_stocks numeric(14,2),
    gas_storage numeric(14,2),
    gas_production numeric(14,2),
    hdd numeric(10,2),
    cdd numeric(10,2),
    opec_spare numeric(10,4),
    global_inv numeric(12,4),
    dollar_index numeric(10,4),
    industrial_production numeric(10,4),
    treasury_10y numeric(8,4),
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.gold_features OWNER TO prajwalanand;

--
-- Name: gold_forecast_results; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.gold_forecast_results (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    run_id uuid DEFAULT public.uuid_generate_v4(),
    target character varying(40) NOT NULL,
    model_name character varying(20) NOT NULL,
    forecast_period date NOT NULL,
    forecast_value numeric(10,4) NOT NULL,
    lower_bound numeric(10,4),
    upper_bound numeric(10,4),
    rmse numeric(10,4),
    mape numeric(10,4),
    trained_on_periods integer,
    created_at timestamp with time zone DEFAULT now(),
    mae numeric(10,4)
);


ALTER TABLE public.gold_forecast_results OWNER TO prajwalanand;

--
-- Name: pipeline_runs; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.pipeline_runs (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    run_name character varying(80) NOT NULL,
    status character varying(20) NOT NULL,
    rows_ingested integer DEFAULT 0,
    rows_failed integer DEFAULT 0,
    error_message text,
    started_at timestamp with time zone DEFAULT now(),
    finished_at timestamp with time zone
);


ALTER TABLE public.pipeline_runs OWNER TO prajwalanand;

--
-- Name: silver_natural_gas; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.silver_natural_gas (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    period date NOT NULL,
    henry_hub_price numeric(10,4),
    us_storage_bcf numeric(12,2),
    us_production numeric(14,2),
    price_mom_change numeric(10,4),
    is_outlier boolean DEFAULT false,
    transformed_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.silver_natural_gas OWNER TO prajwalanand;

--
-- Name: silver_petroleum; Type: TABLE; Schema: public; Owner: prajwalanand
--

CREATE TABLE public.silver_petroleum (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    period date NOT NULL,
    wti_price numeric(10,4),
    brent_price numeric(10,4),
    us_production numeric(14,2),
    price_spread numeric(10,4),
    wti_mom_change numeric(10,4),
    is_outlier boolean DEFAULT false,
    transformed_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.silver_petroleum OWNER TO prajwalanand;

--

--

--

--
-- Name: bronze_cooling_degree_days bronze_cooling_degree_days_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_cooling_degree_days
    ADD CONSTRAINT bronze_cooling_degree_days_pkey PRIMARY KEY (id);


--
-- Name: bronze_crude_imports bronze_crude_imports_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_crude_imports
    ADD CONSTRAINT bronze_crude_imports_pkey PRIMARY KEY (id);


--
-- Name: bronze_distillate_stocks bronze_distillate_stocks_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_distillate_stocks
    ADD CONSTRAINT bronze_distillate_stocks_pkey PRIMARY KEY (id);


--
-- Name: bronze_dollar_index bronze_dollar_index_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_dollar_index
    ADD CONSTRAINT bronze_dollar_index_pkey PRIMARY KEY (id);


--
-- Name: bronze_gasoline_stocks bronze_gasoline_stocks_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_gasoline_stocks
    ADD CONSTRAINT bronze_gasoline_stocks_pkey PRIMARY KEY (id);


--
-- Name: bronze_global_oil_inventory bronze_global_oil_inventory_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_global_oil_inventory
    ADD CONSTRAINT bronze_global_oil_inventory_pkey PRIMARY KEY (id);


--
-- Name: bronze_heating_degree_days bronze_heating_degree_days_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_heating_degree_days
    ADD CONSTRAINT bronze_heating_degree_days_pkey PRIMARY KEY (id);


--
-- Name: bronze_industrial_production bronze_industrial_production_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_industrial_production
    ADD CONSTRAINT bronze_industrial_production_pkey PRIMARY KEY (id);


--
-- Name: bronze_natural_gas bronze_natural_gas_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_natural_gas
    ADD CONSTRAINT bronze_natural_gas_pkey PRIMARY KEY (id);


--
-- Name: bronze_opec_spare_capacity bronze_opec_spare_capacity_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_opec_spare_capacity
    ADD CONSTRAINT bronze_opec_spare_capacity_pkey PRIMARY KEY (id);


--
-- Name: bronze_petroleum bronze_petroleum_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_petroleum
    ADD CONSTRAINT bronze_petroleum_pkey PRIMARY KEY (id);


--
-- Name: bronze_refinery_utilization bronze_refinery_utilization_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_refinery_utilization
    ADD CONSTRAINT bronze_refinery_utilization_pkey PRIMARY KEY (id);


--
-- Name: bronze_treasury_10y bronze_treasury_10y_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.bronze_treasury_10y
    ADD CONSTRAINT bronze_treasury_10y_pkey PRIMARY KEY (id);


--
-- Name: data_quality_results data_quality_results_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.data_quality_results
    ADD CONSTRAINT data_quality_results_pkey PRIMARY KEY (id);


--
-- Name: gold_energy_prices gold_energy_prices_period_key; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.gold_energy_prices
    ADD CONSTRAINT gold_energy_prices_period_key UNIQUE (period);


--
-- Name: gold_energy_prices gold_energy_prices_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.gold_energy_prices
    ADD CONSTRAINT gold_energy_prices_pkey PRIMARY KEY (id);


--
-- Name: gold_features gold_features_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.gold_features
    ADD CONSTRAINT gold_features_pkey PRIMARY KEY (period);


--
-- Name: gold_forecast_results gold_forecast_results_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.gold_forecast_results
    ADD CONSTRAINT gold_forecast_results_pkey PRIMARY KEY (id);


--
-- Name: pipeline_runs pipeline_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.pipeline_runs
    ADD CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id);


--
-- Name: silver_natural_gas silver_natural_gas_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.silver_natural_gas
    ADD CONSTRAINT silver_natural_gas_pkey PRIMARY KEY (id);


--
-- Name: silver_petroleum silver_petroleum_pkey; Type: CONSTRAINT; Schema: public; Owner: prajwalanand
--

ALTER TABLE ONLY public.silver_petroleum
    ADD CONSTRAINT silver_petroleum_pkey PRIMARY KEY (id);


--
-- Name: idx_bronze_cdd_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_cdd_series ON public.bronze_cooling_degree_days USING btree (series_id, period);


--
-- Name: idx_bronze_distillate_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_distillate_series ON public.bronze_distillate_stocks USING btree (series_id, period);


--
-- Name: idx_bronze_dollar_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_dollar_series ON public.bronze_dollar_index USING btree (series_id, period);


--
-- Name: idx_bronze_gas_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_gas_series ON public.bronze_natural_gas USING btree (series_id, period);


--
-- Name: idx_bronze_gasoline_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_gasoline_series ON public.bronze_gasoline_stocks USING btree (series_id, period);


--
-- Name: idx_bronze_globalinv_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_globalinv_series ON public.bronze_global_oil_inventory USING btree (series_id, period);


--
-- Name: idx_bronze_hdd_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_hdd_series ON public.bronze_heating_degree_days USING btree (series_id, period);


--
-- Name: idx_bronze_imports_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_imports_series ON public.bronze_crude_imports USING btree (series_id, period);


--
-- Name: idx_bronze_indpro_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_indpro_series ON public.bronze_industrial_production USING btree (series_id, period);


--
-- Name: idx_bronze_opec_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_opec_series ON public.bronze_opec_spare_capacity USING btree (series_id, period);


--
-- Name: idx_bronze_pet_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_pet_series ON public.bronze_petroleum USING btree (series_id, period);


--
-- Name: idx_bronze_refinery_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_refinery_series ON public.bronze_refinery_utilization USING btree (series_id, period);


--
-- Name: idx_bronze_treasury_series; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_bronze_treasury_series ON public.bronze_treasury_10y USING btree (series_id, period);


--
-- Name: idx_gold_energy_period; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_gold_energy_period ON public.gold_energy_prices USING btree (period);


--
-- Name: idx_gold_forecast_period; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_gold_forecast_period ON public.gold_forecast_results USING btree (forecast_period, target);


--
-- Name: idx_gold_forecast_run; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_gold_forecast_run ON public.gold_forecast_results USING btree (run_id, target, model_name);


--
-- Name: idx_gold_period; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE INDEX idx_gold_period ON public.gold_energy_prices USING btree (period);


--
-- Name: idx_silver_gas_period; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE UNIQUE INDEX idx_silver_gas_period ON public.silver_natural_gas USING btree (period);


--
-- Name: idx_silver_pet_period; Type: INDEX; Schema: public; Owner: prajwalanand
--

CREATE UNIQUE INDEX idx_silver_pet_period ON public.silver_petroleum USING btree (period);


--
-- PostgreSQL database dump complete
--

\unrestrict mxPpaI1femCLtB1mmudma5ZZWW6EaN62Rt7kBVvBdl5TZEXPUh6XQS0HoKRJJ12

