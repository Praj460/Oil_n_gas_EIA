-- oil_gas_pipeline | dbt/models/stg_natural_gas.sql
-- Staging model — cleans and pivots bronze_natural_gas into silver_natural_gas
-- Reads from: bronze_natural_gas (raw EIA API data)
-- Writes to:  silver_natural_gas (one row per month, wide format)
-- Materialized as: view (rebuilt on every dbt run)



with

-- ── Step 1: pull raw bronze data ──────────────────────────────────────────────
raw as (
    select
        series_id,
        series_name,
        date_trunc('month', period::date)::date  as period,
        value::numeric(12, 4)                    as value,
        unit,
        ingested_at
    from bronze_natural_gas
    where
        value is not null
        and period is not null
        and value >= 0
),

-- ── Step 2: deduplicate — keep latest ingested row per series+period ─────────
deduplicated as (
    select distinct on (series_id, period)
        series_id,
        period,
        value,
        ingested_at
    from raw
    order by series_id, period, ingested_at desc
),

-- ── Step 3: pivot — one row per month ────────────────────────────────────────
pivoted as (
    select
        period,

        max(case when series_id = 'NG.RNGWHHD.M'              then value end) as henry_hub_price,
        max(case when series_id = 'NG.NW2_EPG0_SWO_R48_BCF.M' then value end) as us_storage_bcf,
        max(case when series_id = 'NG.N9010US2.M'              then value end) as us_production

    from deduplicated
    group by period
),

-- ── Step 4: add derived features ─────────────────────────────────────────────
with_features as (
    select
        period,
        henry_hub_price,
        us_storage_bcf,
        us_production,

        -- Month-over-month % change in Henry Hub price
        round(
            (henry_hub_price - lag(henry_hub_price) over (order by period))
            / nullif(lag(henry_hub_price) over (order by period), 0) * 100,
            4
        )                                                                   as price_mom_change,

        -- Outlier flag — Henry Hub price > 3 std devs from mean
        abs(henry_hub_price - avg(henry_hub_price) over ())
            > 3 * stddev(henry_hub_price) over ()                           as is_outlier,

        now()                                                               as transformed_at

    from pivoted
    where period is not null
)

-- ── Final select ──────────────────────────────────────────────────────────────
select
    gen_random_uuid()           as id,
    period,
    henry_hub_price,
    us_storage_bcf,
    us_production,
    price_mom_change,
    coalesce(is_outlier, false) as is_outlier,
    transformed_at
from with_features
order by period asc