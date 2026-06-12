
  create view "oil_gas_db"."public_public"."silver_petroleum__dbt_tmp"
    
    
  as (
    -- oil_gas_pipeline | dbt/models/stg_petroleum.sql
-- Staging model — cleans and pivots bronze_petroleum into silver_petroleum
-- Reads from: bronze_petroleum (raw EIA API data)
-- Writes to:  silver_petroleum (one row per month, wide format)
-- Materialized as: view (rebuilt on every dbt run)



with

-- ── Step 1: pull raw bronze data ──────────────────────────────────────────────
raw as (
    select
        series_id,
        series_name,
        -- normalize period to first day of the month
        date_trunc('month', period::date)::date  as period,
        value::numeric(12, 4)                    as value,
        unit,
        ingested_at
    from bronze_petroleum
    where
        value is not null
        and period is not null
        -- exclude clearly bad values (negative prices)
        and value >= 0
),

-- ── Step 2: deduplicate — keep latest ingested row per series+period ─────────
deduplicated as (
    select distinct on (series_id, period)
        series_id,
        period,
        value,
        unit,
        ingested_at
    from raw
    order by series_id, period, ingested_at desc
),

-- ── Step 3: pivot — one row per month with each series as its own column ─────
pivoted as (
    select
        period,

        max(case when series_id = 'PET.RWTC.M'     then value end) as wti_price,
        max(case when series_id = 'PET.RBRTE.M'    then value end) as brent_price,
        max(case when series_id = 'PET.MCRFPUS2.M' then value end) as us_production

    from deduplicated
    group by period
),

-- ── Step 4: add derived features ─────────────────────────────────────────────
with_features as (
    select
        period,
        wti_price,
        brent_price,
        us_production,

        -- Brent-WTI spread — useful market indicator
        round(brent_price - wti_price, 4)                           as price_spread,

        -- Month-over-month % change in WTI
        round(
            (wti_price - lag(wti_price) over (order by period))
            / nullif(lag(wti_price) over (order by period), 0) * 100,
            4
        )                                                           as wti_mom_change,

        -- Flag outliers: WTI price more than 3 standard deviations from mean
        -- (computed over the whole dataset)
        abs(wti_price - avg(wti_price) over ())
            > 3 * stddev(wti_price) over ()                         as is_outlier,

        now()                                                       as transformed_at

    from pivoted
    where period is not null
)

-- ── Final select ──────────────────────────────────────────────────────────────
select
    gen_random_uuid()   as id,
    period,
    wti_price,
    brent_price,
    us_production,
    price_spread,
    wti_mom_change,
    coalesce(is_outlier, false) as is_outlier,
    transformed_at
from with_features
order by period asc
  );