
  
    

  create  table "oil_gas_db"."public_public"."gold_energy_prices__dbt_tmp"
  
  
    as
  
  (
    -- oil_gas_pipeline | dbt/models/mart_energy_prices.sql
-- Gold mart model — joins silver petroleum + silver natural gas
-- into one wide, analysis-ready table for forecasting and the dashboard
-- Reads from: silver_petroleum, silver_natural_gas (the two staging views)
-- Writes to:  gold_energy_prices (persisted table)
-- Materialized as: table (persisted — fast for Streamlit queries)



with

-- ── Pull from silver views (built by the two staging models) ──────────────────
petroleum as (
    select
        period,
        wti_price,
        brent_price,
        us_production       as us_oil_production,
        price_spread,
        wti_mom_change,
        is_outlier          as oil_is_outlier
    from "oil_gas_db"."public_public"."silver_petroleum"
),

natural_gas as (
    select
        period,
        henry_hub_price,
        us_storage_bcf      as us_gas_storage_bcf,
        us_production       as us_gas_production,
        price_mom_change    as gas_mom_change,
        is_outlier          as gas_is_outlier
    from "oil_gas_db"."public_public"."silver_natural_gas"
),

-- ── Full outer join — keep months even if one source is missing data ──────────
joined as (
    select
        coalesce(p.period, g.period)    as period,

        -- petroleum columns
        p.wti_price,
        p.brent_price,
        p.price_spread,
        p.us_oil_production,
        p.wti_mom_change,
        p.oil_is_outlier,

        -- natural gas columns
        g.henry_hub_price,
        g.us_gas_storage_bcf,
        g.us_gas_production,
        g.gas_mom_change,
        g.gas_is_outlier

    from petroleum p
    full outer join natural_gas g
        on p.period = g.period
),

-- ── Add cross-commodity derived features ─────────────────────────────────────
with_features as (
    select
        period,
        wti_price,
        brent_price,
        price_spread,
        us_oil_production,
        wti_mom_change,
        henry_hub_price,
        us_gas_storage_bcf,
        us_gas_production,
        gas_mom_change,

        -- Oil-to-gas price ratio — key metric for energy traders
        -- tells you how many MMBtu of gas equals one barrel of oil
        round(
            wti_price / nullif(henry_hub_price, 0),
            4
        )                                                       as oil_gas_ratio,

        -- Rolling 3-month average WTI — smooths out short-term spikes
        round(
            avg(wti_price) over (
                order by period
                rows between 2 preceding and current row
            ),
            4
        )                                                       as wti_3m_avg,

        -- Rolling 3-month average Henry Hub
        round(
            avg(henry_hub_price) over (
                order by period
                rows between 2 preceding and current row
            ),
            4
        )                                                       as henry_hub_3m_avg,

        -- Year-over-year WTI change
        round(
            wti_price - lag(wti_price, 12) over (order by period),
            4
        )                                                       as wti_yoy_change,

        -- Flag if either commodity has an outlier this month
        (coalesce(oil_is_outlier, false) or coalesce(gas_is_outlier, false)) as has_outlier,

        now()                                                   as created_at

    from joined
    where period is not null
)

-- ── Final select ──────────────────────────────────────────────────────────────
select
    gen_random_uuid()   as id,
    period,
    wti_price,
    brent_price,
    price_spread,
    us_oil_production,
    wti_mom_change,
    wti_3m_avg,
    wti_yoy_change,
    henry_hub_price,
    us_gas_storage_bcf,
    us_gas_production,
    gas_mom_change,
    henry_hub_3m_avg,
    oil_gas_ratio,
    has_outlier,
    created_at
from with_features
order by period asc
  );
  