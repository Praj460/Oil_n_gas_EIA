# oil_gas_pipeline | dashboard/app.py
# Main page - Data Quality & Pipeline Monitoring

import sys
sys.path.insert(0, "/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline")

import streamlit as st
import pandas as pd
import psycopg2
from dotenv import load_dotenv
st.set_page_config(page_title="Data Quality", page_icon="✅", layout="wide")

import os
load_dotenv()

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

# All 13 bronze tables grouped by provider — for the high-level data summary
BRONZE_TABLES = {
    "EIA_API": [
        "bronze_petroleum", "bronze_natural_gas",
        "bronze_crude_imports", "bronze_refinery_utilization",
        "bronze_gasoline_stocks", "bronze_distillate_stocks",
    ],
    "EIA_STEO": [
        "bronze_heating_degree_days", "bronze_cooling_degree_days",
        "bronze_opec_spare_capacity", "bronze_global_oil_inventory",
    ],
    "FRED_API": [
        "bronze_dollar_index", "bronze_industrial_production", "bronze_treasury_10y",
    ],
}

# -- Data loading ------------------------------------------------------

@st.cache_data(ttl=60)
def load_counts():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    counts = {}

    # Core layer counts
    for name, table in [
        ("gold",     "gold_features"),
        ("fc",       "gold_forecast_results"),
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[name] = cur.fetchone()[0]

    # Total raw bronze records across all 13 tables, plus per-provider table counts
    total_raw = 0
    provider_tables = {}
    for provider, tables in BRONZE_TABLES.items():
        provider_tables[provider] = len(tables)
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total_raw += cur.fetchone()[0]
    counts["total_raw"] = total_raw
    counts["provider_tables"] = provider_tables
    counts["n_bronze_tables"] = sum(len(t) for t in BRONZE_TABLES.values())

    # Distinct models in the forecast results
    cur.execute("SELECT COUNT(DISTINCT model_name) FROM gold_forecast_results")
    counts["n_models"] = cur.fetchone()[0]

    conn.close()
    return counts

@st.cache_data(ttl=60)
def load_quality_results():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT suite_name, table_name, total_expectations, passed, "
        "failed, success_rate, run_at "
        "FROM data_quality_results ORDER BY run_at DESC", conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def load_pipeline_runs():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql(
        "SELECT run_name, status, rows_ingested, rows_failed, "
        "error_message, started_at, finished_at "
        "FROM pipeline_runs ORDER BY started_at DESC LIMIT 50", conn)
    conn.close()
    return df

counts  = load_counts()
quality = load_quality_results()
runs    = load_pipeline_runs()

# -- Header ------------------------------------------------------------

st.title("✅ Data Quality & Pipeline Monitoring")
st.caption("End-to-end pipeline health: data collected, quality checks, and run history")

st.divider()

# -- Data overview (high-level: series, providers, features, models) ---

st.header("📦 Data in the Pipeline")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Data Series",
    "17",
    help="2 forecast targets (WTI, Henry Hub) + 15 predictor signals across supply, demand, weather, and macro",
)
c2.metric(
    "Data Providers",
    f"{len(BRONZE_TABLES)}",
    help="EIA petroleum/gas API, EIA STEO route, and the FRED macroeconomic API",
)
c3.metric(
    "Raw Records",
    f"{counts['total_raw']:,}",
    help=f"Total rows across all {counts['n_bronze_tables']} bronze tables (the raw ingested layer)",
)
c4.metric(
    "Engineered Features",
    "192",
    help="Lags, rolling stats, momentum, seasonality, and cross-series features built on top of the 17 base series",
)

# Second row — gold layer + modeling
c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Monthly Feature Table",
    f"{counts['gold']} months",
    help="One clean row per month with all 17 series aligned (gold_features) — the model-ready table",
)
c6.metric(
    "Bronze Tables",
    f"{counts['n_bronze_tables']}",
    help="One table per series in the raw layer, each tagged with its source provider",
)
c7.metric(
    "Model Variants",
    f"{counts['n_models']}",
    help="SARIMA, SARIMAX, and XGBoost variants across both targets — see Model Showdown",
)
c8.metric(
    "Stored Forecasts",
    counts["fc"],
    help="Forecast rows across all models and test windows in gold_forecast_results",
)

# Provider breakdown caption
provider_summary = "  •  ".join(
    f"**{prov.replace('_', ' ')}**: {n} series"
    for prov, n in counts["provider_tables"].items()
)
st.markdown(f"By provider:  {provider_summary}")

st.caption(
    "Flow: 3 data providers → 13 bronze tables (raw) → wide monthly feature table (gold) "
    "→ 192 engineered features → forecasting models"
)

st.divider()

# -- Great Expectations section ----------------------------------------

st.header("🧪 Data Quality Checks")

if quality.empty:
    st.info("No quality results yet. Run: python3 scripts/2_quality_check.py")
else:
    latest = quality.drop_duplicates(subset=["suite_name"], keep="first")

    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        passing = row["failed"] == 0
        col.metric(
            label=row["suite_name"].replace("_", " ").title(),
            value=f"{row['passed']}/{row['total_expectations']} passed",
            delta=f"{row['success_rate']:.1f}% success rate",
            delta_color="normal" if passing else "inverse",
        )

    st.subheader("Check History")
    q_table = quality.copy()
    q_table["run_at"] = pd.to_datetime(q_table["run_at"]).dt.strftime("%Y-%m-%d %H:%M")
    q_table.columns = ["Suite", "Table", "Total Checks", "Passed",
                       "Failed", "Success %", "Run At"]
    st.dataframe(q_table, use_container_width=True, hide_index=True)

st.divider()

# -- Pipeline run history ----------------------------------------------

st.header("🔄 Pipeline Run History")

if runs.empty:
    st.info("No pipeline runs yet. Run: python3 scripts/1_ingest.py")
else:
    total      = len(runs)
    successful = (runs["status"] == "success").sum()
    failed     = (runs["status"] == "failed").sum()
    rate       = (successful / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", total)
    c2.metric("Successful", successful)
    c3.metric("Failed", failed)
    c4.metric("Success Rate", f"{rate:.1f}%")

    r_table = runs.copy()
    r_table["started_at"]  = pd.to_datetime(r_table["started_at"]).dt.strftime("%Y-%m-%d %H:%M")
    r_table["finished_at"] = pd.to_datetime(r_table["finished_at"]).dt.strftime("%Y-%m-%d %H:%M")
    r_table["error_message"] = r_table["error_message"].fillna("—")
    r_table.columns = ["Run Name", "Status", "Rows Ingested", "Rows Failed",
                       "Error", "Started", "Finished"]
    st.dataframe(r_table, use_container_width=True, hide_index=True)