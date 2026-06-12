# oil_gas_pipeline | dashboard/app.py
# Main page - Data Quality & Pipeline Monitoring

import sys
sys.path.insert(0, "/Users/prajwalanand/Oil_n_gas/oil_gas_pipeline")

import streamlit as st
import pandas as pd
import psycopg2

st.set_page_config(page_title="Data Quality", page_icon="✅", layout="wide")

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

# -- Data loading ------------------------------------------------------

@st.cache_data(ttl=60)
def load_counts():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    counts = {}
    for name, table in [
        ("pet",  "bronze_petroleum"),
        ("gas",  "bronze_natural_gas"),
        ("gold", "gold_energy_prices"),
        ("fc",   "gold_forecast_results"),
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[name] = cur.fetchone()[0]
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

# -- Data overview (plain language, no empty tables) -------------------

st.header("📦 Data in the Pipeline")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Raw Oil Price Records",
    counts["pet"],
    help="WTI, Brent, and US production data pulled from the EIA API (bronze layer)",
)
c2.metric(
    "Raw Gas Price Records",
    counts["gas"],
    help="Henry Hub, storage, and gas production data from the EIA API (bronze layer)",
)
c3.metric(
    "Monthly Combined Table",
    f"{counts['gold']} months",
    help="One clean row per month with all prices and derived features (gold layer) - used by charts and models",
)
c4.metric(
    "Stored Forecasts",
    counts["fc"],
    help="Forecast rows from SARIMA and Prophet - includes 2024 validation and next-12-month future forecasts",
)

st.caption(
    "Flow: EIA API → raw records → quality checks → combined monthly table → forecasts"
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
