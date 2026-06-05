# oil_gas_pipeline | dashboard/pages/4_data_quality.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Data Quality", page_icon="✅", layout="wide")

DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
          user="prajwalanand", password="India@1947")

def get_conn():
    return psycopg2.connect(**DB)

st.title("✅ Data Quality & Pipeline Monitoring")
st.caption("Great Expectations validation results and pipeline run history")
st.divider()

@st.cache_data(ttl=60)
def load_all():
    conn    = get_conn()
    quality = pd.read_sql("SELECT * FROM data_quality_results ORDER BY run_at DESC LIMIT 100", conn)
    runs    = pd.read_sql("SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 50", conn)
    tables  = {}
    for t in ["bronze_petroleum","bronze_natural_gas","bronze_well_production",
              "silver_petroleum","silver_natural_gas","gold_energy_prices","gold_forecast_results"]:
        try:
            tables[t] = int(pd.read_sql(f"SELECT COUNT(*) as cnt FROM {t}", conn).iloc[0]["cnt"])
        except:
            tables[t] = 0
    conn.close()
    return quality, runs, tables

with st.spinner("Loading monitoring data..."):
    try:
        quality_df, pipeline_df, table_counts = load_all()
    except Exception as e:
        st.error(f"Failed to load: {e}")
        st.stop()

# Medallion row counts
st.subheader("🏛️ Medallion Architecture — Row Counts")
b_col, s_col, g_col = st.columns(3)
with b_col:
    st.markdown("**🟤 Bronze Layer**")
    st.metric("Petroleum Records",      f"{table_counts.get('bronze_petroleum',0):,}")
    st.metric("Natural Gas Records",    f"{table_counts.get('bronze_natural_gas',0):,}")
    st.metric("Well Production Records",f"{table_counts.get('bronze_well_production',0):,}")
with s_col:
    st.markdown("**⚪ Silver Layer**")
    st.metric("Petroleum (cleaned)",    f"{table_counts.get('silver_petroleum',0):,}")
    st.metric("Natural Gas (cleaned)",  f"{table_counts.get('silver_natural_gas',0):,}")
with g_col:
    st.markdown("**🟡 Gold Layer**")
    st.metric("Energy Prices (mart)",   f"{table_counts.get('gold_energy_prices',0):,}")
    st.metric("Forecast Results",       f"{table_counts.get('gold_forecast_results',0):,}")

st.divider()

# Great Expectations results
st.subheader("🧪 Great Expectations — Validation Results")
if quality_df.empty:
    st.info("No quality results yet. Run the ingestion pipeline to trigger validations.")
else:
    latest = quality_df.sort_values("run_at").groupby("suite_name").last().reset_index()
    q1, q2 = st.columns(2)
    for i, row in latest.iterrows():
        col  = q1 if i % 2 == 0 else q2
        rate = row["success_rate"]
        with col:
            st.metric(
                label=f"{row['suite_name']} — {row['table_name']}",
                value=f"{rate:.1f}% {'✅ PASSING' if rate >= 80 else '❌ FAILING'}",
                delta=f"{int(row['passed'])}/{int(row['total_expectations'])} checks passed",
            )

    st.subheader("📉 Quality Score Over Time")
    fig = px.line(quality_df.sort_values("run_at"), x="run_at", y="success_rate",
                  color="suite_name", markers=True,
                  labels={"run_at":"Run Time","success_rate":"Pass Rate (%)","suite_name":"Suite"})
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Quality Gate (80%)")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(range=[0,105]))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Pipeline run history
st.subheader("🔄 Pipeline Run History")
if pipeline_df.empty:
    st.info("No pipeline runs recorded yet.")
else:
    p1, p2, p3, p4 = st.columns(4)
    total   = len(pipeline_df)
    success = (pipeline_df["status"] == "success").sum()
    failed  = (pipeline_df["status"] == "failed").sum()
    rate    = (success / total * 100) if total > 0 else 0
    with p1: st.metric("Total Runs",   total)
    with p2: st.metric("Successful",   success)
    with p3: st.metric("Failed",       failed)
    with p4: st.metric("Success Rate", f"{rate:.1f}%")

    display = pipeline_df[["run_name","status","rows_ingested","rows_failed",
                            "error_message","started_at","finished_at"]].head(20)

    def highlight(row):
        if row["status"] == "success": return ["background-color:#d4edda"]*len(row)
        if row["status"] == "failed":  return ["background-color:#f8d7da"]*len(row)
        return [""]*len(row)

    st.dataframe(display.style.apply(highlight, axis=1), use_container_width=True, height=350)