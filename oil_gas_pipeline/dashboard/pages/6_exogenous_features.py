# oil_gas_pipeline | dashboard/pages/6_exogenous_features.py
# Exogenous Features dashboard — the new predictors we added on top of price history.
# Centerpiece: OPEC spare capacity overlaid against WTI, showing the early-2026 collapse.

import os
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "oil_gas_db"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

st.set_page_config(page_title="Exogenous Features", page_icon="📊", layout="wide")
st.title("📊 Exogenous Features")
st.caption("The 11 predictor signals we add to the WTI forecast beyond price history alone.")


@st.cache_data(ttl=300)
def load_features():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM gold_features ORDER BY period", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df


df = load_features()

# ── Summary cards: provider counts ─────────────────────────────────────────
st.markdown("### Data foundation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Series total",       17, help="Targets + predictors across all sources")
c2.metric("Predictors",         11, help="Exogenous signals fed to the model")
c3.metric("Data providers",      3, help="EIA petroleum/gas, EIA STEO, FRED")
c4.metric("Monthly coverage",   f"{len(df)} months", help="Jan 2015 → present")

st.divider()

# ── THE STAR CHART: OPEC spare capacity overlaid on WTI ─────────────────────
st.markdown("### The supply-fragility story")
st.markdown(
    "When OPEC spare crude capacity collapses, the market loses its shock absorber. "
    "Watch the inverse relationship clearly during the early-2026 spike."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["period"], y=df["wti_price"],
    name="WTI price ($/bbl)", yaxis="y1",
    line=dict(color="#d62728", width=2),
))
fig.add_trace(go.Scatter(
    x=df["period"], y=df["opec_spare"],
    name="OPEC spare capacity (mbd)", yaxis="y2",
    line=dict(color="#2ca02c", width=2),
))
fig.update_layout(
    height=480,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    yaxis=dict(title="WTI price ($/bbl)", side="left", color="#d62728"),
    yaxis2=dict(title="OPEC spare capacity (mbd)", side="right", overlaying="y",
                color="#2ca02c", showgrid=False),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# Quick callout — the spike window numbers
with st.expander("📌 The 2026 spike, month by month", expanded=False):
    spike = df[(df["period"] >= "2025-09-01") & (df["period"] <= "2026-04-01")][
        ["period", "wti_price", "opec_spare", "global_inv"]
    ].copy()
    spike["period"] = spike["period"].dt.strftime("%Y-%m")
    spike.columns = ["Month", "WTI ($)", "OPEC spare (mbd)", "Global inv (Mbbl)"]
    st.dataframe(spike.set_index("Month"), use_container_width=True)
    st.caption("OPEC spare capacity fell from ~3.0 mbd to near zero exactly as WTI doubled.")

st.divider()

# ── Group all features by economic category ────────────────────────────────
st.markdown("### Predictor catalog")
st.markdown("Grouped by economic mechanism — each one targets a different price driver.")

GROUPS = {
    "Oil supply": [
        ("oil_production",    "US oil production",         "thousand bbl/day"),
        ("crude_imports",     "US crude imports",          "thousand bbl/day"),
        ("refinery_util",     "US refinery utilization",   "% of capacity"),
    ],
    "Downstream demand": [
        ("gasoline_stocks",   "US gasoline stocks",        "thousand bbl"),
        ("distillate_stocks", "US distillate stocks",      "thousand bbl"),
    ],
    "Natural gas (paired)": [
        ("gas_storage",       "US gas storage",            "Bcf"),
        ("gas_production",    "US gas production",         "Bcf"),
        ("henry_hub_price",   "Henry Hub price",           "$/MMBtu"),
    ],
    "Weather demand": [
        ("hdd",  "Heating degree days (US pop-weighted)",  "degree-days"),
        ("cdd",  "Cooling degree days (US pop-weighted)",  "degree-days"),
    ],
    "Supply fragility": [
        ("opec_spare",   "OPEC spare crude capacity",   "mbd"),
        ("global_inv",   "Global commercial oil inventory", "Mbbl"),
    ],
    "Macro": [
        ("dollar_index",          "Trade-weighted USD index", "Index"),
        ("industrial_production", "US industrial production", "Index"),
        ("treasury_10y",          "10-Year Treasury yield",   "Percent"),
    ],
}

selected_group = st.selectbox("Pick a group to inspect:", list(GROUPS.keys()), index=4)

for col, label, unit in GROUPS[selected_group]:
    if col not in df.columns:
        continue
    latest = df.dropna(subset=[col]).iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["period"], y=df[col], name=label,
        line=dict(width=1.6),
    ))
    fig.update_layout(
        height=240, title=f"{label}   |   latest: {latest[col]:.2f} {unit}  ({latest['period'].strftime('%b %Y')})",
        margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Source provenance table — three providers tagged in bronze ─────────────
st.markdown("### Source provenance")
st.markdown("Every row in the bronze layer is tagged with where it came from.")
provenance = pd.DataFrame([
    ["EIA_API",  "Petroleum + Natural Gas",  "6 series",  "WTI, Brent, oil prod, Henry Hub, storage, gas prod"],
    ["EIA_API",  "Exogenous petroleum",       "4 series", "crude imports, refinery util, gasoline stocks, distillate stocks"],
    ["EIA_STEO", "STEO weather + fragility",  "4 series", "HDD, CDD, OPEC spare, global inventory"],
    ["FRED_API", "Macro economic",            "3 series", "Dollar index, industrial prod, 10-yr Treasury"],
], columns=["Source tag", "Group", "Count", "Series"])
st.dataframe(provenance, use_container_width=True, hide_index=True)
