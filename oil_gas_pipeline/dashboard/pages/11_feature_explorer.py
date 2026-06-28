# oil_gas_pipeline | dashboard/pages/11_feature_explorer.py
# Feature Explorer — pick any feature from the sidebar, see full stats,
# history chart, recent values, and why we added it.
# Designed for screen-share demos with Kedar.

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

st.set_page_config(page_title="Feature Explorer", page_icon="🔍", layout="wide")
st.title("🔍 Feature Explorer")
st.caption(
    "Pick any feature from the sidebar. See its full history, key stats, "
    "recent values, and why it belongs in the model."
)

# ── Feature catalog — every column with metadata ───────────────────────────
FEATURES = {
    # ── WTI Target ──────────────────────────────────────────────────────────
    "WTI Crude Oil Price": {
        "col": "wti_price",
        "unit": "$/barrel",
        "group": "🎯 Target — Oil",
        "why": (
            "The primary forecasting target. WTI (West Texas Intermediate) is the "
            "US benchmark crude oil price, set daily at Cushing, Oklahoma. "
            "Everything else in the oil model is built to predict this."
        ),
        "wti_corr": None,
        "hh_corr": None,
    },
    # ── Henry Hub Target ────────────────────────────────────────────────────
    "Henry Hub Gas Price": {
        "col": "henry_hub_price",
        "unit": "$/MMBtu",
        "group": "🎯 Target — Gas",
        "why": (
            "The secondary forecasting target. Henry Hub is the US benchmark natural "
            "gas price, set at a pipeline hub in Louisiana. Driven by completely "
            "different mechanisms than oil — weather and storage, not geopolitics."
        ),
        "wti_corr": True,
        "hh_corr": None,
    },
    # ── Oil supply ──────────────────────────────────────────────────────────
    "Brent Crude Oil Price": {
        "col": "brent_price",
        "unit": "$/barrel",
        "group": "🛢️ Oil Supply & Prices",
        "why": (
            "The international oil benchmark. Brent and WTI move together but "
            "diverge on supply events. Adding Brent gives the model a global "
            "supply signal beyond the US market alone."
        ),
        "wti_corr": True,
        "hh_corr": True,
    },
    "US Oil Production": {
        "col": "oil_production",
        "unit": "thousand bbl/day",
        "group": "🛢️ Oil Supply & Prices",
        "why": (
            "Domestic supply. When US production ramps up (e.g. shale boom), "
            "it adds to global supply and pushes WTI down. A key supply-side lever "
            "that the model needs to see."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    "US Crude Imports": {
        "col": "crude_imports",
        "unit": "thousand bbl/day",
        "group": "🛢️ Oil Supply & Prices",
        "why": (
            "How much crude the US is importing. High imports signal domestic "
            "demand exceeding production — a bullish supply pressure on price."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    "US Refinery Utilization": {
        "col": "refinery_util",
        "unit": "% of capacity",
        "group": "🛢️ Oil Supply & Prices",
        "why": (
            "How hard refineries are running. High utilization = strong demand "
            "for crude to process. When refineries are at capacity, they compete "
            "for crude barrels and push prices up."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    # ── Oil demand ──────────────────────────────────────────────────────────
    "US Gasoline Stocks": {
        "col": "gasoline_stocks",
        "unit": "thousand barrels",
        "group": "⛽ Oil Demand",
        "why": (
            "Finished motor gasoline inventory. High gasoline stocks signal "
            "oversupply downstream — refineries made more than consumers needed. "
            "A bearish signal for crude demand and WTI price."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    "US Distillate Stocks": {
        "col": "distillate_stocks",
        "unit": "thousand barrels",
        "group": "⛽ Oil Demand",
        "why": (
            "Diesel and heating oil inventory. Distillates serve both transportation "
            "and industrial use. Low distillate stocks heading into winter is a "
            "tight-market signal that supports crude prices."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    # ── Natural gas ─────────────────────────────────────────────────────────
    "US Gas Storage": {
        "col": "gas_storage",
        "unit": "Bcf",
        "group": "🔥 Natural Gas",
        "why": (
            "The single most important Henry Hub predictor. Storage builds in "
            "summer and draws down in winter. Going into a cold winter with low "
            "storage means no cushion — prices spike. The gas equivalent of OPEC "
            "spare capacity for oil."
        ),
        "wti_corr": False,
        "hh_corr": True,
    },
    "US Gas Production": {
        "col": "gas_production",
        "unit": "Bcf/month",
        "group": "🔥 Natural Gas",
        "why": (
            "Domestic gas supply. Higher production keeps storage levels healthy "
            "and dampens price spikes during demand surges. A key supply-side "
            "offset to weather-driven demand shocks."
        ),
        "wti_corr": False,
        "hh_corr": True,
    },
    # ── Weather ─────────────────────────────────────────────────────────────
    "Heating Degree Days (HDD)": {
        "col": "hdd",
        "unit": "degree-days",
        "group": "🌡️ Weather",
        "why": (
            "Population-weighted measure of how cold the US was each month. "
            "Every degree below 65°F = 1 HDD. High HDD → people turn on heating "
            "→ gas demand surges. The textbook leading indicator for Henry Hub."
        ),
        "wti_corr": False,
        "hh_corr": True,
    },
    "Cooling Degree Days (CDD)": {
        "col": "cdd",
        "unit": "degree-days",
        "group": "🌡️ Weather",
        "why": (
            "Same concept but for summer heat. High CDD → air conditioning runs hard "
            "→ power plants burn more gas for electricity. Drives the secondary "
            "summer demand peak visible in Henry Hub's seasonal pattern."
        ),
        "wti_corr": False,
        "hh_corr": True,
    },
    # ── Supply fragility ────────────────────────────────────────────────────
    "OPEC Spare Capacity": {
        "col": "opec_spare",
        "unit": "million barrels/day",
        "group": "⚠️ Supply Fragility",
        "why": (
            "How much extra oil OPEC could pump immediately but isn't. This is "
            "the market's insurance policy against supply shocks. When spare "
            "capacity collapsed to near zero in early 2026, WTI doubled — the "
            "single most important causal finding in this project."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    "Global Oil Inventory": {
        "col": "global_inv",
        "unit": "million barrels",
        "group": "⚠️ Supply Fragility",
        "why": (
            "Total commercial oil sitting in storage worldwide. The world's "
            "physical oil buffer. When inventory draws down alongside collapsing "
            "spare capacity, the market has neither a present nor a future cushion "
            "against disruption."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    # ── Macro ───────────────────────────────────────────────────────────────
    "Trade-Weighted Dollar Index": {
        "col": "dollar_index",
        "unit": "Index",
        "group": "📊 Macro",
        "why": (
            "Oil is priced in US dollars globally. When the dollar strengthens, "
            "oil becomes more expensive for non-US buyers, reducing demand and "
            "pushing prices down. The currency channel is a consistent macro "
            "driver of oil prices."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
    "US Industrial Production": {
        "col": "industrial_production",
        "unit": "Index (2017=100)",
        "group": "📊 Macro",
        "why": (
            "Real economy demand proxy. Industrial activity drives energy consumption "
            "— factories burn fuel and generate heat. The most statistically "
            "significant exogenous feature in the SARIMAX model for WTI."
        ),
        "wti_corr": True,
        "hh_corr": True,
    },
    "10-Year Treasury Yield": {
        "col": "treasury_10y",
        "unit": "Percent",
        "group": "📊 Macro",
        "why": (
            "The prevailing long-term interest rate. Higher rates increase the "
            "cost of holding oil inventory (financing cost) and signal tighter "
            "financial conditions, which typically dampen commodity demand and prices."
        ),
        "wti_corr": True,
        "hh_corr": False,
    },
}

# ── Group ordering for sidebar ─────────────────────────────────────────────
GROUP_ORDER = [
    "🎯 Target — Oil",
    "🎯 Target — Gas",
    "🛢️ Oil Supply & Prices",
    "⛽ Oil Demand",
    "🔥 Natural Gas",
    "🌡️ Weather",
    "⚠️ Supply Fragility",
    "📊 Macro",
]


@st.cache_data(ttl=300)
def load():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM gold_features ORDER BY period", conn)
    conn.close()
    df["period"] = pd.to_datetime(df["period"])
    return df


df = load()

# ── Sidebar — group then feature picker ───────────────────────────────────
st.sidebar.markdown("## Feature Explorer")
st.sidebar.markdown("Pick a group and feature to explore:")

groups = GROUP_ORDER
selected_group = st.sidebar.selectbox("Group:", groups)

group_features = {k: v for k, v in FEATURES.items() if v["group"] == selected_group}
selected_feature = st.sidebar.selectbox("Feature:", list(group_features.keys()))

meta = FEATURES[selected_feature]
col = meta["col"]
series = df[["period", col]].dropna(subset=[col])

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"## {selected_feature}")
st.markdown(f"**Group:** {meta['group']}   |   **Unit:** {meta['unit']}")

# Why we need this
st.info(f"**Why this feature?** {meta['why']}")

st.divider()

# ── Stats row ──────────────────────────────────────────────────────────────
st.markdown("### Key statistics (full history)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Minimum",  f"{series[col].min():.2f}")
c2.metric("Maximum",  f"{series[col].max():.2f}")
c3.metric("Mean",     f"{series[col].mean():.2f}")
c4.metric("Std Dev",  f"{series[col].std():.2f}")
c5.metric("Latest",   f"{series[col].iloc[-1]:.2f}",
          help=series["period"].iloc[-1].strftime("%b %Y"))

st.divider()

# ── Full history chart ──────────────────────────────────────────────────────
st.markdown("### Full history")

# Overlay WTI or Henry Hub on a second axis if relevant
show_wti = meta.get("wti_corr") and col != "wti_price"
show_hh  = meta.get("hh_corr")  and col != "henry_hub_price" and not show_wti

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=series["period"], y=series[col],
    name=f"{selected_feature} ({meta['unit']})",
    line=dict(color="#1f77b4", width=2),
    yaxis="y1",
))

if show_wti:
    wti = df[["period", "wti_price"]].dropna()
    fig.add_trace(go.Scatter(
        x=wti["period"], y=wti["wti_price"],
        name="WTI price ($/bbl)", yaxis="y2",
        line=dict(color="#d62728", width=1.5, dash="dot"), opacity=0.7,
    ))
    fig.update_layout(
        yaxis2=dict(title="WTI ($/bbl)", overlaying="y", side="right",
                    color="#d62728", showgrid=False),
    )
elif show_hh:
    hh = df[["period", "henry_hub_price"]].dropna()
    fig.add_trace(go.Scatter(
        x=hh["period"], y=hh["henry_hub_price"],
        name="Henry Hub ($/MMBtu)", yaxis="y2",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"), opacity=0.7,
    ))
    fig.update_layout(
        yaxis2=dict(title="Henry Hub ($/MMBtu)", overlaying="y", side="right",
                    color="#ff7f0e", showgrid=False),
    )

fig.update_layout(
    height=420,
    yaxis=dict(title=f"{selected_feature} ({meta['unit']})"),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Two targets side by side ───────────────────────────────────────────────
st.markdown("### Relationship to both targets")
col1, col2 = st.columns(2)

# Safe correlation — use index alignment to avoid duplicate column issue
# when col == "wti_price" or col == "henry_hub_price"
feat    = df.set_index("period")[col].dropna().rename("feat")
wti_t   = df.set_index("period")["wti_price"].dropna().rename("wti")
hh_t    = df.set_index("period")["henry_hub_price"].dropna().rename("hh")

wti_df  = pd.concat([feat, wti_t], axis=1).dropna()
hh_df   = pd.concat([feat, hh_t],  axis=1).dropna()

wti_corr = float(wti_df["feat"].corr(wti_df["wti"]))
hh_corr  = float(hh_df["feat"].corr(hh_df["hh"]))

with col1:
    st.markdown("**vs WTI Crude Oil**")
    delta_color = "normal" if abs(wti_corr) > 0.2 else "off"
    st.metric("Pearson correlation", f"{wti_corr:.3f}",
              delta=f"{'strong' if abs(wti_corr) > 0.5 else 'weak'} {'positive' if wti_corr > 0 else 'negative'}",
              delta_color=delta_color)
    fig_wti = go.Figure(go.Scatter(
        x=wti_df["feat"], y=wti_df["wti"],
        mode="markers",
        marker=dict(color="#d62728", size=5, opacity=0.6),
    ))
    fig_wti.update_layout(
        height=280,
        xaxis_title=f"{selected_feature} ({meta['unit']})",
        yaxis_title="WTI ($/bbl)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_wti, use_container_width=True)

with col2:
    st.markdown("**vs Henry Hub Gas**")
    delta_color = "normal" if abs(hh_corr) > 0.2 else "off"
    st.metric("Pearson correlation", f"{hh_corr:.3f}",
              delta=f"{'strong' if abs(hh_corr) > 0.5 else 'weak'} {'positive' if hh_corr > 0 else 'negative'}",
              delta_color=delta_color)
    fig_hh = go.Figure(go.Scatter(
        x=hh_df["feat"], y=hh_df["hh"],
        mode="markers",
        marker=dict(color="#ff7f0e", size=5, opacity=0.6),
    ))
    fig_hh.update_layout(
        height=280,
        xaxis_title=f"{selected_feature} ({meta['unit']})",
        yaxis_title="Henry Hub ($/MMBtu)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_hh, use_container_width=True)

st.divider()

# ── Recent 6 months ────────────────────────────────────────────────────────
st.markdown("### Recent values (last 6 months)")
recent = series.tail(6).copy()
recent["period"] = recent["period"].dt.strftime("%b %Y")
recent.columns = ["Month", f"{selected_feature} ({meta['unit']})"]
recent = recent.set_index("Month").T
st.dataframe(recent, use_container_width=True)

# Flag if latest value is near historical extremes
latest_val = series[col].iloc[-1]
pct_rank = (series[col] < latest_val).mean() * 100
if pct_rank > 90:
    st.warning(f"⚠️ Latest value is in the **top 10%** of all historical readings — historically high.")
elif pct_rank < 10:
    st.warning(f"⚠️ Latest value is in the **bottom 10%** of all historical readings — historically low.")
else:
    st.success(f"✅ Latest value is at the **{pct_rank:.0f}th percentile** of historical range — within normal bounds.")