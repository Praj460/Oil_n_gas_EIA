# oil_gas_pipeline | dashboard/pages/8_feature_importance.py
# Feature Importance — re-trains XGBoost on cached features and shows
# which signals it actually leaned on. Confirms the story isn't just narrative.

import os
import numpy as np
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Feature Importance", page_icon="🎯", layout="wide")
st.title("🎯 Feature Importance")
st.caption("What did the model actually learn? Re-trained on demand to keep the numbers honest.")

CURATED_FEATURES = [
    "wti_price_lag_1", "wti_price_lag_3",
    "wti_price_roll3_mean", "wti_price_roll6_mean", "wti_price_mom_pct",
    "brent_price_lag_1", "henry_hub_price_lag_1",
    "crude_imports_lag_1", "refinery_util_lag_1",
    "opec_spare_lag_1", "global_inv_lag_1",
    "gasoline_stocks_lag_1", "distillate_stocks_lag_1",
    "industrial_production_lag_1",
    "dollar_index_lag_1", "treasury_10y_lag_1",
    "month_sin", "month_cos",
    "opec_spare_roll3_std",
]

# Map each feature to its economic group for color-coding
FEATURE_GROUP = {
    "wti_price_lag_1": "Own-price",  "wti_price_lag_3": "Own-price",
    "wti_price_roll3_mean": "Own-price", "wti_price_roll6_mean": "Own-price",
    "wti_price_mom_pct": "Own-price",
    "brent_price_lag_1": "Co-moving", "henry_hub_price_lag_1": "Co-moving",
    "crude_imports_lag_1": "Supply", "refinery_util_lag_1": "Supply",
    "opec_spare_lag_1": "Supply fragility", "global_inv_lag_1": "Supply fragility",
    "opec_spare_roll3_std": "Supply fragility",
    "gasoline_stocks_lag_1": "Demand", "distillate_stocks_lag_1": "Demand",
    "industrial_production_lag_1": "Demand",
    "dollar_index_lag_1": "Macro", "treasury_10y_lag_1": "Macro",
    "month_sin": "Seasonality", "month_cos": "Seasonality",
}
GROUP_COLORS = {
    "Own-price":         "#1f77b4",
    "Co-moving":         "#17becf",
    "Supply":            "#ff7f0e",
    "Supply fragility":  "#d62728",
    "Demand":            "#9467bd",
    "Macro":             "#2ca02c",
    "Seasonality":       "#7f7f7f",
}


@st.cache_data(ttl=600)
def fit_model(window: str):
    """Train tuned XGBoost on the chosen window and return feature importances."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None, "xgboost not installed — `pip install xgboost`"

    df = pd.read_csv(f"{PROJECT_ROOT}/gold_features_engineered_raw.csv",
                     parse_dates=["period"]).set_index("period").sort_index()
    conn = psycopg2.connect(**DB)
    gold = pd.read_sql("SELECT period, wti_price FROM gold_features ORDER BY period", conn).set_index("period")
    conn.close()
    gold.index = pd.to_datetime(gold.index)
    df["wti_price"] = gold["wti_price"]

    if window == "2024 calm year":
        train = df.loc["2016-01-01":"2023-12-01"]
        params = dict(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, random_state=42, n_jobs=-1)
    else:  # 2026 spike window
        train = df.loc["2016-01-01":"2025-12-01"]
        params = dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.8, random_state=42, n_jobs=-1)

    train = train.dropna(subset=CURATED_FEATURES + ["wti_price"])
    X, y = train[CURATED_FEATURES], train["wti_price"]
    model = XGBRegressor(**params)
    model.fit(X, y)
    imp = pd.DataFrame({
        "feature": CURATED_FEATURES,
        "importance": model.feature_importances_,
        "group": [FEATURE_GROUP.get(f, "Other") for f in CURATED_FEATURES],
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp, None


window_choice = st.radio(
    "Which experiment's model do you want to see?",
    ["2024 calm year", "2026 spike window"],
    horizontal=True,
)

with st.spinner("Re-fitting the tuned XGBoost model…"):
    imp, err = fit_model(window_choice)

if err:
    st.error(err)
    st.stop()

# ── Top features bar chart ─────────────────────────────────────────────────
st.markdown(f"### Feature importance — {window_choice}")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=imp["importance"][::-1],
    y=imp["feature"][::-1],
    orientation="h",
    marker_color=[GROUP_COLORS[g] for g in imp["group"][::-1]],
    text=[f"{i:.3f}" for i in imp["importance"][::-1]],
    textposition="outside",
    hovertemplate="%{y}<br>importance %{x:.4f}<extra></extra>",
))
fig.update_layout(
    height=600,
    xaxis_title="XGBoost feature importance",
    yaxis_title=None,
    margin=dict(l=10, r=60, t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# ── Group totals — where the predictive power actually lives ──────────────
st.markdown("### Importance by economic group")
st.markdown("Aggregating individual features into the categories they belong to.")

group_totals = (
    imp.groupby("group")["importance"].sum()
    .sort_values(ascending=False)
    .reset_index()
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=group_totals["group"],
    y=group_totals["importance"],
    marker_color=[GROUP_COLORS.get(g, "#999") for g in group_totals["group"]],
    text=[f"{v:.1%}" for v in group_totals["importance"]],
    textposition="outside",
))
fig2.update_layout(
    height=320,
    yaxis_title="Sum of feature importances",
    yaxis_tickformat=".0%",
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "**Reading this honestly:** the model leans heavily on **own-price momentum** "
    "and **co-moving Brent**. The supply-fragility and macro features make economic sense "
    "and tell a story, but their individual contribution to the prediction is small. "
    "That's a limitation worth stating out loud, not hiding."
)
