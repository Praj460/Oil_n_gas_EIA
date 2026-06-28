# oil_gas_pipeline | tests/test_feature_engineer.py
# Unit tests for the FeatureEngineer class
# Run with: python3 -m pytest tests/test_feature_engineer.py -v

import pytest
import numpy as np
import pandas as pd
from models.feature_engineer import FeatureEngineer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_monthly_df():
    """Well-formed gold_features-shaped monthly DataFrame, 5 years, no nulls."""
    periods = pd.date_range("2018-01-01", periods=60, freq="MS")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "period":           periods,
        "wti_price":        np.linspace(50, 90, 60) + rng.normal(0, 2, 60),
        "henry_hub_price":  np.linspace(2, 5, 60) + rng.normal(0, 0.1, 60),
        "brent_price":      np.linspace(55, 95, 60) + rng.normal(0, 2, 60),
        "oil_production":   np.linspace(10000, 13000, 60),
        "crude_imports":    np.linspace(6000, 7000, 60),
        "refinery_util":    np.linspace(85, 92, 60),
        "gasoline_stocks":  np.linspace(220000, 240000, 60),
        "distillate_stocks":np.linspace(110000, 130000, 60),
        "gas_storage":      np.linspace(2000000, 2200000, 60),
        "gas_production":   np.linspace(3000000, 3800000, 60),
        "hdd":              200 + 600 * np.sin(2 * np.pi * np.arange(60) / 12),
        "cdd":              200 - 200 * np.sin(2 * np.pi * np.arange(60) / 12),
        "opec_spare":       np.linspace(3.0, 2.5, 60),
        "global_inv":       np.linspace(1280, 1260, 60),
        "dollar_index":     np.linspace(95, 120, 60),
        "industrial_production": np.linspace(98, 103, 60),
        "treasury_10y":     np.linspace(2.0, 4.5, 60),
    })
    return df


@pytest.fixture
def df_with_edge_nulls(clean_monthly_df):
    """Same as clean, but the last 2 rows have NaN in a couple of columns
    (the realistic reporting-lag situation in our gold_features table)."""
    df = clean_monthly_df.copy()
    df.loc[df.index[-2:], "henry_hub_price"] = np.nan
    df.loc[df.index[-1:], "crude_imports"]   = np.nan
    return df


# ── Edge-null handling ────────────────────────────────────────────────────────

class TestForwardFill:

    def test_fills_recent_edge_nulls(self, df_with_edge_nulls):
        """Forward-fill should eliminate all base-column nulls."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(df_with_edge_nulls)
        # After transform, original base columns should have no NaN at the edges
        for col in ["henry_hub_price", "crude_imports"]:
            assert raw[col].isna().sum() == 0, f"{col} still has nulls after ffill"

    def test_filled_value_matches_previous_month(self, df_with_edge_nulls):
        """Forward-fill should propagate the LAST non-null value, not a global mean."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(df_with_edge_nulls)
        # The last 2 rows of henry_hub_price should equal the 3rd-from-last (its last real value)
        last_real = df_with_edge_nulls["henry_hub_price"].dropna().iloc[-1]
        assert raw["henry_hub_price"].iloc[-1] == pytest.approx(last_real)
        assert raw["henry_hub_price"].iloc[-2] == pytest.approx(last_real)


# ── Lag features ──────────────────────────────────────────────────────────────

class TestLagFeatures:

    def test_lag_columns_created(self, clean_monthly_df):
        """All four lag depths (1, 3, 6, 12) should exist for each base column."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        for col in ["wti_price", "opec_spare", "dollar_index"]:
            for lag in [1, 3, 6, 12]:
                assert f"{col}_lag_{lag}" in raw.columns

    def test_lag_1_equals_previous_row_value(self, clean_monthly_df):
        """col_lag_1 at row N must equal col at row N-1 (no shift sign errors)."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        # Pick row 30 — well past any forward-fill weirdness
        actual_lag_1 = raw["wti_price_lag_1"].iloc[30]
        original_prev = clean_monthly_df["wti_price"].iloc[29]
        assert actual_lag_1 == pytest.approx(original_prev)

    def test_lag_12_uses_year_ago_value(self, clean_monthly_df):
        """col_lag_12 must equal the value from exactly 12 months earlier."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        actual_lag_12 = raw["wti_price_lag_12"].iloc[24]
        original_year_ago = clean_monthly_df["wti_price"].iloc[12]
        assert actual_lag_12 == pytest.approx(original_year_ago)

    def test_lag_12_is_nan_in_first_year(self, clean_monthly_df):
        """Rows 0-11 cannot have a 12-month lag — they predate the start."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        assert raw["wti_price_lag_12"].iloc[:12].isna().all()


# ── Rolling features ──────────────────────────────────────────────────────────

class TestRollingFeatures:

    def test_rolling_columns_created(self, clean_monthly_df):
        """Mean and std for each rolling window (3, 6, 12) for each base col."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        for w in [3, 6, 12]:
            assert f"wti_price_roll{w}_mean" in raw.columns
            assert f"wti_price_roll{w}_std"  in raw.columns

    def test_rolling_does_not_include_current_row(self, clean_monthly_df):
        """CRITICAL: rolling stats must use shift(1) — current row never in the window.
        Otherwise the model would 'see' its own target during training (lookahead leak)."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        # Roll-3-mean at row N should equal mean of rows N-3, N-2, N-1 (not N-2, N-1, N)
        idx = 20
        expected = clean_monthly_df["wti_price"].iloc[idx-3:idx].mean()
        actual   = raw["wti_price_roll3_mean"].iloc[idx]
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_rolling_std_non_negative(self, clean_monthly_df):
        """Standard deviation, by definition, is never negative."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        stds = raw.filter(regex="_std$").dropna()
        assert (stds >= 0).all().all()


# ── Momentum (% change) ───────────────────────────────────────────────────────

class TestMomentumFeatures:

    def test_mom_pct_column_created(self, clean_monthly_df):
        """Every base column should get a _mom_pct version."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        for col in ["wti_price", "opec_spare", "dollar_index"]:
            assert f"{col}_mom_pct" in raw.columns

    def test_mom_pct_matches_pct_change(self, clean_monthly_df):
        """col_mom_pct at row N = (col[N] - col[N-1]) / col[N-1] * 100"""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        idx = 10
        a = clean_monthly_df["wti_price"].iloc[idx]
        b = clean_monthly_df["wti_price"].iloc[idx-1]
        expected = (a - b) / b * 100
        assert raw["wti_price_mom_pct"].iloc[idx] == pytest.approx(expected)


# ── Seasonality ───────────────────────────────────────────────────────────────

class TestSeasonalityFeatures:

    def test_month_sin_cos_created(self, clean_monthly_df):
        """Cyclical month encoding columns should exist."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        assert "month_sin" in raw.columns
        assert "month_cos" in raw.columns

    def test_month_sin_in_unit_range(self, clean_monthly_df):
        """sin output is always in [-1, 1]."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        assert raw["month_sin"].between(-1, 1).all()
        assert raw["month_cos"].between(-1, 1).all()

    def test_dec_and_jan_are_adjacent_in_feature_space(self, clean_monthly_df):
        """The whole point of sin/cos encoding: December (12) and January (1) should
        be close in feature space, not far apart like raw integers."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        dec_row = raw[raw["month"] == 12].iloc[0]
        jan_row = raw[raw["month"] == 1].iloc[0]
        # Euclidean distance in (sin, cos) space between Dec and Jan
        dist_dec_jan = ((dec_row["month_sin"] - jan_row["month_sin"]) ** 2 +
                        (dec_row["month_cos"] - jan_row["month_cos"]) ** 2) ** 0.5
        # Distance between Jan and July (opposite months) should be much larger
        jul_row = raw[raw["month"] == 7].iloc[0]
        dist_jan_jul = ((jul_row["month_sin"] - jan_row["month_sin"]) ** 2 +
                        (jul_row["month_cos"] - jan_row["month_cos"]) ** 2) ** 0.5
        assert dist_dec_jan < dist_jan_jul


# ── Scaling ───────────────────────────────────────────────────────────────────

class TestScaling:

    def test_transform_returns_tuple(self, clean_monthly_df):
        """transform() should return (raw, scaled) as separate DataFrames."""
        fe = FeatureEngineer()
        result = fe.transform(clean_monthly_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scaled_has_same_shape_as_raw(self, clean_monthly_df):
        """Scaling does not add or remove columns — same shape as raw."""
        fe = FeatureEngineer()
        raw, scaled = fe.transform(clean_monthly_df)
        assert raw.shape == scaled.shape

    def test_scaled_columns_have_unit_std(self, clean_monthly_df):
        """Z-scoring: each scaled column has std ≈ 1 (after dropping NaN rows)."""
        fe = FeatureEngineer()
        _, scaled = fe.transform(clean_monthly_df)
        # Pick a column that has no NaN early on (no lag dependency)
        col = "month_sin"
        if scaled[col].std() > 0:
            assert scaled[col].std() == pytest.approx(1.0, rel=0.1)

    def test_scaler_parameters_stored(self, clean_monthly_df):
        """After transform, _scale_means and _scale_stds must be set so we
        can later apply the SAME scaling to new data (no refitting)."""
        fe = FeatureEngineer()
        fe.transform(clean_monthly_df)
        assert fe._scale_means is not None
        assert fe._scale_stds is not None
        assert len(fe._scale_means) > 0


# ── Output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:

    def test_row_count_preserved(self, clean_monthly_df):
        """Engineering shouldn't drop rows — same row count in and out."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        assert len(raw) == len(clean_monthly_df)

    def test_period_column_retained(self, clean_monthly_df):
        """Period column must survive transformation — needed for indexing."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        assert "period" in raw.columns

    def test_many_features_added(self, clean_monthly_df):
        """We expect ~190+ engineered features on top of the 17 base columns."""
        fe = FeatureEngineer()
        raw, _ = fe.transform(clean_monthly_df)
        # 17 base + at least 150 engineered (4 lags + 6 rolling + 1 mom_pct per base ≈ 11/base × 17 = ~190)
        assert raw.shape[1] > 150
