# oil_gas_pipeline | tests/test_preprocessor.py
# Unit tests for the Preprocessor class
# Run with: python3 -m pytest tests/test_preprocessor.py -v

import pytest
import numpy as np
import pandas as pd
from models.preprocessor import Preprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_monthly_df():
    """Well-formed monthly time series — no gaps, no nulls."""
    periods = pd.date_range("2015-01-01", periods=60, freq="MS")
    return pd.DataFrame({
        "period":       periods,
        "wti_price":    np.random.uniform(40, 100, 60).round(2),
        "brent_price":  np.random.uniform(45, 105, 60).round(2),
    })


@pytest.fixture
def df_with_gaps():
    """Monthly series with two missing months."""
    periods = pd.date_range("2015-01-01", periods=60, freq="MS")
    df = pd.DataFrame({
        "period":    periods,
        "wti_price": np.random.uniform(40, 100, 60).round(2),
    })
    # Introduce gaps by removing two rows
    df = df.drop(index=[10, 25]).reset_index(drop=True)
    return df


@pytest.fixture
def df_with_nulls():
    """Monthly series with some null values in target column."""
    periods = pd.date_range("2015-01-01", periods=60, freq="MS")
    values  = np.random.uniform(40, 100, 60).round(2)
    values[5]  = np.nan
    values[20] = np.nan
    return pd.DataFrame({"period": periods, "wti_price": values})


@pytest.fixture
def df_with_outliers():
    """Monthly series with two extreme outlier values."""
    periods = pd.date_range("2015-01-01", periods=60, freq="MS")
    values  = np.random.uniform(60, 80, 60).round(2)
    values[15] = 5000.0    # extreme high outlier
    values[30] = -200.0    # extreme low outlier
    return pd.DataFrame({"period": periods, "wti_price": values})


# ── DatetimeIndex tests ───────────────────────────────────────────────────────

class TestSetDatetimeIndex:

    def test_sets_period_as_index(self, clean_monthly_df):
        """After _set_datetime_index, index should be DatetimeIndex."""
        prep = Preprocessor()
        result = prep._set_datetime_index(clean_monthly_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_period_column_removed(self, clean_monthly_df):
        """After setting index, 'period' should no longer be a column."""
        prep = Preprocessor()
        result = prep._set_datetime_index(clean_monthly_df)
        assert "period" not in result.columns

    def test_raises_if_no_period_column(self):
        """Should raise ValueError if 'period' column is missing."""
        prep = Preprocessor()
        df   = pd.DataFrame({"wti_price": [50, 60, 70]})
        with pytest.raises(ValueError):
            prep._set_datetime_index(df)

    def test_index_sorted_ascending(self, clean_monthly_df):
        """Index should be sorted in ascending order."""
        shuffled = clean_monthly_df.sample(frac=1).reset_index(drop=True)
        prep     = Preprocessor()
        result   = prep._set_datetime_index(shuffled)
        assert result.index.is_monotonic_increasing


# ── Resample tests ────────────────────────────────────────────────────────────

class TestResampleMonthly:

    def test_fills_missing_months(self, df_with_gaps):
        """Resampling should fill in the two removed months."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(df_with_gaps)
        result = prep._resample_monthly(df, "wti_price", [])
        # Should have 60 months (Jan 2015 to Dec 2019)
        assert len(result) == 60

    def test_output_is_monthly_frequency(self, clean_monthly_df):
        """Resampled DataFrame should have monthly frequency."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(clean_monthly_df)
        result = prep._resample_monthly(df, "wti_price", [])
        assert result.index.freqstr in ("MS", "M", "<MonthBegin>")


# ── Missing value tests ───────────────────────────────────────────────────────

class TestFillMissing:

    def test_no_nulls_after_fill(self, df_with_nulls):
        """After _fill_missing, there should be no null values."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(df_with_nulls)
        df     = prep._resample_monthly(df, "wti_price", [])
        result = prep._fill_missing(df)
        assert result["wti_price"].isna().sum() == 0

    def test_values_within_original_range(self, df_with_nulls):
        """Filled values should be within the original min-max range."""
        original_min = df_with_nulls["wti_price"].min()
        original_max = df_with_nulls["wti_price"].max()
        prep         = Preprocessor()
        df           = prep._set_datetime_index(df_with_nulls)
        df           = prep._resample_monthly(df, "wti_price", [])
        result       = prep._fill_missing(df)
        # Allow small tolerance for interpolation edge cases
        assert result["wti_price"].min() >= original_min - 1
        assert result["wti_price"].max() <= original_max + 1


# ── Outlier tests ─────────────────────────────────────────────────────────────

class TestRemoveOutliers:

    def test_outliers_replaced(self, df_with_outliers):
        """Values of 5000 and -200 should be replaced after outlier removal."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(df_with_outliers)
        df     = prep._resample_monthly(df, "wti_price", [])
        result = prep._remove_outliers(df, "wti_price", iqr_multiplier=3.0)
        assert result["wti_price"].max() < 5000.0
        assert result["wti_price"].min() >= 0.0

    def test_no_nulls_after_outlier_removal(self, df_with_outliers):
        """After outlier removal + interpolation, no nulls should remain."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(df_with_outliers)
        df     = prep._resample_monthly(df, "wti_price", [])
        result = prep._remove_outliers(df, "wti_price")
        assert result["wti_price"].isna().sum() == 0


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_lag_features_created(self, clean_monthly_df):
        """Lag features for 1, 3, 6, 12 months should be added."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(clean_monthly_df)
        result = prep._engineer_features(df, "wti_price")
        for lag in [1, 3, 6, 12]:
            assert f"wti_price_lag_{lag}" in result.columns

    def test_rolling_features_created(self, clean_monthly_df):
        """Rolling mean and std features should be added."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(clean_monthly_df)
        result = prep._engineer_features(df, "wti_price")
        assert "wti_price_rolling_3_mean"  in result.columns
        assert "wti_price_rolling_6_mean"  in result.columns
        assert "wti_price_rolling_12_mean" in result.columns
        assert "wti_price_rolling_3_std"   in result.columns

    def test_calendar_features_created(self, clean_monthly_df):
        """Month, quarter, and year columns should be added."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(clean_monthly_df)
        result = prep._engineer_features(df, "wti_price")
        assert "month"   in result.columns
        assert "quarter" in result.columns
        assert "year"    in result.columns

    def test_month_values_valid(self, clean_monthly_df):
        """Month values should be between 1 and 12."""
        prep   = Preprocessor()
        df     = prep._set_datetime_index(clean_monthly_df)
        result = prep._engineer_features(df, "wti_price")
        assert result["month"].between(1, 12).all()


# ── Train/test split tests ────────────────────────────────────────────────────

class TestTrainTestSplit:

    def test_split_sizes_correct(self, clean_monthly_df):
        """Train should be 80%, test 20% of total rows."""
        prep       = Preprocessor()
        df         = prep._set_datetime_index(clean_monthly_df)
        train, test = prep.train_test_split(df, "wti_price", test_size=0.2)
        assert len(train) == 48   # 80% of 60
        assert len(test)  == 12   # 20% of 60

    def test_no_overlap_between_train_and_test(self, clean_monthly_df):
        """Train and test sets must not share any index values."""
        prep        = Preprocessor()
        df          = prep._set_datetime_index(clean_monthly_df)
        train, test = prep.train_test_split(df, "wti_price")
        overlap     = train.index.intersection(test.index)
        assert len(overlap) == 0

    def test_train_comes_before_test(self, clean_monthly_df):
        """All train periods must be before all test periods."""
        prep        = Preprocessor()
        df          = prep._set_datetime_index(clean_monthly_df)
        train, test = prep.train_test_split(df, "wti_price")
        assert train.index.max() < test.index.min()


# ── Full fit_transform test ───────────────────────────────────────────────────

class TestFitTransform:

    def test_fit_transform_returns_dataframe(self, clean_monthly_df):
        """fit_transform should return a non-empty DataFrame."""
        prep   = Preprocessor()
        result = prep.fit_transform(clean_monthly_df, target_col="wti_price")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_fit_transform_index_is_datetime(self, clean_monthly_df):
        """Output DataFrame should have DatetimeIndex."""
        prep   = Preprocessor()
        result = prep.fit_transform(clean_monthly_df, target_col="wti_price")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_fit_transform_no_nulls_in_target(self, df_with_nulls):
        """Target column should have no nulls after fit_transform."""
        prep   = Preprocessor()
        result = prep.fit_transform(df_with_nulls, target_col="wti_price")
        assert result["wti_price"].isna().sum() == 0