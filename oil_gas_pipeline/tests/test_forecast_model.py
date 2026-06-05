# oil_gas_pipeline | tests/test_forecast_model.py
# Unit tests for ForecastModel and Evaluator classes
# Run with: python3 -m pytest tests/test_forecast_model.py -v

import pytest
import numpy as np
import pandas as pd
from models.forecast_model import ForecastModel, ForecastResult
from models.evaluator import Evaluator, EvalMetrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def monthly_train_df():
    """
    Clean 5-year monthly time series for training.
    Uses a simple trend + seasonality pattern so models can fit it.
    """
    periods = pd.date_range("2018-01-01", periods=60, freq="MS")
    trend   = np.linspace(60, 90, 60)
    season  = 5 * np.sin(2 * np.pi * np.arange(60) / 12)
    noise   = np.random.normal(0, 1, 60)
    values  = (trend + season + noise).round(2)

    df = pd.DataFrame({
        "wti_price":   values,
        "brent_price": (values + 3).round(2),
    }, index=periods)
    df.index.freq = "MS"
    return df


@pytest.fixture
def actual_series():
    """12-month actual values for evaluation."""
    periods = pd.date_range("2023-01-01", periods=12, freq="MS")
    return pd.Series(
        np.random.uniform(70, 90, 12).round(2),
        index=periods,
        name="wti_price",
    )


@pytest.fixture
def predicted_series(actual_series):
    """Predicted values close to actual — small random noise added."""
    noise = np.random.normal(0, 2, 12)
    return pd.Series(
        (actual_series.values + noise).round(2),
        index=actual_series.index,
        name="wti_price",
    )


# ── ForecastResult tests ──────────────────────────────────────────────────────

class TestForecastResult:

    def test_to_db_df_has_required_columns(self):
        """to_db_df() should return DataFrame with all columns needed by db_manager."""
        forecast_df = pd.DataFrame({
            "period":   pd.date_range("2024-01-01", periods=3, freq="MS"),
            "forecast": [75.0, 76.5, 74.2],
            "lower":    [70.0, 71.0, 69.0],
            "upper":    [80.0, 82.0, 79.0],
        })
        result = ForecastResult(
            model_name="prophet",
            target="wti_price",
            forecast_df=forecast_df,
            train_periods=60,
        )
        db_df    = result.to_db_df()
        required = {
            "target", "model_name", "forecast_period",
            "forecast_value", "lower_bound", "upper_bound",
            "trained_on_periods",
        }
        assert required.issubset(set(db_df.columns))

    def test_to_db_df_row_count_matches(self):
        """to_db_df() should have same number of rows as forecast_df."""
        forecast_df = pd.DataFrame({
            "period":   pd.date_range("2024-01-01", periods=12, freq="MS"),
            "forecast": np.random.uniform(70, 90, 12),
            "lower":    np.random.uniform(65, 85, 12),
            "upper":    np.random.uniform(75, 95, 12),
        })
        result = ForecastResult(
            model_name="sarima",
            target="wti_price",
            forecast_df=forecast_df,
            train_periods=60,
        )
        assert len(result.to_db_df()) == 12


# ── ForecastModel — SARIMA tests ──────────────────────────────────────────────

class TestForecastModelSARIMA:

    def test_fit_sarima_sets_model(self, monthly_train_df):
        """After fit_sarima, _sarima_model should not be None."""
        fm = ForecastModel()
        fm.fit_sarima(monthly_train_df["wti_price"].asfreq("MS"))
        assert fm._sarima_model is not None

    def test_predict_sarima_returns_forecast_result(self, monthly_train_df):
        """predict_sarima should return a ForecastResult."""
        fm = ForecastModel()
        fm.fit_sarima(monthly_train_df["wti_price"].asfreq("MS"))
        result = fm.predict_sarima(horizon=12)
        assert isinstance(result, ForecastResult)

    def test_predict_sarima_correct_horizon(self, monthly_train_df):
        """Forecast DataFrame should have exactly horizon rows."""
        fm = ForecastModel()
        fm.fit_sarima(monthly_train_df["wti_price"].asfreq("MS"))
        result = fm.predict_sarima(horizon=6)
        assert len(result.forecast_df) == 6

    def test_predict_sarima_has_required_columns(self, monthly_train_df):
        """Forecast DataFrame must have period, forecast, lower, upper."""
        fm = ForecastModel()
        fm.fit_sarima(monthly_train_df["wti_price"].asfreq("MS"))
        result = fm.predict_sarima(horizon=12)
        required = {"period", "forecast", "lower", "upper"}
        assert required.issubset(set(result.forecast_df.columns))

    def test_predict_before_fit_raises(self):
        """predict_sarima before fit_sarima should raise ValueError."""
        fm = ForecastModel()
        with pytest.raises(ValueError):
            fm.predict_sarima(horizon=12)

    def test_confidence_interval_ordering(self, monthly_train_df):
        """Lower bound must always be <= forecast <= upper bound."""
        fm = ForecastModel()
        fm.fit_sarima(monthly_train_df["wti_price"].asfreq("MS"))
        result = fm.predict_sarima(horizon=12)
        df     = result.forecast_df
        assert (df["lower"] <= df["forecast"]).all()
        assert (df["forecast"] <= df["upper"]).all()


# ── ForecastModel — Prophet tests ─────────────────────────────────────────────

class TestForecastModelProphet:

    def test_fit_prophet_sets_model(self, monthly_train_df):
        """After fit_prophet, _prophet_model should not be None."""
        fm = ForecastModel()
        fm.fit_prophet(monthly_train_df, "wti_price")
        assert fm._prophet_model is not None

    def test_predict_prophet_returns_forecast_result(self, monthly_train_df):
        """predict_prophet should return a ForecastResult."""
        fm = ForecastModel()
        fm.fit_prophet(monthly_train_df, "wti_price")
        result = fm.predict_prophet(horizon=12)
        assert isinstance(result, ForecastResult)

    def test_predict_prophet_correct_horizon(self, monthly_train_df):
        """Forecast DataFrame should have exactly horizon rows."""
        fm = ForecastModel()
        fm.fit_prophet(monthly_train_df, "wti_price")
        result = fm.predict_prophet(horizon=6)
        assert len(result.forecast_df) == 6

    def test_predict_prophet_has_required_columns(self, monthly_train_df):
        """Forecast DataFrame must have period, forecast, lower, upper."""
        fm = ForecastModel()
        fm.fit_prophet(monthly_train_df, "wti_price")
        result = fm.predict_prophet(horizon=12)
        required = {"period", "forecast", "lower", "upper"}
        assert required.issubset(set(result.forecast_df.columns))

    def test_predict_before_fit_raises(self):
        """predict_prophet before fit_prophet should raise ValueError."""
        fm = ForecastModel()
        with pytest.raises(ValueError):
            fm.predict_prophet(horizon=12)


# ── ForecastModel — unified interface tests ───────────────────────────────────

class TestFitPredict:

    def test_fit_predict_sarima(self, monthly_train_df):
        """fit_predict with model='sarima' should return ForecastResult."""
        fm     = ForecastModel()
        result = fm.fit_predict(monthly_train_df, "wti_price", model="sarima", horizon=6)
        assert isinstance(result, ForecastResult)
        assert result.model_name == "sarima"
        assert result.target     == "wti_price"

    def test_fit_predict_prophet(self, monthly_train_df):
        """fit_predict with model='prophet' should return ForecastResult."""
        fm     = ForecastModel()
        result = fm.fit_predict(monthly_train_df, "wti_price", model="prophet", horizon=6)
        assert isinstance(result, ForecastResult)
        assert result.model_name == "prophet"

    def test_fit_predict_invalid_model_raises(self, monthly_train_df):
        """fit_predict with unknown model name should raise ValueError."""
        fm = ForecastModel()
        with pytest.raises(ValueError):
            fm.fit_predict(monthly_train_df, "wti_price", model="xgboost")

    def test_run_both_returns_two_results(self, monthly_train_df):
        """run_both should return dict with sarima and prophet keys."""
        fm      = ForecastModel()
        results = fm.run_both(monthly_train_df, "wti_price", horizon=6)
        assert "sarima"  in results
        assert "prophet" in results
        assert isinstance(results["sarima"],  ForecastResult)
        assert isinstance(results["prophet"], ForecastResult)


# ── Evaluator tests ───────────────────────────────────────────────────────────

class TestEvaluator:

    def test_evaluate_returns_eval_metrics(self, actual_series, predicted_series):
        """evaluate() should return an EvalMetrics dataclass."""
        ev      = Evaluator()
        metrics = ev.evaluate(actual_series, predicted_series, "prophet", "wti_price")
        assert isinstance(metrics, EvalMetrics)

    def test_rmse_is_non_negative(self, actual_series, predicted_series):
        """RMSE must always be >= 0."""
        ev      = Evaluator()
        metrics = ev.evaluate(actual_series, predicted_series, "prophet", "wti_price")
        assert metrics.rmse >= 0

    def test_mape_is_percentage(self, actual_series, predicted_series):
        """MAPE should be expressed as a percentage (0-100 range typically)."""
        ev      = Evaluator()
        metrics = ev.evaluate(actual_series, predicted_series, "prophet", "wti_price")
        assert metrics.mape >= 0

    def test_perfect_forecast_gives_zero_rmse(self, actual_series):
        """Perfect predictions should give RMSE=0 and MAPE=0."""
        ev      = Evaluator()
        metrics = ev.evaluate(actual_series, actual_series, "perfect", "wti_price")
        assert metrics.rmse == 0.0
        assert metrics.mape == 0.0

    def test_perfect_forecast_gives_r2_of_one(self, actual_series):
        """Perfect predictions should give R²=1.0."""
        ev      = Evaluator()
        metrics = ev.evaluate(actual_series, actual_series, "perfect", "wti_price")
        assert metrics.r2 == 1.0

    def test_compare_models_returns_dataframe(self, actual_series, predicted_series):
        """compare_models() should return a DataFrame with two rows."""
        ev     = Evaluator()
        result = ev.compare_models(
            actual=actual_series,
            sarima_pred=predicted_series,
            prophet_pred=predicted_series,
            target="wti_price",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result["model"]) == {"sarima", "prophet"}

    def test_to_dataframe_accumulates_results(self, actual_series, predicted_series):
        """to_dataframe() should include all evaluations run so far."""
        ev = Evaluator()
        ev.evaluate(actual_series, predicted_series, "sarima",  "wti_price")
        ev.evaluate(actual_series, predicted_series, "prophet", "wti_price")
        df = ev.to_dataframe()
        assert len(df) == 2
        assert set(df["model"]) == {"sarima", "prophet"}