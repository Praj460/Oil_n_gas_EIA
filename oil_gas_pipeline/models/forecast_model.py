# oil_gas_pipeline | models/forecast_model.py
# ForecastModel class — wraps SARIMA and Prophet forecasting models
# Switchable via config — returns unified output format from both models
# Usage: from models.forecast_model import ForecastModel

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

from config.config import config

warnings.filterwarnings("ignore")   # suppress statsmodels convergence warnings
logger = logging.getLogger(__name__)


class ForecastResult:
    """
    Unified output format for both SARIMA and Prophet forecasts.
    Regardless of which model ran, downstream code always gets
    the same structure.
    """

    def __init__(
        self,
        model_name:      str,
        target:          str,
        forecast_df:     pd.DataFrame,
        fitted_values:   Optional[pd.Series] = None,
        train_periods:   int = 0,
    ):
        """
        Args:
            model_name:    "sarima" or "prophet"
            target:        column that was forecasted e.g. "wti_price"
            forecast_df:   DataFrame with columns [period, forecast, lower, upper]
            fitted_values: in-sample fitted values on training data
            train_periods: number of months used for training
        """
        self.model_name    = model_name
        self.target        = target
        self.forecast_df   = forecast_df
        self.fitted_values = fitted_values
        self.train_periods = train_periods
        self.created_at    = datetime.now()

    def to_db_df(self) -> pd.DataFrame:
        """
        Converts forecast result into a DataFrame ready to be written
        to gold_forecast_results by DatabaseManager.write_forecast_results()
        """
        df = self.forecast_df.copy()
        df["model_name"]       = self.model_name
        df["target"]           = self.target
        df["trained_on_periods"] = self.train_periods
        df = df.rename(columns={
            "period":   "forecast_period",
            "forecast": "forecast_value",
            "lower":    "lower_bound",
            "upper":    "upper_bound",
        })
        return df[["target", "model_name", "forecast_period",
                   "forecast_value", "lower_bound", "upper_bound",
                   "trained_on_periods"]]


class ForecastModel:
    """
    Wraps SARIMA (statsmodels) and Prophet (Meta) under a unified interface.

    Both models:
    - Accept a clean time series from Preprocessor
    - Return a ForecastResult with forecast + confidence intervals
    - Support saving and loading trained models to disk

    The primary model is set in config (default: prophet).
    Both models can be run and compared via run_both().

    Usage:
        model  = ForecastModel()
        result = model.fit_predict(train_df, target_col="wti_price")
        print(result.forecast_df)
    """

    def __init__(self):
        self.model_cfg      = config.model
        self.paths          = config.paths
        self._sarima_model  = None
        self._prophet_model = None
        logger.info(
            f"ForecastModel initialized | "
            f"primary={self.model_cfg.primary_model} | "
            f"horizon={self.model_cfg.forecast_horizon} months"
        )

    # ── SARIMA ────────────────────────────────────────────────────────────────

    def fit_sarima(
        self,
        train: pd.Series,
        order:          Optional[tuple] = None,
        seasonal_order: Optional[tuple] = None,
    ) -> None:
        """
        Fits a SARIMA model on the training series.

        SARIMA(p,d,q)(P,D,Q,s):
        - p,d,q: non-seasonal AR order, differencing, MA order
        - P,D,Q: seasonal AR, differencing, MA order
        - s: seasonal period (12 for monthly data)

        Args:
            train:          pd.Series with DatetimeIndex (monthly frequency)
            order:          (p,d,q) — defaults to config value (1,1,1)
            seasonal_order: (P,D,Q,s) — defaults to config value (1,1,1,12)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        order          = order          or self.model_cfg.sarima_order
        seasonal_order = seasonal_order or self.model_cfg.sarima_seasonal_order

        logger.info(f"Fitting SARIMA{order}x{seasonal_order} on {len(train)} periods...")

        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._sarima_model = model.fit(disp=False)
        logger.info("SARIMA fit complete")

    def predict_sarima(
        self,
        horizon: Optional[int] = None,
        alpha:   float = 0.05,          # 95% confidence interval
    ) -> ForecastResult:
        """
        Generates SARIMA forecast for the next `horizon` months.

        Args:
            horizon: number of months to forecast (default from config)
            alpha:   significance level for confidence intervals (0.05 = 95%)

        Returns:
            ForecastResult with forecast DataFrame and fitted values
        """
        if self._sarima_model is None:
            raise ValueError("Call fit_sarima() before predict_sarima()")

        horizon = horizon or self.model_cfg.forecast_horizon

        forecast_obj = self._sarima_model.get_forecast(steps=horizon)
        forecast_mean = forecast_obj.predicted_mean
        conf_int      = forecast_obj.conf_int(alpha=alpha)

        forecast_df = pd.DataFrame({
            "period":   forecast_mean.index,
            "forecast": forecast_mean.values.round(4),
            "lower":    conf_int.iloc[:, 0].values.round(4),
            "upper":    conf_int.iloc[:, 1].values.round(4),
        })

        fitted_values = pd.Series(
            self._sarima_model.fittedvalues,
            index=self._sarima_model.fittedvalues.index,
        )

        logger.info(f"SARIMA forecast | horizon={horizon} | periods={forecast_df['period'].tolist()}")
        return ForecastResult(
            model_name="sarima",
            target="",             # caller sets this
            forecast_df=forecast_df,
            fitted_values=fitted_values,
            train_periods=len(self._sarima_model.fittedvalues),
        )

    # ── Prophet ───────────────────────────────────────────────────────────────

    def fit_prophet(self, train: pd.DataFrame, target_col: str) -> None:
        """
        Fits a Prophet model on the training DataFrame.

        Prophet expects a DataFrame with columns [ds, y]:
        - ds: datetime column
        - y:  target values

        Args:
            train:      DataFrame with DatetimeIndex and target_col
            target_col: name of the column to forecast
        """
        from prophet import Prophet

        prophet_df = pd.DataFrame({
            "ds": train.index,
            "y":  train[target_col].values,
        })

        logger.info(f"Fitting Prophet on {len(prophet_df)} periods...")

        self._prophet_model = Prophet(
            changepoint_prior_scale=self.model_cfg.prophet_changepoint_prior_scale,
            seasonality_mode=self.model_cfg.prophet_seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,   # monthly data — no weekly pattern
            daily_seasonality=False,
        )
        self._prophet_model.fit(prophet_df)
        logger.info("Prophet fit complete")

    def predict_prophet(
        self,
        horizon: Optional[int] = None,
    ) -> ForecastResult:
        """
        Generates Prophet forecast for the next `horizon` months.

        Args:
            horizon: number of months to forecast (default from config)

        Returns:
            ForecastResult with forecast DataFrame and fitted values
        """
        if self._prophet_model is None:
            raise ValueError("Call fit_prophet() before predict_prophet()")

        horizon = horizon or self.model_cfg.forecast_horizon

        future = self._prophet_model.make_future_dataframe(
            periods=horizon,
            freq="MS",      # month start frequency
        )
        forecast = self._prophet_model.predict(future)

        # Prophet returns full history + future — keep only the future rows
        future_forecast = forecast.tail(horizon)

        forecast_df = pd.DataFrame({
            "period":   pd.to_datetime(future_forecast["ds"]).values,
            "forecast": future_forecast["yhat"].round(4).values,
            "lower":    future_forecast["yhat_lower"].round(4).values,
            "upper":    future_forecast["yhat_upper"].round(4).values,
        })

        # In-sample fitted values (all historical rows in the forecast output)
        fitted_df = forecast.iloc[:-horizon]
        fitted_values = pd.Series(
            fitted_df["yhat"].values,
            index=pd.to_datetime(fitted_df["ds"]),
        )

        logger.info(f"Prophet forecast | horizon={horizon}")
        return ForecastResult(
            model_name="prophet",
            target="",
            forecast_df=forecast_df,
            fitted_values=fitted_values,
            train_periods=len(fitted_values),
        )

    # ── Unified interface ─────────────────────────────────────────────────────

    def fit_predict(
        self,
        train:      pd.DataFrame,
        target_col: str,
        model:      Optional[str] = None,
        horizon:    Optional[int] = None,
    ) -> ForecastResult:
        """
        Fits and forecasts using the specified model.
        This is the main method called by the Airflow forecast DAG.

        Args:
            train:      preprocessed training DataFrame with DatetimeIndex
            target_col: column to forecast e.g. "wti_price"
            model:      "sarima" or "prophet" (defaults to config primary_model)
            horizon:    months to forecast (defaults to config forecast_horizon)

        Returns:
            ForecastResult with forecast + confidence intervals
        """
        model   = model   or self.model_cfg.primary_model
        horizon = horizon or self.model_cfg.forecast_horizon

        logger.info(f"fit_predict | model={model} | target={target_col} | horizon={horizon}")

        if model == "sarima":
            series = train[target_col].asfreq("MS")
            self.fit_sarima(series)
            result = self.predict_sarima(horizon=horizon)

        elif model == "prophet":
            self.fit_prophet(train, target_col)
            result = self.predict_prophet(horizon=horizon)

        else:
            raise ValueError(f"Unknown model '{model}' — use 'sarima' or 'prophet'")

        result.target = target_col
        return result

    def run_both(
        self,
        train:      pd.DataFrame,
        target_col: str,
        horizon:    Optional[int] = None,
    ) -> dict[str, ForecastResult]:
        """
        Runs both SARIMA and Prophet and returns results for comparison.
        Used by the Streamlit model comparison page.

        Args:
            train:      preprocessed training DataFrame
            target_col: column to forecast
            horizon:    months ahead to forecast

        Returns:
            {"sarima": ForecastResult, "prophet": ForecastResult}
        """
        logger.info(f"Running both models | target={target_col}")
        return {
            "sarima":  self.fit_predict(train, target_col, model="sarima",  horizon=horizon),
            "prophet": self.fit_predict(train, target_col, model="prophet", horizon=horizon),
        }

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, model_name: str, target_col: str) -> Path:
        """
        Saves the trained model to disk as a pickle file.

        Args:
            model_name: "sarima" or "prophet"
            target_col: e.g. "wti_price" (used in filename)

        Returns:
            Path to the saved file
        """
        filename = f"{model_name}_{target_col}_{pd.Timestamp.now().strftime('%Y%m%d')}.pkl"
        path     = self.paths.models_saved / filename

        model_obj = self._sarima_model if model_name == "sarima" else self._prophet_model
        if model_obj is None:
            raise ValueError(f"No fitted {model_name} model to save")

        with open(path, "wb") as f:
            pickle.dump(model_obj, f)

        logger.info(f"Model saved | path={path}")
        return path

    def load(self, path: Path, model_name: str) -> None:
        """
        Loads a previously saved model from disk.

        Args:
            path:       path to the .pkl file
            model_name: "sarima" or "prophet"
        """
        with open(path, "rb") as f:
            model_obj = pickle.load(f)

        if model_name == "sarima":
            self._sarima_model = model_obj
        else:
            self._prophet_model = model_obj

        logger.info(f"Model loaded | path={path} | model={model_name}")