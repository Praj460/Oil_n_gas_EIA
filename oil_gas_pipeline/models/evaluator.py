# oil_gas_pipeline | models/evaluator.py
# Evaluator class — calculates RMSE, MAPE, and generates model comparison reports
# Works with ForecastResult objects from ForecastModel
# Usage: from models.evaluator import Evaluator

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Holds evaluation metrics for one model on one target."""
    model_name: str
    target:     str
    rmse:       float    # Root Mean Squared Error — penalizes large errors
    mae:        float    # Mean Absolute Error — average absolute error
    mape:       float    # Mean Absolute Percentage Error — scale-independent
    r2:         float    # R-squared — how much variance is explained
    n_periods:  int      # how many test periods were evaluated

    def summary(self) -> str:
        return (
            f"{self.model_name.upper()} | {self.target} | "
            f"RMSE={self.rmse:.4f} | MAE={self.mae:.4f} | "
            f"MAPE={self.mape:.2f}% | R²={self.r2:.4f} | n={self.n_periods}"
        )


class Evaluator:
    """
    Evaluates forecast accuracy by comparing predicted vs actual values.

    Metrics used:
    - RMSE (Root Mean Squared Error):
        Square root of the average squared error.
        Penalizes large errors heavily. Good for energy prices where
        big misses are costly. Lower is better.

    - MAE (Mean Absolute Error):
        Average of absolute errors. Easier to interpret than RMSE
        (same unit as the target — dollars per barrel). Lower is better.

    - MAPE (Mean Absolute Percentage Error):
        Average % error. Scale-independent — lets you compare models
        across different targets (WTI vs Henry Hub have different scales).
        Lower is better.

    - R² (R-squared):
        Proportion of variance explained by the model.
        1.0 = perfect, 0.0 = no better than the mean, <0 = worse than mean.
        Higher is better.

    Usage:
        evaluator = Evaluator()
        metrics   = evaluator.evaluate(actual=test["wti_price"],
                                       predicted=result.fitted_values,
                                       model_name="prophet",
                                       target="wti_price")
        print(metrics.summary())
    """

    def __init__(self):
        self.results: list[EvalMetrics] = []

    # ── Core metric calculations ──────────────────────────────────────────────

    def _rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))

    def _mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(actual - predicted)))

    def _mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        Excludes periods where actual == 0 to avoid division by zero.
        """
        mask = actual != 0
        if mask.sum() == 0:
            logger.warning("All actual values are 0 — MAPE undefined, returning 0")
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def _r2(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    # ── Evaluate single model ─────────────────────────────────────────────────

    def evaluate(
        self,
        actual:     pd.Series,
        predicted:  pd.Series,
        model_name: str,
        target:     str,
    ) -> EvalMetrics:
        """
        Evaluates one model's predictions against actual values.

        Aligns actual and predicted on their index before computing metrics
        so missing periods don't cause silent errors.

        Args:
            actual:     pd.Series of actual values (test set)
            predicted:  pd.Series of predicted values from the model
            model_name: "sarima" or "prophet"
            target:     column name e.g. "wti_price"

        Returns:
            EvalMetrics dataclass with all metrics
        """
        # Align on shared index
        aligned = pd.concat(
            [actual.rename("actual"), predicted.rename("predicted")],
            axis=1,
        ).dropna()

        if aligned.empty:
            logger.error(f"No overlapping periods between actual and predicted for {target}")
            return EvalMetrics(
                model_name=model_name, target=target,
                rmse=0, mae=0, mape=0, r2=0, n_periods=0
            )

        a = aligned["actual"].values
        p = aligned["predicted"].values

        metrics = EvalMetrics(
            model_name=model_name,
            target=target,
            rmse=round(self._rmse(a, p), 4),
            mae=round(self._mae(a, p), 4),
            mape=round(self._mape(a, p), 4),
            r2=round(self._r2(a, p), 4),
            n_periods=len(aligned),
        )

        self.results.append(metrics)
        logger.info(f"Evaluation | {metrics.summary()}")
        return metrics

    # ── Backtesting ───────────────────────────────────────────────────────────

    def backtest(
        self,
        df:           pd.DataFrame,
        target_col:   str,
        model_name:   str,
        n_splits:     int = 3,
        horizon:      int = 12,
    ) -> pd.DataFrame:
        """
        Walk-forward backtesting — trains on expanding windows and
        evaluates on each hold-out period.

        This is more realistic than a single train/test split because
        it shows how the model would have performed if deployed at
        different points in time.

        Example with n_splits=3, horizon=12:
            Split 1: train on 2000-2018, test on 2019
            Split 2: train on 2000-2019, test on 2020
            Split 3: train on 2000-2020, test on 2021

        Args:
            df:         preprocessed DataFrame with DatetimeIndex
            target_col: column to forecast
            model_name: "sarima" or "prophet"
            n_splits:   number of walk-forward splits
            horizon:    months per evaluation window

        Returns:
            DataFrame with RMSE, MAE, MAPE, R² for each split
        """
        from models.forecast_model import ForecastModel

        logger.info(f"Backtesting | model={model_name} | splits={n_splits} | horizon={horizon}")

        total_periods = len(df)
        min_train     = total_periods - (n_splits * horizon)

        if min_train < 24:
            raise ValueError(
                f"Not enough data for {n_splits} splits with horizon={horizon}. "
                f"Need at least {n_splits * horizon + 24} periods, have {total_periods}."
            )

        backtest_results = []

        for i in range(n_splits):
            train_end   = min_train + (i * horizon)
            test_start  = train_end
            test_end    = test_start + horizon

            train_df = df.iloc[:train_end]
            test_df  = df.iloc[test_start:test_end]

            logger.info(
                f"Split {i+1}/{n_splits} | "
                f"train: {train_df.index[0].date()} → {train_df.index[-1].date()} | "
                f"test:  {test_df.index[0].date()} → {test_df.index[-1].date()}"
            )

            try:
                fm     = ForecastModel()
                result = fm.fit_predict(train_df, target_col, model=model_name, horizon=horizon)

                # Align forecast periods with test actual values
                forecast_series = pd.Series(
                    result.forecast_df["forecast"].values,
                    index=pd.to_datetime(result.forecast_df["period"]),
                )
                actual_series = test_df[target_col]

                metrics = self.evaluate(
                    actual=actual_series,
                    predicted=forecast_series,
                    model_name=model_name,
                    target=target_col,
                )

                backtest_results.append({
                    "split":      i + 1,
                    "train_end":  train_df.index[-1].date(),
                    "test_start": test_df.index[0].date(),
                    "test_end":   test_df.index[-1].date(),
                    "rmse":       metrics.rmse,
                    "mae":        metrics.mae,
                    "mape":       metrics.mape,
                    "r2":         metrics.r2,
                    "n_periods":  metrics.n_periods,
                })

            except Exception as e:
                logger.error(f"Backtest split {i+1} failed | {e}")
                backtest_results.append({
                    "split": i + 1, "error": str(e),
                    "rmse": None, "mae": None, "mape": None, "r2": None,
                })

        results_df = pd.DataFrame(backtest_results)
        logger.info(
            f"Backtest complete | "
            f"avg RMSE={results_df['rmse'].mean():.4f} | "
            f"avg MAPE={results_df['mape'].mean():.2f}%"
        )
        return results_df

    # ── Model comparison ──────────────────────────────────────────────────────

    def compare_models(
        self,
        actual:         pd.Series,
        sarima_pred:    pd.Series,
        prophet_pred:   pd.Series,
        target:         str,
    ) -> pd.DataFrame:
        """
        Compares SARIMA vs Prophet side by side.
        Used by the Streamlit model comparison page.

        Args:
            actual:       actual test values
            sarima_pred:  SARIMA fitted/forecast values
            prophet_pred: Prophet fitted/forecast values
            target:       column name e.g. "wti_price"

        Returns:
            DataFrame with one row per model and all metrics as columns
        """
        sarima_metrics  = self.evaluate(actual, sarima_pred,  "sarima",  target)
        prophet_metrics = self.evaluate(actual, prophet_pred, "prophet", target)

        comparison = pd.DataFrame([
            {
                "model":  sarima_metrics.model_name,
                "target": sarima_metrics.target,
                "rmse":   sarima_metrics.rmse,
                "mae":    sarima_metrics.mae,
                "mape":   sarima_metrics.mape,
                "r2":     sarima_metrics.r2,
            },
            {
                "model":  prophet_metrics.model_name,
                "target": prophet_metrics.target,
                "rmse":   prophet_metrics.rmse,
                "mae":    prophet_metrics.mae,
                "mape":   prophet_metrics.mape,
                "r2":     prophet_metrics.r2,
            },
        ])

        # Add a winner column per metric (lower RMSE/MAE/MAPE, higher R² is better)
        comparison["best_rmse"] = comparison["rmse"] == comparison["rmse"].min()
        comparison["best_mape"] = comparison["mape"] == comparison["mape"].min()
        comparison["best_r2"]   = comparison["r2"]   == comparison["r2"].max()

        logger.info(f"Model comparison complete | target={target}")
        return comparison

    # ── Full report ───────────────────────────────────────────────────────────

    def report(self) -> str:
        """Prints a formatted report of all evaluations run so far."""
        if not self.results:
            return "No evaluations run yet."

        lines = ["\n" + "=" * 65, "  EVALUATION REPORT", "=" * 65]
        for m in self.results:
            lines.append(f"  {m.summary()}")
        lines.append("=" * 65 + "\n")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all evaluation results as a DataFrame."""
        return pd.DataFrame([
            {
                "model":     m.model_name,
                "target":    m.target,
                "rmse":      m.rmse,
                "mae":       m.mae,
                "mape":      m.mape,
                "r2":        m.r2,
                "n_periods": m.n_periods,
            }
            for m in self.results
        ])