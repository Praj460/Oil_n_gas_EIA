# oil_gas_pipeline | models/future_forecaster.py
# FutureForecaster class - trains on ALL available data and
# forecasts the next 12 months (true future forecast)
# Usage: from models.future_forecaster import FutureForecaster

import logging
import pandas as pd
import psycopg2

from models.preprocessor import Preprocessor
from models.forecast_model import ForecastModel
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class FutureForecaster:
    """
    Produces true future forecasts.

    Difference from the validation run (scripts/4_run_forecast.py):
    - Validation: train on first 80%, forecast 2024, compare to actual 2024
    - Future:     train on ALL data, forecast the next 12 unseen months

    Usage:
        ff = FutureForecaster()
        ff.run()
    """

    DB = dict(host="localhost", port=5432, dbname="oil_gas_db",
              user="prajwalanand", password="India@1947")

    TARGETS = ["wti_price", "henry_hub_price"]
    MODELS  = ["sarima", "prophet"]

    def __init__(self, horizon: int = 12):
        self.horizon = horizon
        self.db      = DatabaseManager()

    # -- Load ----------------------------------------------------------

    def load_gold(self) -> pd.DataFrame:
        """Reads the full gold table - all months, no split."""
        conn = psycopg2.connect(**self.DB)
        df = pd.read_sql(
            "SELECT * FROM gold_energy_prices ORDER BY period ASC", conn
        )
        conn.close()
        logger.info(f"Loaded {len(df)} gold rows | latest={df['period'].max()}")
        return df

    # -- Forecast one combination --------------------------------------

    def forecast_one(self, df: pd.DataFrame, target: str, model_name: str) -> int:
        """
        Trains one model on ALL data for one target,
        forecasts the next `horizon` months, saves to DB.
        Returns number of rows saved.
        """
        prep  = Preprocessor()
        clean = prep.fit_transform(df, target_col=target)

        # No train_test_split here - that is the whole point.
        fm     = ForecastModel()
        result = fm.fit_predict(clean, target, model=model_name, horizon=self.horizon)
        result.target = target

        db_df = result.to_db_df()
        db_df["rmse"] = None     # future has no actuals yet
        db_df["mape"] = None
        rows = self.db.write_forecast_results(db_df)

        logger.info(
            f"Future forecast saved | model={model_name} | target={target} | rows={rows}"
        )
        return rows

    # -- Run all combinations ------------------------------------------

    def run(self) -> dict:
        """Runs all targets x all models. Returns summary dict."""
        df      = self.load_gold()
        summary = {}

        for target in self.TARGETS:
            for model_name in self.MODELS:
                key = f"{model_name}_{target}"
                try:
                    summary[key] = self.forecast_one(df, target, model_name)
                except Exception as e:
                    logger.error(f"Future forecast failed | {key} | {e}")
                    summary[key] = 0

        return summary


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ff      = FutureForecaster()
    summary = ff.run()

    print("\n" + "=" * 50)
    print("FUTURE FORECAST SUMMARY (next 12 months)")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v} rows saved")
    print("=" * 50)
