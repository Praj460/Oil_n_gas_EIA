# oil_gas_pipeline | models/preprocessor.py
# Preprocessor class — cleans silver layer data and engineers features
# for time series forecasting
# Usage: from models.preprocessor import Preprocessor

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

from config.config import config

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Cleans and prepares silver layer data for forecasting models.

    Steps applied in order:
    1. Set period as DatetimeIndex
    2. Resample to monthly frequency — fills any gaps in the time series
    3. Handle missing values via forward fill then backward fill
    4. Remove outliers using IQR method — replaces with interpolated values
    5. Engineer lag features and rolling statistics
    6. Normalize if requested (for neural network compatibility)

    Usage:
        prep  = Preprocessor()
        clean = prep.fit_transform(df, target_col="wti_price")
    """

    def __init__(self):
        self.target_col:   Optional[str]   = None
        self.feature_cols: list[str]       = []
        self._mean:        Optional[float] = None
        self._std:         Optional[float] = None
        self._is_fitted:   bool            = False

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df:           pd.DataFrame,
        target_col:   str,
        feature_cols: Optional[list[str]] = None,
        normalize:    bool = False,
    ) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline on a DataFrame.

        Args:
            df:           DataFrame from silver or gold layer with 'period' column
            target_col:   column to forecast e.g. "wti_price"
            feature_cols: additional feature columns to keep e.g. ["brent_price"]
            normalize:    if True, z-score normalize the target column

        Returns:
            Clean DataFrame indexed by period with target + feature columns
            and engineered lag/rolling features added
        """
        self.target_col   = target_col
        self.feature_cols = feature_cols or []

        logger.info(f"Preprocessing | target={target_col} | rows={len(df)}")

        df = df.copy()

        # Step 1 — set DatetimeIndex
        df = self._set_datetime_index(df)

        # Step 2 — resample to monthly, fill gaps
        df = self._resample_monthly(df, target_col, self.feature_cols)

        # Step 3 — handle missing values
        df = self._fill_missing(df)

        # Step 4 — remove outliers
        df = self._remove_outliers(df, target_col)

        # Step 5 — engineer features
        df = self._engineer_features(df, target_col)

        # Step 6 — normalize target if requested
        if normalize:
            df = self._normalize(df, target_col)

        self._is_fitted = True

        # Drop rows with NaN introduced by lag features at the start
        before = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows with NaN after feature engineering")

        logger.info(f"Preprocessing complete | output rows={len(df)} | cols={list(df.columns)}")
        return df

    # ── Step 1: DatetimeIndex ─────────────────────────────────────────────────

    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts 'period' column to DatetimeIndex."""
        if "period" not in df.columns:
            raise ValueError("DataFrame must have a 'period' column")
        df["period"] = pd.to_datetime(df["period"])
        df = df.set_index("period").sort_index()
        logger.debug("DatetimeIndex set on 'period'")
        return df

    # ── Step 2: Resample ──────────────────────────────────────────────────────

    def _resample_monthly(
        self,
        df:           pd.DataFrame,
        target_col:   str,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """
        Resamples DataFrame to monthly frequency (MS = month start).
        Fills any missing months in the time series so there are no gaps.
        Uses mean aggregation in case there are duplicate months.
        """
        cols_to_keep = [target_col] + [c for c in feature_cols if c in df.columns]
        df = df[cols_to_keep]
        numeric_df = df.select_dtypes(include="number")   # keep only numbers
        df = numeric_df.resample("MS").mean()

        gap_count = df[target_col].isna().sum()
        if gap_count > 0:
            logger.warning(f"Resampling introduced {gap_count} missing months in {target_col}")

        logger.debug(f"Resampled to monthly | rows={len(df)}")
        return df

    # ── Step 3: Fill missing ──────────────────────────────────────────────────

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using:
        - Forward fill first (carries last known value forward)
        - Backward fill second (fills any NaN at the very start)
        - Linear interpolation as a final fallback
        """
        null_before = df.isna().sum().sum()

        df = df.ffill()
        df = df.bfill()
        df = df.interpolate(method="linear")

        null_after = df.isna().sum().sum()
        logger.info(f"Missing value fill | before={null_before} | after={null_after}")
        return df

    # ── Step 4: Outlier removal ───────────────────────────────────────────────

    def _remove_outliers(
        self,
        df:  pd.DataFrame,
        col: str,
        iqr_multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detects and replaces outliers in the target column using the IQR method.

        Outlier definition: value outside [Q1 - k*IQR, Q3 + k*IQR]
        where k = iqr_multiplier (default 3.0 — conservative for energy prices
        which have legitimate large moves like the 2008 spike or 2020 crash)

        Outliers are replaced with NaN then linearly interpolated — this
        preserves the time series structure better than dropping rows.

        Args:
            df:             DataFrame with DatetimeIndex
            col:            column to check for outliers
            iqr_multiplier: how many IQRs define an outlier (default 3.0)
        """
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.warning(
                f"Outliers in {col}: {outlier_count} values outside "
                f"[{lower:.2f}, {upper:.2f}] — replacing with interpolated values"
            )
            df.loc[outlier_mask, col] = np.nan
            df[col] = df[col].interpolate(method="linear")

        return df

    # ── Step 5: Feature engineering ───────────────────────────────────────────

    def _engineer_features(
        self,
        df:         pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """
        Adds time series features that help forecasting models:

        Lag features:
            lag_1, lag_3, lag_6, lag_12 — price N months ago
            (captures autocorrelation — this month's price depends on last month's)

        Rolling statistics:
            rolling_3_mean  — 3-month moving average (short-term trend)
            rolling_6_mean  — 6-month moving average (medium-term trend)
            rolling_12_mean — 12-month moving average (annual trend)
            rolling_3_std   — 3-month volatility

        Calendar features:
            month     — seasonality (energy demand is seasonal)
            quarter   — quarterly patterns
            year      — long-term trend

        Args:
            df:         clean DataFrame with DatetimeIndex
            target_col: column to generate features for

        Returns:
            DataFrame with original columns + new feature columns
        """
        t = target_col

        # Lag features
        df[f"{t}_lag_1"]  = df[t].shift(1)
        df[f"{t}_lag_3"]  = df[t].shift(3)
        df[f"{t}_lag_6"]  = df[t].shift(6)
        df[f"{t}_lag_12"] = df[t].shift(12)

        # Rolling statistics (min_periods=1 avoids NaN at the start)
        df[f"{t}_rolling_3_mean"]  = df[t].rolling(window=3,  min_periods=1).mean()
        df[f"{t}_rolling_6_mean"]  = df[t].rolling(window=6,  min_periods=1).mean()
        df[f"{t}_rolling_12_mean"] = df[t].rolling(window=12, min_periods=1).mean()
        df[f"{t}_rolling_3_std"]   = df[t].rolling(window=3,  min_periods=1).std()

        # Month-over-month % change
        df[f"{t}_mom_pct"] = df[t].pct_change(1) * 100

        # Calendar features
        df["month"]   = df.index.month
        df["quarter"] = df.index.quarter
        df["year"]    = df.index.year

        logger.debug(f"Feature engineering complete | new cols added for {target_col}")
        return df

    # ── Step 6: Normalize ─────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Z-score normalizes the target column.
        Stores mean and std so inverse_transform can undo it later.
        """
        self._mean = df[col].mean()
        self._std  = df[col].std()

        if self._std == 0:
            logger.warning(f"Std dev is 0 for {col} — skipping normalization")
            return df

        df[col] = (df[col] - self._mean) / self._std
        logger.info(f"Normalized {col} | mean={self._mean:.4f} | std={self._std:.4f}")
        return df

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """
        Reverses normalization — call this on forecast output
        to get prices back in original dollars.
        """
        if self._mean is None or self._std is None:
            raise ValueError("Preprocessor was not used with normalize=True")
        return values * self._std + self._mean

    # ── Train / test split ────────────────────────────────────────────────────

    def train_test_split(
        self,
        df:         pd.DataFrame,
        target_col: str,
        test_size:  float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a time series DataFrame into train and test sets.
        Uses the last `test_size` fraction as the test set —
        never shuffles because order matters in time series.

        Args:
            df:         preprocessed DataFrame with DatetimeIndex
            target_col: column being forecasted
            test_size:  fraction of data to use as test set (default 0.2 = 20%)

        Returns:
            (train_df, test_df) tuple
        """
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]

        logger.info(
            f"Train/test split | "
            f"train={len(train)} rows ({train.index[0].date()} → {train.index[-1].date()}) | "
            f"test={len(test)} rows ({test.index[0].date()} → {test.index[-1].date()})"
        )
        return train, test

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, df: pd.DataFrame) -> dict:
        """Returns a quick summary of the preprocessed DataFrame."""
        return {
            "rows":       len(df),
            "columns":    list(df.columns),
            "date_range": f"{df.index.min().date()} → {df.index.max().date()}",
            "nulls":      df.isna().sum().to_dict(),
            "target_stats": {
                "mean": round(df[self.target_col].mean(), 4) if self.target_col else None,
                "std":  round(df[self.target_col].std(), 4)  if self.target_col else None,
                "min":  round(df[self.target_col].min(), 4)  if self.target_col else None,
                "max":  round(df[self.target_col].max(), 4)  if self.target_col else None,
            }
        }