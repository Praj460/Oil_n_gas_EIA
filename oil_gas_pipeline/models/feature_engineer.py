# oil_gas_pipeline | models/feature_engineer.py
# FeatureEngineer class — builds derived features for SARIMAX / ML models.
# Designed to be re-runnable: same .transform() for training and live data.
# Usage:
#   from models.feature_engineer import FeatureEngineer
#   fe = FeatureEngineer()
#   raw_features, scaled_features = fe.transform(gold_features_df)

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Base columns to engineer features ON. Excludes period, created_at.
# Targets are included so we can build their own lags (for SARIMA's own use).
BASE_COLS = [
    "wti_price", "henry_hub_price",                      # targets
    "brent_price", "oil_production",                     # petroleum
    "crude_imports", "refinery_util",
    "gasoline_stocks", "distillate_stocks",
    "gas_storage", "gas_production",                     # natural gas
    "hdd", "cdd",                                        # weather
    "opec_spare", "global_inv",                          # supply fragility
    "dollar_index", "industrial_production", "treasury_10y",  # macro
]

LAGS = [1, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]


class FeatureEngineer:
    """
    Builds derived features on top of the wide gold_features table.

    Produces:
    - Lag features:        col_lag_{1,3,6,12}        (last month, quarter, half-year, year)
    - Rolling stats:       col_roll{3,6,12}_{mean,std}
    - Momentum:            col_mom_pct                (month-over-month % change)
    - Seasonality:         month_sin, month_cos, quarter
    - Cross-series:        storage_vs_12mo_avg
    - Scaled versions:     same features, z-score normalized (for SARIMAX/linear)

    Why two outputs (raw + scaled):
    - Tree models (XGBoost): scaling is unnecessary, raw is fine
    - SARIMAX / linear:      scaling is required, otherwise large-magnitude
                              features (storage in thousands) dwarf small-magnitude
                              ones (rates in single digits) purely on units
    """

    def __init__(self,
                 base_cols: List[str] = BASE_COLS,
                 lags: List[int] = LAGS,
                 roll_windows: List[int] = ROLL_WINDOWS):
        self.base_cols    = base_cols
        self.lags         = lags
        self.roll_windows = roll_windows
        # Scaling parameters stored after fit — applied identically to future data
        self._scale_means: pd.Series = None
        self._scale_stds:  pd.Series = None

    # ── Forward-fill edge nulls ──────────────────────────────────────────────

    def _fill_edge_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fills small reporting-lag gaps at the recent edge of the data."""
        before = df.isna().sum().sum()
        df = df.copy()
        df[self.base_cols] = df[self.base_cols].ffill()
        after = df.isna().sum().sum()
        logger.info(f"Edge nulls forward-filled: {before} → {after} null cells")
        return df

    # ── Per-column derivation ────────────────────────────────────────────────

    def _add_lags(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        for lag in self.lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def _add_rolling(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        for w in self.roll_windows:
            # shift(1) so the rolling window only sees PAST data — no current-month leak
            r = df[col].shift(1).rolling(window=w, min_periods=max(2, w // 2))
            df[f"{col}_roll{w}_mean"] = r.mean()
            df[f"{col}_roll{w}_std"]  = r.std()
        return df

    def _add_momentum(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        df[f"{col}_mom_pct"] = df[col].pct_change() * 100
        return df

    # ── Seasonality + cross-series ───────────────────────────────────────────

    def _add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        # Month / quarter as integers
        df["month"]   = df["period"].dt.month
        df["quarter"] = df["period"].dt.quarter
        # Cyclical encoding — Jan & Dec are adjacent in time, must be adjacent in feature space
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    def _add_cross_series(self, df: pd.DataFrame) -> pd.DataFrame:
        # Storage vs its own rolling 12-month average — captures glut/shortage
        if "gas_storage" in df.columns:
            roll12 = df["gas_storage"].shift(1).rolling(12, min_periods=6).mean()
            df["storage_vs_12mo_avg"] = df["gas_storage"] - roll12
        return df

    # ── Build derived features ───────────────────────────────────────────────

    def _build_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.base_cols:
            df = self._add_lags(df, col)
            df = self._add_rolling(df, col)
            df = self._add_momentum(df, col)
        df = self._add_seasonality(df)
        df = self._add_cross_series(df)
        return df

    # ── Scaling (z-score) ────────────────────────────────────────────────────

    def _fit_scaler(self, df: pd.DataFrame, cols: List[str]) -> None:
        """Compute mean/std for each numeric column — to be applied to future data unchanged."""
        self._scale_means = df[cols].mean()
        self._scale_stds  = df[cols].std().replace(0, 1)   # avoid divide-by-zero

    def _apply_scaler(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df_scaled = df.copy()
        df_scaled[cols] = (df[cols] - self._scale_means) / self._scale_stds
        return df_scaled

    # ── Public transform ─────────────────────────────────────────────────────

    def transform(self, gold_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds the engineered feature table.

        Args:
            gold_features: DataFrame from the gold_features table, with a
                           datetime 'period' column and the 17 base feature columns.

        Returns:
            (raw_features, scaled_features) — both with the same shape, scaled is
            z-score normalized for SARIMAX / linear models.
        """
        df = gold_features.copy()

        # Ensure period is datetime + sorted ascending
        df["period"] = pd.to_datetime(df["period"])
        df = df.sort_values("period").reset_index(drop=True)

        # 1. Forward-fill edge gaps
        df = self._fill_edge_nulls(df)

        # 2. Build derived columns
        df = self._build_derived(df)

        # 3. Identify numeric feature columns (exclude period, created_at, integer month/quarter)
        exclude = {"period", "created_at", "month", "quarter"}
        numeric_cols = [c for c in df.columns
                        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        # 4. Fit + apply scaler → scaled version
        self._fit_scaler(df, numeric_cols)
        df_scaled = self._apply_scaler(df, numeric_cols)

        n_engineered = len(df.columns) - len(gold_features.columns)
        logger.info(f"Built {n_engineered} engineered features (total cols: {len(df.columns)})")
        logger.info(f"Scaled {len(numeric_cols)} numeric columns")

        return df, df_scaled
