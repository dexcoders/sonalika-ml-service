"""
Feature engineering for the XGBoost demand forecasting model.

XGBoost does not understand time natively — we encode temporal patterns
and external signals as explicit numeric features.

Feature groups:
  1. Time features       — encode seasonality (month, quarter, season flags)
  2. Identity features   — which region and SKU (label encoded)
  3. External signals    — the 4 market signals from market_signals table
  4. Lag features        — give XGBoost "memory" of past demand
     units_lag_12        — same month last year = the naive baseline
     units_lag_3         — 3 months ago
     units_lag_1         — 1 month ago
     units_rolling_3m    — 3-month rolling average

Why lag features matter:
  units_lag_12 is literally "what Sonalika's Excel process would predict"
  XGBoost's prediction using ALL features is the AI forecast
  The difference = the value of the AI model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS  = ["Tiger DI 35", "Tiger DI 50", "Tiger DI 75", "Worldtrac 60"]
REGIONS = ["Punjab-Haryana", "UP-Bihar", "MP-Rajasthan", "Maharashtra-Gujarat", "South"]

SIGNAL_FEATURES   = ["monsoon_index", "msp_wheat", "diesel_price", "industry_sales"]
LAG_FEATURES      = ["units_lag_1", "units_lag_3", "units_lag_12", "units_rolling_3m"]
TIME_FEATURES     = ["month", "quarter", "is_rabi_peak", "is_kharif_prep",
                     "is_monsoon_trough", "month_sin", "month_cos"]
IDENTITY_FEATURES = ["region_enc", "model_enc"]

ALL_FEATURES = TIME_FEATURES + IDENTITY_FEATURES + SIGNAL_FEATURES + LAG_FEATURES

# Label encoders — fit once, reuse for inference
_region_encoder = LabelEncoder().fit(REGIONS)
_model_encoder  = LabelEncoder().fit(MODELS)


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_region(region: str) -> int:
    return int(_region_encoder.transform([region])[0])

def encode_model(model_name: str) -> int:
    return int(_model_encoder.transform([model_name])[0])

def decode_region(enc: int) -> str:
    return str(_region_encoder.inverse_transform([enc])[0])

def decode_model(enc: int) -> str:
    return str(_model_encoder.inverse_transform([enc])[0])


# ── Time feature builder ──────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all time-based features to a DataFrame that has a 'month' column (1-12).
    These encode the seasonality patterns XGBoost needs to learn.
    """
    df = df.copy()
    df["quarter"]           = ((df["month"] - 1) // 3) + 1
    df["is_rabi_peak"]      = df["month"].isin([9, 10, 11]).astype(int)
    df["is_kharif_prep"]    = df["month"].isin([3, 4]).astype(int)
    df["is_monsoon_trough"] = df["month"].isin([6, 7]).astype(int)

    # Cyclical encoding — month 1 and month 12 are adjacent, not far apart
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ── Training feature matrix ───────────────────────────────────────────────────

def build_training_features(
    sales_df: pd.DataFrame,
    signals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the complete feature matrix for XGBoost training.

    Input:
      sales_df   — 480 rows from sales_history (24 months x 4 models x 5 regions)
      signals_df — 24 rows from market_signals

    Output:
      DataFrame with ALL_FEATURES columns + 'units_sold' target
      Rows with NaN lag features (first 12 months) are dropped —
      XGBoost cannot train on NaN values.
    """
    df = sales_df.copy()

    # ── Merge external signals ────────────────────────────────────────────────
    signal_cols = ["ds"] + SIGNAL_FEATURES
    df = df.merge(signals_df[signal_cols], on="ds", how="left")
    df[SIGNAL_FEATURES] = df[SIGNAL_FEATURES].ffill().bfill()

    # ── Time features ─────────────────────────────────────────────────────────
    df = add_time_features(df)

    # ── Identity encoding ─────────────────────────────────────────────────────
    df["region_enc"] = df["region"].map(encode_region)
    df["model_enc"]  = df["model"].map(encode_model)

    # ── Lag features (computed per region+model group) ────────────────────────
    df = df.sort_values(["region", "model", "ds"]).reset_index(drop=True)

    lag_dfs = []
    for (region, model_name), group in df.groupby(["region", "model"]):
        group = group.copy().sort_values("ds")
        group["units_lag_1"]      = group["units_sold"].shift(1)
        group["units_lag_3"]      = group["units_sold"].shift(3)
        group["units_lag_12"]     = group["units_sold"].shift(12)
        group["units_rolling_3m"] = group["units_sold"].shift(1).rolling(3).mean()
        lag_dfs.append(group)

    df = pd.concat(lag_dfs, ignore_index=True)

    # Drop rows where lag_12 is NaN (first 12 months of each series)
    df = df.dropna(subset=["units_lag_12"]).reset_index(drop=True)

    return df


# ── Inference feature matrix ──────────────────────────────────────────────────

def build_inference_features(
    region: str,
    model_name: str,
    horizon_months: int,
    sales_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    signal_overrides: dict = None
) -> pd.DataFrame:
    """
    Build feature rows for forecasting future months.

    For each future month (Apr 2026 -> Apr+horizon):
    - Time features computed from the future date
    - Identity features from region + model
    - Signal features from last known values (with overrides applied)
    - Lag features looked up from historical sales

    signal_overrides: dict with optional keys:
      monsoon_index, msp_wheat, diesel_price, industry_sales
      If provided, these override the last known signal values.
    """
    if signal_overrides is None:
        signal_overrides = {}

    # Last date in training data
    last_ds = sales_df["ds"].max()

    # Last known signal values
    last_signals = signals_df.sort_values("ds").iloc[-1]
    base_signals = {col: float(last_signals[col]) for col in SIGNAL_FEATURES}
    # Apply overrides
    for key, val in signal_overrides.items():
        if key in base_signals:
            base_signals[key] = float(val)

    # Historical lookup for lag features: (region, model, year, month) -> units
    hist_mask = (sales_df["region"] == region) & (sales_df["model"] == model_name)
    hist = sales_df[hist_mask].set_index("ds")["units_sold"].to_dict()

    rows = []
    for i in range(horizon_months):
        future_ds    = last_ds + pd.DateOffset(months=i + 1)
        future_year  = future_ds.year
        future_month = future_ds.month

        # Lag lookups from historical data
        def get_lag(months_back: int) -> float:
            lag_ds = future_ds - pd.DateOffset(months=months_back)
            return float(hist.get(lag_ds, np.nan))

        lag_1  = get_lag(1)
        lag_3  = get_lag(3)
        lag_12 = get_lag(12)   # same month last year = naive baseline

        # Rolling 3m average (lags 1,2,3)
        lag_2 = get_lag(2)
        rolling_3m = np.nanmean([lag_1, lag_2, lag_3])

        row = {
            "ds":          future_ds,
            "year":        future_year,
            "month":       future_month,
            "region_enc":  encode_region(region),
            "model_enc":   encode_model(model_name),
            **base_signals,
            "units_lag_1":      lag_1  if not np.isnan(lag_1)  else 0,
            "units_lag_3":      lag_3  if not np.isnan(lag_3)  else 0,
            "units_lag_12":     lag_12 if not np.isnan(lag_12) else 0,
            "units_rolling_3m": rolling_3m if not np.isnan(rolling_3m) else 0,
            # Naive forecast = same month last year (what Excel does)
            "units_naive":  int(lag_12) if not np.isnan(lag_12) else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_time_features(df)
    return df
