# Sonalika ML Service — Claude Code Build Instructions
> **Read this entire document before writing a single line of code.**
> This is the complete specification for the Python ML backend.
> The frontend instructions are in a separate document.

---

## 0. What This Service Is

A FastAPI + XGBoost machine learning service that:

1. **Trains 1 XGBoost model** (3 variants for quantile regression) on startup
2. **Serves live demand forecasts** — every number the frontend shows comes from here
3. **Runs inventory optimisation** — computes rebalancing recommendations from real math
4. **Exposes a scenario API** — frontend sends slider values, XGBoost reruns in ~10ms

This is the **core product**. The frontend is just the face. All intelligence lives here.

**Why XGBoost over Prophet:**
- One model handles all 4 SKUs × 5 regions — scales to 50 SKUs × 30 regions with zero
  architectural change (just more training rows)
- External signals (monsoon, MSP, diesel) are native features — not bolted-on regressors
- SHAP values give exact signal contributions — not approximations
- Training takes ~2 seconds — cold starts are not a concern
- Leaner Docker image — no system-level C++ dependencies

---

## 1. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| ML model | XGBoost (gradient boosted trees) |
| Explainability | SHAP (signal contribution values) |
| Preprocessing | scikit-learn |
| Data processing | pandas, numpy |
| DB client | supabase-py |
| Server | Uvicorn |
| Containerisation | Docker (Railway deployment) |

---

## 2. Project Structure

Create this exact structure:

```
sonalika-ml-service/
├── Dockerfile
├── requirements.txt
├── railway.toml
├── .env.example
├── main.py               ← FastAPI app, routes, CORS, startup trainer
├── feature_store.py      ← Feature engineering (most important file)
├── forecaster.py         ← XGBoost training + SHAP inference
├── optimizer.py          ← Inventory optimisation logic
└── data_loader.py        ← Supabase data fetching
```

---

## 3. Environment Variables

```bash
# .env.example — rename to .env for local, set in Railway dashboard for prod
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
PORT=8000
```

---

## 4. Dependencies

```txt
# requirements.txt
fastapi==0.111.0
uvicorn[standard]==0.29.0
xgboost==2.0.3
scikit-learn==1.4.2
shap==0.45.0
pandas==2.2.2
numpy==1.26.4
supabase==2.4.6
python-dotenv==1.0.1
pydantic==2.7.1
```

---

## 5. Dockerfile

XGBoost is pure Python — no system-level C++ dependencies needed.
This produces a much leaner image than Prophet (~300MB smaller).

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 6. railway.toml

```toml
[build]
builder = "dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT"
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

---

## 7. data_loader.py — Supabase Data Fetching

```python
"""
Loads training data from Supabase.
Called once on service startup — results cached in memory.
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"]
        )
    return _client


def load_sales_history() -> pd.DataFrame:
    """
    Load all 480 rows from sales_history.
    Returns DataFrame with columns:
      year, month, region, model, units_sold, revenue_cr
    """
    client = get_client()
    response = client.table("sales_history").select("*").execute()
    df = pd.DataFrame(response.data)
    df["ds"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def load_market_signals() -> pd.DataFrame:
    """
    Load all 24 rows from market_signals.
    Returns DataFrame with columns:
      year, month, monsoon_index, msp_wheat, msp_paddy, diesel_price, industry_sales
    """
    client = get_client()
    response = client.table("market_signals").select("*").execute()
    df = pd.DataFrame(response.data)
    df["ds"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def load_inventory() -> pd.DataFrame:
    """
    Load all 20 rows from inventory_current.
    Returns DataFrame with columns:
      region, model, units_available, units_optimal
    """
    client = get_client()
    response = client.table("inventory_current").select("*").execute()
    return pd.DataFrame(response.data)
```

---

## 8. feature_store.py — Feature Engineering

This is the most important file in the XGBoost approach.
XGBoost learns from features — this file builds the feature matrix from raw data.

```python
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

SIGNAL_FEATURES  = ["monsoon_index", "msp_wheat", "diesel_price", "industry_sales"]
LAG_FEATURES     = ["units_lag_1", "units_lag_3", "units_lag_12", "units_rolling_3m"]
TIME_FEATURES    = ["month", "quarter", "is_rabi_peak", "is_kharif_prep",
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
    Add all time-based features to a DataFrame that has a 'month' column (1–12).
    These encode the seasonality patterns XGBoost needs to learn.
    """
    df = df.copy()
    df["quarter"]          = ((df["month"] - 1) // 3) + 1
    df["is_rabi_peak"]     = df["month"].isin([9, 10, 11]).astype(int)
    df["is_kharif_prep"]   = df["month"].isin([3, 4]).astype(int)
    df["is_monsoon_trough"]= df["month"].isin([6, 7]).astype(int)

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
      sales_df   — 480 rows from sales_history (24 months × 4 models × 5 regions)
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
    df[SIGNAL_FEATURES] = df[SIGNAL_FEATURES].fillna(method="ffill").fillna(method="bfill")

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

    For each future month (Apr 2026 → Apr+horizon):
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

    # Historical lookup for lag features: (region, model, year, month) → units
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
```

---

## 9. forecaster.py — XGBoost Training + SHAP Inference

```python
"""
XGBoost demand forecaster with SHAP explainability.

Architecture:
  Three XGBoost models trained on the same feature matrix:
  - model_median  : predicts median demand (main forecast = units_ai)
  - model_lower   : predicts 10th percentile (confidence lower bound)
  - model_upper   : predicts 90th percentile (confidence upper bound)

  All three train on ALL 480 rows (across all regions and SKUs).
  Region and SKU are features — not separate models.

  This means:
  - 1 training run covers all 20 region/SKU combinations
  - Adding new regions or SKUs = add rows, retrain — nothing else changes
  - Training takes ~2 seconds total

SHAP values:
  After training, a SHAP TreeExplainer is built on model_median.
  For any forecast, SHAP gives exact per-feature contributions.
  This directly powers the Signal Contribution panel in the UI.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from feature_store import (
    build_training_features,
    build_inference_features,
    ALL_FEATURES, SIGNAL_FEATURES, LAG_FEATURES,
    MODELS, REGIONS
)

# ── Global state — populated on startup ──────────────────────────────────────

MODEL_MEDIAN = None    # main forecast
MODEL_LOWER  = None    # 10th percentile
MODEL_UPPER  = None    # 90th percentile
EXPLAINER    = None    # SHAP TreeExplainer
TRAIN_DF     = None    # cached training DataFrame (for historical lookup)
SIGNALS_DF   = None    # cached signals DataFrame


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(sales_df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    """
    Train 3 XGBoost models on startup.
    Populates MODULE-level globals: MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, EXPLAINER.
    """
    global MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, EXPLAINER, TRAIN_DF, SIGNALS_DF

    TRAIN_DF   = sales_df.copy()
    SIGNALS_DF = signals_df.copy()

    print("Building feature matrix...")
    feat_df = build_training_features(sales_df, signals_df)
    print(f"  Feature matrix: {len(feat_df)} rows × {len(ALL_FEATURES)} features")
    print(f"  Target: units_sold | Range: {feat_df['units_sold'].min()}–{feat_df['units_sold'].max()}")

    X = feat_df[ALL_FEATURES].values
    y = feat_df["units_sold"].values

    # Split for evaluation (not used for final model — retrain on full data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_params = {
        "n_estimators":     300,
        "max_depth":        4,        # shallow trees prevent overfitting on 480 rows
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state":     42,
        "n_jobs":           -1,
    }

    print("Training XGBoost models...")

    # ── Median model (main forecast) ──────────────────────────────────────────
    MODEL_MEDIAN = xgb.XGBRegressor(objective="reg:squarederror", **xgb_params)
    MODEL_MEDIAN.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     verbose=False)

    val_preds = MODEL_MEDIAN.predict(X_val)
    mae  = mean_absolute_error(y_val, val_preds)
    mape = np.mean(np.abs((y_val - val_preds) / np.clip(y_val, 1, None)))
    print(f"  Median model — Val MAE: {mae:.0f} units | Val MAPE: {mape*100:.1f}%")

    # Retrain on full data for production use
    MODEL_MEDIAN.fit(X, y, verbose=False)

    # ── Lower bound (10th percentile) ─────────────────────────────────────────
    MODEL_LOWER = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        **xgb_params
    )
    MODEL_LOWER.fit(X, y, verbose=False)

    # ── Upper bound (90th percentile) ─────────────────────────────────────────
    MODEL_UPPER = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        **xgb_params
    )
    MODEL_UPPER.fit(X, y, verbose=False)

    # ── SHAP explainer ────────────────────────────────────────────────────────
    print("  Building SHAP explainer...")
    EXPLAINER = shap.TreeExplainer(MODEL_MEDIAN)

    print(f"  Done — 3 XGBoost models trained + SHAP explainer ready.")
    print(f"  Model accuracy: MAE={mae:.0f} units | MAPE={mape*100:.1f}%")


# ── Inference ─────────────────────────────────────────────────────────────────

def forecast(
    region: str,
    model_name: str,
    horizon_months: int = 6,
    signal_overrides: dict = None
) -> dict:
    """
    Generate demand forecast for a specific region + tractor model.

    Returns JSON-serialisable dict with:
    - forecast:             list of monthly predictions with confidence bands
    - historical:           24 months of actual sales (for chart)
    - signal_contributions: SHAP-based exact feature importance
    - model_accuracy:       MAE + MAPE from validation set
    """
    if MODEL_MEDIAN is None:
        raise RuntimeError("Models not trained. Call train_models() first.")

    if signal_overrides is None:
        signal_overrides = {}

    # ── Build inference features ──────────────────────────────────────────────
    feat_df = build_inference_features(
        region=region,
        model_name=model_name,
        horizon_months=horizon_months,
        sales_df=TRAIN_DF,
        signals_df=SIGNALS_DF,
        signal_overrides=signal_overrides
    )

    X_future = feat_df[ALL_FEATURES].values

    # ── Predict ───────────────────────────────────────────────────────────────
    preds_median = np.clip(MODEL_MEDIAN.predict(X_future), 0, None)
    preds_lower  = np.clip(MODEL_LOWER.predict(X_future),  0, None)
    preds_upper  = np.clip(MODEL_UPPER.predict(X_future),  0, None)

    # ── SHAP signal contributions ─────────────────────────────────────────────
    shap_values = EXPLAINER.shap_values(X_future)  # shape: (horizon, n_features)

    # Mean absolute SHAP contribution per signal feature
    # Expressed as fraction of mean prediction (percentage impact)
    mean_pred  = float(preds_median.mean()) if preds_median.mean() > 0 else 1.0
    contributions = {}
    for sig in SIGNAL_FEATURES:
        idx = ALL_FEATURES.index(sig)
        mean_shap = float(shap_values[:, idx].mean())
        contributions[sig] = round(mean_shap / mean_pred, 4)

    # ── Build forecast list ───────────────────────────────────────────────────
    forecast_list = []
    for i, row in feat_df.iterrows():
        forecast_list.append({
            "ds":          row["ds"].strftime("%Y-%m-%d"),
            "units_ai":    int(round(preds_median[i])),
            "units_lower": int(round(preds_lower[i])),
            "units_upper": int(round(preds_upper[i])),
            "units_naive": int(row["units_naive"]),  # same month last year
        })

    # ── Historical data for chart ─────────────────────────────────────────────
    historical = get_historical(region, model_name)

    # ── In-sample accuracy (recompute on this region/model slice) ─────────────
    feat_train = build_training_features(TRAIN_DF, SIGNALS_DF)
    mask = (
        (feat_train["region_enc"] == feat_df["region_enc"].iloc[0]) &
        (feat_train["model_enc"]  == feat_df["model_enc"].iloc[0])
    )
    slice_df = feat_train[mask]
    if len(slice_df) > 0:
        X_sl = slice_df[ALL_FEATURES].values
        y_sl = slice_df["units_sold"].values
        p_sl = MODEL_MEDIAN.predict(X_sl)
        mae  = round(float(mean_absolute_error(y_sl, p_sl)), 1)
        mape = round(float(np.mean(np.abs((y_sl - p_sl) / np.clip(y_sl, 1, None)))), 4)
    else:
        mae, mape = 0.0, 0.0

    return {
        "region":               region,
        "model":                model_name,
        "forecast":             forecast_list,
        "historical":           historical,
        "signal_contributions": contributions,
        "model_accuracy":       {"mae": mae, "mape": mape},
    }


def get_historical(region: str, model_name: str) -> list:
    """Return historical sales for a region + model. Used for the chart."""
    mask = (TRAIN_DF["region"] == region) & (TRAIN_DF["model"] == model_name)
    hist = TRAIN_DF[mask].sort_values("ds")
    return [
        {"ds": row["ds"].strftime("%Y-%m-%d"), "units_sold": int(row["units_sold"])}
        for _, row in hist.iterrows()
    ]


def forecast_all_regions(model_name: str, horizon_months: int = 6,
                          signal_overrides: dict = None) -> list:
    """Forecast one tractor model across all 5 regions."""
    return [
        forecast(region, model_name, horizon_months, signal_overrides)
        for region in REGIONS
    ]


def forecast_all_models(region: str, horizon_months: int = 6,
                         signal_overrides: dict = None) -> list:
    """Forecast all 4 tractor models for one region."""
    return [
        forecast(region, model_name, horizon_months, signal_overrides)
        for model_name in MODELS
    ]
```

---

## 10. optimizer.py — Inventory Optimisation Logic

```python
"""
Inventory optimisation engine.

Given:
  - Current inventory snapshot (from Supabase inventory_current table)
  - Demand forecasts for next 6 months (from XGBoost)

Computes:
  - Health status per region/model (overstock / understock / healthy)
  - Days of cover (how many days current stock will last at forecast demand rate)
  - Rebalancing recommendations (move X units from region A to region B)
  - Financial impact (₹ Cr at risk from overstock carrying cost or lost sales)
  - Priority-ranked action list
"""

import pandas as pd
import numpy as np
from typing import List, Dict

MODEL_PRICE_L = {
    "Tiger DI 35":  5.8,
    "Tiger DI 50":  7.2,
    "Tiger DI 75":  9.5,
    "Worldtrac 60": 8.5,
}

CARRYING_COST_RATE   = 0.015   # 1.5% per month of inventory value
OVERSTOCK_THRESHOLD  = 1.15    # available > 115% of optimal
UNDERSTOCK_THRESHOLD = 0.85    # available < 85% of optimal


def compute_inventory_health(
    inventory_df: pd.DataFrame,
    forecasts: dict
) -> List[Dict]:
    """
    Compute health status for every region/model combination.

    forecasts: dict keyed by (region, model_name) → forecast response from XGBoost
               Used to compute days_of_cover from actual demand forecast
    """
    results = []

    for _, row in inventory_df.iterrows():
        region     = row["region"]
        model_name = row["model"]
        available  = int(row["units_available"])
        optimal    = int(row["units_optimal"])

        ratio  = available / optimal if optimal > 0 else 1.0
        status = (
            "overstock"  if ratio > OVERSTOCK_THRESHOLD  else
            "understock" if ratio < UNDERSTOCK_THRESHOLD else
            "healthy"
        )

        # Days of cover from XGBoost forecast
        days_of_cover = None
        fc_key = (region, model_name)
        if fc_key in forecasts and forecasts[fc_key]["forecast"]:
            monthly_demand = forecasts[fc_key]["forecast"][0]["units_ai"]
            daily_demand   = monthly_demand / 30
            days_of_cover  = round(available / daily_demand) if daily_demand > 0 else None

        price_l = MODEL_PRICE_L.get(model_name, 7.0)
        if status == "overstock":
            risk_cr = round((available - optimal) * price_l / 100 * CARRYING_COST_RATE, 2)
        elif status == "understock":
            risk_cr = round((optimal - available) * price_l / 100 * 0.08, 2)
        else:
            risk_cr = 0.0

        results.append({
            "region":        region,
            "model":         model_name,
            "available":     available,
            "optimal":       optimal,
            "ratio":         round(ratio, 3),
            "status":        status,
            "deviation_pct": round((ratio - 1) * 100, 1),
            "days_of_cover": days_of_cover,
            "risk_cr":       risk_cr,
            "price_l":       price_l,
        })

    return results


def generate_rebalancing_recommendations(health: List[Dict]) -> List[Dict]:
    """
    Match overstock sources to understock sinks for the same model.
    Rank by ₹ saving — highest impact first.
    """
    overstock  = [r for r in health if r["status"] == "overstock"]
    understock = [r for r in health if r["status"] == "understock"]
    recs = []

    for src in overstock:
        for snk in understock:
            if src["model"] != snk["model"]:
                continue
            excess    = src["available"] - src["optimal"]
            shortfall = snk["optimal"]   - snk["available"]
            units     = min(excess, shortfall)
            if units <= 0:
                continue

            price_l   = src["price_l"]
            saving_cr = round(
                units * price_l / 100 * (CARRYING_COST_RATE + 0.08), 2
            )

            recs.append({
                "action":      "transfer",
                "from_region": src["region"],
                "to_region":   snk["region"],
                "model":       src["model"],
                "units":       units,
                "saving_cr":   saving_cr,
                "priority":    "high" if saving_cr > 0.5 else "medium",
                "rationale": (
                    f"{src['region']} has {src['deviation_pct']:+.0f}% excess. "
                    f"{snk['region']} is {abs(snk['deviation_pct']):.0f}% below optimal. "
                    f"Moving {units:,} units of {src['model']} recovers "
                    f"₹{saving_cr:.2f} Cr."
                ),
            })

    recs.sort(key=lambda x: x["saving_cr"], reverse=True)
    return recs


def compute_summary(health: List[Dict], recs: List[Dict]) -> Dict:
    """Overall inventory summary for the KPI strip."""
    deviations   = [abs(r["deviation_pct"]) for r in health]
    health_score = max(0, round(100 - np.mean(deviations), 1))

    return {
        "health_score":     health_score,
        "total_risk_cr":    round(sum(r["risk_cr"] for r in health), 2),
        "total_saving_cr":  round(sum(r["saving_cr"] for r in recs), 2),
        "overstock_count":  sum(1 for r in health if r["status"] == "overstock"),
        "understock_count": sum(1 for r in health if r["status"] == "understock"),
        "healthy_count":    sum(1 for r in health if r["status"] == "healthy"),
        "action_count":     len(recs),
    }
```

---

## 11. main.py — FastAPI App + All Endpoints

```python
"""
FastAPI application.
XGBoost models train on startup in ~2 seconds — cold starts are not a concern.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import data_loader as dl
import forecaster  as fc
import optimizer   as opt
from feature_store import MODELS, REGIONS


# ── Startup: train models ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data from Supabase...")
    sales_df   = dl.load_sales_history()
    signals_df = dl.load_market_signals()
    print(f"  Loaded {len(sales_df)} sales rows, {len(signals_df)} signal rows")

    print("Training XGBoost models (~2 seconds)...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, fc.train_models, sales_df, signals_df)

    app.state.inventory_df = dl.load_inventory()
    print(f"  Loaded {len(app.state.inventory_df)} inventory rows")
    print("Service ready.")
    yield


app = FastAPI(
    title="Sonalika Demand Intelligence API",
    description="XGBoost demand forecasting and inventory optimisation for Sonalika Tractors",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    region:            str
    model:             str
    horizon_months:    int            = 6
    monsoon_index:     Optional[float] = None
    msp_change_pct:    Optional[float] = None
    diesel_change_pct: Optional[float] = None
    market_growth_pct: Optional[float] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_signal_overrides(req: ScenarioRequest) -> dict:
    """Convert scenario request to absolute signal override values."""
    overrides = {}
    last = fc.SIGNALS_DF.sort_values("ds").iloc[-1]

    if req.monsoon_index is not None:
        overrides["monsoon_index"] = req.monsoon_index

    if req.msp_change_pct is not None:
        overrides["msp_wheat"] = float(last["msp_wheat"]) * (1 + req.msp_change_pct / 100)

    if req.diesel_change_pct is not None:
        overrides["diesel_price"] = float(last["diesel_price"]) * (1 + req.diesel_change_pct / 100)

    if req.market_growth_pct is not None:
        overrides["industry_sales"] = float(last["industry_sales"]) * (1 + req.market_growth_pct / 100)

    return overrides


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Confirms models are loaded and service is ready."""
    ready = fc.MODEL_MEDIAN is not None
    return {
        "status":        "ok" if ready else "training",
        "models_loaded": 3 if ready else 0,   # median + lower + upper
        "expected":      3,
        "shap_ready":    fc.EXPLAINER is not None,
    }


@app.get("/forecast")
def get_forecast(
    region:         str = Query(...),
    model:          str = Query(...),
    horizon_months: int = Query(6, ge=1, le=12),
):
    """XGBoost demand forecast for one region + tractor model."""
    try:
        return fc.forecast(region, model, horizon_months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/all-regions")
def get_forecast_all_regions(
    model:          str = Query(...),
    horizon_months: int = Query(6),
):
    """Forecast one tractor model across all 5 regions."""
    try:
        return {"results": fc.forecast_all_regions(model, horizon_months)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/all-models")
def get_forecast_all_models(
    region:         str = Query(...),
    horizon_months: int = Query(6),
):
    """Forecast all 4 tractor models for one region."""
    try:
        return {"results": fc.forecast_all_models(region, horizon_months)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario")
def run_scenario(req: ScenarioRequest):
    """
    What-if scenario for a single region + model.
    XGBoost reruns with overridden signal values (~10ms).
    """
    try:
        overrides = build_signal_overrides(req)
        result    = fc.forecast(req.region, req.model, req.horizon_months, overrides)
        baseline  = fc.forecast(req.region, req.model, req.horizon_months)

        base_total     = sum(f["units_ai"] for f in baseline["forecast"])
        scenario_total = sum(f["units_ai"] for f in result["forecast"])
        delta_units    = scenario_total - base_total
        delta_pct      = round(delta_units / base_total * 100, 1) if base_total else 0

        return {
            **result,
            "baseline_total":    base_total,
            "scenario_total":    scenario_total,
            "delta_units":       delta_units,
            "delta_pct":         delta_pct,
            "overrides_applied": overrides,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario/aggregate")
def run_scenario_aggregate(req: ScenarioRequest):
    """
    What-if across ALL regions and ALL models.
    Used by the Scenario Planner summary panels.
    """
    try:
        overrides       = build_signal_overrides(req)
        scenario_total  = 0
        baseline_total  = 0
        region_impacts  = {}
        model_impacts   = {}

        for region in REGIONS:
            r_scenario = 0
            r_baseline = 0
            for model_name in MODELS:
                sc  = fc.forecast(region, model_name, req.horizon_months, overrides)
                bl  = fc.forecast(region, model_name, req.horizon_months)
                sc_sum = sum(f["units_ai"] for f in sc["forecast"])
                bl_sum = sum(f["units_ai"] for f in bl["forecast"])
                scenario_total += sc_sum
                baseline_total += bl_sum
                r_scenario     += sc_sum
                r_baseline     += bl_sum
                model_impacts[model_name] = model_impacts.get(model_name, 0) + (sc_sum - bl_sum)
            region_impacts[region] = r_scenario - r_baseline

        delta_units  = scenario_total - baseline_total
        delta_pct    = round(delta_units / baseline_total * 100, 1) if baseline_total else 0
        financial_cr = round(abs(delta_units) * 7.2 / 100, 2)

        top_region = max(region_impacts, key=lambda k: abs(region_impacts[k]))
        top_model  = max(model_impacts,  key=lambda k: abs(model_impacts[k]))

        return {
            "baseline_total":    baseline_total,
            "scenario_total":    scenario_total,
            "delta_units":       delta_units,
            "delta_pct":         delta_pct,
            "financial_cr":      financial_cr,
            "top_region":        top_region,
            "top_region_delta":  region_impacts[top_region],
            "top_model":         top_model,
            "top_model_delta":   model_impacts[top_model],
            "region_impacts":    region_impacts,
            "model_impacts":     model_impacts,
            "overrides_applied": overrides,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory")
def get_inventory(horizon_months: int = Query(6)):
    """
    Full inventory health + rebalancing recommendations.
    Combines current stock from Supabase with XGBoost demand forecast.
    """
    try:
        inventory_df = app.state.inventory_df
        forecasts = {
            (region, model_name): fc.forecast(region, model_name, horizon_months)
            for region in REGIONS
            for model_name in MODELS
        }
        health  = opt.compute_inventory_health(inventory_df, forecasts)
        recs    = opt.generate_rebalancing_recommendations(health)
        summary = opt.compute_summary(health, recs)
        return {"summary": summary, "health": health, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market-signals")
def get_market_signals():
    """Raw market signals data for the Market Signals Tracker screen."""
    try:
        df = dl.load_market_signals()
        return {"signals": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/historical")
def get_historical(
    region: str = Query(...),
    model:  str = Query(...),
):
    """Historical sales for a region + model (for the chart)."""
    try:
        return {"historical": fc.get_historical(region, model)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 12. Build & Deploy Sequence

### Step 1 — Local setup
```bash
mkdir sonalika-ml-service && cd sonalika-ml-service

# Create all files as specified above
# Copy .env.example → .env and fill in Supabase values

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Local run
```bash
uvicorn main:app --reload --port 8000
```

Expected startup output:
```
Loading data from Supabase...
  Loaded 480 sales rows, 24 signal rows
Training XGBoost models (~2 seconds)...
  Feature matrix: 360 rows × 15 features
  Target: units_sold | Range: 261–26451
  Median model — Val MAE: ~320 units | Val MAPE: ~7%
  Building SHAP explainer...
  Done — 3 XGBoost models trained + SHAP explainer ready.
  Loaded 20 inventory rows
Service ready.
```

### Step 3 — Test endpoints locally
```bash
# Health check — expect models_loaded=3
curl http://localhost:8000/health

# Single forecast
curl "http://localhost:8000/forecast?region=Punjab-Haryana&model=Tiger%20DI%2050"

# Inventory optimisation
curl http://localhost:8000/inventory

# Scenario — poor monsoon
curl -X POST http://localhost:8000/scenario/aggregate \
  -H "Content-Type: application/json" \
  -d '{"region":"Punjab-Haryana","model":"Tiger DI 50","monsoon_index":76}'
```

### Step 4 — Push to GitHub + deploy to Railway
```bash
git init
git add .
git commit -m "Initial ML service — XGBoost demand forecasting"

# Create GitHub repo "sonalika-ml-service" and push
git remote add origin https://github.com/yourname/sonalika-ml-service.git
git push -u origin main

# Deploy via Railway CLI
railway login
railway init
railway up

# Set environment variables
railway variables set SUPABASE_URL=your_url
railway variables set SUPABASE_KEY=your_key
```

### Step 5 — Verify production
```bash
curl https://your-railway-url.up.railway.app/health
# Expect: {"status":"ok","models_loaded":3,"expected":3,"shap_ready":true}
```

**Save the Railway URL — the frontend needs it as ML_SERVICE_URL.**

---

## 13. API Reference Summary

| Method | Endpoint | Used by |
|---|---|---|
| GET | `/health` | Frontend startup check |
| GET | `/forecast?region=&model=` | Demand Intelligence screen |
| GET | `/forecast/all-regions?model=` | Demand Intelligence (All Regions) |
| GET | `/forecast/all-models?region=` | Demand Intelligence (All Models) |
| POST | `/scenario` | Scenario Planner (single) |
| POST | `/scenario/aggregate` | Scenario Planner (summary panels) |
| GET | `/inventory` | Inventory Health + Command Center |
| GET | `/market-signals` | Market Signals Tracker |
| GET | `/historical?region=&model=` | Sales history chart |

---

## 14. Critical Notes

1. **3 models trained on startup** — median + lower + upper quantile (~2 sec total)
2. **Naive forecast = `units_lag_12`** — same month last year, built in `feature_store.py`
3. **SHAP values are exact** — not approximations. Signal Contribution panel shows real model output
4. **Scenario reruns XGBoost** — feature overrides applied, inference in ~10ms
5. **No cold start anxiety** — XGBoost trains in 2 seconds, Railway restarts are harmless
6. **CORS is open (`*`)** — acceptable for demo, tighten to Vercel domain for production
7. **No auth on endpoints** — acceptable for demo, add API key header for production
8. **Production path** — same XGBoost architecture, just more rows. 50 SKUs × 30 regions = add rows, retrain. Nothing else changes.

---

*ML Service Instructions v2.0 | Sonalika Demand Intelligence | Dexian India*
