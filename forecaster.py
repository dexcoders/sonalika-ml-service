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
    Populates module-level globals: MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, EXPLAINER.
    """
    global MODEL_MEDIAN, MODEL_LOWER, MODEL_UPPER, EXPLAINER, TRAIN_DF, SIGNALS_DF

    TRAIN_DF   = sales_df.copy()
    SIGNALS_DF = signals_df.copy()

    print("Building feature matrix...")
    feat_df = build_training_features(sales_df, signals_df)
    print(f"  Feature matrix: {len(feat_df)} rows x {len(ALL_FEATURES)} features")
    print(f"  Target: units_sold | Range: {feat_df['units_sold'].min()}-{feat_df['units_sold'].max()}")

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
