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
    horizon_months:    int             = 6
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
