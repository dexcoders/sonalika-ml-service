# Sonalika ML Service — Claude Code Context

## What This Project Is
This is the Python ML backend for the **Sonalika Demand Intelligence Platform** —
a sales demo built by Dexian India for Sonalika Tractors (International Tractors Limited).

The demo is for **Shailendra**, Head of IT / Digital Transformation at Sonalika.
The goal is to convince him to move to a paid POC using Sonalika's actual SAP data.

This service is the **core product** — not a helper. Every forecast number, every
inventory recommendation, every scenario output in the frontend comes from here.

---

## Business Context

**The problem Sonalika has today:**
- Demand planning done via Excel + dealer calls + gut feel
- No structured way to factor in monsoon, MSP announcements, or crop cycles
- Result: overstock in wrong regions, stockouts in others — crores in working capital tied up

**What this service does:**
- Trains 1 XGBoost model (3 variants for quantile regression) on startup in ~2 seconds
- Serves live demand forecasts with SHAP-based signal contributions and confidence bands
- Runs inventory optimisation and generates rebalancing recommendations
- Powers a scenario planner — user moves sliders, XGBoost reruns in ~10ms

**The key proof point** (what wins the demo):
- Q3 2025, Punjab-Haryana: Naive forecast (`units_lag_12` = prior year actuals) → ~25% error
- AI forecast (XGBoost with all features including monsoon signal) → ~2–5% error
- This is computed dynamically from real model output — never hardcoded

**Why XGBoost over Prophet:**
- One model handles all 4 SKUs × 5 regions — scales to production without re-architecting
- SHAP values give exact signal contributions (not approximations)
- Trains in ~2 seconds — no cold start anxiety on Railway
- Leaner Docker image — no C++ system dependencies

---

## Data in Supabase (3 tables — XGBoost training inputs)

| Table | Rows | What it is |
|---|---|---|
| `sales_history` | 480 | 24 months × 4 models × 5 regions — training data |
| `market_signals` | 24 | Monthly external signals (monsoon, MSP, diesel, TMA) |
| `inventory_current` | 20 | Snapshot as of Mar 2026 — optimizer input |

**Key signal in the data:**
- 2024 monsoon = 102% of normal (good year)
- 2025 monsoon = 84% of normal (weak year)
- Punjab-Haryana Aug–Sep 2025 demand dropped ~27–31% vs 2024
- XGBoost learns this correlation via `monsoon_index` feature + lag features

**No `demand_forecast` or `production_plan` tables** — XGBoost generates these live.

---

## Tractor Models and Regions

**Models:**
- Tiger DI 35 — entry segment, highest volume
- Tiger DI 50 — mid segment
- Tiger DI 75 — premium, lowest volume
- Worldtrac 60 — export-driven, flatter seasonality

**Regions:**
- Punjab-Haryana (most monsoon-sensitive)
- UP-Bihar
- MP-Rajasthan (currently understocked)
- Maharashtra-Gujarat
- South

**Model prices (₹ Lakhs):**
- Tiger DI 35: ₹5.8L | Tiger DI 50: ₹7.2L | Tiger DI 75: ₹9.5L | Worldtrac 60: ₹8.5L

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| ML model | XGBoost (gradient boosted trees) |
| Explainability | SHAP (exact feature contributions) |
| Preprocessing | scikit-learn |
| Data processing | pandas, numpy |
| DB client | supabase-py |
| Server | Uvicorn |
| Deployment | Railway (Docker) |

---

## Project Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, all endpoints, CORS, startup trainer |
| `feature_store.py` | Feature engineering — builds training + inference matrices |
| `forecaster.py` | XGBoost training + SHAP inference + naive baseline |
| `optimizer.py` | Inventory health, rebalancing, ₹ impact |
| `data_loader.py` | Supabase data fetching |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Railway deployment (lean — no system deps) |
| `railway.toml` | Railway config |
| `.env.example` | Environment variable template |

---

## Feature Engineering (feature_store.py)

XGBoost learns from features — not raw time series. The feature matrix has:

| Group | Features | Purpose |
|---|---|---|
| Time | month, quarter, month_sin, month_cos | Encode cyclical seasonality |
| Season flags | is_rabi_peak, is_kharif_prep, is_monsoon_trough | Binary season markers |
| Identity | region_enc, model_enc | Which region and SKU (label encoded) |
| Signals | monsoon_index, msp_wheat, diesel_price, industry_sales | External market signals |
| Lag 1 month | units_lag_1 | Recent demand momentum |
| Lag 3 months | units_lag_3 | Short-term trend |
| Lag 12 months | units_lag_12 | **Same month last year = naive baseline** |
| Rolling | units_rolling_3m | 3-month smoothed demand |

**`units_lag_12` is the naive forecast** — it's literally "what Sonalika's Excel process predicts."
XGBoost's output using all features is the AI forecast.
The difference = the value of the model. This is the proof point.

---

## Environment Variables

```
SUPABASE_URL   — Supabase project URL
SUPABASE_KEY   — Supabase anon key
PORT           — 8000 (default)
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Confirms 3 models loaded + SHAP ready |
| GET | `/forecast` | Single region + model forecast |
| GET | `/forecast/all-regions` | One model, all 5 regions |
| GET | `/forecast/all-models` | All 4 models, one region |
| POST | `/scenario` | Single region/model what-if |
| POST | `/scenario/aggregate` | All regions + models what-if (Scenario Planner) |
| GET | `/inventory` | Full inventory health + rebalancing recommendations |
| GET | `/market-signals` | Raw signal data for Market Signals screen |
| GET | `/historical` | Historical sales for chart |

---

## Key Design Decisions

- **3 XGBoost models** — median (main forecast) + lower/upper quantile (confidence bands)
- **1 SHAP explainer** built on median model — gives exact per-feature contributions
- **Naive forecast = `units_lag_12`** — prior year actuals, built in feature_store.py
- **Scenario reruns XGBoost** — feature overrides applied, inference in ~10ms
- **Training on startup** — ~2 seconds, Railway restarts are harmless
- **CORS is open (`*`)** — acceptable for demo, tighten for production
- **No auth on endpoints** — acceptable for demo
- **Inventory carrying cost rate** — 1.5% per month of inventory value

---

## Build Instructions

Full detailed instructions with all code are in: `INSTRUCTIONS_ML_SERVICE.md`
Read that file for complete implementation, build order, and deployment steps.

---

## Connected Tools

- **Supabase MCP** — use to verify tables and debug data issues
- **Railway CLI** — `railway up` (deploy), `railway logs` (debug), `railway variables` (env vars)
- **Git** — push to GitHub before Railway deployment

---

## Critical Rules

1. **Never hardcode forecast numbers** — all data comes from XGBoost
2. **Naive forecast = `units_lag_12`** — prior year actual for same month, built in feature_store.py
3. **Proof point is dynamic** — computed from model output, never hardcoded
4. **Always verify `/health` returns `models_loaded=3, shap_ready=true`** after any restart
5. **Before demo** — hit `/health` once to confirm service is warm
6. **Production path** — same XGBoost architecture, just more rows. Never go back to N×M models.

---

## Production Architecture Note (for reference — not for demo build)

In production, this service evolves to:
- Global LightGBM or XGBoost model trained on full SAP history (millions of rows)
- Weekly retraining pipeline via Airflow or Prefect
- Model versioning via MLflow
- Redis caching of forecast results
- API key authentication on all endpoints
- CORS restricted to frontend domain only

The demo architecture is a direct subset of this — not a throwaway prototype.

---

*Project: Sonalika Demand Intelligence | Client: Sonalika Tractors (ITL)*
*Built by: Dexian India | Contact: Kiran Monish Mahalingam*
