"""
Inventory optimisation engine.

Given:
  - Current inventory snapshot (from Supabase inventory_current table)
  - Demand forecasts for next 6 months (from XGBoost)

Computes:
  - Health status per region/model (overstock / understock / healthy)
  - Days of cover (how many days current stock will last at forecast demand rate)
  - Rebalancing recommendations (move X units from region A to region B)
  - Financial impact (Rs Cr at risk from overstock carrying cost or lost sales)
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

    forecasts: dict keyed by (region, model_name) -> forecast response from XGBoost
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
    Rank by Rs saving — highest impact first.
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
                    f"Rs{saving_cr:.2f} Cr."
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
