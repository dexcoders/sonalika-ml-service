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
