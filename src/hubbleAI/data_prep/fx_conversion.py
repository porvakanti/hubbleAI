"""
FX conversion utilities for hubbleAI.

Converts LP amounts to EUR using historical FX rates.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np


def build_fx_lookup(fx_df: pd.DataFrame) -> Dict:
    """
    Build a lookup dictionary for FX rates by date.

    Args:
        fx_df: DataFrame with Date, USD, CHF columns.

    Returns:
        Dictionary mapping date to {currency: rate}.
    """
    fx_df = fx_df.copy()
    fx_df["Date"] = pd.to_datetime(fx_df["Date"])
    return fx_df.set_index("Date")[["USD", "CHF"]].to_dict("index")


def convert_to_eur(
    row: pd.Series,
    fx_lookup: Dict,
    fx_min_date: pd.Timestamp,
) -> float:
    """
    Convert a single LP row amount to EUR.

    Args:
        row: Row from LP DataFrame with Plan Currency, Amount in plan currency,
             Amount, and Item's Date columns.
        fx_lookup: Dictionary from build_fx_lookup.
        fx_min_date: Minimum date in FX data.

    Returns:
        Amount in EUR.
    """
    if row["Plan Currency"] == "EUR":
        return row["Amount in plan currency"]

    # Need conversion
    item_date = pd.to_datetime(row["Item's Date"])
    currency = row["Plan Currency"]

    # Find closest FX rate (handle weekends/missing dates)
    while item_date not in fx_lookup and item_date >= fx_min_date:
        item_date -= pd.Timedelta(days=1)

    if item_date in fx_lookup:
        rate = fx_lookup[item_date].get(currency)
        if rate and not pd.isna(rate):
            return row["Amount"] / float(rate)

    # Fallback: use Amount in plan currency as-is
    return row["Amount in plan currency"]


def convert_lp_to_eur(lp_df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert LP amounts to EUR and pivot to wide format.

    Args:
        lp_df: Filtered LP DataFrame from load_liquidity_plan.
        fx_df: FX rates DataFrame from load_fx_rates.

    Returns:
        Wide-format LP DataFrame with W1_Forecast through W4_Forecast columns.
    """
    lp_df = lp_df.copy()
    fx_lookup = build_fx_lookup(fx_df)
    fx_min_date = fx_df["Date"].min()

    # Convert amounts to EUR
    lp_df["Amount_EUR"] = lp_df.apply(
        lambda row: convert_to_eur(row, fx_lookup, fx_min_date), axis=1
    )

    # Rename liquidity group column
    lp_df.rename(
        columns={"Liquidity Group/Super Liquidity Group": "Liquidity_Group"},
        inplace=True,
    )

    # Sort and create week number within each group
    lp_df["Item_Date"] = pd.to_datetime(lp_df["Item's Date"])
    lp_df = lp_df.sort_values(
        ["Entity", "Entity Name", "Year Title", "Liquidity_Group", "Item_Date"]
    )

    lp_df["Week_Num"] = (
        lp_df.groupby(["Entity", "Entity Name", "Year Title", "Liquidity_Group"])
        .cumcount()
        + 1
    )

    # Keep only first 4 weeks
    lp_df = lp_df[lp_df["Week_Num"] <= 4]

    # Pivot to wide format
    lp_wide = lp_df.pivot_table(
        index=["Entity", "Entity Name", "Year Title", "Liquidity_Group"],
        columns="Week_Num",
        values="Amount_EUR",
        aggfunc="first",
    ).reset_index()

    # Rename columns
    lp_wide.columns = [
        "Entity",
        "Entity Name",
        "Year_Title",
        "Liquidity_Group",
        "W1_Forecast",
        "W2_Forecast",
        "W3_Forecast",
        "W4_Forecast",
    ]

    # Add availability flags
    for col in ["W1_Forecast", "W2_Forecast", "W3_Forecast", "W4_Forecast"]:
        lp_wide[f"{col}_Available"] = ~lp_wide[col].isna()

    lp_wide["Available_Forecast_Count"] = (
        lp_wide["W1_Forecast_Available"].astype(int)
        + lp_wide["W2_Forecast_Available"].astype(int)
        + lp_wide["W3_Forecast_Available"].astype(int)
        + lp_wide["W4_Forecast_Available"].astype(int)
    )

    return lp_wide


def yearweek_to_monday(yt: str) -> pd.Timestamp:
    """
    Convert Year Title (e.g., '2023/CW12') to Monday date.

    Args:
        yt: Year title string like '2023/CW12'.

    Returns:
        Timestamp for the Monday of that week.
    """
    from datetime import date as dt_date

    year, cw = yt.split("/CW")
    year = int(year)
    week = int(cw)
    return pd.Timestamp(dt_date.fromisocalendar(year, week, 1))
