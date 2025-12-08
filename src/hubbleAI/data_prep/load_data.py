"""
Data loading functions for hubbleAI.

This module handles loading raw data from files. In the future, this can be
extended to load from Denodo / Databricks / Reval without changing callers.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from hubbleAI.config import (
    DATA_RAW_DIR,
    ACTUALS_FILENAME,
    LP_FILENAME,
    FX_FILENAME,
)


def load_actuals(
    file_path: Optional[Path] = None,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load and filter actuals data.

    Args:
        file_path: Path to actuals CSV. If None, uses default from config.
        as_of_date: Optional as-of date to filter data up to.

    Returns:
        DataFrame with filtered actuals data.
    """
    if file_path is None:
        file_path = DATA_RAW_DIR / ACTUALS_FILENAME

    actuals_df = pd.read_csv(file_path)

    # Standardize entity codes
    actuals_df["Entity"] = actuals_df["Entity"].astype(str).replace("57", "057")
    actuals_df["Entity"] = actuals_df["Entity"].astype(str).replace("10H2", "14C1")
    actuals_df["Entity"] = actuals_df["Entity"].astype(str).replace("10G6", "17C7")

    # Filter columns
    actuals_filtered = actuals_df[
        [
            "Entity",
            "Value Date",
            "Amount Functional Currency",
            "Liquidity Group",
            "Counterpart",
            "Status",
            "ISO Country Code",
        ]
    ].copy()

    # Keep only TRR and TRP
    actuals_filtered = actuals_filtered[
        actuals_filtered["Liquidity Group"].isin(["TRR", "TRP"])
    ].copy()

    # Normalize types / clean strings
    actuals_filtered["Counterpart"] = (
        actuals_filtered["Counterpart"].astype(str).str.strip()
    )
    actuals_filtered["Status"] = actuals_filtered["Status"].astype(str).str.strip()
    actuals_filtered["ISO Country Code"] = (
        actuals_filtered["ISO Country Code"].astype(str).str.strip()
    )

    # Parse dates
    actuals_filtered["Value Date"] = pd.to_datetime(
        actuals_filtered["Value Date"], errors="coerce"
    )

    # Filter by as_of_date if provided
    if as_of_date is not None:
        actuals_filtered = actuals_filtered[
            actuals_filtered["Value Date"] <= pd.Timestamp(as_of_date)
        ]

    return actuals_filtered


def load_liquidity_plan(
    file_path: Optional[Path] = None,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load and filter liquidity plan data.

    Args:
        file_path: Path to LP CSV. If None, uses default from config.
        as_of_date: Optional as-of date to filter data up to.

    Returns:
        DataFrame with filtered LP data.
    """
    if file_path is None:
        file_path = DATA_RAW_DIR / LP_FILENAME

    lp_df = pd.read_csv(file_path)

    # Standardize entity codes
    lp_df["Entity"] = lp_df["Entity"].astype(str).replace("57", "057")
    lp_df["Entity"] = lp_df["Entity"].astype(str).replace("10H2", "14C1")
    lp_df["Entity"] = lp_df["Entity"].astype(str).replace("10G6", "17C7")

    # Filter columns
    lp_filtered = lp_df[
        [
            "Entity",
            "Entity Name",
            "Liquidity Group/Super Liquidity Group",
            "Year Title",
            "Item's Date",
            "Amount",
            "Currency",
            "Plan Currency",
            "Amount in plan currency",
            "Rate",
            "Comment",
        ]
    ].copy()

    # Keep only TRR and TRP
    lp_filtered = lp_filtered[
        lp_filtered["Liquidity Group/Super Liquidity Group"].isin(["TRR", "TRP"])
    ].copy()

    # Handle duplicate year titles (e.g., 2022/CW21/2 -> 2022/CW21)
    lp_filtered = lp_filtered[lp_filtered["Year Title"] != "2022/CW21"]
    lp_filtered["Year Title"] = lp_filtered["Year Title"].str.extract(
        r"^(\d{4}/CW\d{2})"
    )[0]

    # Parse item date
    lp_filtered["Item_Date"] = pd.to_datetime(lp_filtered["Item's Date"])

    # Filter by as_of_date if provided
    if as_of_date is not None:
        lp_filtered = lp_filtered[
            lp_filtered["Item_Date"] <= pd.Timestamp(as_of_date)
        ]

    return lp_filtered


def load_fx_rates(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load FX rates data.

    Args:
        file_path: Path to FX rates CSV. If None, uses default from config.

    Returns:
        DataFrame with FX rates indexed by date.
    """
    if file_path is None:
        file_path = DATA_RAW_DIR / FX_FILENAME

    fx_df = pd.read_csv(file_path)
    fx_df["Date"] = pd.to_datetime(fx_df["Date"])

    return fx_df
