"""
Main data preparation orchestration for hubbleAI.

This module ties together loading, filtering, and merging of all data sources.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from hubbleAI.config import TIER2_LIST, MIN_HISTORY_WEEKS
from hubbleAI.data_prep.load_data import (
    load_actuals,
    load_liquidity_plan,
    load_fx_rates,
)
from hubbleAI.data_prep.aggregation import (
    aggregate_actuals_weekly,
    build_trp_weekly_features,
    _add_week_has_month_middle,
    _add_week_has_day,
    _add_week_has_eom_cluster,
    _add_week_has_bom_cluster,
)
from hubbleAI.data_prep.fx_conversion import convert_lp_to_eur, yearweek_to_monday


def prepare_weekly_data(
    as_of_date: Optional[date] = None,
    actuals_path: Optional[Path] = None,
    lp_path: Optional[Path] = None,
    fx_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all data sources, merge them, and return the prepared dataset.

    Args:
        as_of_date: Optional as-of date to filter data up to.
        actuals_path: Path to actuals file (uses default if None).
        lp_path: Path to LP file (uses default if None).
        fx_path: Path to FX file (uses default if None).

    Returns:
        Tuple of:
        - merged_df: Full merged dataset with all features ready for ML
        - actuals_filtered: Raw filtered actuals (needed for TRP feature building)
    """
    # Load raw data
    actuals_filtered = load_actuals(actuals_path, as_of_date)
    lp_df = load_liquidity_plan(lp_path, as_of_date)
    fx_df = load_fx_rates(fx_path)

    # Aggregate actuals to weekly
    actuals_weekly = aggregate_actuals_weekly(actuals_filtered)

    # Build TRP-specific features
    trp_weekly_features = build_trp_weekly_features(actuals_filtered)

    # Merge TRP features into weekly actuals
    actuals_weekly = actuals_weekly.merge(
        trp_weekly_features,
        on=["Entity", "Liquidity Group", "week_start"],
        how="left",
    )

    # Convert LP to EUR and pivot to wide format
    lp_wide = convert_lp_to_eur(lp_df, fx_df)

    # Add week_start to LP (shifted back by 7 days per notebook logic)
    lp_wide["week_start"] = lp_wide["Year_Title"].apply(yearweek_to_monday)
    lp_wide["week_start"] = pd.to_datetime(lp_wide["week_start"]) - pd.Timedelta(days=7)

    # Rename columns for consistency
    lp_wide = lp_wide.rename(
        columns={
            "Entity": "entity",
            "Entity Name": "entity_name",
            "Liquidity_Group": "liquidity_group",
            "Available_Forecast_Count": "available_forecast_count",
        }
    )
    actuals_weekly = actuals_weekly.rename(
        columns={
            "Entity": "entity",
            "Liquidity Group": "liquidity_group",
        }
    )

    # Merge actuals with LP using outer join to retain all LP predictions
    # even when actuals data is missing for a given week
    merged_df = actuals_weekly.merge(
        lp_wide,
        how="outer",
        on=["entity", "liquidity_group", "week_start"],
    )
    merged_df = merged_df.sort_values(["entity", "liquidity_group", "week_start"])

    # Fill total_amount_week with 0 for LP-only rows (no actuals)
    merged_df["total_amount_week"] = merged_df["total_amount_week"].fillna(0)

    # Compute calendar flags for LP-only rows that are missing them
    # These flags can be derived purely from week_start
    calendar_flag_cols = [
        "week_has_month_start",
        "week_has_month_end",
        "week_has_month_middle",
        "week_has_10th",
        "week_has_15th",
        "week_has_20th",
        "week_has_eom_cluster",
        "week_has_bom_cluster",
    ]

    # Identify rows missing calendar flags (LP-only rows)
    missing_calendar_mask = merged_df["week_has_month_start"].isna()

    if missing_calendar_mask.any():
        # Compute week_has_month_start: check if 1st of month falls in the week
        s = pd.to_datetime(merged_df.loc[missing_calendar_mask, "week_start"])
        week_end = s + pd.Timedelta(days=6)
        first_of_month = s.dt.to_period("M").dt.to_timestamp()
        # Check if 1st falls within [week_start, week_end]
        merged_df.loc[missing_calendar_mask, "week_has_month_start"] = (
            (first_of_month >= s) & (first_of_month <= week_end)
        ).astype(int).values

        # Compute week_has_month_end: check if last day of month falls in the week
        month_end = s.dt.to_period("M").dt.to_timestamp("M")
        merged_df.loc[missing_calendar_mask, "week_has_month_end"] = (
            (month_end >= s) & (month_end <= week_end)
        ).astype(int).values

    # Apply helper functions to fill remaining calendar flags for all rows
    # (they compute from week_start, so safe to apply to entire df)
    merged_df = _add_week_has_month_middle(merged_df)
    merged_df = _add_week_has_day(merged_df, 10, new_col="week_has_10th")
    merged_df = _add_week_has_day(merged_df, 15, new_col="week_has_15th")
    merged_df = _add_week_has_day(merged_df, 20, new_col="week_has_20th")
    merged_df = _add_week_has_eom_cluster(merged_df, window=4, new_col="week_has_eom_cluster")
    merged_df = _add_week_has_bom_cluster(merged_df, window=4, new_col="week_has_bom_cluster")

    # Compute history_weeks
    merged_df["history_weeks"] = merged_df.groupby(
        ["entity", "liquidity_group"]
    ).cumcount()

    # Identify tier2_dynamic: entity+LG combinations with zero actuals ever
    # These are brand new combinations that should be LP pass-through candidates
    actuals_sum_by_combo = merged_df.groupby(
        ["entity", "liquidity_group"]
    )["total_amount_week"].transform("sum")
    tier2_dynamic_mask = actuals_sum_by_combo == 0
    tier2_dynamic = set(
        merged_df.loc[tier2_dynamic_mask, ["entity", "liquidity_group"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )

    # Add tier column (includes both static TIER2_LIST and dynamic tier2)
    merged_df["tier"] = merged_df.apply(
        lambda row: (
            "Tier2"
            if (row["entity"], row["liquidity_group"]) in TIER2_LIST
            or (row["entity"], row["liquidity_group"]) in tier2_dynamic
            else "Tier1"
        ),
        axis=1,
    )

    # Convert to category types
    merged_df["entity"] = merged_df["entity"].astype("category")
    merged_df["liquidity_group"] = merged_df["liquidity_group"].astype("category")

    return merged_df, actuals_filtered


def filter_tier1_with_history(
    merged_df: pd.DataFrame,
    min_history: int = MIN_HISTORY_WEEKS,
) -> pd.DataFrame:
    """
    Filter to Tier-1 entities with sufficient history for ML training.

    Args:
        merged_df: Full merged DataFrame.
        min_history: Minimum history weeks required.

    Returns:
        Filtered DataFrame with Tier-1 entities having enough history.
    """
    tier1_df = merged_df[
        (merged_df["tier"] == "Tier1") & (merged_df["history_weeks"] >= min_history)
    ].copy()
    return tier1_df


def add_target_columns(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """
    Add target columns (y_h1, y_h2, ..., y_h8) for each horizon.

    Args:
        df: DataFrame with entity, liquidity_group, week_start, total_amount_week.
        horizons: List of horizons to create targets for (default 1-8).

    Returns:
        DataFrame with target columns added.
    """
    if horizons is None:
        from hubbleAI.config import HORIZONS
        horizons = HORIZONS

    df = df.copy()
    df = df.sort_values(["entity", "liquidity_group", "week_start"])

    for h in horizons:
        df[f"y_h{h}"] = df.groupby(["entity", "liquidity_group"])[
            "total_amount_week"
        ].shift(-h)

    return df
