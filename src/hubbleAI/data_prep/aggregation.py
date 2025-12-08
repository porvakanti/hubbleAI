"""
Weekly aggregation functions for hubbleAI.

Aggregates daily actuals to weekly level and builds TRP-specific features.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def aggregate_actuals_weekly(actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily actuals to weekly level.

    Args:
        actuals_df: Filtered actuals DataFrame from load_actuals.

    Returns:
        Weekly aggregated DataFrame with entity, liquidity_group, week_start,
        total_amount_week, and calendar features.
    """
    df = actuals_df.copy()

    # Ensure Value Date is datetime
    df["Value Date"] = pd.to_datetime(df["Value Date"], errors="coerce")

    # Compute ISO week info
    df["iso_year"] = df["Value Date"].dt.isocalendar().year
    df["iso_week"] = df["Value Date"].dt.isocalendar().week

    # Compute week_start (Monday of the week)
    df["week_start"] = pd.to_datetime(
        df["iso_year"].astype(str)
        + "-W"
        + df["iso_week"].astype(str)
        + "-1",
        format="%G-W%V-%u",
    )

    # Calendar flags
    df["is_month_start"] = df["Value Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Value Date"].dt.is_month_end.astype(int)

    # Aggregate to weekly
    actuals_weekly = (
        df.groupby(["Entity", "Liquidity Group", "week_start"])
        .agg(
            total_amount_week=("Amount Functional Currency", "sum"),
            week_has_month_start=("is_month_start", "max"),
            week_has_month_end=("is_month_end", "max"),
        )
        .reset_index()
    )

    # Add week-has-month-middle flag
    actuals_weekly = _add_week_has_month_middle(actuals_weekly)

    # Add day-of-month flags
    actuals_weekly = _add_week_has_day(actuals_weekly, 10, new_col="week_has_10th")
    actuals_weekly = _add_week_has_day(actuals_weekly, 15, new_col="week_has_15th")
    actuals_weekly = _add_week_has_day(actuals_weekly, 20, new_col="week_has_20th")

    # Add EOM/BOM cluster flags
    actuals_weekly = _add_week_has_eom_cluster(
        actuals_weekly, window=4, new_col="week_has_eom_cluster"
    )
    actuals_weekly = _add_week_has_bom_cluster(
        actuals_weekly, window=4, new_col="week_has_bom_cluster"
    )

    return actuals_weekly


def _add_week_has_month_middle(
    weekly: pd.DataFrame,
    week_start_col: str = "week_start",
    new_col: str = "week_has_month_middle",
) -> pd.DataFrame:
    """Flag whether the week contains the middle of the month."""
    weekly = weekly.copy()
    s = pd.to_datetime(weekly[week_start_col])

    days_in_month = s.dt.days_in_month
    mid_day = days_in_month // 2

    first_of_month = s.dt.to_period("M").dt.to_timestamp()
    mid_date = first_of_month + pd.to_timedelta(mid_day - 1, unit="D")
    week_end = s + pd.Timedelta(days=6)

    weekly[new_col] = ((mid_date >= s) & (mid_date <= week_end)).astype(int)
    return weekly


def _add_week_has_day(
    weekly: pd.DataFrame,
    day: int,
    week_start_col: str = "week_start",
    new_col: str = None,
) -> pd.DataFrame:
    """Flag whether the week contains a specific calendar day of the month."""
    if new_col is None:
        new_col = f"week_has_{day}"

    weekly = weekly.copy()
    s = pd.to_datetime(weekly[week_start_col])

    first_of_month = s.dt.to_period("M").dt.to_timestamp()
    target_date = first_of_month + pd.to_timedelta(day - 1, unit="D")
    week_end = s + pd.Timedelta(days=6)

    weekly[new_col] = ((target_date >= s) & (target_date <= week_end)).astype(int)
    return weekly


def _add_week_has_eom_cluster(
    weekly: pd.DataFrame,
    week_start_col: str = "week_start",
    window: int = 4,
    new_col: str = "week_has_eom_cluster",
) -> pd.DataFrame:
    """Flag if the week contains any of the last N days of the month."""
    weekly = weekly.copy()
    s = pd.to_datetime(weekly[week_start_col])

    month_end = s.dt.to_period("M").dt.to_timestamp("M")
    week_end = s + pd.Timedelta(days=6)

    cluster_dates = [month_end - pd.Timedelta(days=i) for i in range(window)]

    flag = np.zeros(len(weekly), dtype=int)
    for d in cluster_dates:
        flag |= ((d >= s) & (d <= week_end)).astype(int)

    weekly[new_col] = flag
    return weekly


def _add_week_has_bom_cluster(
    weekly: pd.DataFrame,
    week_start_col: str = "week_start",
    window: int = 3,
    new_col: str = "week_has_bom_cluster",
) -> pd.DataFrame:
    """Flag if the week contains any of the first N days of the month."""
    weekly = weekly.copy()
    s = pd.to_datetime(weekly[week_start_col])

    first_of_month = s.dt.to_period("M").dt.to_timestamp()
    week_end = s + pd.Timedelta(days=6)

    cluster_dates = [first_of_month + pd.to_timedelta(i, unit="D") for i in range(window)]

    flag = np.zeros(len(weekly), dtype=int)
    for d in cluster_dates:
        flag |= ((d >= s) & (d <= week_end)).astype(int)

    weekly[new_col] = flag
    return weekly


def build_trp_weekly_features(actuals_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Build TRP-specific weekly features from raw transaction-level actuals.

    Args:
        actuals_filtered: Filtered actuals with week_start column already computed.

    Returns:
        DataFrame with TRP-specific features per entity/week.
    """
    df = actuals_filtered.copy()

    # Ensure week_start exists
    if "week_start" not in df.columns:
        df["iso_year"] = df["Value Date"].dt.isocalendar().year
        df["iso_week"] = df["Value Date"].dt.isocalendar().week
        df["week_start"] = pd.to_datetime(
            df["iso_year"].astype(str)
            + "-W"
            + df["iso_week"].astype(str)
            + "-1",
            format="%G-W%V-%u",
        )

    # Restrict to TRP only
    df_trp = df[df["Liquidity Group"] == "TRP"].copy()
    if df_trp.empty:
        return pd.DataFrame(
            columns=[
                "Entity",
                "Liquidity Group",
                "week_start",
                "trp_vendor_count",
                "trp_top_vendor_share",
                "trp_country_count",
                "trp_top_country_share",
                "trp_reconciled_share",
            ]
        )

    df_trp["amount_abs"] = df_trp["Amount Functional Currency"].abs()

    # Vendor-level stats
    vendor_week = (
        df_trp.groupby(["Entity", "Liquidity Group", "week_start", "Counterpart"], as_index=False)["amount_abs"]
        .sum()
        .rename(columns={"amount_abs": "vendor_amount"})
    )

    weekly_vendor = vendor_week.groupby(
        ["Entity", "Liquidity Group", "week_start"], as_index=False
    ).agg(
        trp_vendor_count=("Counterpart", "nunique"),
        trp_top_vendor_amount=("vendor_amount", "max"),
        trp_total_abs_vendor=("vendor_amount", "sum"),
    )

    weekly_vendor["trp_top_vendor_share"] = weekly_vendor["trp_top_vendor_amount"] / (
        weekly_vendor["trp_total_abs_vendor"] + 1e-6
    )

    # Country-level stats
    country_week = (
        df_trp.groupby(["Entity", "Liquidity Group", "week_start", "ISO Country Code"], as_index=False)["amount_abs"]
        .sum()
        .rename(columns={"amount_abs": "country_amount"})
    )

    weekly_country = country_week.groupby(
        ["Entity", "Liquidity Group", "week_start"], as_index=False
    ).agg(
        trp_country_count=("ISO Country Code", "nunique"),
        trp_top_country_amount=("country_amount", "max"),
    )

    weekly_totals = (
        df_trp.groupby(["Entity", "Liquidity Group", "week_start"], as_index=False)["amount_abs"]
        .sum()
        .rename(columns={"amount_abs": "trp_total_abs"})
    )

    weekly_country = weekly_country.merge(
        weekly_totals, on=["Entity", "Liquidity Group", "week_start"], how="left"
    )

    weekly_country["trp_top_country_share"] = weekly_country["trp_top_country_amount"] / (
        weekly_country["trp_total_abs"] + 1e-6
    )

    # Status-based stats
    df_trp["is_reconciled"] = (
        df_trp["Status"].str.contains("Reconciled", case=False, na=False).astype(int)
    )
    df_trp["amount_abs_reconciled"] = df_trp["amount_abs"] * df_trp["is_reconciled"]

    weekly_status = df_trp.groupby(
        ["Entity", "Liquidity Group", "week_start"], as_index=False
    ).agg(
        trp_total_abs_status=("amount_abs", "sum"),
        trp_reconciled_abs=("amount_abs_reconciled", "sum"),
    )

    weekly_status["trp_reconciled_share"] = weekly_status["trp_reconciled_abs"] / (
        weekly_status["trp_total_abs_status"] + 1e-6
    )

    # Combine all TRP weekly features
    trp_weekly = weekly_totals.copy()

    trp_weekly = trp_weekly.merge(
        weekly_vendor[["Entity", "Liquidity Group", "week_start", "trp_vendor_count", "trp_top_vendor_share"]],
        on=["Entity", "Liquidity Group", "week_start"],
        how="left",
    )

    trp_weekly = trp_weekly.merge(
        weekly_country[["Entity", "Liquidity Group", "week_start", "trp_country_count", "trp_top_country_share"]],
        on=["Entity", "Liquidity Group", "week_start"],
        how="left",
    )

    trp_weekly = trp_weekly.merge(
        weekly_status[["Entity", "Liquidity Group", "week_start", "trp_reconciled_share"]],
        on=["Entity", "Liquidity Group", "week_start"],
        how="left",
    )

    # Drop intermediate columns
    trp_weekly = trp_weekly.drop(columns=["trp_total_abs"], errors="ignore")

    return trp_weekly
