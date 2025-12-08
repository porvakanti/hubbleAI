"""
Calendar feature engineering for hubbleAI.
"""

from __future__ import annotations

import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    week_start_col: str = "week_start",
) -> pd.DataFrame:
    """
    Add weekly calendar features based on `week_start`.

    Features added:
      - year, month, quarter
      - iso_week_of_year
      - is_quarter_start, is_quarter_end
      - is_year_start, is_year_end

    Args:
        df: Input DataFrame.
        week_start_col: Column containing week start dates.

    Returns:
        DataFrame with calendar features added.
    """
    df = df.copy()
    s = pd.to_datetime(df[week_start_col])

    df["year"] = s.dt.year
    df["month"] = s.dt.month
    df["quarter"] = s.dt.quarter
    df["iso_week_of_year"] = s.dt.isocalendar().week.astype("int32")

    df["is_quarter_start"] = s.dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = s.dt.is_quarter_end.astype(int)
    df["is_year_start"] = s.dt.is_year_start.astype(int)
    df["is_year_end"] = s.dt.is_year_end.astype(int)

    return df
