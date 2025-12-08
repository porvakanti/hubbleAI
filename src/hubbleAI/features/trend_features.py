"""
Trend feature engineering for hubbleAI.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from hubbleAI.config import TREND_WINDOWS


def add_trend_features(
    df: pd.DataFrame,
    value_col: str = "total_amount_week",
    group_cols: Union[Sequence[str], tuple] = ("entity", "liquidity_group"),
    trend_windows: Tuple[int, ...] = TREND_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling trend features.

    Features added per window:
      - trend_{w}w_slope: linear regression slope over last w weeks (shifted)
      - trend_12w_accel: week-over-week change in 12w slope

    Uses shift(1) to ensure only past data is used.

    Args:
        df: Input DataFrame.
        value_col: Column to compute trends from.
        group_cols: Columns to group by.
        trend_windows: Window sizes for trend computation.

    Returns:
        DataFrame with trend features added.
    """
    df = df.copy()
    sort_cols = list(group_cols) + ["week_start"]
    df = df.sort_values(sort_cols)

    grouped = df.groupby(list(group_cols), group_keys=False)

    for w in trend_windows:
        slopes = grouped[value_col].apply(
            lambda s: s.shift(1)
            .rolling(window=w, min_periods=w)
            .apply(_rolling_slope, raw=True)
        )
        df[f"trend_{w}w_slope"] = slopes

    # Acceleration: change in 12-week slope week over week (within group)
    if 12 in trend_windows:
        df["trend_12w_accel"] = df.groupby(list(group_cols))["trend_12w_slope"].diff()

    return df


def _rolling_slope(arr: np.ndarray) -> float:
    """Compute slope of y ~ x over the window (x = 0..n-1)."""
    n = len(arr)
    if n < 2:
        return np.nan
    x = np.arange(n)
    y = arr.astype(float)
    if np.all(np.isnan(y)):
        return np.nan
    x_mean = x.mean()
    y_mean = np.nanmean(y)
    cov = np.nanmean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2)
    if var == 0:
        return 0.0
    return cov / var
