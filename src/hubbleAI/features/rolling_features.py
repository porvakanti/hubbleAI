"""
Rolling window feature engineering for hubbleAI.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from hubbleAI.config import ROLLING_WINDOWS


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str = "total_amount_week",
    group_cols: Union[Sequence[str], tuple] = ("entity", "liquidity_group"),
    windows: Tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling statistics over given windows for `value_col` within each group.

    Metrics per window:
      - mean, std, sum, min, max, median, coefficient of variation (cv = std/mean)

    Uses shift(1) so that the current week's value is NOT included
    in the window (avoids data leakage).

    Args:
        df: Input DataFrame.
        value_col: Column to compute rolling stats from.
        group_cols: Columns to group by.
        windows: Window sizes to compute stats for.

    Returns:
        DataFrame with rolling features added.
    """
    df = df.copy()
    sort_cols = list(group_cols) + ["week_start"]
    df = df.sort_values(sort_cols)

    grouped = df.groupby(list(group_cols), group_keys=False)

    for w in windows:
        # Shift to exclude current week from rolling window
        shifted = grouped[value_col].apply(lambda s: s.shift(1))
        roll = shifted.rolling(window=w, min_periods=1)

        mean_ = roll.mean()
        std_ = roll.std()
        sum_ = roll.sum()
        min_ = roll.min()
        max_ = roll.max()
        med_ = roll.median()

        # Coefficient of variation = std / mean (handle division by zero)
        cv_ = std_ / mean_.replace(0, np.nan)

        df[f"roll_{w}w_mean"] = mean_
        df[f"roll_{w}w_std"] = std_
        df[f"roll_{w}w_sum"] = sum_
        df[f"roll_{w}w_min"] = min_
        df[f"roll_{w}w_max"] = max_
        df[f"roll_{w}w_median"] = med_
        df[f"roll_{w}w_cv"] = cv_

    return df
