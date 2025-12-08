"""
Lag feature engineering for hubbleAI.
"""

from __future__ import annotations

from typing import Sequence, Union

import pandas as pd

from hubbleAI.config import LAG_WEEKS


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "total_amount_week",
    group_cols: Union[Sequence[str], tuple] = ("entity", "liquidity_group"),
    n_lags: int = LAG_WEEKS,
    prefix: str = "lag",
) -> pd.DataFrame:
    """
    Add lag features for `value_col` within each group in `group_cols`.

    Creates columns like: lag_1w_total, lag_2w_total, ..., lag_52w_total.

    Args:
        df: Input DataFrame.
        value_col: Column to create lags from.
        group_cols: Columns to group by for lagging.
        n_lags: Number of lag periods to create.
        prefix: Prefix for lag column names.

    Returns:
        DataFrame with lag features added.
    """
    df = df.copy()
    sort_cols = list(group_cols) + ["week_start"]
    df = df.sort_values(sort_cols)

    for lag in range(1, n_lags + 1):
        df[f"{prefix}_{lag}w_total"] = df.groupby(list(group_cols))[value_col].shift(lag)

    return df
