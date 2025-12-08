"""
Liquidity Plan (LP) feature engineering for hubbleAI.

This module handles horizon-specific LP feature injection as per Claude.md:
  - H1 → W1_Forecast only
  - H2 → W2_Forecast only
  - H3 → W3_Forecast only
  - H4 → W4_Forecast only
  - H5-H8 → no LP feature
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import pandas as pd

from hubbleAI.config import LP_FORECAST_COLS, LP_ACCURACY_WINDOW


def get_feature_cols_for_horizon(
    horizon: int,
    base_feature_cols: List[str],
    all_cols: List[str] = None,
) -> List[str]:
    """
    Build feature column list for a specific horizon, injecting horizon-specific LP.

    IMPORTANT: Base feature_cols should NOT contain any LP columns.
    LP feature(s) are injected dynamically per horizon via this helper.

    Args:
        horizon: Forecast horizon (1-8).
        base_feature_cols: List of base feature columns (without LP columns).
        all_cols: All available columns in the DataFrame (to verify LP col exists).

    Returns:
        List of feature columns including horizon-specific LP column (if applicable).
    """
    cols = list(base_feature_cols)

    # Get the LP column for this horizon (if any)
    lp_col = LP_FORECAST_COLS.get(horizon)

    # Add LP column if it exists for this horizon
    if lp_col is not None:
        if all_cols is None or lp_col in all_cols:
            cols.append(lp_col)

    # Remove 'split' column if present (not a feature)
    cols = [x for x in cols if x != "split"]

    return cols


def add_lp_accuracy_features(
    df: pd.DataFrame,
    value_col: str = "total_amount_week",
    group_cols: Union[Sequence[str], tuple] = ("entity", "liquidity_group"),
    horizons: Tuple[int, ...] = (1, 2, 3, 4),
    lp_prefix: str = "W",
    window_bias: int = LP_ACCURACY_WINDOW,
) -> pd.DataFrame:
    """
    Add historical liquidity plan accuracy features.

    For each horizon h in horizons:
      - lp_W{h}_error      = actual_t - W{h}_Forecast at t-h
      - lp_W{h}_abs_error  = |error|
      - lp_W{h}_bias_12w   = rolling mean(error) over past 12 weeks (shifted)
      - lp_W{h}_mae_12w    = rolling mean(abs_error) over past 12 weeks (shifted)

    All calculations are done within each (entity, liquidity_group) group.

    Args:
        df: Input DataFrame with LP forecast columns.
        value_col: Actual value column.
        group_cols: Columns to group by.
        horizons: LP horizons to compute accuracy for.
        lp_prefix: Prefix for LP columns (default "W").
        window_bias: Rolling window for bias/MAE computation.

    Returns:
        DataFrame with LP accuracy features added.
    """
    df = df.copy()
    sort_cols = list(group_cols) + ["week_start"]
    df = df.sort_values(sort_cols)

    grouped = df.groupby(list(group_cols), group_keys=False)

    for h in horizons:
        fc_col = f"{lp_prefix}{h}_Forecast"
        if fc_col not in df.columns:
            continue

        error_col = f"lp_W{h}_error"
        abs_error_col = f"lp_W{h}_abs_error"

        # Forecast made h weeks ago for this week's actual
        shifted_fc = grouped[fc_col].shift(h)
        df[error_col] = df[value_col] - shifted_fc
        df[abs_error_col] = df[error_col].abs()

        # Rolling bias and MAE (only past data)
        df[f"lp_W{h}_bias_{window_bias}w"] = grouped[error_col].apply(
            lambda s: s.shift(1).rolling(window=window_bias, min_periods=4).mean()
        )
        df[f"lp_W{h}_mae_{window_bias}w"] = grouped[abs_error_col].apply(
            lambda s: s.shift(1).rolling(window=window_bias, min_periods=4).mean()
        )

    return df
