"""
Feature building orchestration for hubbleAI.

This module ties together all feature engineering steps.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from hubbleAI.config import (
    DROP_COLS,
    ID_COLS,
    TRP_EXTRA_FEATURES,
    ALL_LP_COLS,
    HORIZONS,
)
from hubbleAI.features.lag_features import add_lag_features
from hubbleAI.features.rolling_features import add_rolling_features
from hubbleAI.features.calendar_features import add_calendar_features
from hubbleAI.features.trend_features import add_trend_features
from hubbleAI.features.lp_features import add_lp_accuracy_features


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for the forecasting model.

    This function applies all feature engineering steps in sequence:
    1. Lag features (1-52 weeks)
    2. Rolling features (4, 8, 13, 26, 52 weeks)
    3. Calendar features
    4. Trend features
    5. LP accuracy features

    Args:
        df: Merged DataFrame from prepare_weekly_data.

    Returns:
        DataFrame with all features added.
    """
    df = df.copy()

    # Add lag features
    df = add_lag_features(df)

    # Add rolling features
    df = add_rolling_features(df)

    # Add calendar features
    df = add_calendar_features(df)

    # Add trend features
    df = add_trend_features(df)

    # Add LP accuracy features
    df = add_lp_accuracy_features(df)

    return df


def get_base_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Get the base feature columns (excluding LP, targets, and ID columns).

    IMPORTANT: The returned list does NOT include LP forecast columns.
    LP columns are injected dynamically per horizon using get_feature_cols_for_horizon().

    Args:
        df: DataFrame with all features.

    Returns:
        List of base feature column names.
    """
    # Target columns
    target_cols = [f"y_h{h}" for h in HORIZONS]

    # Columns to exclude
    exclude_cols = (
        DROP_COLS
        + ID_COLS
        + target_cols
        + TRP_EXTRA_FEATURES
        + ALL_LP_COLS
    )

    # Get feature columns (everything not in exclude list)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols


def get_trp_extra_features(df: pd.DataFrame) -> List[str]:
    """
    Get TRP-specific extra features that exist in the DataFrame.

    Args:
        df: DataFrame with features.

    Returns:
        List of TRP extra feature column names present in df.
    """
    return [f for f in TRP_EXTRA_FEATURES if f in df.columns]
