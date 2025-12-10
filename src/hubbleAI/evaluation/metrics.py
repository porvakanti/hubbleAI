"""
Evaluation metrics for hubbleAI.

Supports WAPE, MAE, RMSE, and direction accuracy.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Weighted Absolute Percentage Error.

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        WAPE value.
    """
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def wape_series(actual: pd.Series, pred: pd.Series, eps: float = 1e-6) -> float:
    """
    Compute WAPE from pandas Series inputs.

    WAPE = sum(|actual - pred|) / sum(|actual|)

    Args:
        actual: Actual values as pandas Series.
        pred: Predicted values as pandas Series.
        eps: Small constant to avoid division by zero.

    Returns:
        WAPE value as float.
    """
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return np.nan
    y_true = actual[mask].values
    y_pred = pred[mask].values
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return float(mean_absolute_error(y_true, y_pred))


def mae_series(actual: pd.Series, pred: pd.Series) -> float:
    """
    Compute MAE from pandas Series inputs.

    MAE = mean(|actual - pred|)

    Args:
        actual: Actual values as pandas Series.
        pred: Predicted values as pandas Series.

    Returns:
        MAE value as float.
    """
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask].values - pred[mask].values)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray = None,
) -> float:
    """
    Compute direction accuracy (correct sign of change vs previous).

    If y_prev is not provided, computes accuracy based on sign of y_true and y_pred.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        y_prev: Previous actual values (optional).

    Returns:
        Direction accuracy as a fraction (0-1).
    """
    if y_prev is not None:
        true_dir = np.sign(y_true - y_prev)
        pred_dir = np.sign(y_pred - y_prev)
    else:
        true_dir = np.sign(y_true)
        pred_dir = np.sign(y_pred)

    return float(np.mean(true_dir == pred_dir))


def directional_accuracy_series(
    actual: pd.Series,
    pred: pd.Series,
    prev_actual: pd.Series,
) -> float:
    """
    Compute directional accuracy from pandas Series inputs.

    Direction accuracy measures whether the predicted direction of change
    (increase/decrease) from the previous actual matches the true direction.

    Formula:
        directional_accuracy = fraction of rows where:
            sign(pred - prev_actual) == sign(actual - prev_actual)

    Interpretation:
        - 1.0 = perfect direction prediction
        - 0.5 = random (no better than coin flip)
        - 0.0 = perfectly wrong direction

    Notes:
        - Rows where prev_actual is NaN are excluded from computation
        - Rows where actual or pred is NaN are excluded
        - If no valid rows exist, returns NaN

    Args:
        actual: Actual values for the target period.
        pred: Predicted values for the target period.
        prev_actual: Actual values from the previous period (lag-1 actual).

    Returns:
        Directional accuracy as a fraction (0-1), or NaN if no valid rows.
    """
    # Create mask for valid rows (all three values must be non-null)
    mask = actual.notna() & pred.notna() & prev_actual.notna()

    if mask.sum() == 0:
        return np.nan

    y_true = actual[mask].values
    y_pred = pred[mask].values
    y_prev = prev_actual[mask].values

    # Compute direction of change
    true_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)

    return float(np.mean(true_direction == pred_direction))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all standard metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with 'mae', 'rmse', 'wape' keys.
    """
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Backtest Metrics Computation
# ---------------------------------------------------------------------------


def compute_group_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
    prev_actual_col: str = "prev_actual",
) -> Dict[str, float]:
    """
    Compute ML and LP metrics for a group of rows.

    Args:
        df: DataFrame with actual, ML prediction, and LP prediction columns.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.
        prev_actual_col: Column name for previous period actuals.

    Returns:
        Dictionary with metric values:
        - n_obs: Number of observations
        - ml_wape, ml_mae, ml_directional_accuracy
        - lp_wape, lp_mae, lp_directional_accuracy
    """
    result = {"n_obs": len(df)}

    # ML metrics
    actual = df[actual_col]
    ml_pred = df[ml_pred_col]

    result["ml_wape"] = wape_series(actual, ml_pred)
    result["ml_mae"] = mae_series(actual, ml_pred)

    # LP metrics (may have NaN for h>=5)
    lp_pred = df[lp_pred_col] if lp_pred_col in df.columns else pd.Series([np.nan] * len(df))

    result["lp_wape"] = wape_series(actual, lp_pred)
    result["lp_mae"] = mae_series(actual, lp_pred)

    # Directional accuracy (requires prev_actual)
    if prev_actual_col in df.columns:
        prev_actual = df[prev_actual_col]
        result["ml_directional_accuracy"] = directional_accuracy_series(
            actual, ml_pred, prev_actual
        )
        result["lp_directional_accuracy"] = directional_accuracy_series(
            actual, lp_pred, prev_actual
        )
    else:
        result["ml_directional_accuracy"] = np.nan
        result["lp_directional_accuracy"] = np.nan

    return result


def compute_metrics_by_lg(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute LG-level metrics: grouped by (week_start, liquidity_group, horizon).

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with metrics per (week_start, liquidity_group, horizon).
    """
    # Add prev_actual for directional accuracy
    df = _add_prev_actual(df)

    results = []
    group_cols = ["week_start", "liquidity_group", "horizon"]

    for keys, group in df.groupby(group_cols, observed=True):
        week_start, lg, horizon = keys
        metrics = compute_group_metrics(
            group,
            actual_col=actual_col,
            ml_pred_col=ml_pred_col,
            lp_pred_col=lp_pred_col,
        )
        metrics["week_start"] = week_start
        metrics["liquidity_group"] = lg
        metrics["horizon"] = horizon
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # Reorder columns
    col_order = ["week_start", "liquidity_group", "horizon", "n_obs",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_metrics_by_entity(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute Entity-level metrics: grouped by (week_start, entity, liquidity_group, horizon).

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with metrics per (week_start, entity, liquidity_group, horizon).
    """
    # Add prev_actual for directional accuracy
    df = _add_prev_actual(df)

    results = []
    group_cols = ["week_start", "entity", "liquidity_group", "horizon"]

    for keys, group in df.groupby(group_cols, observed=True):
        week_start, entity, lg, horizon = keys
        metrics = compute_group_metrics(
            group,
            actual_col=actual_col,
            ml_pred_col=ml_pred_col,
            lp_pred_col=lp_pred_col,
        )
        metrics["week_start"] = week_start
        metrics["entity"] = entity
        metrics["liquidity_group"] = lg
        metrics["horizon"] = horizon
        results.append(metrics)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # Reorder columns
    col_order = ["week_start", "entity", "liquidity_group", "horizon", "n_obs",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_metrics_net(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute Net-level metrics: TRR + TRP summed, grouped by (week_start, horizon).

    This aggregates across both liquidity groups (TRR and TRP) by summing
    actual values and predictions, then computing metrics on the sums.

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with net metrics per (week_start, horizon).
    """
    # Aggregate by summing across entities and LGs per (week_start, horizon)
    agg_cols = {
        actual_col: "sum",
        ml_pred_col: "sum",
    }
    if lp_pred_col in df.columns:
        agg_cols[lp_pred_col] = "sum"

    net_df = df.groupby(["week_start", "horizon"], observed=True).agg(agg_cols).reset_index()

    # Add prev_actual for directional accuracy (sum of previous week's actuals)
    net_df = net_df.sort_values(["horizon", "week_start"])
    net_df["prev_actual"] = net_df.groupby("horizon", observed=True)[actual_col].shift(1)

    results = []
    for keys, group in net_df.groupby(["week_start", "horizon"], observed=True):
        week_start, horizon = keys

        actual = group[actual_col].iloc[0]
        ml_pred = group[ml_pred_col].iloc[0]
        lp_pred = group[lp_pred_col].iloc[0] if lp_pred_col in group.columns else np.nan
        prev = group["prev_actual"].iloc[0]

        metrics = {
            "week_start": week_start,
            "horizon": horizon,
            "n_obs": 1,  # Net is a single aggregated value
            "net_actual": actual,
            "net_ml_pred": ml_pred,
            "net_lp_pred": lp_pred,
        }

        # Compute metrics
        if pd.notna(actual) and pd.notna(ml_pred):
            metrics["ml_wape"] = abs(actual - ml_pred) / (abs(actual) + 1e-6)
            metrics["ml_mae"] = abs(actual - ml_pred)
        else:
            metrics["ml_wape"] = np.nan
            metrics["ml_mae"] = np.nan

        if pd.notna(actual) and pd.notna(lp_pred):
            metrics["lp_wape"] = abs(actual - lp_pred) / (abs(actual) + 1e-6)
            metrics["lp_mae"] = abs(actual - lp_pred)
        else:
            metrics["lp_wape"] = np.nan
            metrics["lp_mae"] = np.nan

        # Directional accuracy
        if pd.notna(prev):
            if pd.notna(ml_pred):
                metrics["ml_directional_accuracy"] = float(
                    np.sign(actual - prev) == np.sign(ml_pred - prev)
                )
            else:
                metrics["ml_directional_accuracy"] = np.nan
            if pd.notna(lp_pred):
                metrics["lp_directional_accuracy"] = float(
                    np.sign(actual - prev) == np.sign(lp_pred - prev)
                )
            else:
                metrics["lp_directional_accuracy"] = np.nan
        else:
            metrics["ml_directional_accuracy"] = np.nan
            metrics["lp_directional_accuracy"] = np.nan

        results.append(metrics)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["week_start", "horizon", "n_obs", "net_actual", "net_ml_pred", "net_lp_pred",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_metrics_net_entity(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute Net-Entity-level metrics: TRR + TRP summed per entity,
    grouped by (week_start, entity, horizon).

    For each entity, sums TRR and TRP values (if both exist) and computes metrics.

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with net-entity metrics per (week_start, entity, horizon).
    """
    # Aggregate by summing TRR+TRP for each entity per (week_start, entity, horizon)
    agg_cols = {
        actual_col: "sum",
        ml_pred_col: "sum",
    }
    if lp_pred_col in df.columns:
        agg_cols[lp_pred_col] = "sum"

    net_entity_df = df.groupby(
        ["week_start", "entity", "horizon"], observed=True
    ).agg(agg_cols).reset_index()

    # Add prev_actual for directional accuracy
    net_entity_df = net_entity_df.sort_values(["entity", "horizon", "week_start"])
    net_entity_df["prev_actual"] = net_entity_df.groupby(
        ["entity", "horizon"], observed=True
    )[actual_col].shift(1)

    results = []
    for keys, group in net_entity_df.groupby(["week_start", "entity", "horizon"], observed=True):
        week_start, entity, horizon = keys

        actual = group[actual_col].iloc[0]
        ml_pred = group[ml_pred_col].iloc[0]
        lp_pred = group[lp_pred_col].iloc[0] if lp_pred_col in group.columns else np.nan
        prev = group["prev_actual"].iloc[0]

        metrics = {
            "week_start": week_start,
            "entity": entity,
            "horizon": horizon,
            "n_obs": 1,
            "net_actual": actual,
            "net_ml_pred": ml_pred,
            "net_lp_pred": lp_pred,
        }

        # Compute metrics
        if pd.notna(actual) and pd.notna(ml_pred):
            metrics["ml_wape"] = abs(actual - ml_pred) / (abs(actual) + 1e-6)
            metrics["ml_mae"] = abs(actual - ml_pred)
        else:
            metrics["ml_wape"] = np.nan
            metrics["ml_mae"] = np.nan

        if pd.notna(actual) and pd.notna(lp_pred):
            metrics["lp_wape"] = abs(actual - lp_pred) / (abs(actual) + 1e-6)
            metrics["lp_mae"] = abs(actual - lp_pred)
        else:
            metrics["lp_wape"] = np.nan
            metrics["lp_mae"] = np.nan

        # Directional accuracy
        if pd.notna(prev):
            if pd.notna(ml_pred):
                metrics["ml_directional_accuracy"] = float(
                    np.sign(actual - prev) == np.sign(ml_pred - prev)
                )
            else:
                metrics["ml_directional_accuracy"] = np.nan
            if pd.notna(lp_pred):
                metrics["lp_directional_accuracy"] = float(
                    np.sign(actual - prev) == np.sign(lp_pred - prev)
                )
            else:
                metrics["lp_directional_accuracy"] = np.nan
        else:
            metrics["ml_directional_accuracy"] = np.nan
            metrics["lp_directional_accuracy"] = np.nan

        results.append(metrics)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["week_start", "entity", "horizon", "n_obs", "net_actual", "net_ml_pred", "net_lp_pred",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def _add_prev_actual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prev_actual column: the lag-1 actual value for directional accuracy.

    For each (entity, liquidity_group, horizon) group, shift actual_value by 1 week.

    Args:
        df: Backtest predictions DataFrame.

    Returns:
        DataFrame with prev_actual column added.
    """
    df = df.copy()
    df = df.sort_values(["entity", "liquidity_group", "horizon", "week_start"])
    df["prev_actual"] = df.groupby(
        ["entity", "liquidity_group", "horizon"], observed=True
    )["actual_value"].shift(1)
    return df
