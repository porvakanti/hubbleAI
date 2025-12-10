"""
Evaluation metrics for hubbleAI.

Supports WAPE, MAE, RMSE, and direction accuracy.

WAPE Approaches:
- Standard WAPE: sum(|actual - pred|) / sum(|actual|) - measures total absolute error
- Aggregate-then-Error WAPE: |sum(actual) - sum(pred)| / |sum(actual)| - Treasury-aligned
  (errors can cancel out, which is relevant for total cash position)

For LG-level and Net-level metrics, we use Aggregate-then-Error WAPE because
Treasury cares about the total position, not individual entity errors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Standard Weighted Absolute Percentage Error.

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        WAPE value.
    """
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))


def wape_aggregate(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Aggregate-then-Error WAPE (Treasury-aligned).

    WAPE = |sum(y_true) - sum(y_pred)| / |sum(y_true)|

    This approach allows over-predictions and under-predictions to cancel out,
    which is relevant for Treasury's total cash position view.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        WAPE value.
    """
    actual_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)
    return float(np.abs(actual_sum - pred_sum) / (np.abs(actual_sum) + eps))


def wape_series(actual: pd.Series, pred: pd.Series, eps: float = 1e-6) -> float:
    """
    Compute Standard WAPE from pandas Series inputs.

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


def wape_aggregate_series(actual: pd.Series, pred: pd.Series, eps: float = 1e-6) -> float:
    """
    Compute Aggregate-then-Error WAPE from pandas Series inputs (Treasury-aligned).

    WAPE = |sum(actual) - sum(pred)| / |sum(actual)|

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
    actual_sum = actual[mask].sum()
    pred_sum = pred[mask].sum()
    return float(np.abs(actual_sum - pred_sum) / (np.abs(actual_sum) + eps))


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


def mae_aggregate_series(actual: pd.Series, pred: pd.Series) -> float:
    """
    Compute MAE on aggregated sums (absolute error on totals).

    MAE = |sum(actual) - sum(pred)|

    Args:
        actual: Actual values as pandas Series.
        pred: Predicted values as pandas Series.

    Returns:
        Absolute error on the aggregated sums.
    """
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.abs(actual[mask].sum() - pred[mask].sum()))


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


def compute_group_metrics_aggregate(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
    prev_actual_col: str = "prev_actual",
) -> Dict[str, float]:
    """
    Compute ML and LP metrics for a group using Aggregate-then-Error approach.

    This is Treasury-aligned: sum all actuals and predictions first,
    then compute error on the aggregated totals.

    Args:
        df: DataFrame with actual, ML prediction, and LP prediction columns.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.
        prev_actual_col: Column name for previous period actuals.

    Returns:
        Dictionary with metric values:
        - n_obs: Number of observations
        - actual_sum, ml_pred_sum, lp_pred_sum: Aggregated sums
        - ml_wape, ml_mae, ml_directional_accuracy
        - lp_wape, lp_mae, lp_directional_accuracy
    """
    result = {"n_obs": len(df)}

    # Get values
    actual = df[actual_col]
    ml_pred = df[ml_pred_col]
    lp_pred = df[lp_pred_col] if lp_pred_col in df.columns else pd.Series([np.nan] * len(df), index=df.index)

    # Aggregated sums
    result["actual_sum"] = actual.sum() if actual.notna().any() else np.nan
    result["ml_pred_sum"] = ml_pred.sum() if ml_pred.notna().any() else np.nan
    result["lp_pred_sum"] = lp_pred.sum() if lp_pred.notna().any() else np.nan

    # ML metrics (aggregate-then-error)
    result["ml_wape"] = wape_aggregate_series(actual, ml_pred)
    result["ml_mae"] = mae_aggregate_series(actual, ml_pred)

    # LP metrics (aggregate-then-error)
    result["lp_wape"] = wape_aggregate_series(actual, lp_pred)
    result["lp_mae"] = mae_aggregate_series(actual, lp_pred)

    # Directional accuracy on aggregated sums
    # For this, we need the previous period's aggregated actual
    if prev_actual_col in df.columns and df[prev_actual_col].notna().any():
        prev_sum = df[prev_actual_col].sum()
        actual_sum = result["actual_sum"]
        ml_sum = result["ml_pred_sum"]
        lp_sum = result["lp_pred_sum"]

        if pd.notna(prev_sum) and pd.notna(actual_sum) and pd.notna(ml_sum):
            result["ml_directional_accuracy"] = float(
                np.sign(actual_sum - prev_sum) == np.sign(ml_sum - prev_sum)
            )
        else:
            result["ml_directional_accuracy"] = np.nan

        if pd.notna(prev_sum) and pd.notna(actual_sum) and pd.notna(lp_sum):
            result["lp_directional_accuracy"] = float(
                np.sign(actual_sum - prev_sum) == np.sign(lp_sum - prev_sum)
            )
        else:
            result["lp_directional_accuracy"] = np.nan
    else:
        result["ml_directional_accuracy"] = np.nan
        result["lp_directional_accuracy"] = np.nan

    return result


def compute_metrics_by_lg(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
    include_passthrough: bool = True,
) -> pd.DataFrame:
    """
    Compute LG-level metrics: grouped by (week_start, liquidity_group, horizon).

    Uses Aggregate-then-Error WAPE (Treasury-aligned):
    - Sum actuals and predictions across all entities in each group
    - Compute WAPE on the aggregated totals

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.
        include_passthrough: If False, exclude Tier-2 passthrough rows (clean ML metrics).

    Returns:
        DataFrame with metrics per (week_start, liquidity_group, horizon).
    """
    df = df.copy()

    # Filter out passthroughs if requested (for clean ML metrics)
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    if df.empty:
        return pd.DataFrame()

    # Add prev_actual for directional accuracy (per entity first, then we'll aggregate)
    df = _add_prev_actual(df)

    results = []
    group_cols = ["week_start", "liquidity_group", "horizon"]

    for keys, group in df.groupby(group_cols, observed=True):
        week_start, lg, horizon = keys
        metrics = compute_group_metrics_aggregate(
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
                 "actual_sum", "ml_pred_sum", "lp_pred_sum",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_metrics_by_entity(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
    include_passthrough: bool = True,
) -> pd.DataFrame:
    """
    Compute Entity-level metrics: grouped by (week_start, entity, liquidity_group, horizon).

    At entity level, each group has 1 row, so standard WAPE and aggregate WAPE are equivalent.

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.
        include_passthrough: If False, exclude Tier-2 passthrough rows.

    Returns:
        DataFrame with metrics per (week_start, entity, liquidity_group, horizon).
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    if df.empty:
        return pd.DataFrame()

    # Add prev_actual for directional accuracy
    df = _add_prev_actual(df)

    results = []
    group_cols = ["week_start", "entity", "liquidity_group", "horizon"]

    for keys, group in df.groupby(group_cols, observed=True):
        week_start, entity, lg, horizon = keys

        # For entity level, use aggregate metrics (equivalent to standard when n=1)
        metrics = compute_group_metrics_aggregate(
            group,
            actual_col=actual_col,
            ml_pred_col=ml_pred_col,
            lp_pred_col=lp_pred_col,
        )
        metrics["week_start"] = week_start
        metrics["entity"] = entity
        metrics["liquidity_group"] = lg
        metrics["horizon"] = horizon

        # Add is_pass_through flag for entity-level metrics
        if "is_pass_through" in group.columns:
            metrics["is_pass_through"] = group["is_pass_through"].iloc[0]

        results.append(metrics)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    # Reorder columns
    col_order = ["week_start", "entity", "liquidity_group", "horizon", "n_obs",
                 "actual_sum", "ml_pred_sum", "lp_pred_sum",
                 "ml_wape", "ml_mae", "ml_directional_accuracy",
                 "lp_wape", "lp_mae", "lp_directional_accuracy",
                 "is_pass_through"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_metrics_net(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
    include_passthrough: bool = True,
) -> pd.DataFrame:
    """
    Compute Net-level metrics: TRR + TRP summed, grouped by (week_start, horizon).

    Uses Aggregate-then-Error approach:
    - Sum actuals and predictions across ALL entities and BOTH LGs
    - Compute WAPE on the aggregated totals

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.
        include_passthrough: If False, exclude Tier-2 passthrough rows (clean ML metrics).

    Returns:
        DataFrame with net metrics per (week_start, horizon).
    """
    df = df.copy()

    # Filter out passthroughs if requested (for clean ML metrics)
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    if df.empty:
        return pd.DataFrame()

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
    for _, row in net_df.iterrows():
        week_start = row["week_start"]
        horizon = row["horizon"]

        actual = row[actual_col]
        ml_pred = row[ml_pred_col]
        lp_pred = row[lp_pred_col] if lp_pred_col in row.index else np.nan
        prev = row["prev_actual"]

        metrics = {
            "week_start": week_start,
            "horizon": horizon,
            "n_obs": 1,  # Net is a single aggregated value
            "net_actual": actual,
            "net_ml_pred": ml_pred,
            "net_lp_pred": lp_pred,
        }

        # Compute metrics (already aggregated, so just compute error)
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
    include_passthrough: bool = True,
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
        include_passthrough: If False, exclude Tier-2 passthrough rows.

    Returns:
        DataFrame with net-entity metrics per (week_start, entity, horizon).
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    if df.empty:
        return pd.DataFrame()

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
    for _, row in net_entity_df.iterrows():
        week_start = row["week_start"]
        entity = row["entity"]
        horizon = row["horizon"]

        actual = row[actual_col]
        ml_pred = row[ml_pred_col]
        lp_pred = row[lp_pred_col] if lp_pred_col in row.index else np.nan
        prev = row["prev_actual"]

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
