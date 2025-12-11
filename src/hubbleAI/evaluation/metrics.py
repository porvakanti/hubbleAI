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

from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Backtest Diagnostics (Task 2.2)
# ---------------------------------------------------------------------------


def compute_horizon_profiles(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute horizon-wise error profiles: WAPE, MAE, MSE, RMSE per horizon.

    Uses Aggregate-then-Error WAPE (Treasury-aligned):
    - WAPE = |sum(actual) - sum(pred)| / |sum(actual)|

    For LP metrics, horizons 5-8 will have NaN (no LP baseline).

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with metrics per horizon:
        horizon, ml_wape, ml_mae, ml_mse, ml_rmse,
        lp_wape, lp_mae, lp_mse, lp_rmse, n_obs
    """
    results = []

    for horizon, group in df.groupby("horizon", observed=True):
        actual = group[actual_col]
        ml_pred = group[ml_pred_col]
        lp_pred = group[lp_pred_col] if lp_pred_col in group.columns else pd.Series([np.nan] * len(group))

        # Valid ML observations
        ml_mask = actual.notna() & ml_pred.notna()
        n_ml = ml_mask.sum()

        # Valid LP observations
        lp_mask = actual.notna() & lp_pred.notna()
        n_lp = lp_mask.sum()

        row = {"horizon": horizon, "n_obs": n_ml}

        # ML metrics
        if n_ml > 0:
            y_true = actual[ml_mask].values
            y_pred = ml_pred[ml_mask].values
            row["ml_wape"] = wape_aggregate(y_true, y_pred)
            row["ml_mae"] = float(np.mean(np.abs(y_true - y_pred)))
            row["ml_mse"] = float(np.mean((y_true - y_pred) ** 2))
            row["ml_rmse"] = float(np.sqrt(row["ml_mse"]))
        else:
            row["ml_wape"] = np.nan
            row["ml_mae"] = np.nan
            row["ml_mse"] = np.nan
            row["ml_rmse"] = np.nan

        # LP metrics
        if n_lp > 0:
            y_true_lp = actual[lp_mask].values
            y_pred_lp = lp_pred[lp_mask].values
            row["lp_wape"] = wape_aggregate(y_true_lp, y_pred_lp)
            row["lp_mae"] = float(np.mean(np.abs(y_true_lp - y_pred_lp)))
            row["lp_mse"] = float(np.mean((y_true_lp - y_pred_lp) ** 2))
            row["lp_rmse"] = float(np.sqrt(row["lp_mse"]))
        else:
            row["lp_wape"] = np.nan
            row["lp_mae"] = np.nan
            row["lp_mse"] = np.nan
            row["lp_rmse"] = np.nan

        results.append(row)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["horizon", "ml_wape", "ml_mae", "ml_mse", "ml_rmse",
                 "lp_wape", "lp_mae", "lp_mse", "lp_rmse", "n_obs"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_residual_diagnostics(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
) -> pd.DataFrame:
    """
    Compute residual distribution summaries per LG × horizon.

    Residual = actual_value - y_pred_point

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.

    Returns:
        DataFrame with residual statistics per (liquidity_group, horizon):
        liquidity_group, horizon, count, mean_residual, median_residual,
        std_residual, p10_residual, p25_residual, p75_residual, p90_residual
    """
    results = []

    for (lg, horizon), group in df.groupby(["liquidity_group", "horizon"], observed=True):
        actual = group[actual_col]
        ml_pred = group[ml_pred_col]

        # Valid observations
        mask = actual.notna() & ml_pred.notna()
        n = mask.sum()

        if n == 0:
            results.append({
                "liquidity_group": lg,
                "horizon": horizon,
                "count": 0,
                "mean_residual": np.nan,
                "median_residual": np.nan,
                "std_residual": np.nan,
                "p10_residual": np.nan,
                "p25_residual": np.nan,
                "p75_residual": np.nan,
                "p90_residual": np.nan,
            })
            continue

        residuals = actual[mask].values - ml_pred[mask].values

        results.append({
            "liquidity_group": lg,
            "horizon": horizon,
            "count": int(n),
            "mean_residual": float(np.mean(residuals)),
            "median_residual": float(np.median(residuals)),
            "std_residual": float(np.std(residuals, ddof=1)) if n > 1 else np.nan,
            "p10_residual": float(np.percentile(residuals, 10)),
            "p25_residual": float(np.percentile(residuals, 25)),
            "p75_residual": float(np.percentile(residuals, 75)),
            "p90_residual": float(np.percentile(residuals, 90)),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["liquidity_group", "horizon", "count", "mean_residual",
                 "median_residual", "std_residual", "p10_residual",
                 "p25_residual", "p75_residual", "p90_residual"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_entity_stability(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
) -> pd.DataFrame:
    """
    Compute entity-level stability across time per entity × horizon.

    For each entity × horizon:
    - abs_error = |actual_value - y_pred_point|
    - Rolling 4-week volatility of abs_error (std with window=4)

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.

    Returns:
        DataFrame with stability metrics per (entity, liquidity_group, horizon):
        entity, liquidity_group, horizon, mean_abs_error, median_abs_error,
        std_abs_error, mean_4w_volatility, max_4w_volatility, n_weeks
    """
    df = df.copy()

    # Compute absolute error
    mask = df[actual_col].notna() & df[ml_pred_col].notna()
    df["abs_error"] = np.nan
    df.loc[mask, "abs_error"] = np.abs(
        df.loc[mask, actual_col].values - df.loc[mask, ml_pred_col].values
    )

    # Sort by entity, LG, horizon, week for rolling calculation
    df = df.sort_values(["entity", "liquidity_group", "horizon", "week_start"])

    # Compute rolling 4-week volatility (std) per entity × LG × horizon
    df["error_volatility_4w"] = df.groupby(
        ["entity", "liquidity_group", "horizon"], observed=True
    )["abs_error"].transform(lambda x: x.rolling(window=4, min_periods=2).std())

    results = []

    for (entity, lg, horizon), group in df.groupby(
        ["entity", "liquidity_group", "horizon"], observed=True
    ):
        abs_errors = group["abs_error"].dropna()
        volatilities = group["error_volatility_4w"].dropna()

        n_weeks = len(abs_errors)

        if n_weeks == 0:
            results.append({
                "entity": entity,
                "liquidity_group": lg,
                "horizon": horizon,
                "mean_abs_error": np.nan,
                "median_abs_error": np.nan,
                "std_abs_error": np.nan,
                "mean_4w_volatility": np.nan,
                "max_4w_volatility": np.nan,
                "n_weeks": 0,
            })
            continue

        results.append({
            "entity": entity,
            "liquidity_group": lg,
            "horizon": horizon,
            "mean_abs_error": float(abs_errors.mean()),
            "median_abs_error": float(abs_errors.median()),
            "std_abs_error": float(abs_errors.std(ddof=1)) if n_weeks > 1 else np.nan,
            "mean_4w_volatility": float(volatilities.mean()) if len(volatilities) > 0 else np.nan,
            "max_4w_volatility": float(volatilities.max()) if len(volatilities) > 0 else np.nan,
            "n_weeks": int(n_weeks),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["entity", "liquidity_group", "horizon", "mean_abs_error",
                 "median_abs_error", "std_abs_error", "mean_4w_volatility",
                 "max_4w_volatility", "n_weeks"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_model_vs_lp_wins(
    df: pd.DataFrame,
    actual_col: str = "actual_value",
    ml_pred_col: str = "y_pred_point",
    lp_pred_col: str = "lp_baseline_point",
) -> pd.DataFrame:
    """
    Compute model vs LP win-loss analysis per LG × horizon.

    For each observation, compare |ml_error| vs |lp_error|:
    - ML wins if |ml_error| < |lp_error|
    - LP wins if |lp_error| < |ml_error|
    - Tie if equal

    Note: Horizons 5-8 have NaN LP baseline, so no comparison possible.

    Args:
        df: Backtest predictions DataFrame.
        actual_col: Column name for actual values.
        ml_pred_col: Column name for ML predictions.
        lp_pred_col: Column name for LP baseline predictions.

    Returns:
        DataFrame with win-loss counts per (liquidity_group, horizon):
        liquidity_group, horizon, ml_better_count, lp_better_count,
        tie_count, ml_win_rate, lp_win_rate, total
    """
    results = []

    for (lg, horizon), group in df.groupby(["liquidity_group", "horizon"], observed=True):
        actual = group[actual_col]
        ml_pred = group[ml_pred_col]
        lp_pred = group[lp_pred_col] if lp_pred_col in group.columns else pd.Series([np.nan] * len(group))

        # Valid observations where both ML and LP are available
        mask = actual.notna() & ml_pred.notna() & lp_pred.notna()
        n = mask.sum()

        if n == 0:
            results.append({
                "liquidity_group": lg,
                "horizon": horizon,
                "ml_better_count": 0,
                "lp_better_count": 0,
                "tie_count": 0,
                "ml_win_rate": np.nan,
                "lp_win_rate": np.nan,
                "total": 0,
            })
            continue

        y_true = actual[mask].values
        y_ml = ml_pred[mask].values
        y_lp = lp_pred[mask].values

        ml_error = np.abs(y_true - y_ml)
        lp_error = np.abs(y_true - y_lp)

        ml_better = int(np.sum(ml_error < lp_error))
        lp_better = int(np.sum(lp_error < ml_error))
        tie = int(np.sum(ml_error == lp_error))
        total = int(n)

        results.append({
            "liquidity_group": lg,
            "horizon": horizon,
            "ml_better_count": ml_better,
            "lp_better_count": lp_better,
            "tie_count": tie,
            "ml_win_rate": ml_better / total if total > 0 else np.nan,
            "lp_win_rate": lp_better / total if total > 0 else np.nan,
            "total": total,
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["liquidity_group", "horizon", "ml_better_count", "lp_better_count",
                 "tie_count", "ml_win_rate", "lp_win_rate", "total"]
    return result_df[[c for c in col_order if c in result_df.columns]]


# ---------------------------------------------------------------------------
# Probabilistic Diagnostics (Task 3.2)
# ---------------------------------------------------------------------------


def pinball_loss(
    actual: pd.Series,
    pred: pd.Series,
    alpha: float,
) -> float:
    """
    Compute pinball (quantile) loss for a given alpha.

    For each observation:
        u = actual - pred
        loss = max(alpha * u, (alpha - 1) * u)

    This is equivalent to:
        - If actual > pred (under-prediction): loss = alpha * |error|
        - If actual < pred (over-prediction): loss = (1 - alpha) * |error|

    Args:
        actual: Actual values as pandas Series.
        pred: Predicted values as pandas Series.
        alpha: Quantile level (e.g., 0.10, 0.50, 0.90).

    Returns:
        Mean pinball loss over all non-null pairs, or NaN if no valid pairs.
    """
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return np.nan

    y_true = actual[mask].values
    y_pred = pred[mask].values

    u = y_true - y_pred
    loss = np.maximum(alpha * u, (alpha - 1) * u)
    return float(np.mean(loss))


def compute_quantile_coverage_by_horizon(
    df: pd.DataFrame,
    *,
    include_passthrough: bool = False,
) -> pd.DataFrame:
    """
    Compute quantile coverage metrics aggregated by horizon only.

    Coverage metrics show how well-calibrated the quantile predictions are:
    - prob_below_p10 should be ≈ 0.10 for well-calibrated P10
    - prob_between_p10_p90 should be ≈ 0.80 for well-calibrated P10/P90
    - prob_above_p90 should be ≈ 0.10 for well-calibrated P90
    - prob_above_p50 / prob_below_p50 should each be ≈ 0.50 for well-calibrated median

    Args:
        df: Backtest predictions DataFrame with actual_value and quantile columns.
        include_passthrough: If False (default), exclude Tier-2 passthrough rows.
            Note: Tier-2 rows have NaN quantiles, so they're effectively excluded anyway.

    Returns:
        DataFrame with columns:
            horizon, n, prob_below_p10, prob_between_p10_p90, prob_above_p90,
            prob_above_p50, prob_below_p50
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    # Filter to rows with valid actual and all quantile predictions
    mask = (
        df["actual_value"].notna() &
        df["y_pred_p10"].notna() &
        df["y_pred_p50"].notna() &
        df["y_pred_p90"].notna()
    )
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    for horizon, group in df.groupby("horizon", observed=True):
        actual = group["actual_value"].values
        p10 = group["y_pred_p10"].values
        p50 = group["y_pred_p50"].values
        p90 = group["y_pred_p90"].values

        n = len(group)

        # Coverage indicators
        below_p10 = actual <= p10
        above_p90 = actual >= p90
        between_p10_p90 = (actual > p10) & (actual < p90)
        above_p50 = actual >= p50
        below_p50 = actual < p50

        results.append({
            "horizon": horizon,
            "n": n,
            "prob_below_p10": float(np.mean(below_p10)),
            "prob_between_p10_p90": float(np.mean(between_p10_p90)),
            "prob_above_p90": float(np.mean(above_p90)),
            "prob_above_p50": float(np.mean(above_p50)),
            "prob_below_p50": float(np.mean(below_p50)),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["horizon", "n", "prob_below_p10", "prob_between_p10_p90",
                 "prob_above_p90", "prob_above_p50", "prob_below_p50"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_quantile_coverage_by_lg_horizon(
    df: pd.DataFrame,
    *,
    include_passthrough: bool = False,
) -> pd.DataFrame:
    """
    Compute quantile coverage metrics grouped by (liquidity_group, horizon).

    Same as compute_quantile_coverage_by_horizon but with additional LG grouping.

    Args:
        df: Backtest predictions DataFrame with actual_value and quantile columns.
        include_passthrough: If False (default), exclude Tier-2 passthrough rows.

    Returns:
        DataFrame with columns:
            liquidity_group, horizon, n, prob_below_p10, prob_between_p10_p90,
            prob_above_p90, prob_above_p50, prob_below_p50
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    # Filter to rows with valid actual and all quantile predictions
    mask = (
        df["actual_value"].notna() &
        df["y_pred_p10"].notna() &
        df["y_pred_p50"].notna() &
        df["y_pred_p90"].notna()
    )
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    for (lg, horizon), group in df.groupby(["liquidity_group", "horizon"], observed=True):
        actual = group["actual_value"].values
        p10 = group["y_pred_p10"].values
        p50 = group["y_pred_p50"].values
        p90 = group["y_pred_p90"].values

        n = len(group)

        # Coverage indicators
        below_p10 = actual <= p10
        above_p90 = actual >= p90
        between_p10_p90 = (actual > p10) & (actual < p90)
        above_p50 = actual >= p50
        below_p50 = actual < p50

        results.append({
            "liquidity_group": lg,
            "horizon": horizon,
            "n": n,
            "prob_below_p10": float(np.mean(below_p10)),
            "prob_between_p10_p90": float(np.mean(between_p10_p90)),
            "prob_above_p90": float(np.mean(above_p90)),
            "prob_above_p50": float(np.mean(above_p50)),
            "prob_below_p50": float(np.mean(below_p50)),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["liquidity_group", "horizon", "n", "prob_below_p10",
                 "prob_between_p10_p90", "prob_above_p90", "prob_above_p50",
                 "prob_below_p50"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_pinball_by_horizon(
    df: pd.DataFrame,
    *,
    include_passthrough: bool = False,
) -> pd.DataFrame:
    """
    Compute pinball loss per horizon for P10, P50, P90.

    Pinball loss measures quantile prediction accuracy (lower is better).
    It penalizes under-predictions and over-predictions asymmetrically
    based on the quantile level.

    Args:
        df: Backtest predictions DataFrame with actual_value and quantile columns.
        include_passthrough: If False (default), exclude Tier-2 passthrough rows.

    Returns:
        DataFrame with columns:
            horizon, n, pinball_p10, pinball_p50, pinball_p90
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    # Filter to rows with valid actual and all quantile predictions
    mask = (
        df["actual_value"].notna() &
        df["y_pred_p10"].notna() &
        df["y_pred_p50"].notna() &
        df["y_pred_p90"].notna()
    )
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    for horizon, group in df.groupby("horizon", observed=True):
        actual = group["actual_value"]
        p10 = group["y_pred_p10"]
        p50 = group["y_pred_p50"]
        p90 = group["y_pred_p90"]

        n = len(group)

        results.append({
            "horizon": horizon,
            "n": n,
            "pinball_p10": pinball_loss(actual, p10, alpha=0.10),
            "pinball_p50": pinball_loss(actual, p50, alpha=0.50),
            "pinball_p90": pinball_loss(actual, p90, alpha=0.90),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["horizon", "n", "pinball_p10", "pinball_p50", "pinball_p90"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def compute_pinball_by_lg_horizon(
    df: pd.DataFrame,
    *,
    include_passthrough: bool = False,
) -> pd.DataFrame:
    """
    Compute pinball loss per (liquidity_group, horizon) for P10, P50, P90.

    Same as compute_pinball_by_horizon but with additional LG grouping.

    Args:
        df: Backtest predictions DataFrame with actual_value and quantile columns.
        include_passthrough: If False (default), exclude Tier-2 passthrough rows.

    Returns:
        DataFrame with columns:
            liquidity_group, horizon, n, pinball_p10, pinball_p50, pinball_p90
    """
    df = df.copy()

    # Filter out passthroughs if requested
    if not include_passthrough and "is_pass_through" in df.columns:
        df = df[df["is_pass_through"] == False].copy()

    # Filter to rows with valid actual and all quantile predictions
    mask = (
        df["actual_value"].notna() &
        df["y_pred_p10"].notna() &
        df["y_pred_p50"].notna() &
        df["y_pred_p90"].notna()
    )
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    for (lg, horizon), group in df.groupby(["liquidity_group", "horizon"], observed=True):
        actual = group["actual_value"]
        p10 = group["y_pred_p10"]
        p50 = group["y_pred_p50"]
        p90 = group["y_pred_p90"]

        n = len(group)

        results.append({
            "liquidity_group": lg,
            "horizon": horizon,
            "n": n,
            "pinball_p10": pinball_loss(actual, p10, alpha=0.10),
            "pinball_p50": pinball_loss(actual, p50, alpha=0.50),
            "pinball_p90": pinball_loss(actual, p90, alpha=0.90),
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["liquidity_group", "horizon", "n", "pinball_p10",
                 "pinball_p50", "pinball_p90"]
    return result_df[[c for c in col_order if c in result_df.columns]]


# ---------------------------------------------------------------------------
# Hybrid ML+LP Alpha Tuning (Task 4.1)
# ---------------------------------------------------------------------------


def tune_hybrid_alpha(
    df: pd.DataFrame,
    alphas: Optional[List[float]] = None,
    n_folds: int = 4,
    min_train_weeks: int = 20,
    val_weeks: int = 4,
) -> pd.DataFrame:
    """
    Tune alpha for TRP hybrid model using time-series cross-validation
    on TRAIN+VALID splits (NOT test), for each horizon.

    Hybrid formula: y_hybrid = alpha * y_ml + (1 - alpha) * y_lp

    Only TRP Tier-1 rows are used for tuning.
    TRR is excluded (uses pure ML, alpha=1.0).

    Args:
        df: Full backtest DataFrame with all splits (must include train+valid).
            Should have columns: liquidity_group, horizon, week_start,
            actual_value, y_pred_point, lp_baseline_point, is_pass_through, split.
        alphas: Alpha grid to search. Default: [0.0, 0.1, ..., 1.0].
        n_folds: Maximum number of time-series folds.
        min_train_weeks: Minimum weeks required for training in each fold.
        val_weeks: Number of weeks per validation fold.

    Returns:
        DataFrame with columns:
            liquidity_group, horizon, alpha, wape_ml, wape_lp, wape_hybrid, n_folds_used
    """
    if alphas is None:
        alphas = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0

    results = []

    # Process both LGs and all horizons
    for lg in ["TRR", "TRP"]:
        for horizon in range(1, 9):
            # For TRR: always use pure ML (alpha=1.0)
            if lg == "TRR":
                # Compute ML WAPE for reference
                df_lg_h = df[
                    (df["liquidity_group"] == lg) &
                    (df["horizon"] == horizon) &
                    (df["is_pass_through"] == False) &
                    (df["actual_value"].notna()) &
                    (df["y_pred_point"].notna())
                ]
                if len(df_lg_h) > 0:
                    wape_ml = wape_aggregate_series(
                        df_lg_h["actual_value"],
                        df_lg_h["y_pred_point"]
                    )
                    # LP WAPE for horizons 1-4
                    if horizon <= 4 and "lp_baseline_point" in df_lg_h.columns:
                        lp_mask = df_lg_h["lp_baseline_point"].notna()
                        if lp_mask.sum() > 0:
                            wape_lp = wape_aggregate_series(
                                df_lg_h.loc[lp_mask, "actual_value"],
                                df_lg_h.loc[lp_mask, "lp_baseline_point"]
                            )
                        else:
                            wape_lp = np.nan
                    else:
                        wape_lp = np.nan
                else:
                    wape_ml = np.nan
                    wape_lp = np.nan

                results.append({
                    "liquidity_group": lg,
                    "horizon": horizon,
                    "alpha": 1.0,
                    "wape_ml": wape_ml,
                    "wape_lp": wape_lp,
                    "wape_hybrid": wape_ml,  # Hybrid = ML for TRR
                    "n_folds_used": 0,
                })
                continue

            # For TRP horizons 5-8: no LP, use pure ML
            if horizon > 4:
                df_lg_h = df[
                    (df["liquidity_group"] == lg) &
                    (df["horizon"] == horizon) &
                    (df["is_pass_through"] == False) &
                    (df["actual_value"].notna()) &
                    (df["y_pred_point"].notna())
                ]
                if len(df_lg_h) > 0:
                    wape_ml = wape_aggregate_series(
                        df_lg_h["actual_value"],
                        df_lg_h["y_pred_point"]
                    )
                else:
                    wape_ml = np.nan

                results.append({
                    "liquidity_group": lg,
                    "horizon": horizon,
                    "alpha": 1.0,
                    "wape_ml": wape_ml,
                    "wape_lp": np.nan,
                    "wape_hybrid": wape_ml,
                    "n_folds_used": 0,
                })
                continue

            # For TRP horizons 1-4: tune alpha using time-series CV
            # Filter to TRP Tier-1 rows with valid data
            df_trp = df[
                (df["liquidity_group"] == "TRP") &
                (df["horizon"] == horizon) &
                (df["is_pass_through"] == False) &
                (df["actual_value"].notna()) &
                (df["y_pred_point"].notna()) &
                (df["lp_baseline_point"].notna())
            ].copy()

            if df_trp.empty:
                results.append({
                    "liquidity_group": lg,
                    "horizon": horizon,
                    "alpha": 1.0,
                    "wape_ml": np.nan,
                    "wape_lp": np.nan,
                    "wape_hybrid": np.nan,
                    "n_folds_used": 0,
                })
                continue

            # Ensure week_start is datetime
            df_trp["week_start"] = pd.to_datetime(df_trp["week_start"])

            # Get unique weeks sorted
            weeks = sorted(df_trp["week_start"].unique())
            n_weeks = len(weeks)

            # Build time-series CV folds
            folds = []
            start_idx = 0
            while len(folds) < n_folds:
                train_end_idx = start_idx + min_train_weeks - 1
                val_start_idx = train_end_idx + 1
                val_end_idx = val_start_idx + val_weeks - 1

                if val_end_idx >= n_weeks:
                    break

                train_weeks_set = set(weeks[:train_end_idx + 1])
                val_weeks_set = set(weeks[val_start_idx:val_end_idx + 1])

                folds.append((train_weeks_set, val_weeks_set))
                start_idx += val_weeks

            if not folds:
                # Not enough data for CV, use all data with alpha=0.5 as default
                wape_ml = wape_aggregate_series(
                    df_trp["actual_value"],
                    df_trp["y_pred_point"]
                )
                wape_lp = wape_aggregate_series(
                    df_trp["actual_value"],
                    df_trp["lp_baseline_point"]
                )
                # Default to 0.5 blend
                hybrid_pred = 0.5 * df_trp["y_pred_point"] + 0.5 * df_trp["lp_baseline_point"]
                wape_hybrid = wape_aggregate_series(df_trp["actual_value"], hybrid_pred)

                results.append({
                    "liquidity_group": lg,
                    "horizon": horizon,
                    "alpha": 0.5,
                    "wape_ml": wape_ml,
                    "wape_lp": wape_lp,
                    "wape_hybrid": wape_hybrid,
                    "n_folds_used": 0,
                })
                continue

            # Evaluate each alpha across all folds
            alpha_results = {a: [] for a in alphas}

            for train_weeks_set, val_weeks_set in folds:
                df_val = df_trp[df_trp["week_start"].isin(val_weeks_set)]

                if df_val.empty:
                    continue

                for a in alphas:
                    blended = a * df_val["y_pred_point"] + (1 - a) * df_val["lp_baseline_point"]
                    wape_val = wape_aggregate_series(df_val["actual_value"], blended)
                    if not np.isnan(wape_val):
                        alpha_results[a].append(wape_val)

            # Find best alpha (lowest mean WAPE across folds)
            best_alpha = 1.0
            best_mean_wape = float("inf")

            for a in alphas:
                if alpha_results[a]:
                    mean_wape = np.mean(alpha_results[a])
                    if mean_wape < best_mean_wape:
                        best_mean_wape = mean_wape
                        best_alpha = a

            # Compute final WAPE values on all data
            wape_ml = wape_aggregate_series(
                df_trp["actual_value"],
                df_trp["y_pred_point"]
            )
            wape_lp = wape_aggregate_series(
                df_trp["actual_value"],
                df_trp["lp_baseline_point"]
            )
            hybrid_pred = best_alpha * df_trp["y_pred_point"] + (1 - best_alpha) * df_trp["lp_baseline_point"]
            wape_hybrid = wape_aggregate_series(df_trp["actual_value"], hybrid_pred)

            results.append({
                "liquidity_group": lg,
                "horizon": horizon,
                "alpha": best_alpha,
                "wape_ml": wape_ml,
                "wape_lp": wape_lp,
                "wape_hybrid": wape_hybrid,
                "n_folds_used": len(folds),
            })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    col_order = ["liquidity_group", "horizon", "alpha", "wape_ml", "wape_lp",
                 "wape_hybrid", "n_folds_used"]
    return result_df[[c for c in col_order if c in result_df.columns]]


def get_alpha_mapping(alpha_df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """
    Convert alpha tuning DataFrame to dict: {(liquidity_group, horizon): alpha}.

    Args:
        alpha_df: DataFrame from tune_hybrid_alpha() with columns
            liquidity_group, horizon, alpha.

    Returns:
        Dict mapping (liquidity_group, horizon) to alpha value.
    """
    mapping = {}
    for _, row in alpha_df.iterrows():
        key = (row["liquidity_group"], int(row["horizon"]))
        mapping[key] = float(row["alpha"])
    return mapping
