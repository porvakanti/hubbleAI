"""
LightGBM model training and inference for hubbleAI.

This module provides functions for training LightGBM models for each
(Liquidity Group × Horizon) combination.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from hubbleAI.config import (
    DEFAULT_LGBM_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    TRAIN_RATIO,
    VALID_RATIO,
)


def wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Weighted Absolute Percentage Error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        eps: Small constant to avoid division by zero.

    Returns:
        WAPE value.
    """
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps)


def eval_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute MAE, RMSE, and WAPE metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Tuple of (MAE, RMSE, WAPE).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape_val = wape(y_true, y_pred)
    return mae, rmse, wape_val


def assign_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
) -> pd.DataFrame:
    """
    Assign train/valid/test split based on time.

    Args:
        df: DataFrame with week_start column.
        train_ratio: Fraction of weeks for training (default 0.85).
        valid_ratio: Fraction of weeks for train+valid (default 0.95).

    Returns:
        DataFrame with 'split' column added.
    """
    df = df.copy()
    unique_weeks = df["week_start"].drop_duplicates().sort_values().tolist()
    n = len(unique_weeks)

    train_end = unique_weeks[int(n * train_ratio)]
    valid_end = unique_weeks[int(n * valid_ratio)]

    def _assign(row):
        if row["week_start"] <= train_end:
            return "train"
        elif row["week_start"] <= valid_end:
            return "valid"
        else:
            return "test"

    df["split"] = df.apply(_assign, axis=1)
    return df


def train_lgbm_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    params: Optional[Dict] = None,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> Tuple[lgb.Booster, Dict[str, float], int]:
    """
    Train a single LightGBM model.

    Expects the DataFrame to have a 'split' column with values 'train', 'valid', 'test'.

    Args:
        df: DataFrame with features and target.
        feature_cols: List of feature column names.
        target_col: Target column name.
        params: LightGBM parameters (uses defaults if None).
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Early stopping patience.

    Returns:
        Tuple of (model, validation_metrics, best_iteration).
    """
    if params is None:
        params = DEFAULT_LGBM_PARAMS.copy()

    # Split data
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
    )

    # Compute validation metrics
    val_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    mae, rmse, wape_val = eval_metrics(y_valid.values, val_pred)

    val_metrics = {
        "mae": mae,
        "rmse": rmse,
        "wape": wape_val,
    }

    return model, val_metrics, model.best_iteration


def predict_lgbm(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Generate predictions using a trained LightGBM model.

    Args:
        model: Trained LightGBM model.
        df: DataFrame with features.
        feature_cols: List of feature column names.

    Returns:
        Array of predictions.
    """
    X = df[feature_cols]
    return model.predict(X, num_iteration=model.best_iteration)


def train_models_for_lg_horizon(
    df: pd.DataFrame,
    liquidity_group: str,
    horizon: int,
    base_feature_cols: List[str],
    extra_feature_cols: Optional[List[str]] = None,
    params: Optional[Dict] = None,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Train a model for a specific (Liquidity Group × Horizon) combination.

    Args:
        df: Full feature DataFrame with split column.
        liquidity_group: 'TRR' or 'TRP'.
        horizon: Forecast horizon (1-8).
        base_feature_cols: Base feature columns (without LP).
        extra_feature_cols: Additional features for this liquidity group.
        params: LightGBM parameters.

    Returns:
        Tuple of (model, metadata_dict).
    """
    from hubbleAI.features.lp_features import get_feature_cols_for_horizon

    # Filter to specific liquidity group
    df_lg = df[df["liquidity_group"] == liquidity_group].copy()

    # Build feature columns for this horizon
    feature_cols = get_feature_cols_for_horizon(
        horizon, base_feature_cols, all_cols=df_lg.columns.tolist()
    )

    # Add extra features if provided
    if extra_feature_cols:
        feature_cols = feature_cols + [
            f for f in extra_feature_cols if f not in feature_cols and f in df_lg.columns
        ]

    target_col = f"y_h{horizon}"

    # Drop rows with missing target
    df_lg = df_lg.dropna(subset=[target_col])

    # Train model
    model, val_metrics, best_iter = train_lgbm_model(
        df_lg, feature_cols, target_col, params
    )

    metadata = {
        "liquidity_group": liquidity_group,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "val_metrics": val_metrics,
        "best_iteration": best_iter,
    }

    return model, metadata


def generate_predictions_for_lg_horizon(
    model: lgb.Booster,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate predictions for a trained model.

    Args:
        model: Trained LightGBM model.
        df: DataFrame with features.
        metadata: Model metadata from train_models_for_lg_horizon.

    Returns:
        DataFrame with predictions and metadata columns.
    """
    liquidity_group = metadata["liquidity_group"]
    horizon = metadata["horizon"]
    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]

    # Filter to specific liquidity group
    df_lg = df[df["liquidity_group"] == liquidity_group].copy()

    # Drop rows with missing target
    df_lg = df_lg.dropna(subset=[target_col])

    # Generate predictions
    predictions = predict_lgbm(model, df_lg, feature_cols)

    # Build output DataFrame
    output = df_lg[["entity", "liquidity_group", "week_start"]].copy()
    output["horizon"] = horizon
    output["target_week"] = output["week_start"] + pd.Timedelta(weeks=horizon)
    output["y_actual"] = df_lg[target_col].values
    output["y_pred_point"] = predictions

    # TODO: implement proper quantile models for p10/p50/p90 in a later task
    output["y_pred_p10"] = np.nan
    output["y_pred_p50"] = np.nan
    output["y_pred_p90"] = np.nan

    output["model_type"] = "lightgbm"
    output["is_pass_through"] = False

    return output
