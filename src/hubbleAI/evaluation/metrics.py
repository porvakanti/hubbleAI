"""
Evaluation metrics for hubbleAI.

Supports WAPE, MAE, RMSE, and direction accuracy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
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
