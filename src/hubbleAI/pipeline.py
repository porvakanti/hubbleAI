"""
Core forecasting pipeline for hubbleAI.

This module defines the main `run_forecast` entry point that both the
scheduler and the Streamlit app can call.

IMPORTANT:
- For now, this pipeline works with local raw files under `data/raw`.
- Outputs (forecasts, run status, metrics) are written under `data/processed`.
- In the future, data sources may move to Denodo / Databricks / Reval / etc.
  Please keep I/O logic well isolated so that we can swap out the data backend
  without changing the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from hubbleAI.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    RUN_STATUS_DIR,
    FORECASTS_DIR,
    BACKTESTS_DIR,
    METRICS_DIR,
    BACKTEST_METRICS_DIR,
    HORIZONS,
    LIQUIDITY_GROUPS,
    TIER2_LIST,
    LP_FORECAST_COLS,
    TRP_EXTRA_FEATURES,
    ACTUALS_FILENAME,
    LP_FILENAME,
    FX_FILENAME,
    MIN_HISTORY_WEEKS,
    FORECAST_OUTPUT_COLS,
    BACKTEST_OUTPUT_COLS,
)

# Ensure directories exist
RUN_STATUS_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_METRICS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

RunStatus = Literal["success", "data_missing", "skipped", "error"]


ForecastMode = Literal["forward", "backtest"]


@dataclass
class ForecastRunStatus:
    """
    Summary of a single forecast pipeline run.

    This is what the Streamlit UI will read to show "last run" status.
    """

    run_id: str
    as_of_date: date
    ref_week_start: date  # Monday reference week (always a Monday)
    mode: ForecastMode  # "forward" or "backtest"
    trigger_source: Literal["scheduler", "manual", "notebook"]
    status: RunStatus
    created_at: datetime
    message: str
    missing_inputs: List[str]
    output_paths: Dict[str, str]
    metrics_paths: Dict[str, str]

    @property
    def latest_forecasts_path(self) -> Optional[Path]:
        return (
            Path(self.output_paths.get("forecasts"))
            if "forecasts" in self.output_paths
            else None
        )

    @property
    def latest_backtest_path(self) -> Optional[Path]:
        return (
            Path(self.output_paths.get("backtest"))
            if "backtest" in self.output_paths
            else None
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_default_as_of_date(today: Optional[date] = None) -> date:
    """
    Return the default as_of_date for a run.

    Convention:
    - Use the last fully closed week whose data should be available
      by the time we run (e.g. previous Friday / banking day).
    - For now we use (today - 3 days) as a simple placeholder.
    """
    if today is None:
        today = date.today()
    return today - timedelta(days=3)


def check_data_availability(as_of_date: date) -> Tuple[bool, List[str]]:
    """
    Check whether input data is available and fresh enough for as_of_date.

    Returns:
        (is_ready, missing_inputs)

    For now, this function only checks for the *presence* of expected files
    under data/raw. Later we can extend it to include content validation,
    completeness checks, and Denodo / Databricks / Reval connectors.
    """
    missing: List[str] = []

    actuals_path = DATA_RAW_DIR / ACTUALS_FILENAME
    lp_path = DATA_RAW_DIR / LP_FILENAME
    fx_path = DATA_RAW_DIR / FX_FILENAME

    if not actuals_path.exists():
        missing.append("ACTUALS_FILE")
    if not lp_path.exists():
        missing.append("LP_FILE")
    if not fx_path.exists():
        missing.append("FX_FILE")

    # TODO: In the future, inspect dates inside the files and verify that
    # they cover as_of_date and the required horizons.

    is_ready = len(missing) == 0
    return is_ready, missing


def _load_and_prepare_data(
    as_of_date: date,
) -> Tuple[pd.DataFrame, pd.DataFrame, date]:
    """
    Load and prepare all data sources for forecasting.

    Returns:
        Tuple of (model_ready_df, tier2_df, ref_week_start)
        - model_ready_df: DataFrame with features and targets for Tier-1 entities
        - tier2_df: DataFrame with Tier-2 entities for LP pass-through
        - ref_week_start: The latest Monday in the dataset (reference week)
    """
    from hubbleAI.data_prep import prepare_weekly_data
    from hubbleAI.data_prep.prepare import filter_tier1_with_history, add_target_columns
    from hubbleAI.features import build_all_features

    # Prepare weekly data (loads and merges all sources)
    merged_df, _ = prepare_weekly_data(as_of_date=as_of_date)

    # Build all features
    featured_df = build_all_features(merged_df)

    # Add target columns
    featured_df = add_target_columns(featured_df)

    # Compute ref_week_start (latest Monday in dataset)
    # week_start is already Monday-aligned from aggregation.py
    ref_week_start = pd.to_datetime(featured_df["week_start"].max()).date()

    # Verify ref_week_start is a Monday (weekday() == 0)
    if ref_week_start.weekday() != 0:
        # Normalize to previous Monday if not already Monday
        days_since_monday = ref_week_start.weekday()
        ref_week_start = ref_week_start - timedelta(days=days_since_monday)
        logger.warning(
            f"ref_week_start normalized to Monday: {ref_week_start}"
        )

    # Filter Tier-1 entities with sufficient history for ML
    tier1_df = filter_tier1_with_history(featured_df, MIN_HISTORY_WEEKS)

    # Get Tier-2 entities (for LP pass-through)
    tier2_df = featured_df[featured_df["tier"] == "Tier2"].copy()

    # Add target_week_start for output purposes
    tier1_df["target_week_start"] = tier1_df["week_start"] + pd.Timedelta(days=7)

    return tier1_df, tier2_df, ref_week_start


def _assign_backtest_split(
    df: pd.DataFrame,
    train_ratio: float = 0.85,
    valid_ratio: float = 0.10,
) -> pd.DataFrame:
    """
    Assign train/valid/test split based on chronological 85/10/5 split.

    Args:
        df: DataFrame with week_start column.
        train_ratio: Fraction of weeks for training (default 0.85).
        valid_ratio: Fraction of weeks for validation (default 0.10).

    Returns:
        DataFrame with 'split' column added.
    """
    df = df.copy()
    unique_weeks = sorted(df["week_start"].drop_duplicates().tolist())
    n = len(unique_weeks)

    train_end_idx = int(n * train_ratio)
    valid_end_idx = int(n * (train_ratio + valid_ratio))

    train_weeks = set(unique_weeks[:train_end_idx])
    valid_weeks = set(unique_weeks[train_end_idx:valid_end_idx])
    test_weeks = set(unique_weeks[valid_end_idx:])

    def _assign(row):
        ws = row["week_start"]
        if ws in train_weeks:
            return "train"
        elif ws in valid_weeks:
            return "valid"
        else:
            return "test"

    df["split"] = df.apply(_assign, axis=1)
    return df


def _build_and_run_models_forward(
    ref_week_start: date,
    tier1_df: pd.DataFrame,
    tier2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run models in FORWARD mode: train on all data, predict only for ref_week_start.

    Forward mode outputs ONLY forecasts for the next 8 weeks after ref_week_start:
    - week_start == ref_week_start (a single Monday)
    - horizon ∈ {1..8}
    - target_week_start = ref_week_start + 7 * horizon days (ALWAYS a Monday)

    Args:
        ref_week_start: The reference Monday (latest Monday in data).
        tier1_df: Prepared DataFrame for Tier-1 ML forecasting.
        tier2_df: DataFrame with Tier-2 entities for LP pass-through.

    Returns:
        Unified forward forecast DataFrame.
    """
    from hubbleAI.features import get_base_feature_cols, get_feature_cols_for_horizon
    from hubbleAI.features.builder import get_trp_extra_features
    from hubbleAI.models.lightgbm_model import (
        train_lgbm_model,
        train_lgbm_quantile_model,
        predict_lgbm,
        assign_split,
    )

    all_forecasts = []

    # Assign split for training (use all data for training in forward mode)
    tier1_df = assign_split(tier1_df)

    # Get base feature columns (without LP)
    base_feature_cols = get_base_feature_cols(tier1_df)

    # Get TRP-specific extra features
    trp_extra_features = get_trp_extra_features(tier1_df)

    # Convert ref_week_start to timestamp for comparison
    ref_week_ts = pd.Timestamp(ref_week_start)

    # Train and predict for each (LG × Horizon) combination
    for lg in LIQUIDITY_GROUPS:
        # Filter to liquidity group
        df_lg = tier1_df[tier1_df["liquidity_group"] == lg].copy()

        if df_lg.empty:
            logger.warning(f"No data for liquidity group {lg}, skipping")
            continue

        # Determine extra features for this LG
        extra_features = trp_extra_features if lg == "TRP" else []

        for horizon in HORIZONS:
            logger.info(f"[Forward] Training model for {lg} - Horizon {horizon}")

            # Get feature columns for this horizon (with horizon-specific LP)
            feature_cols = get_feature_cols_for_horizon(
                horizon, base_feature_cols, all_cols=df_lg.columns.tolist()
            )

            # Add extra features if available
            if extra_features:
                feature_cols = feature_cols + [
                    f
                    for f in extra_features
                    if f not in feature_cols and f in df_lg.columns
                ]

            target_col = f"y_h{horizon}"

            # For training: use rows with non-null targets
            df_train = df_lg.dropna(subset=[target_col]).copy()

            if df_train.empty:
                logger.warning(
                    f"No valid targets for {lg} H{horizon}, skipping"
                )
                continue

            try:
                # Train point model on all available data with targets
                model, val_metrics, best_iter = train_lgbm_model(
                    df_train, feature_cols, target_col
                )

                logger.info(
                    f"{lg} H{horizon}: val_wape={val_metrics['wape']:.4f}, "
                    f"best_iter={best_iter}"
                )

                # Train quantile models (p10, p50, p90)
                model_q10, _, _ = train_lgbm_quantile_model(
                    df_train, feature_cols, target_col, alpha=0.10
                )
                model_q50, _, _ = train_lgbm_quantile_model(
                    df_train, feature_cols, target_col, alpha=0.50
                )
                model_q90, _, _ = train_lgbm_quantile_model(
                    df_train, feature_cols, target_col, alpha=0.90
                )

                # For forward mode: predict ONLY for ref_week_start
                df_ref = df_lg[df_lg["week_start"] == ref_week_ts].copy()

                if df_ref.empty:
                    logger.warning(
                        f"No rows for ref_week_start {ref_week_start} in {lg}, skipping H{horizon}"
                    )
                    continue

                # Point predictions
                predictions = predict_lgbm(model, df_ref, feature_cols)

                # Quantile predictions
                q10 = predict_lgbm(model_q10, df_ref, feature_cols)
                q50 = predict_lgbm(model_q50, df_ref, feature_cols)
                q90 = predict_lgbm(model_q90, df_ref, feature_cols)

                # Build output DataFrame
                output = df_ref[
                    ["entity", "liquidity_group", "week_start"]
                ].copy()
                output["horizon"] = horizon
                output["target_week_start"] = output["week_start"] + pd.Timedelta(
                    days=7 * horizon
                )

                # Forward mode: actual_value is NaN (future not yet observed)
                output["actual_value"] = np.nan

                output["y_pred_point"] = predictions
                output["y_pred_p10"] = q10
                output["y_pred_p50"] = q50
                output["y_pred_p90"] = q90

                output["model_type"] = "lightgbm"
                output["is_pass_through"] = False

                all_forecasts.append(output)

            except Exception as e:
                logger.error(f"Error training {lg} H{horizon}: {e}")
                continue

    # Add Tier-2 LP pass-through forecasts (only for ref_week_start)
    tier2_forecasts = _build_tier2_passthrough_forward(ref_week_start, tier2_df)
    if not tier2_forecasts.empty:
        all_forecasts.append(tier2_forecasts)

    # Combine all forecasts
    if all_forecasts:
        forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    else:
        forecasts_df = pd.DataFrame(columns=FORECAST_OUTPUT_COLS)

    # Ensure correct output columns
    forecasts_df = forecasts_df[FORECAST_OUTPUT_COLS]

    return forecasts_df


def _build_and_run_models_backtest(
    ref_week_start: date,
    tier1_df: pd.DataFrame,
    tier2_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Run models in BACKTEST mode: 85/10/5 split, predict only for test (last 5%) weeks.

    Backtest mode:
    - First 85% of weeks → training
    - Next 10% of weeks → validation
    - Last 5% of weeks → BACKTEST (test) - predictions generated here

    Also computes and saves metrics at 4 aggregation levels:
    - LG-level: (week_start, liquidity_group, horizon)
    - Entity-level: (week_start, entity, liquidity_group, horizon)
    - Net-level: (week_start, horizon) - TRR+TRP summed
    - Net-Entity-level: (week_start, entity, horizon) - TRR+TRP summed per entity

    Args:
        ref_week_start: The reference Monday (latest Monday in validation set).
        tier1_df: Prepared DataFrame for Tier-1 ML forecasting.
        tier2_df: DataFrame with Tier-2 entities for LP pass-through.

    Returns:
        Tuple of:
        - Backtest predictions DataFrame for the last 5% weeks (includes lp_baseline_point)
        - Dict mapping metric file names to their paths
    """
    from hubbleAI.features import get_base_feature_cols, get_feature_cols_for_horizon
    from hubbleAI.features.builder import get_trp_extra_features
    from hubbleAI.models.lightgbm_model import (
        train_lgbm_model,
        train_lgbm_quantile_model,
        predict_lgbm,
    )
    from hubbleAI.evaluation.metrics import (
        compute_metrics_by_lg,
        compute_metrics_by_entity,
        compute_metrics_net,
        compute_metrics_net_entity,
    )

    all_forecasts = []

    # Assign 85/10/5 split
    tier1_df = _assign_backtest_split(tier1_df, train_ratio=0.85, valid_ratio=0.10)

    # Get base feature columns (without LP)
    base_feature_cols = get_base_feature_cols(tier1_df)

    # Get TRP-specific extra features
    trp_extra_features = get_trp_extra_features(tier1_df)

    # Train and predict for each (LG × Horizon) combination
    for lg in LIQUIDITY_GROUPS:
        # Filter to liquidity group
        df_lg = tier1_df[tier1_df["liquidity_group"] == lg].copy()

        if df_lg.empty:
            logger.warning(f"No data for liquidity group {lg}, skipping")
            continue

        # Determine extra features for this LG
        extra_features = trp_extra_features if lg == "TRP" else []

        for horizon in HORIZONS:
            logger.info(f"[Backtest] Training model for {lg} - Horizon {horizon}")

            # Get feature columns for this horizon (with horizon-specific LP)
            feature_cols = get_feature_cols_for_horizon(
                horizon, base_feature_cols, all_cols=df_lg.columns.tolist()
            )

            # Add extra features if available
            if extra_features:
                feature_cols = feature_cols + [
                    f
                    for f in extra_features
                    if f not in feature_cols and f in df_lg.columns
                ]

            target_col = f"y_h{horizon}"

            # For training: use rows with non-null targets in train/valid splits
            df_trainval = df_lg[
                (df_lg["split"].isin(["train", "valid"])) &
                (df_lg[target_col].notna())
            ].copy()

            if df_trainval.empty:
                logger.warning(
                    f"No valid targets for {lg} H{horizon}, skipping"
                )
                continue

            try:
                # Train point model on train+valid data
                model, val_metrics, best_iter = train_lgbm_model(
                    df_trainval, feature_cols, target_col
                )

                logger.info(
                    f"{lg} H{horizon}: val_wape={val_metrics['wape']:.4f}, "
                    f"best_iter={best_iter}"
                )

                # Train quantile models (p10, p50, p90)
                model_q10, _, _ = train_lgbm_quantile_model(
                    df_trainval, feature_cols, target_col, alpha=0.10
                )
                model_q50, _, _ = train_lgbm_quantile_model(
                    df_trainval, feature_cols, target_col, alpha=0.50
                )
                model_q90, _, _ = train_lgbm_quantile_model(
                    df_trainval, feature_cols, target_col, alpha=0.90
                )

                # For backtest mode: predict ONLY for test split (last 5%)
                df_test = df_lg[df_lg["split"] == "test"].copy()

                if df_test.empty:
                    logger.warning(
                        f"No test rows for {lg} H{horizon}, skipping"
                    )
                    continue

                # Point predictions
                predictions = predict_lgbm(model, df_test, feature_cols)

                # Quantile predictions
                q10 = predict_lgbm(model_q10, df_test, feature_cols)
                q50 = predict_lgbm(model_q50, df_test, feature_cols)
                q90 = predict_lgbm(model_q90, df_test, feature_cols)

                # Build output DataFrame
                output = df_test[
                    ["entity", "liquidity_group", "week_start"]
                ].copy()
                output["horizon"] = horizon
                output["target_week_start"] = output["week_start"] + pd.Timedelta(
                    days=7 * horizon
                )

                # Backtest mode: include actual_value for evaluation
                output["actual_value"] = df_test[target_col].values

                output["y_pred_point"] = predictions

                # Add LP baseline point for comparison
                # h=1-4: use W{h}_Forecast, h>=5: NaN
                lp_col = LP_FORECAST_COLS.get(horizon)
                if lp_col is not None and lp_col in df_test.columns:
                    output["lp_baseline_point"] = df_test[lp_col].values
                else:
                    output["lp_baseline_point"] = np.nan

                output["y_pred_p10"] = q10
                output["y_pred_p50"] = q50
                output["y_pred_p90"] = q90

                output["model_type"] = "lightgbm"
                output["is_pass_through"] = False

                all_forecasts.append(output)

            except Exception as e:
                logger.error(f"Error training {lg} H{horizon}: {e}")
                continue

    # Add Tier-2 LP pass-through forecasts for test weeks
    tier2_forecasts = _build_tier2_passthrough_backtest(tier2_df)
    if not tier2_forecasts.empty:
        all_forecasts.append(tier2_forecasts)

    # Combine all forecasts
    if all_forecasts:
        forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    else:
        forecasts_df = pd.DataFrame(columns=BACKTEST_OUTPUT_COLS)

    # Ensure correct output columns for backtest (includes lp_baseline_point)
    forecasts_df = forecasts_df[BACKTEST_OUTPUT_COLS]

    # Compute and save metrics at 4 aggregation levels
    metrics_paths = _compute_and_save_backtest_metrics(forecasts_df, ref_week_start)

    return forecasts_df, metrics_paths


def _compute_and_save_backtest_metrics(
    forecasts_df: pd.DataFrame,
    ref_week_start: date,
) -> Dict[str, str]:
    """
    Compute and save backtest metrics at 4 aggregation levels.

    For each level, saves TWO versions:
    - Full metrics: includes all data (Tier-1 + Tier-2 passthroughs)
    - Clean metrics: Tier-1 only (true ML predictions, no LP passthrough)

    Uses Aggregate-then-Error WAPE (Treasury-aligned):
    - WAPE = |sum(actual) - sum(pred)| / |sum(actual)|
    - Over/under predictions cancel out, relevant for total cash position

    Args:
        forecasts_df: Backtest predictions DataFrame with lp_baseline_point.
        ref_week_start: Reference week for output directory naming.

    Returns:
        Dict mapping metric file names to their paths.
    """
    from hubbleAI.evaluation.metrics import (
        compute_metrics_by_lg,
        compute_metrics_by_entity,
        compute_metrics_net,
        compute_metrics_net_entity,
    )

    metrics_paths: Dict[str, str] = {}

    # Create output directory
    metrics_dir = BACKTEST_METRICS_DIR / ref_week_start.isoformat()
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing backtest metrics (full + clean versions)...")

    # 1. LG-level metrics: (week_start, liquidity_group, horizon)
    # 1a. Full (includes passthroughs)
    try:
        metrics_lg = compute_metrics_by_lg(forecasts_df, include_passthrough=True)
        lg_path = metrics_dir / "metrics_by_lg.parquet"
        metrics_lg.to_parquet(lg_path, index=False)
        metrics_paths["metrics_by_lg"] = str(lg_path)
        logger.info(f"LG-level metrics (full) saved: {len(metrics_lg)} rows")
    except Exception as e:
        logger.error(f"Error computing LG-level metrics: {e}")

    # 1b. Clean (Tier-1 only - true ML performance)
    try:
        metrics_lg_clean = compute_metrics_by_lg(forecasts_df, include_passthrough=False)
        lg_clean_path = metrics_dir / "metrics_by_lg_clean.parquet"
        metrics_lg_clean.to_parquet(lg_clean_path, index=False)
        metrics_paths["metrics_by_lg_clean"] = str(lg_clean_path)
        logger.info(f"LG-level metrics (clean) saved: {len(metrics_lg_clean)} rows")
    except Exception as e:
        logger.error(f"Error computing LG-level clean metrics: {e}")

    # 2. Entity-level metrics: (week_start, entity, liquidity_group, horizon)
    # Full only (entity-level already shows passthrough flag)
    try:
        metrics_entity = compute_metrics_by_entity(forecasts_df, include_passthrough=True)
        entity_path = metrics_dir / "metrics_by_entity.parquet"
        metrics_entity.to_parquet(entity_path, index=False)
        metrics_paths["metrics_by_entity"] = str(entity_path)
        logger.info(f"Entity-level metrics saved: {len(metrics_entity)} rows")
    except Exception as e:
        logger.error(f"Error computing Entity-level metrics: {e}")

    # 3. Net-level metrics: (week_start, horizon) - TRR+TRP summed
    # 3a. Full (includes passthroughs)
    try:
        metrics_net = compute_metrics_net(forecasts_df, include_passthrough=True)
        net_path = metrics_dir / "metrics_net.parquet"
        metrics_net.to_parquet(net_path, index=False)
        metrics_paths["metrics_net"] = str(net_path)
        logger.info(f"Net-level metrics (full) saved: {len(metrics_net)} rows")
    except Exception as e:
        logger.error(f"Error computing Net-level metrics: {e}")

    # 3b. Clean (Tier-1 only - true ML performance)
    try:
        metrics_net_clean = compute_metrics_net(forecasts_df, include_passthrough=False)
        net_clean_path = metrics_dir / "metrics_net_clean.parquet"
        metrics_net_clean.to_parquet(net_clean_path, index=False)
        metrics_paths["metrics_net_clean"] = str(net_clean_path)
        logger.info(f"Net-level metrics (clean) saved: {len(metrics_net_clean)} rows")
    except Exception as e:
        logger.error(f"Error computing Net-level clean metrics: {e}")

    # 4. Net-Entity-level metrics: (week_start, entity, horizon) - TRR+TRP summed per entity
    try:
        metrics_net_entity = compute_metrics_net_entity(forecasts_df, include_passthrough=True)
        net_entity_path = metrics_dir / "metrics_net_entity.parquet"
        metrics_net_entity.to_parquet(net_entity_path, index=False)
        metrics_paths["metrics_net_entity"] = str(net_entity_path)
        logger.info(f"Net-Entity-level metrics saved: {len(metrics_net_entity)} rows")
    except Exception as e:
        logger.error(f"Error computing Net-Entity-level metrics: {e}")

    # 5. Compute and save diagnostics (always uses full dataset)
    diagnostic_paths = _compute_and_save_diagnostics(forecasts_df, ref_week_start)
    metrics_paths.update(diagnostic_paths)

    return metrics_paths


def _compute_and_save_diagnostics(
    forecasts_df: pd.DataFrame,
    ref_week_start: date,
) -> Dict[str, str]:
    """
    Compute and save backtest diagnostics.

    Diagnostics provide deeper analysis of model performance:
    - Horizon profiles: WAPE, MAE, MSE, RMSE per horizon
    - Residual diagnostics: Distribution of prediction errors per LG × horizon
    - Entity stability: Error volatility across time per entity × horizon
    - Model vs LP wins: Win-loss analysis per LG × horizon

    Note: Diagnostics always use the FULL dataset (includes Tier-2 passthroughs).

    Args:
        forecasts_df: Backtest predictions DataFrame with lp_baseline_point.
        ref_week_start: Reference week for output directory naming.

    Returns:
        Dict mapping diagnostic file names to their paths.
    """
    from hubbleAI.evaluation.metrics import (
        compute_horizon_profiles,
        compute_residual_diagnostics,
        compute_entity_stability,
        compute_model_vs_lp_wins,
    )

    diagnostic_paths: Dict[str, str] = {}

    # Create diagnostics subdirectory
    diagnostics_dir = BACKTEST_METRICS_DIR / ref_week_start.isoformat() / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing backtest diagnostics...")

    # 1. Horizon profiles
    try:
        horizon_profiles = compute_horizon_profiles(forecasts_df)
        horizon_path = diagnostics_dir / "metrics_horizon_profiles.parquet"
        horizon_profiles.to_parquet(horizon_path, index=False)
        diagnostic_paths["metrics_horizon_profiles"] = str(horizon_path)
        logger.info(f"Horizon profiles saved: {len(horizon_profiles)} rows")
    except Exception as e:
        logger.error(f"Error computing horizon profiles: {e}")

    # 2. Residual diagnostics
    try:
        residuals = compute_residual_diagnostics(forecasts_df)
        residuals_path = diagnostics_dir / "residual_diagnostics.parquet"
        residuals.to_parquet(residuals_path, index=False)
        diagnostic_paths["residual_diagnostics"] = str(residuals_path)
        logger.info(f"Residual diagnostics saved: {len(residuals)} rows")
    except Exception as e:
        logger.error(f"Error computing residual diagnostics: {e}")

    # 3. Entity stability
    try:
        stability = compute_entity_stability(forecasts_df)
        stability_path = diagnostics_dir / "entity_stability.parquet"
        stability.to_parquet(stability_path, index=False)
        diagnostic_paths["entity_stability"] = str(stability_path)
        logger.info(f"Entity stability saved: {len(stability)} rows")
    except Exception as e:
        logger.error(f"Error computing entity stability: {e}")

    # 4. Model vs LP wins
    try:
        wins = compute_model_vs_lp_wins(forecasts_df)
        wins_path = diagnostics_dir / "model_vs_lp_wins.parquet"
        wins.to_parquet(wins_path, index=False)
        diagnostic_paths["model_vs_lp_wins"] = str(wins_path)
        logger.info(f"Model vs LP wins saved: {len(wins)} rows")
    except Exception as e:
        logger.error(f"Error computing model vs LP wins: {e}")

    return diagnostic_paths


def _build_tier2_passthrough_forward(
    ref_week_start: date,
    tier2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build LP pass-through forecasts for Tier-2 entities in FORWARD mode.

    For Tier-2 entities:
    - Horizons 1-4: Use LP forecast values (only for ref_week_start)
    - Horizons 5-8: No forecast (LP doesn't extend that far)

    Args:
        ref_week_start: The reference Monday.
        tier2_df: DataFrame with Tier-2 entity data.

    Returns:
        DataFrame with Tier-2 pass-through forecasts for ref_week_start only.
    """
    if tier2_df.empty:
        return pd.DataFrame()

    # Filter to ref_week_start only
    ref_week_ts = pd.Timestamp(ref_week_start)
    tier2_ref = tier2_df[tier2_df["week_start"] == ref_week_ts].copy()

    if tier2_ref.empty:
        return pd.DataFrame()

    tier2_forecasts = []

    # Only horizons 1-4 have LP forecasts
    for horizon in [1, 2, 3, 4]:
        lp_col = LP_FORECAST_COLS.get(horizon)
        if lp_col is None or lp_col not in tier2_ref.columns:
            continue

        # Get rows with valid LP for this horizon
        df_h = tier2_ref[tier2_ref[lp_col].notna()].copy()

        if df_h.empty:
            continue

        output = df_h[["entity", "liquidity_group", "week_start"]].copy()
        output["horizon"] = horizon
        output["target_week_start"] = output["week_start"] + pd.Timedelta(days=7 * horizon)

        # Forward mode: actual_value is NaN
        output["actual_value"] = np.nan

        output["y_pred_point"] = df_h[lp_col].values
        output["y_pred_p10"] = np.nan
        output["y_pred_p50"] = np.nan
        output["y_pred_p90"] = np.nan

        output["model_type"] = "lp_passthrough"
        output["is_pass_through"] = True

        tier2_forecasts.append(output)

    if tier2_forecasts:
        return pd.concat(tier2_forecasts, ignore_index=True)
    return pd.DataFrame()


def _build_tier2_passthrough_backtest(
    tier2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build LP pass-through forecasts for Tier-2 entities in BACKTEST mode.

    For Tier-2 entities:
    - Horizons 1-4: Use LP forecast values (only for test split weeks)
    - Horizons 5-8: No forecast (LP doesn't extend that far)

    Note: For Tier-2 rows, both y_pred_point and lp_baseline_point are set
    to the LP forecast value (since ML prediction = LP pass-through).
    This ensures consistency in metric computation.

    Args:
        tier2_df: DataFrame with Tier-2 entity data.

    Returns:
        DataFrame with Tier-2 pass-through forecasts for test weeks.
    """
    if tier2_df.empty:
        return pd.DataFrame()

    # Apply 85/10/5 split to tier2_df
    tier2_df = _assign_backtest_split(tier2_df, train_ratio=0.85, valid_ratio=0.10)

    # Filter to test split only
    tier2_test = tier2_df[tier2_df["split"] == "test"].copy()

    if tier2_test.empty:
        return pd.DataFrame()

    tier2_forecasts = []

    # Only horizons 1-4 have LP forecasts
    for horizon in [1, 2, 3, 4]:
        lp_col = LP_FORECAST_COLS.get(horizon)
        if lp_col is None or lp_col not in tier2_test.columns:
            continue

        target_col = f"y_h{horizon}"

        # Get rows with valid LP for this horizon
        df_h = tier2_test[tier2_test[lp_col].notna()].copy()

        if df_h.empty:
            continue

        output = df_h[["entity", "liquidity_group", "week_start"]].copy()
        output["horizon"] = horizon
        output["target_week_start"] = output["week_start"] + pd.Timedelta(days=7 * horizon)

        # Backtest mode: include actual_value for evaluation (if available)
        if target_col in df_h.columns:
            output["actual_value"] = df_h[target_col].values
        else:
            output["actual_value"] = np.nan

        # For Tier-2: y_pred_point is LP pass-through
        output["y_pred_point"] = df_h[lp_col].values
        # Also set lp_baseline_point for consistency in metrics
        output["lp_baseline_point"] = df_h[lp_col].values

        output["y_pred_p10"] = np.nan
        output["y_pred_p50"] = np.nan
        output["y_pred_p90"] = np.nan

        output["model_type"] = "lp_passthrough"
        output["is_pass_through"] = True

        tier2_forecasts.append(output)

    if tier2_forecasts:
        return pd.concat(tier2_forecasts, ignore_index=True)
    return pd.DataFrame()


def _save_run_status(status: ForecastRunStatus) -> Path:
    """Persist run status as JSON for later use by the UI."""
    RUN_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_STATUS_DIR / f"run_status_{status.run_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(status),
                "as_of_date": status.as_of_date.isoformat(),
                "ref_week_start": status.ref_week_start.isoformat(),
                "created_at": status.created_at.isoformat(),
            },
            f,
            indent=2,
        )
    # Also update a "latest" pointer file
    latest_pointer = RUN_STATUS_DIR / "latest_run_status.json"
    latest_pointer.write_text(path.name, encoding="utf-8")
    return path


def _generate_run_id(as_of_date: date, trigger_source: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{as_of_date.isoformat()}_{trigger_source}_{timestamp}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_forecast(
    trigger_source: Literal["scheduler", "manual", "notebook"] = "scheduler",
    mode: ForecastMode = "forward",
    as_of_week: Optional[date] = None,
    *,
    force_run: bool = False,
) -> ForecastRunStatus:
    """
    Run the end-to-end forecast pipeline.

    This function supports two modes:
    - "forward": Normal operational forecast. Outputs ONLY forecasts for the next
      8 weeks after ref_week_start (the latest Monday in the dataset).
    - "backtest": Evaluation mode. Uses 85/10/5 chronological split and outputs
      predictions only for the last 5% (test split) weeks.

    IMPORTANT: All weeks are Monday-based:
    - week_start is always a Monday
    - target_week_start is always a Monday (ref_week_start + 7*horizon days)

    Args:
        trigger_source: Whether triggered by 'scheduler', 'manual', or 'notebook'.
        mode: "forward" for operational forecasts, "backtest" for evaluation.
        as_of_week: Optional Monday date to use as reference. If None, uses the
                    latest Monday found in the dataset.
        force_run: If True, run even if some data is missing.

    Returns:
        ForecastRunStatus with run details including ref_week_start and mode.

    Forward mode output:
        data/processed/forecasts/{ref_week_start}/forecasts.parquet
        - Only 8 horizons × (#entities × #LG)
        - week_start == ref_week_start (Monday)
        - target_week_start == ref_week_start + (7*H) days (Monday)
        - actual_value = NaN (future not yet observed)

    Backtest mode output:
        data/processed/backtests/{ref_week_start}/backtest_predictions.parquet
        - Predictions for ALL historical Monday weeks within the last 5% test split
        - actual_value = observed amount for that target_week_start
    """
    # Use a placeholder as_of_date for initial checks
    as_of_date = as_of_week if as_of_week else get_default_as_of_date()

    run_id = _generate_run_id(as_of_date, f"{trigger_source}_{mode}")
    created_at = datetime.utcnow()

    logger.info(
        "Starting forecast run %s mode=%s (trigger=%s)",
        run_id,
        mode,
        trigger_source,
    )

    is_ready, missing = check_data_availability(as_of_date)
    if not is_ready and not force_run:
        message = "Forecast not run: missing required inputs: " + ", ".join(missing)
        logger.warning(message)
        # Return placeholder ref_week_start (will be updated after data load)
        status = ForecastRunStatus(
            run_id=run_id,
            as_of_date=as_of_date,
            ref_week_start=as_of_date,  # Placeholder
            mode=mode,
            trigger_source=trigger_source,
            status="data_missing",
            created_at=created_at,
            message=message,
            missing_inputs=missing,
            output_paths={},
            metrics_paths={},
        )
        _save_run_status(status)
        return status

    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        tier1_df, tier2_df, ref_week_start = _load_and_prepare_data(as_of_date)

        # Override ref_week_start if as_of_week was explicitly provided
        if as_of_week is not None:
            # Validate that as_of_week is a Monday
            if as_of_week.weekday() != 0:
                raise ValueError(
                    f"as_of_week must be a Monday, got {as_of_week} "
                    f"(weekday={as_of_week.weekday()})"
                )
            ref_week_start = as_of_week

        logger.info(f"ref_week_start (Monday): {ref_week_start}")

        # Build and run models based on mode
        if mode == "forward":
            logger.info("Running in FORWARD mode...")
            forecasts_df = _build_and_run_models_forward(
                ref_week_start, tier1_df, tier2_df
            )

            # Save forecasts to forward path
            output_dir = FORECASTS_DIR / ref_week_start.isoformat()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "forecasts.parquet"
            forecasts_df.to_parquet(output_path, index=False)

            output_paths = {"forecasts": str(output_path)}
            message = (
                f"Forward forecast completed. Generated {len(forecasts_df)} forecasts "
                f"for ref_week_start={ref_week_start}."
            )

        elif mode == "backtest":
            logger.info("Running in BACKTEST mode (85/10/5 split)...")

            # For backtest, ref_week_start is the last Monday in validation set
            # (i.e., the cutoff before the 5% test weeks)
            unique_weeks = sorted(tier1_df["week_start"].drop_duplicates().tolist())
            n = len(unique_weeks)
            valid_end_idx = int(n * 0.95)  # 85% train + 10% valid = 95%
            if valid_end_idx > 0:
                backtest_ref_week = pd.Timestamp(unique_weeks[valid_end_idx - 1]).date()
            else:
                backtest_ref_week = ref_week_start

            # Build backtest predictions and compute metrics
            forecasts_df, metrics_paths = _build_and_run_models_backtest(
                backtest_ref_week, tier1_df, tier2_df
            )

            # Save backtest predictions
            output_dir = BACKTESTS_DIR / backtest_ref_week.isoformat()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "backtest_predictions.parquet"
            forecasts_df.to_parquet(output_path, index=False)

            output_paths = {"backtest": str(output_path)}
            ref_week_start = backtest_ref_week  # Update for status
            message = (
                f"Backtest completed. Generated {len(forecasts_df)} predictions "
                f"for test weeks (last 5%). ref_week_start={ref_week_start}. "
                f"Metrics saved for {len(metrics_paths)} aggregation levels."
            )

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'forward' or 'backtest'.")

        # For forward mode, no metrics are computed (no actuals available)
        if mode == "forward":
            metrics_paths = {}

        logger.info(message)
        status = ForecastRunStatus(
            run_id=run_id,
            as_of_date=as_of_date,
            ref_week_start=ref_week_start,
            mode=mode,
            trigger_source=trigger_source,
            status="success",
            created_at=created_at,
            message=message,
            missing_inputs=[],
            output_paths=output_paths,
            metrics_paths=metrics_paths,
        )
        _save_run_status(status)
        return status

    except Exception as exc:
        logger.exception("Error during forecast run %s", run_id)
        status = ForecastRunStatus(
            run_id=run_id,
            as_of_date=as_of_date,
            ref_week_start=as_of_date,  # Placeholder on error
            mode=mode,
            trigger_source=trigger_source,
            status="error",
            created_at=created_at,
            message=str(exc),
            missing_inputs=[],
            output_paths={},
            metrics_paths={},
        )
        _save_run_status(status)
        return status
