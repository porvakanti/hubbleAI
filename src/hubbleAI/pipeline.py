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
    METRICS_DIR,
    HORIZONS,
    LIQUIDITY_GROUPS,
    TIER2_LIST,
    LP_FORECAST_COLS,
    TRP_EXTRA_FEATURES,
    ACTUALS_FILENAME,
    LP_FILENAME,
    FX_FILENAME,
    MIN_HISTORY_WEEKS,
)

# Ensure directories exist
RUN_STATUS_DIR.mkdir(parents=True, exist_ok=True)
FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

RunStatus = Literal["success", "data_missing", "skipped", "error"]


@dataclass
class ForecastRunStatus:
    """
    Summary of a single forecast pipeline run.

    This is what the Streamlit UI will read to show "last run" status.
    """

    run_id: str
    as_of_date: date
    trigger_source: Literal["scheduler", "manual"]
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all data sources for forecasting.

    Returns:
        Tuple of (model_ready_df, tier2_df)
        - model_ready_df: DataFrame with features and targets for Tier-1 entities
        - tier2_df: DataFrame with Tier-2 entities for LP pass-through
    """
    from hubbleAI.data_prep import prepare_weekly_data
    from hubbleAI.data_prep.prepare import filter_tier1_with_history, add_target_columns
    from hubbleAI.features import build_all_features
    from hubbleAI.models.lightgbm_model import assign_split

    # Prepare weekly data (loads and merges all sources)
    merged_df, _ = prepare_weekly_data(as_of_date=as_of_date)

    # Build all features
    featured_df = build_all_features(merged_df)

    # Add target columns
    featured_df = add_target_columns(featured_df)

    # Filter Tier-1 entities with sufficient history for ML
    tier1_df = filter_tier1_with_history(featured_df, MIN_HISTORY_WEEKS)

    # Get Tier-2 entities (for LP pass-through)
    tier2_df = featured_df[featured_df["tier"] == "Tier2"].copy()

    # Add target_week_start for output purposes
    tier1_df["target_week_start"] = tier1_df["week_start"] + pd.Timedelta(days=7)

    # Assign train/valid/test split
    tier1_df = assign_split(tier1_df)

    # Drop rows without full 8-week future targets
    target_cols = [f"y_h{h}" for h in HORIZONS]
    model_ready_df = tier1_df.dropna(subset=target_cols).copy()

    return model_ready_df, tier2_df


def _build_and_run_models(
    as_of_date: date,
    model_ready_df: pd.DataFrame,
    tier2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the LG × horizon models and return a unified forecast DataFrame.

    This function:
    - Trains LightGBM models for each (Liquidity Group × Horizon) combination
    - Uses horizon-specific LP features (W1..W4 for H1..H4; none for H5..H8)
    - Generates predictions for Tier-1 entities
    - Adds Tier-2 LP pass-through forecasts

    Args:
        as_of_date: The as-of date for the forecast.
        model_ready_df: Prepared DataFrame for Tier-1 ML forecasting.
        tier2_df: DataFrame with Tier-2 entities for LP pass-through.

    Returns:
        Unified forecast DataFrame.
    """
    from hubbleAI.features import get_base_feature_cols, get_feature_cols_for_horizon
    from hubbleAI.features.builder import get_trp_extra_features
    from hubbleAI.models.lightgbm_model import (
        train_lgbm_model,
        predict_lgbm,
    )

    all_forecasts = []

    # Get base feature columns (without LP)
    base_feature_cols = get_base_feature_cols(model_ready_df)

    # Get TRP-specific extra features
    trp_extra_features = get_trp_extra_features(model_ready_df)

    # Train and predict for each (LG × Horizon) combination
    for lg in LIQUIDITY_GROUPS:
        # Filter to liquidity group
        df_lg = model_ready_df[model_ready_df["liquidity_group"] == lg].copy()

        if df_lg.empty:
            logger.warning(f"No data for liquidity group {lg}, skipping")
            continue

        # Determine extra features for this LG
        extra_features = trp_extra_features if lg == "TRP" else []

        for horizon in HORIZONS:
            logger.info(f"Training model for {lg} - Horizon {horizon}")

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

            # Drop rows with missing target
            df_horizon = df_lg.dropna(subset=[target_col]).copy()

            if df_horizon.empty:
                logger.warning(
                    f"No valid targets for {lg} H{horizon}, skipping"
                )
                continue

            try:
                # Train model
                model, val_metrics, best_iter = train_lgbm_model(
                    df_horizon, feature_cols, target_col
                )

                logger.info(
                    f"{lg} H{horizon}: val_wape={val_metrics['wape']:.4f}, "
                    f"best_iter={best_iter}"
                )

                # Generate predictions for all splits
                predictions = predict_lgbm(model, df_horizon, feature_cols)

                # Build output DataFrame
                output = df_horizon[
                    ["entity", "liquidity_group", "week_start"]
                ].copy()
                output["horizon"] = horizon
                output["target_week_start"] = output["week_start"] + pd.Timedelta(
                    weeks=horizon
                )
                output["y_pred_point"] = predictions

                # TODO: implement proper quantile models for p10/p50/p90 in a later task
                output["y_pred_p10"] = np.nan
                output["y_pred_p50"] = np.nan
                output["y_pred_p90"] = np.nan

                output["model_type"] = "lightgbm"
                output["is_pass_through"] = False

                all_forecasts.append(output)

            except Exception as e:
                logger.error(f"Error training {lg} H{horizon}: {e}")
                continue

    # Add Tier-2 LP pass-through forecasts
    tier2_forecasts = _build_tier2_passthrough(as_of_date, tier2_df)
    if not tier2_forecasts.empty:
        all_forecasts.append(tier2_forecasts)

    # Combine all forecasts
    if all_forecasts:
        forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    else:
        # Return empty DataFrame with correct schema
        forecasts_df = pd.DataFrame(
            columns=[
                "entity",
                "liquidity_group",
                "week_start",
                "target_week_start",
                "horizon",
                "y_pred_point",
                "y_pred_p10",
                "y_pred_p50",
                "y_pred_p90",
                "model_type",
                "is_pass_through",
            ]
        )

    # Clean up output columns
    output_cols = [
        "entity",
        "liquidity_group",
        "week_start",
        "target_week_start",
        "horizon",
        "y_pred_point",
        "y_pred_p10",
        "y_pred_p50",
        "y_pred_p90",
        "model_type",
        "is_pass_through",
    ]
    forecasts_df = forecasts_df[output_cols]

    return forecasts_df


def _build_tier2_passthrough(
    as_of_date: date,
    tier2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build LP pass-through forecasts for Tier-2 entities.

    For Tier-2 entities:
    - Horizons 1-4: Use LP forecast values
    - Horizons 5-8: No forecast (LP doesn't extend that far)

    Args:
        as_of_date: The as-of date for the forecast.
        tier2_df: DataFrame with Tier-2 entity data.

    Returns:
        DataFrame with Tier-2 pass-through forecasts.
    """
    if tier2_df.empty:
        return pd.DataFrame()

    tier2_forecasts = []

    # Only horizons 1-4 have LP forecasts
    for horizon in [1, 2, 3, 4]:
        lp_col = LP_FORECAST_COLS.get(horizon)
        if lp_col is None or lp_col not in tier2_df.columns:
            continue

        # Get rows with valid LP for this horizon
        df_h = tier2_df[tier2_df[lp_col].notna()].copy()

        if df_h.empty:
            continue

        output = df_h[["entity", "liquidity_group", "week_start"]].copy()
        output["horizon"] = horizon
        output["target_week_start"] = output["week_start"] + pd.Timedelta(weeks=horizon)
        output["y_pred_point"] = df_h[lp_col].values

        # TODO: implement proper quantile models for p10/p50/p90 in a later task
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
    as_of_date: Optional[date] = None,
    *,
    trigger_source: Literal["scheduler", "manual"] = "scheduler",
    force_run: bool = False,
    compute_backtest_metrics: bool = False,
) -> ForecastRunStatus:
    """
    Run the end-to-end forecast pipeline for a given as_of_date.

    - If as_of_date is None, use a default business rule (e.g. last closed week).
    - Check data availability (actuals, LP, FX) using file presence for now.
    - If data is missing and force_run is False:
        - Do NOT run forecasts.
        - Return a ForecastRunStatus with status="data_missing" and the missing inputs.
    - If data is sufficient (or force_run is True):
        - Load and prepare input data.
        - Build features and run models to produce point predictions.
        - Merge Tier-2 LP pass-through outputs.
        - Store forecast outputs under data/processed/forecasts/{as_of_date}/.
        - Optionally compute backtest metrics (future).
        - Return a ForecastRunStatus with status="success".

    Args:
        as_of_date: The as-of date for the forecast run. Defaults to latest valid date.
        trigger_source: Whether triggered by 'scheduler' or 'manual'.
        force_run: If True, run even if some data is missing.
        compute_backtest_metrics: If True, compute and store backtest metrics.

    Returns:
        ForecastRunStatus with run details.
    """
    if as_of_date is None:
        as_of_date = get_default_as_of_date()

    run_id = _generate_run_id(as_of_date, trigger_source)
    created_at = datetime.utcnow()

    logger.info(
        "Starting forecast run %s for as_of_date=%s (trigger=%s)",
        run_id,
        as_of_date,
        trigger_source,
    )

    is_ready, missing = check_data_availability(as_of_date)
    if not is_ready and not force_run:
        message = "Forecast not run: missing required inputs: " + ", ".join(missing)
        logger.warning(message)
        status = ForecastRunStatus(
            run_id=run_id,
            as_of_date=as_of_date,
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
        model_ready_df, tier2_df = _load_and_prepare_data(as_of_date)

        # Build and run models
        logger.info("Training models and generating forecasts...")
        forecasts_df = _build_and_run_models(as_of_date, model_ready_df, tier2_df)

        # Save forecasts
        as_of_dir = FORECASTS_DIR / as_of_date.isoformat()
        as_of_dir.mkdir(parents=True, exist_ok=True)
        forecasts_path = as_of_dir / "forecasts.parquet"
        forecasts_df.to_parquet(forecasts_path, index=False)

        output_paths = {"forecasts": str(forecasts_path)}

        metrics_paths: Dict[str, str] = {}
        if compute_backtest_metrics:
            # TODO: compute and save metrics under METRICS_DIR
            pass

        message = f"Forecast run completed successfully. Generated {len(forecasts_df)} forecasts."
        logger.info(message)
        status = ForecastRunStatus(
            run_id=run_id,
            as_of_date=as_of_date,
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
