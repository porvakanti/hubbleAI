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

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# These paths assume the following layout at repo root:
#   data/raw/        <-- input files (actuals, LP, FX)
#   data/processed/  <-- outputs written by the pipeline
#
# In a real project, these could come from environment variables or a config
# file; for now we keep them simple and file-based.
REPO_ROOT = Path(__file__).resolve().parents[2]  # src/hubbleAI/pipeline.py -> repo root
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RUN_STATUS_DIR = DATA_PROCESSED_DIR / "run_status"
FORECASTS_DIR = DATA_PROCESSED_DIR / "forecasts"
METRICS_DIR = DATA_PROCESSED_DIR / "metrics"

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
        return Path(self.output_paths.get("forecasts")) if "forecasts" in self.output_paths else None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_default_as_of_date(today: Optional[date] = None) -> date:
    """
    Return the default as_of_date for a run.

    Convention (can be adjusted):
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

    # These are placeholders; adjust names to match actual files in data/raw.
    actuals_path = DATA_RAW_DIR / "actuals.csv"
    lp_path = DATA_RAW_DIR / "liquidity_plan.csv"
    fx_path = DATA_RAW_DIR / "fx_rates.csv"

    if not actuals_path.exists():
        missing.append("ACTUALS_FILE")
    if not lp_path.exists():
        missing.append("LP_FILE")
    if not fx_path.exists():
        missing.append("FX_FILE")

    # In the future we could inspect dates inside the files and verify that
    # they cover as_of_date and the required horizons.

    is_ready = len(missing) == 0
    return is_ready, missing


def _load_and_prepare_data(as_of_date: date) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and minimally prepare actuals, LP, and FX data.

    Currently file-based (data/raw). In the future, this function will be
    the main place to swap in Denodo / Databricks / Reval connectors.

    Returns:
        (actuals_df, lp_df, fx_df)
    """
    # TODO: Replace placeholder filenames with the actual names used in the project.
    actuals_path = DATA_RAW_DIR / "actuals.csv"
    lp_path = DATA_RAW_DIR / "liquidity_plan.csv"
    fx_path = DATA_RAW_DIR / "fx_rates.csv"

    actuals_df = pd.read_csv(actuals_path)
    lp_df = pd.read_csv(lp_path)
    fx_df = pd.read_csv(fx_path)

    # TODO: filter by as_of_date, apply any necessary preprocessing
    # (e.g. convert to weekly, join FX, etc.). For now we just return
    # the raw DataFrames.
    return actuals_df, lp_df, fx_df


def _build_and_run_models(
    as_of_date: date,
    actuals_df: pd.DataFrame,
    lp_df: pd.DataFrame,
    fx_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the LG × horizon models and return a unified forecast DataFrame.

    This function is a placeholder for the real modeling logic. It should:

    - Build feature matrices per (Liquidity Group × horizon) using horizon-specific
      LP features (W1..W4 only for H1..H4; none for H5..H8).
    - Load trained LightGBM models for each (LG × horizon) and for each prediction
      type (point, p10, p50, p90), or train them if we are still in dev mode.
    - Generate a DataFrame with 4 predictions per row:
        y_pred_point, y_pred_p10, y_pred_p50, y_pred_p90
    - Merge Tier-2 entities with LP pass-through forecasts.

    For now, this returns an empty DataFrame as a stub.
    """
    # TODO: implement actual modeling logic using src/hubbleAI/features and src/hubbleAI/models
    columns = [
        "entity",
        "liquidity_group",
        "as_of_date",
        "target_week",
        "horizon",
        "y_pred_point",
        "y_pred_p10",
        "y_pred_p50",
        "y_pred_p90",
        "model_type",
        "is_pass_through",
    ]
    return pd.DataFrame(columns=columns)


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
        - Build features and run models to produce 4 predictions per row
          (point, p10, p50, p90).
        - Merge Tier-2 LP pass-through outputs.
        - Store forecast outputs under data/processed/forecasts/{as_of_date}/.
        - Optionally compute backtest metrics (future).
        - Return a ForecastRunStatus with status="success".
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
        actuals_df, lp_df, fx_df = _load_and_prepare_data(as_of_date)
        forecasts_df = _build_and_run_models(as_of_date, actuals_df, lp_df, fx_df)

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

        message = "Forecast run completed successfully."
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
