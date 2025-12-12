"""
Service Layer for hubbleAI.

Provides clean API functions for loading forecasts, metrics, and health checks.
Used by Streamlit UI and external integrations.

This is a read-only wrapper around pipeline outputs - does NOT modify data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Tuple

import pandas as pd

from hubbleAI.config import (
    FORECASTS_DIR,
    BACKTESTS_DIR,
    BACKTEST_METRICS_DIR,
    RUN_STATUS_DIR,
    DATA_RAW_DIR,
    ACTUALS_FILENAME,
    LP_FILENAME,
    FX_FILENAME,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ForecastView:
    """Container for forward forecast data."""
    ref_week_start: date
    forecasts_df: pd.DataFrame
    mode: Literal["forward", "backtest"]
    run_status: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestView:
    """Container for backtest results with all metrics and diagnostics."""
    ref_week_start: date
    backtest_df: pd.DataFrame
    metrics: Dict[str, pd.DataFrame] = field(default_factory=dict)
    diagnostics: Dict[str, pd.DataFrame] = field(default_factory=dict)
    run_status: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Run Status Helpers
# ---------------------------------------------------------------------------


def _load_run_status_from_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a run status JSON file."""
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.warning(f"Failed to load run status from {path}: {e}")
        return None


def get_latest_run_status() -> Optional[Dict[str, Any]]:
    """
    Load the latest pipeline run status (any mode).

    Returns:
        Dict with run status fields or None if no status found.
    """
    try:
        latest_pointer = RUN_STATUS_DIR / "latest_run_status.json"
        if not latest_pointer.exists():
            return None

        latest_filename = latest_pointer.read_text(encoding="utf-8").strip()
        status_path = RUN_STATUS_DIR / latest_filename
        return _load_run_status_from_file(status_path)
    except Exception as e:
        logger.warning(f"Failed to get latest run status: {e}")
        return None


def get_last_run_by_mode(mode: Literal["forward", "backtest"]) -> Optional[Dict[str, Any]]:
    """
    Get the most recent run status for a specific mode.

    Args:
        mode: "forward" or "backtest"

    Returns:
        Dict with run status or None if no matching run found.
    """
    try:
        if not RUN_STATUS_DIR.exists():
            return None

        # Find all run status files
        status_files = list(RUN_STATUS_DIR.glob("run_status_*.json"))
        if not status_files:
            return None

        # Load and filter by mode
        matching_runs = []
        for path in status_files:
            data = _load_run_status_from_file(path)
            if data and data.get("mode") == mode:
                matching_runs.append(data)

        if not matching_runs:
            return None

        # Sort by created_at (descending) and return most recent
        matching_runs.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        return matching_runs[0]

    except Exception as e:
        logger.warning(f"Failed to get last run by mode {mode}: {e}")
        return None


# ---------------------------------------------------------------------------
# Forecast Loaders
# ---------------------------------------------------------------------------


def load_latest_forward_forecast() -> Optional[ForecastView]:
    """
    Load the latest forward forecast data.

    Returns:
        ForecastView with forecast DataFrame and metadata, or None.
    """
    try:
        # Get latest forward run status
        run_status = get_last_run_by_mode("forward")

        # Try to load from run status path first
        if run_status and run_status.get("output_paths", {}).get("forecasts"):
            forecasts_path = Path(run_status["output_paths"]["forecasts"])
            if forecasts_path.exists():
                df = pd.read_parquet(forecasts_path)
                ref_week = run_status.get("ref_week_start", "")
                if isinstance(ref_week, str):
                    ref_week = date.fromisoformat(ref_week)
                return ForecastView(
                    ref_week_start=ref_week,
                    forecasts_df=df,
                    mode="forward",
                    run_status=run_status,
                    extra={"n_rows": len(df), "source": "run_status"}
                )

        # Fallback: find latest forecast directory
        if FORECASTS_DIR.exists():
            subdirs = sorted(
                [d for d in FORECASTS_DIR.iterdir() if d.is_dir()],
                reverse=True
            )
            for subdir in subdirs:
                forecast_file = subdir / "forecasts.parquet"
                if forecast_file.exists():
                    df = pd.read_parquet(forecast_file)
                    ref_week = date.fromisoformat(subdir.name)
                    return ForecastView(
                        ref_week_start=ref_week,
                        forecasts_df=df,
                        mode="forward",
                        run_status=run_status,
                        extra={"n_rows": len(df), "source": "fallback"}
                    )

        return None

    except Exception as e:
        logger.error(f"Failed to load forward forecast: {e}")
        return None


def load_latest_backtest_results() -> Optional[BacktestView]:
    """
    Load the latest backtest results with all metrics and diagnostics.

    Returns:
        BacktestView with backtest predictions and all metrics/diagnostics.
    """
    try:
        # Get latest backtest run status
        run_status = get_last_run_by_mode("backtest")

        # Find latest metrics directory
        if not BACKTEST_METRICS_DIR.exists():
            logger.warning("Metrics directory not found")
            return None

        subdirs = sorted(
            [d for d in BACKTEST_METRICS_DIR.iterdir() if d.is_dir()],
            reverse=True
        )
        if not subdirs:
            logger.warning("No metrics subdirectories found")
            return None

        latest_dir = subdirs[0]
        ref_date_str = latest_dir.name
        ref_week = date.fromisoformat(ref_date_str)

        # Initialize result
        metrics: Dict[str, pd.DataFrame] = {}
        diagnostics: Dict[str, pd.DataFrame] = {}

        # Load main metrics files
        metrics_files = [
            "metrics_by_lg",
            "metrics_by_lg_clean",
            "metrics_by_entity",
            "metrics_net",
            "metrics_net_clean",
            "metrics_net_entity",
            "alpha_by_lg_horizon",
            "weekly_hybrid_breakdown",
        ]

        for name in metrics_files:
            path = latest_dir / f"{name}.parquet"
            if path.exists():
                try:
                    metrics[name] = pd.read_parquet(path)
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

        # Load diagnostics files (in diagnostics subfolder or same folder)
        diagnostics_dir = latest_dir.parent.parent / "diagnostics" / ref_date_str
        if not diagnostics_dir.exists():
            diagnostics_dir = latest_dir  # Fallback to same directory

        diagnostics_files = [
            "metrics_horizon_profiles",
            "residual_diagnostics",
            "entity_stability",
            "model_vs_lp_wins",
            "quantile_coverage_by_horizon",
            "quantile_coverage_by_lg_horizon",
            "pinball_by_horizon",
            "pinball_by_lg_horizon",
        ]

        for name in diagnostics_files:
            path = diagnostics_dir / f"{name}.parquet"
            if path.exists():
                try:
                    diagnostics[name] = pd.read_parquet(path)
                except Exception as e:
                    logger.warning(f"Failed to load diagnostic {name}: {e}")

        # Load backtest predictions
        backtest_df = pd.DataFrame()
        backtest_dir = BACKTESTS_DIR / ref_date_str
        backtest_path = backtest_dir / "backtest_predictions.parquet"
        if backtest_path.exists():
            backtest_df = pd.read_parquet(backtest_path)

        return BacktestView(
            ref_week_start=ref_week,
            backtest_df=backtest_df,
            metrics=metrics,
            diagnostics=diagnostics,
            run_status=run_status,
        )

    except Exception as e:
        logger.error(f"Failed to load backtest results: {e}")
        return None


# ---------------------------------------------------------------------------
# Data Health Helpers
# ---------------------------------------------------------------------------


def get_data_health_summary(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Get data health summary for raw input files.

    Args:
        as_of_date: Date to check (defaults to today)

    Returns:
        Dict with health status for each input file.
    """
    if as_of_date is None:
        as_of_date = date.today()

    result = {
        "as_of_date": as_of_date.isoformat(),
        "has_actuals": False,
        "has_lp": False,
        "has_fx": False,
        "is_ready": False,
        "missing_inputs": [],
        "message": "",
        "details": {},
    }

    # Check each required file
    files_to_check = [
        ("actuals", ACTUALS_FILENAME, "has_actuals"),
        ("lp", LP_FILENAME, "has_lp"),
        ("fx", FX_FILENAME, "has_fx"),
    ]

    for name, filename, flag in files_to_check:
        filepath = DATA_RAW_DIR / filename
        if filepath.exists():
            result[flag] = True
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            result["details"][name] = {
                "exists": True,
                "path": str(filepath),
                "last_modified": mtime.isoformat(),
                "age_days": age_days,
                "healthy": age_days < 30,
            }
        else:
            result["missing_inputs"].append(name.upper())
            result["details"][name] = {
                "exists": False,
                "path": str(filepath),
                "healthy": False,
            }

    # Determine overall status
    result["is_ready"] = len(result["missing_inputs"]) == 0

    if result["is_ready"]:
        result["message"] = f"All input files present for as_of_date={as_of_date}"
    else:
        missing = ", ".join(result["missing_inputs"])
        result["message"] = f"Missing: {missing}"

    return result


# ---------------------------------------------------------------------------
# Performance Summary Helpers
# ---------------------------------------------------------------------------


def get_hybrid_performance_summary() -> Optional[pd.DataFrame]:
    """
    Get summary of hybrid model performance (win rates by horizon).

    Returns:
        DataFrame with performance summary for TRP horizons 1-4, or None.
    """
    backtest = load_latest_backtest_results()
    if backtest is None:
        return None

    alpha_table = backtest.metrics.get("alpha_by_lg_horizon")
    if alpha_table is None or alpha_table.empty:
        return None

    # Filter to TRP horizons 1-4 (where hybrid is used)
    trp = alpha_table[
        (alpha_table["liquidity_group"] == "TRP") &
        (alpha_table["horizon"] <= 4)
    ].copy()

    if trp.empty:
        return None

    return trp[[
        "liquidity_group", "horizon", "alpha",
        "weekly_wins_vs_lp", "total_weeks", "win_rate_vs_lp",
        "avg_wape_ml", "avg_wape_lp", "avg_wape_hybrid"
    ]]


def get_net_wape_with_hybrid() -> Optional[pd.DataFrame]:
    """
    Get net WAPE metrics with hybrid, combining metrics_net and weekly_hybrid_breakdown.

    Returns:
        DataFrame with ML, LP, and Hybrid WAPE by horizon, or None.
    """
    backtest = load_latest_backtest_results()
    if backtest is None:
        return None

    metrics_net = backtest.metrics.get("metrics_net_clean")
    weekly_hybrid = backtest.metrics.get("weekly_hybrid_breakdown")

    if metrics_net is None:
        return None

    # Aggregate net metrics by horizon
    net_by_h = (
        metrics_net
        .groupby("horizon", as_index=False)
        .agg({
            "ml_wape": "mean",
            "lp_wape": "mean",
        })
    )

    # Add hybrid from weekly breakdown if available
    if weekly_hybrid is not None and not weekly_hybrid.empty:
        hybrid_by_h = (
            weekly_hybrid
            .groupby("horizon", as_index=False)
            .agg({"hybrid_wape": "mean"})
        )
        net_by_h = net_by_h.merge(hybrid_by_h, on="horizon", how="left")

    return net_by_h
