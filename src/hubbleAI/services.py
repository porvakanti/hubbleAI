"""
Service Layer for hubbleAI.

Provides clean API functions for loading forecasts, metrics, and health checks.
Used by Streamlit UI and external integrations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from hubbleAI.config import (
    FORECASTS_DIR,
    BACKTEST_DIR,
    BACKTEST_METRICS_DIR,
    RUN_STATUS_DIR,
    RAW_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class RunStatus:
    """Status of a pipeline run."""
    run_id: str
    mode: str
    as_of_date: str
    trigger_source: str
    status: str
    created_at: str
    message: str
    missing_inputs: List[str] = field(default_factory=list)
    output_paths: Dict[str, str] = field(default_factory=dict)
    metrics_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Result from loading forecasts."""
    success: bool
    data: Optional[pd.DataFrame] = None
    run_status: Optional[RunStatus] = None
    error: Optional[str] = None


@dataclass
class MetricsResult:
    """Result from loading metrics."""
    success: bool
    alpha_table: Optional[pd.DataFrame] = None
    weekly_breakdown: Optional[pd.DataFrame] = None
    metrics_by_lg: Optional[pd.DataFrame] = None
    metrics_net: Optional[pd.DataFrame] = None
    backtest_predictions: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    ref_date: Optional[str] = None


@dataclass
class HealthCheck:
    """Result of health checks."""
    all_healthy: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service Functions
# ---------------------------------------------------------------------------


def load_latest_run_status() -> Optional[RunStatus]:
    """
    Load the latest pipeline run status.

    Returns:
        RunStatus object or None if no status found.
    """
    try:
        latest_pointer = RUN_STATUS_DIR / "latest_run_status.json"
        if not latest_pointer.exists():
            return None

        latest_filename = latest_pointer.read_text(encoding="utf-8").strip()
        status_path = RUN_STATUS_DIR / latest_filename
        if not status_path.exists():
            return None

        data = json.loads(status_path.read_text(encoding="utf-8"))
        return RunStatus(
            run_id=data.get("run_id", ""),
            mode=data.get("mode", ""),
            as_of_date=data.get("as_of_date", ""),
            trigger_source=data.get("trigger_source", ""),
            status=data.get("status", ""),
            created_at=data.get("created_at", ""),
            message=data.get("message", ""),
            missing_inputs=data.get("missing_inputs", []),
            output_paths=data.get("output_paths", {}),
            metrics_paths=data.get("metrics_paths", {}),
        )
    except Exception:
        return None


def load_latest_forward_forecast() -> ForecastResult:
    """
    Load the latest forward forecast data.

    Returns:
        ForecastResult with forecast DataFrame and run status.
    """
    try:
        # Get latest run status
        run_status = load_latest_run_status()

        # Try to load from run status path first
        if run_status and run_status.output_paths.get("forecasts"):
            forecasts_path = Path(run_status.output_paths["forecasts"])
            if forecasts_path.exists():
                df = pd.read_parquet(forecasts_path)
                return ForecastResult(success=True, data=df, run_status=run_status)

        # Fallback: find latest forecast directory
        if FORECASTS_DIR.exists():
            subdirs = sorted([d for d in FORECASTS_DIR.iterdir() if d.is_dir()], reverse=True)
            for subdir in subdirs:
                forecast_file = subdir / "forecasts.parquet"
                if forecast_file.exists():
                    df = pd.read_parquet(forecast_file)
                    return ForecastResult(success=True, data=df, run_status=run_status)

        return ForecastResult(success=False, error="No forecast data found")

    except Exception as e:
        return ForecastResult(success=False, error=str(e))


def load_latest_backtest_metrics() -> MetricsResult:
    """
    Load the latest backtest metrics (alpha table, weekly breakdown, etc.).

    Returns:
        MetricsResult with all metrics DataFrames.
    """
    try:
        # Find latest metrics directory
        if not BACKTEST_METRICS_DIR.exists():
            return MetricsResult(success=False, error="Metrics directory not found")

        subdirs = sorted([d for d in BACKTEST_METRICS_DIR.iterdir() if d.is_dir()], reverse=True)
        if not subdirs:
            return MetricsResult(success=False, error="No metrics subdirectories found")

        latest_dir = subdirs[0]
        ref_date = latest_dir.name

        result = MetricsResult(success=True, ref_date=ref_date)

        # Load alpha table
        alpha_path = latest_dir / "alpha_by_lg_horizon.parquet"
        if alpha_path.exists():
            result.alpha_table = pd.read_parquet(alpha_path)

        # Load weekly breakdown
        weekly_path = latest_dir / "weekly_hybrid_breakdown.parquet"
        if weekly_path.exists():
            result.weekly_breakdown = pd.read_parquet(weekly_path)

        # Load LG-level metrics (clean = Tier-1 only)
        lg_path = latest_dir / "metrics_by_lg_clean.parquet"
        if lg_path.exists():
            result.metrics_by_lg = pd.read_parquet(lg_path)

        # Load Net metrics
        net_path = latest_dir / "metrics_net_clean.parquet"
        if net_path.exists():
            result.metrics_net = pd.read_parquet(net_path)

        # Load backtest predictions
        backtest_dir = BACKTEST_DIR / ref_date
        backtest_path = backtest_dir / "backtest_predictions.parquet"
        if backtest_path.exists():
            result.backtest_predictions = pd.read_parquet(backtest_path)

        return result

    except Exception as e:
        return MetricsResult(success=False, error=str(e))


def check_raw_data_freshness() -> Dict[str, Any]:
    """
    Check freshness of raw input files.

    Returns:
        Dict with file status and last modified times.
    """
    checks = {}

    expected_files = [
        ("actuals", "New_Actuals_17C7_2014.csv"),
        ("lp", "New_LP_17C7.csv"),
        ("fx", "20251120_eurofxref-hist.csv"),
    ]

    for name, filename in expected_files:
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            checks[name] = {
                "exists": True,
                "path": str(filepath),
                "last_modified": mtime.isoformat(),
                "age_days": age_days,
                "healthy": age_days < 30,  # Consider stale if > 30 days
            }
        else:
            checks[name] = {
                "exists": False,
                "path": str(filepath),
                "healthy": False,
            }

    return checks


def check_pipeline_status() -> Dict[str, Any]:
    """
    Check overall pipeline health.

    Returns:
        Dict with pipeline status information.
    """
    status = {
        "has_run_status": False,
        "last_run": None,
        "last_run_status": None,
        "has_forecasts": False,
        "has_backtest": False,
        "has_metrics": False,
    }

    # Check run status
    run_status = load_latest_run_status()
    if run_status:
        status["has_run_status"] = True
        status["last_run"] = run_status.created_at
        status["last_run_status"] = run_status.status

    # Check forecasts
    if FORECASTS_DIR.exists():
        subdirs = [d for d in FORECASTS_DIR.iterdir() if d.is_dir()]
        status["has_forecasts"] = len(subdirs) > 0

    # Check backtest
    if BACKTEST_DIR.exists():
        subdirs = [d for d in BACKTEST_DIR.iterdir() if d.is_dir()]
        status["has_backtest"] = len(subdirs) > 0

    # Check metrics
    if BACKTEST_METRICS_DIR.exists():
        subdirs = [d for d in BACKTEST_METRICS_DIR.iterdir() if d.is_dir()]
        status["has_metrics"] = len(subdirs) > 0

    return status


def get_health_summary() -> HealthCheck:
    """
    Get overall health summary combining all checks.

    Returns:
        HealthCheck with all_healthy flag and detailed checks.
    """
    data_checks = check_raw_data_freshness()
    pipeline_checks = check_pipeline_status()

    # Determine overall health
    data_healthy = all(c.get("healthy", False) for c in data_checks.values())
    pipeline_healthy = pipeline_checks.get("has_metrics", False)

    return HealthCheck(
        all_healthy=data_healthy and pipeline_healthy,
        checks={
            "data": data_checks,
            "pipeline": pipeline_checks,
        }
    )


def get_hybrid_performance_summary() -> Optional[pd.DataFrame]:
    """
    Get summary of hybrid model performance (win rates by horizon).

    Returns:
        DataFrame with TRP horizons and their win rates, or None.
    """
    metrics = load_latest_backtest_metrics()
    if not metrics.success or metrics.alpha_table is None:
        return None

    # Filter to TRP horizons 1-4
    trp = metrics.alpha_table[
        (metrics.alpha_table["liquidity_group"] == "TRP") &
        (metrics.alpha_table["horizon"] <= 4)
    ].copy()

    if trp.empty:
        return None

    return trp[["horizon", "alpha", "weekly_wins_vs_lp", "total_weeks", "win_rate_vs_lp"]]
