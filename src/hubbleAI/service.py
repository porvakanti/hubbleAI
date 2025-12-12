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


# ---------------------------------------------------------------------------
# Extended Service Functions for UI
# ---------------------------------------------------------------------------

# WAPE computation threshold (in EUR) - below this, WAPE is undefined
WAPE_EPS_THRESHOLD = 500_000  # 0.5 million EUR


@dataclass
class ForecastRunStatus:
    """Status of a forecast run."""
    run_id: str
    mode: Literal["forward", "backtest"]
    status: str  # "success", "failure", "running"
    ref_week_start: Optional[date]
    as_of_date: Optional[date]
    created_at: Optional[datetime]
    trigger_source: str
    output_paths: Dict[str, str]
    metrics_paths: Dict[str, str]
    message: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForecastRunStatus":
        """Create from dictionary."""
        ref_week = data.get("ref_week_start")
        if isinstance(ref_week, str):
            ref_week = date.fromisoformat(ref_week)

        as_of = data.get("as_of_date")
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)

        created = data.get("created_at")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except Exception:
                created = None

        return cls(
            run_id=data.get("run_id", ""),
            mode=data.get("mode", "forward"),
            status=data.get("status", "unknown"),
            ref_week_start=ref_week,
            as_of_date=as_of,
            created_at=created,
            trigger_source=data.get("trigger_source", "unknown"),
            output_paths=data.get("output_paths", {}),
            metrics_paths=data.get("metrics_paths", {}),
            message=data.get("message", ""),
        )


def get_latest_forward_status() -> Optional[ForecastRunStatus]:
    """
    Get the status of the latest forward forecast run.

    Returns:
        ForecastRunStatus or None if not found.
    """
    status_dict = get_last_run_by_mode("forward")
    if status_dict is None:
        return None
    return ForecastRunStatus.from_dict(status_dict)


def get_latest_backtest_status() -> Optional[ForecastRunStatus]:
    """
    Get the status of the latest backtest run.

    Returns:
        ForecastRunStatus or None if not found.
    """
    status_dict = get_last_run_by_mode("backtest")
    if status_dict is None:
        return None
    return ForecastRunStatus.from_dict(status_dict)


def load_forward_predictions(status: Optional[ForecastRunStatus] = None) -> pd.DataFrame:
    """
    Load forward predictions from a run status.

    Args:
        status: ForecastRunStatus object (if None, loads latest)

    Returns:
        DataFrame with forward predictions.
    """
    if status is None:
        status = get_latest_forward_status()

    if status is None:
        return pd.DataFrame()

    forecast_path = status.output_paths.get("forecasts")
    if forecast_path:
        path = Path(forecast_path)
        if path.exists():
            return pd.read_parquet(path)

    # Fallback to directory scan
    view = load_latest_forward_forecast()
    if view:
        return view.forecasts_df
    return pd.DataFrame()


def load_backtest_predictions(status: Optional[ForecastRunStatus] = None) -> pd.DataFrame:
    """
    Load backtest predictions from a run status.

    Args:
        status: ForecastRunStatus object (if None, loads latest)

    Returns:
        DataFrame with backtest predictions.
    """
    if status is None:
        status = get_latest_backtest_status()

    if status is None:
        # Try loading from BacktestView
        view = load_latest_backtest_results()
        if view:
            return view.backtest_df
        return pd.DataFrame()

    backtest_path = status.output_paths.get("backtest_predictions")
    if backtest_path:
        path = Path(backtest_path)
        if path.exists():
            return pd.read_parquet(path)

    # Fallback to BacktestView
    view = load_latest_backtest_results()
    if view:
        return view.backtest_df
    return pd.DataFrame()


def load_metrics_artifact(
    status: Optional[ForecastRunStatus] = None,
    key: str = "metrics_by_lg_clean"
) -> Optional[pd.DataFrame]:
    """
    Load a specific metrics artifact.

    Args:
        status: ForecastRunStatus object (if None, loads from latest backtest)
        key: Artifact key (e.g., "metrics_by_lg_clean", "metrics_net_clean")

    Returns:
        DataFrame or None if not found.
    """
    try:
        # Try from status first
        if status and status.metrics_paths:
            path_str = status.metrics_paths.get(key)
            if path_str:
                path = Path(path_str)
                if path.exists():
                    return pd.read_parquet(path)

        # Fallback to BacktestView
        view = load_latest_backtest_results()
        if view:
            # Check metrics first, then diagnostics
            if key in view.metrics:
                return view.metrics[key]
            if key in view.diagnostics:
                return view.diagnostics[key]

        return None
    except Exception as e:
        logger.warning(f"Failed to load metrics artifact {key}: {e}")
        return None


def get_available_artifacts(status: Optional[ForecastRunStatus] = None) -> Dict[str, str]:
    """
    Get all available artifacts for a run.

    Args:
        status: ForecastRunStatus object (if None, loads from latest)

    Returns:
        Dict mapping artifact key to file path.
    """
    artifacts = {}

    if status:
        artifacts.update(status.output_paths)
        artifacts.update(status.metrics_paths)

    # Also check BacktestView
    view = load_latest_backtest_results()
    if view:
        # Add metrics and diagnostics keys
        for key in view.metrics:
            if key not in artifacts:
                artifacts[key] = f"loaded:{key}"
        for key in view.diagnostics:
            if key not in artifacts:
                artifacts[key] = f"loaded:{key}"

    return artifacts


def validate_forward_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate forward predictions DataFrame.

    Checks:
    - Required columns exist
    - Quantiles are present for Tier-1 rows
    - Values are reasonable

    Args:
        df: Forward predictions DataFrame

    Returns:
        Dict with "ok": bool, "issues": list of strings
    """
    issues = []

    if df.empty:
        return {"ok": False, "issues": ["DataFrame is empty"]}

    # Required columns
    required = ["entity", "liquidity_group", "horizon", "y_pred_point"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check quantile columns exist
    quantile_cols = ["y_pred_p10", "y_pred_p50", "y_pred_p90"]
    missing_quant = [c for c in quantile_cols if c not in df.columns]
    if missing_quant:
        issues.append(f"Missing quantile columns: {missing_quant}")
    else:
        # Check Tier-1 quantiles (is_pass_through == False)
        if "is_pass_through" in df.columns:
            tier1 = df[df["is_pass_through"] == False]
            if not tier1.empty:
                for col in quantile_cols:
                    null_count = tier1[col].isna().sum()
                    if null_count > 0:
                        pct = null_count / len(tier1) * 100
                        issues.append(f"Tier-1 {col}: {null_count} null values ({pct:.1f}%)")

    # Check for extreme values
    if "y_pred_point" in df.columns:
        extreme = (df["y_pred_point"].abs() > 1e12).sum()  # > 1 trillion
        if extreme > 0:
            issues.append(f"{extreme} rows have extreme point predictions (>1T EUR)")

    return {
        "ok": len(issues) == 0,
        "issues": issues
    }


def prepare_forecast_views(df_forward: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare aggregated forecast views for TRR, TRP, and NET by horizon.

    Args:
        df_forward: Forward predictions DataFrame

    Returns:
        Dict with keys "TRR", "TRP", "NET", each containing a summary DataFrame
        with columns: horizon, target_week_start, point, p10, p50, p90 (in millions)
    """
    result = {}

    if df_forward.empty:
        return result

    for lg in ["TRR", "TRP"]:
        lg_df = df_forward[df_forward["liquidity_group"] == lg]
        if lg_df.empty:
            continue

        summary_rows = []
        for h in range(1, 9):
            h_data = lg_df[lg_df["horizon"] == h]
            if h_data.empty:
                continue

            row = {
                "horizon": h,
                "target_week_start": h_data["target_week_start"].iloc[0] if "target_week_start" in h_data.columns else None,
                "point": h_data["y_pred_point"].sum() / 1e6,
            }

            # Quantiles
            for col, key in [("y_pred_p10", "p10"), ("y_pred_p50", "p50"), ("y_pred_p90", "p90")]:
                if col in h_data.columns:
                    row[key] = h_data[col].sum() / 1e6
                else:
                    row[key] = None

            summary_rows.append(row)

        if summary_rows:
            result[lg] = pd.DataFrame(summary_rows)

    # NET = TRR + TRP
    net_rows = []
    for h in range(1, 9):
        h_data = df_forward[df_forward["horizon"] == h]
        if h_data.empty:
            continue

        row = {
            "horizon": h,
            "target_week_start": h_data["target_week_start"].iloc[0] if "target_week_start" in h_data.columns else None,
            "point": h_data["y_pred_point"].sum() / 1e6,
        }

        for col, key in [("y_pred_p10", "p10"), ("y_pred_p50", "p50"), ("y_pred_p90", "p90")]:
            if col in h_data.columns:
                row[key] = h_data[col].sum() / 1e6
            else:
                row[key] = None

        net_rows.append(row)

    if net_rows:
        result["NET"] = pd.DataFrame(net_rows)

    return result


def get_ml_prediction_column(lg: str, horizon: int, df: pd.DataFrame) -> str:
    """
    Determine which prediction column to use for "ML" based on business rules.

    Rules:
    - TRR: always y_pred_point
    - TRP H1-H4: y_pred_hybrid if available, else y_pred_point
    - TRP H5-H8: y_pred_point

    Args:
        lg: Liquidity group ("TRR" or "TRP")
        horizon: Horizon (1-8)
        df: DataFrame with prediction columns

    Returns:
        Column name to use for ML prediction
    """
    if lg == "TRR":
        return "y_pred_point"

    if lg == "TRP" and horizon <= 4:
        if "y_pred_hybrid" in df.columns and not df["y_pred_hybrid"].isna().all():
            return "y_pred_hybrid"
        return "y_pred_point"

    return "y_pred_point"


def compute_weekly_wape_by_lg(
    df: pd.DataFrame,
    lg: str,
    horizon: int,
    tier1_only: bool = True,
    eps: float = WAPE_EPS_THRESHOLD
) -> pd.DataFrame:
    """
    Compute weekly WAPE for ML vs LP for a specific liquidity group and horizon.

    Args:
        df: Backtest predictions DataFrame
        lg: Liquidity group ("TRR", "TRP", or "NET")
        horizon: Horizon (1-8)
        tier1_only: Whether to exclude pass-through entities
        eps: Threshold for near-zero denominators

    Returns:
        DataFrame with columns: week_start, actual_sum, ml_pred_sum, lp_pred_sum,
        ml_wape, lp_wape, ml_wape_undefined, lp_wape_undefined, winner
    """
    if df.empty:
        return pd.DataFrame()

    # Filter data
    filtered = df[df["horizon"] == horizon].copy()

    if tier1_only and "is_pass_through" in filtered.columns:
        filtered = filtered[filtered["is_pass_through"] == False]

    if lg == "NET":
        # Sum across both LGs
        pass
    elif lg in ["TRR", "TRP"]:
        filtered = filtered[filtered["liquidity_group"] == lg]

    if filtered.empty:
        return pd.DataFrame()

    # Determine ML prediction column
    if lg == "NET":
        # For NET, we need to compute ML separately for TRR and TRP, then sum
        ml_col = "y_pred_point"  # Simplified: use point for NET computation
    else:
        ml_col = get_ml_prediction_column(lg, horizon, filtered)

    # Aggregate by week
    rows = []
    for week, grp in filtered.groupby("week_start", observed=True):
        actual_sum = grp["actual_value"].sum() if "actual_value" in grp.columns else 0
        ml_sum = grp[ml_col].sum() if ml_col in grp.columns else 0
        lp_sum = grp["lp_baseline_point"].sum() if "lp_baseline_point" in grp.columns else float("nan")

        # Compute WAPE with guardrails
        ml_wape = float("nan")
        ml_undefined = True
        lp_wape = float("nan")
        lp_undefined = True

        if abs(actual_sum) >= eps:
            ml_wape = abs(actual_sum - ml_sum) / abs(actual_sum)
            ml_undefined = False

            if pd.notna(lp_sum):
                lp_wape = abs(actual_sum - lp_sum) / abs(actual_sum)
                lp_undefined = False

        # Determine winner
        if ml_undefined:
            winner = "N/A"
        elif lp_undefined or pd.isna(lp_wape):
            winner = "ML"  # LP not available
        elif abs(ml_wape - lp_wape) < 0.001:
            winner = "Tie"
        elif ml_wape < lp_wape:
            winner = "ML"
        else:
            winner = "LP"

        rows.append({
            "week_start": week,
            "actual_sum": actual_sum,
            "ml_pred_sum": ml_sum,
            "lp_pred_sum": lp_sum,
            "ml_wape": ml_wape,
            "lp_wape": lp_wape,
            "ml_wape_undefined": ml_undefined,
            "lp_wape_undefined": lp_undefined,
            "winner": winner,
        })

    return pd.DataFrame(rows)


def compute_net_weekly_wape(
    df: pd.DataFrame,
    horizon: int,
    tier1_only: bool = True,
    eps: float = WAPE_EPS_THRESHOLD
) -> pd.DataFrame:
    """
    Compute weekly WAPE for NET (TRR + TRP combined) at a specific horizon.

    For NET:
    - TRR uses y_pred_point
    - TRP uses y_pred_hybrid for H1-H4 (if available), else y_pred_point

    Args:
        df: Backtest predictions DataFrame
        horizon: Horizon (1-8)
        tier1_only: Whether to exclude pass-through entities
        eps: Threshold for near-zero denominators

    Returns:
        DataFrame with weekly WAPE data
    """
    if df.empty:
        return pd.DataFrame()

    filtered = df[df["horizon"] == horizon].copy()

    if tier1_only and "is_pass_through" in filtered.columns:
        filtered = filtered[filtered["is_pass_through"] == False]

    if filtered.empty:
        return pd.DataFrame()

    rows = []
    for week, grp in filtered.groupby("week_start", observed=True):
        # Sum actuals
        actual_sum = grp["actual_value"].sum() if "actual_value" in grp.columns else 0

        # Sum ML predictions (TRR uses point, TRP uses hybrid for H1-4)
        trr_data = grp[grp["liquidity_group"] == "TRR"]
        trp_data = grp[grp["liquidity_group"] == "TRP"]

        trr_ml = trr_data["y_pred_point"].sum() if "y_pred_point" in trr_data.columns else 0

        # TRP: use hybrid for H1-4 if available
        if horizon <= 4 and "y_pred_hybrid" in trp_data.columns and not trp_data["y_pred_hybrid"].isna().all():
            trp_ml = trp_data["y_pred_hybrid"].sum()
        else:
            trp_ml = trp_data["y_pred_point"].sum() if "y_pred_point" in trp_data.columns else 0

        ml_sum = trr_ml + trp_ml

        # Sum LP predictions
        lp_sum = grp["lp_baseline_point"].sum() if "lp_baseline_point" in grp.columns else float("nan")

        # Compute WAPE
        ml_wape = float("nan")
        ml_undefined = True
        lp_wape = float("nan")
        lp_undefined = True

        if abs(actual_sum) >= eps:
            ml_wape = abs(actual_sum - ml_sum) / abs(actual_sum)
            ml_undefined = False

            if pd.notna(lp_sum) and not (horizon > 4):  # LP only for H1-4
                lp_wape = abs(actual_sum - lp_sum) / abs(actual_sum)
                lp_undefined = False

        # Determine winner
        if ml_undefined:
            winner = "N/A"
        elif lp_undefined or pd.isna(lp_wape):
            winner = "ML"
        elif abs(ml_wape - lp_wape) < 0.001:
            winner = "Tie"
        elif ml_wape < lp_wape:
            winner = "ML"
        else:
            winner = "LP"

        rows.append({
            "week_start": week,
            "actual_sum": actual_sum,
            "ml_pred_sum": ml_sum,
            "lp_pred_sum": lp_sum,
            "ml_wape": ml_wape,
            "lp_wape": lp_wape,
            "ml_wape_undefined": ml_undefined,
            "lp_wape_undefined": lp_undefined,
            "winner": winner,
        })

    return pd.DataFrame(rows)


def get_performance_summary(
    df_wape: pd.DataFrame,
    exclude_undefined: bool = True
) -> Dict[str, Any]:
    """
    Compute summary statistics from weekly WAPE data.

    Args:
        df_wape: DataFrame from compute_weekly_wape_by_lg or compute_net_weekly_wape
        exclude_undefined: Whether to exclude weeks with undefined WAPE

    Returns:
        Dict with: total_weeks, ml_wins, lp_wins, ties, na_count, win_rate,
        avg_ml_wape, avg_lp_wape, improvement_pp
    """
    if df_wape.empty:
        return {
            "total_weeks": 0,
            "ml_wins": 0,
            "lp_wins": 0,
            "ties": 0,
            "na_count": 0,
            "win_rate": 0.0,
            "avg_ml_wape": float("nan"),
            "avg_lp_wape": float("nan"),
            "improvement_pp": 0.0,
        }

    df = df_wape.copy()

    if exclude_undefined:
        df = df[~df["ml_wape_undefined"]]

    total = len(df)
    ml_wins = (df["winner"] == "ML").sum()
    lp_wins = (df["winner"] == "LP").sum()
    ties = (df["winner"] == "Tie").sum()
    na_count = (df["winner"] == "N/A").sum()

    # Compute averages (excluding undefined)
    valid_ml = df[~df["ml_wape"].isna()]
    valid_lp = df[~df["lp_wape"].isna()]

    avg_ml = valid_ml["ml_wape"].mean() if not valid_ml.empty else float("nan")
    avg_lp = valid_lp["lp_wape"].mean() if not valid_lp.empty else float("nan")

    # Improvement in percentage points
    improvement = (avg_lp - avg_ml) * 100 if pd.notna(avg_ml) and pd.notna(avg_lp) else 0.0

    # Win rate (excluding NA)
    valid_comparisons = ml_wins + lp_wins + ties
    win_rate = ml_wins / valid_comparisons if valid_comparisons > 0 else 0.0

    return {
        "total_weeks": total,
        "ml_wins": ml_wins,
        "lp_wins": lp_wins,
        "ties": ties,
        "na_count": na_count,
        "win_rate": win_rate,
        "avg_ml_wape": avg_ml,
        "avg_lp_wape": avg_lp,
        "improvement_pp": improvement,
    }
