"""
Central configuration for hubbleAI.

This module contains all constants and configuration values used across
the forecasting pipeline. Centralizing these values makes it easier to
maintain consistency and swap data sources later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/hubbleAI/config.py -> repo root
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RUN_STATUS_DIR = DATA_PROCESSED_DIR / "run_status"
FORECASTS_DIR = DATA_PROCESSED_DIR / "forecasts"
METRICS_DIR = DATA_PROCESSED_DIR / "metrics"
MODELS_DIR = DATA_PROCESSED_DIR / "models"

# ---------------------------------------------------------------------------
# Data file names (current local files in data/raw)
# ---------------------------------------------------------------------------

# These are the actual filenames from the notebook
ACTUALS_FILENAME = "New_Actuals_17C7_2014.csv"
LP_FILENAME = "New_LP_17C7.csv"
FX_FILENAME = "20251120_eurofxref-hist.csv"

# ---------------------------------------------------------------------------
# Horizons and Liquidity Groups
# ---------------------------------------------------------------------------

HORIZONS: List[int] = list(range(1, 9))  # 1 to 8 weeks ahead
LIQUIDITY_GROUPS: List[str] = ["TRR", "TRP"]

# ---------------------------------------------------------------------------
# LP Forecast Column Mapping (horizon -> LP column)
# ---------------------------------------------------------------------------

LP_FORECAST_COLS: Dict[int, str] = {
    1: "W1_Forecast",
    2: "W2_Forecast",
    3: "W3_Forecast",
    4: "W4_Forecast",
    # Horizons 5-8 have no LP forecast
}

# All LP forecast columns for reference
ALL_LP_COLS: List[str] = ["W1_Forecast", "W2_Forecast", "W3_Forecast", "W4_Forecast"]

# ---------------------------------------------------------------------------
# Tier-2 Entities (excluded from ML; LP pass-through)
# ---------------------------------------------------------------------------

# These (entity, liquidity_group) combinations are excluded from ML forecasting
# due to poor/unreliable history. For Tier-2, the final forecast simply passes
# through the Liquidity Plan (LP) values.
TIER2_LIST: List[Tuple[str, str]] = [
    ("11G5", "TRR"),
    ("14C1", "TRR"),
    ("20B2", "TRR"),
    ("82J", "TRR"),
    ("82J", "TRP"),
    ("25A4", "TRR"),
    ("25A4", "TRP"),
    ("17C7", "TRR"),
    ("17C7", "TRP"),
]

# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------

# Lag feature configuration
LAG_WEEKS: int = 52  # Number of lag weeks to create

# Rolling window sizes
ROLLING_WINDOWS: Tuple[int, ...] = (4, 8, 13, 26, 52)

# Trend window sizes
TREND_WINDOWS: Tuple[int, ...] = (12, 26)

# LP accuracy rolling window
LP_ACCURACY_WINDOW: int = 12

# Minimum history weeks required for a series to be included in ML training
MIN_HISTORY_WEEKS: int = 52

# ---------------------------------------------------------------------------
# TRP-specific feature columns
# ---------------------------------------------------------------------------

TRP_EXTRA_FEATURES: List[str] = [
    "trp_vendor_count",
    "trp_top_vendor_share",
    "trp_country_count",
    "trp_top_country_share",
    "trp_reconciled_share",
]

# ---------------------------------------------------------------------------
# Columns to exclude from features
# ---------------------------------------------------------------------------

DROP_COLS: List[str] = [
    "entity_name",
    "Year_Title",
    "W1_Forecast_Available",
    "W2_Forecast_Available",
    "W3_Forecast_Available",
    "W4_Forecast_Available",
    "available_forecast_count",
    "year",
    "month",
    "quarter",
    "iso_week_of_year",
    "history_weeks",
    "lp_W1_error",
    "lp_W1_abs_error",
    "lp_W1_bias_12w",
    "lp_W1_mae_12w",
    "lp_W2_error",
    "lp_W2_abs_error",
    "lp_W2_bias_12w",
    "lp_W2_mae_12w",
    "lp_W3_error",
    "lp_W3_abs_error",
    "lp_W3_bias_12w",
    "lp_W3_mae_12w",
    "lp_W4_error",
    "lp_W4_abs_error",
    "lp_W4_bias_12w",
    "lp_W4_mae_12w",
]

ID_COLS: List[str] = ["liquidity_group", "week_start", "tier", "target_week_start"]

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Default LightGBM parameters
DEFAULT_LGBM_PARAMS: Dict = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
}

# Training settings
NUM_BOOST_ROUND: int = 2000
EARLY_STOPPING_ROUNDS: int = 50

# Train/Valid/Test split ratios (time-based)
TRAIN_RATIO: float = 0.85
VALID_RATIO: float = 0.95  # valid ends at 95%, test is 95%-100%

# ---------------------------------------------------------------------------
# Output Schema Columns
# ---------------------------------------------------------------------------

FORECAST_OUTPUT_COLS: List[str] = [
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
