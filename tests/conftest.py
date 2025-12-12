"""
Pytest fixtures for hubbleAI tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_backtest_df():
    """
    Create a sample backtest DataFrame for testing.

    Includes TRP and TRR data with multiple weeks, horizons, and entities.
    """
    np.random.seed(42)

    data = []
    weeks = pd.date_range("2025-01-06", periods=4, freq="W-MON")
    entities = ["Entity1", "Entity2"]

    for week in weeks:
        for entity in entities:
            for lg in ["TRP", "TRR"]:
                for horizon in range(1, 5):
                    actual = 1000 + np.random.randn() * 100
                    ml_pred = actual + np.random.randn() * 50
                    lp_pred = actual + np.random.randn() * 80 if lg == "TRP" else np.nan

                    data.append({
                        "entity": entity,
                        "liquidity_group": lg,
                        "week_start": week,
                        "target_week_start": week + pd.Timedelta(days=7 * horizon),
                        "horizon": horizon,
                        "actual_value": actual,
                        "y_pred_point": ml_pred,
                        "lp_baseline_point": lp_pred,
                        "is_pass_through": False,
                    })

    return pd.DataFrame(data)


@pytest.fixture
def sample_trp_df():
    """
    Create a TRP-only DataFrame where we control ML vs LP performance.

    - Week 1: ML better than LP
    - Week 2: LP better than ML
    - Week 3: Hybrid (α=0.5) is best
    """
    data = [
        # Week 1: ML wins (ML error = 10, LP error = 50)
        {"entity": "E1", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-06"),
         "horizon": 1, "actual_value": 1000, "y_pred_point": 1010, "lp_baseline_point": 1050,
         "is_pass_through": False},
        {"entity": "E2", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-06"),
         "horizon": 1, "actual_value": 500, "y_pred_point": 510, "lp_baseline_point": 550,
         "is_pass_through": False},

        # Week 2: LP wins (ML error = 100, LP error = 20)
        {"entity": "E1", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-13"),
         "horizon": 1, "actual_value": 1000, "y_pred_point": 1100, "lp_baseline_point": 1020,
         "is_pass_through": False},
        {"entity": "E2", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-13"),
         "horizon": 1, "actual_value": 500, "y_pred_point": 600, "lp_baseline_point": 520,
         "is_pass_through": False},

        # Week 3: Both have similar errors
        {"entity": "E1", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-20"),
         "horizon": 1, "actual_value": 1000, "y_pred_point": 1030, "lp_baseline_point": 970,
         "is_pass_through": False},
        {"entity": "E2", "liquidity_group": "TRP", "week_start": pd.Timestamp("2025-01-20"),
         "horizon": 1, "actual_value": 500, "y_pred_point": 515, "lp_baseline_point": 485,
         "is_pass_through": False},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def alpha_mapping_fixture():
    """Sample alpha mapping for testing."""
    return {
        ("TRP", 1): 0.1,
        ("TRP", 2): 0.2,
        ("TRP", 3): 0.3,
        ("TRP", 4): 0.4,
        ("TRR", 1): 1.0,
        ("TRR", 2): 1.0,
        ("TRR", 3): 1.0,
        ("TRR", 4): 1.0,
    }
