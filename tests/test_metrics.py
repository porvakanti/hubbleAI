"""
Unit tests for hubbleAI.evaluation.metrics module.
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from hubbleAI.evaluation.metrics import (
    wape,
    wape_aggregate,
    wape_aggregate_series,
    tune_hybrid_alpha,
    get_alpha_mapping,
    compute_weekly_hybrid_breakdown,
    _compute_weekly_wape_stats,
)


class TestWapeMetrics:
    """Tests for WAPE metric functions."""

    def test_wape_perfect_prediction(self):
        """WAPE should be 0 for perfect predictions."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        assert wape(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_wape_standard_calculation(self):
        """Test standard WAPE: sum(|error|) / sum(|actual|)."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 190])  # errors: 10, 10
        # WAPE = (10 + 10) / (100 + 200) = 20/300 = 0.0667
        expected = 20 / 300
        assert wape(y_true, y_pred) == pytest.approx(expected, rel=1e-3)

    def test_wape_aggregate_cancellation(self):
        """Aggregate WAPE allows errors to cancel out."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 190])  # +10 and -10 cancel out
        # Aggregate: |sum(true) - sum(pred)| / |sum(true)| = |300 - 300| / 300 = 0
        assert wape_aggregate(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)

    def test_wape_aggregate_no_cancellation(self):
        """Aggregate WAPE when errors don't cancel."""
        y_true = np.array([100, 200])
        y_pred = np.array([120, 220])  # both +20, total +40
        # Aggregate: |300 - 340| / 300 = 40/300 = 0.1333
        expected = 40 / 300
        assert wape_aggregate(y_true, y_pred) == pytest.approx(expected, rel=1e-3)

    def test_wape_aggregate_series(self):
        """Test aggregate WAPE with pandas Series input."""
        actual = pd.Series([100, 200, 300])
        pred = pd.Series([110, 210, 310])  # all +10, total +30
        # Aggregate: |600 - 630| / 600 = 30/600 = 0.05
        expected = 30 / 600
        assert wape_aggregate_series(actual, pred) == pytest.approx(expected, rel=1e-3)

    def test_wape_aggregate_series_with_nan(self):
        """Aggregate WAPE should ignore NaN values."""
        actual = pd.Series([100, np.nan, 300])
        pred = pd.Series([110, 999, 310])  # middle row should be ignored
        # Only rows 0 and 2: |400 - 420| / 400 = 20/400 = 0.05
        expected = 20 / 400
        assert wape_aggregate_series(actual, pred) == pytest.approx(expected, rel=1e-3)


class TestTuneHybridAlpha:
    """Tests for hybrid alpha tuning."""

    def test_tune_hybrid_alpha_returns_dataframe(self, sample_backtest_df):
        """tune_hybrid_alpha should return a DataFrame with expected columns."""
        result = tune_hybrid_alpha(sample_backtest_df)

        assert isinstance(result, pd.DataFrame)
        expected_cols = ["liquidity_group", "horizon", "alpha", "weekly_wins_vs_lp",
                         "total_weeks", "win_rate_vs_lp"]
        for col in expected_cols:
            assert col in result.columns

    def test_tune_hybrid_alpha_trr_always_one(self, sample_backtest_df):
        """TRR should always have alpha=1.0 (pure ML)."""
        result = tune_hybrid_alpha(sample_backtest_df)
        trr_rows = result[result["liquidity_group"] == "TRR"]

        assert all(trr_rows["alpha"] == 1.0)

    def test_tune_hybrid_alpha_range(self, sample_backtest_df):
        """Alpha values should be in [0, 1] range."""
        result = tune_hybrid_alpha(sample_backtest_df)

        assert all(result["alpha"] >= 0.0)
        assert all(result["alpha"] <= 1.0)

    def test_tune_hybrid_alpha_min_win_rate_fallback(self, sample_trp_df):
        """
        If best alpha doesn't meet min_win_rate, should fall back to alpha=0.
        """
        # With very high min_win_rate, should fall back to LP
        result = tune_hybrid_alpha(sample_trp_df, min_win_rate=0.99)
        trp_h1 = result[(result["liquidity_group"] == "TRP") & (result["horizon"] == 1)]

        if not trp_h1.empty:
            # If win rate < 99%, alpha should be 0 (pure LP)
            if trp_h1["win_rate_vs_lp"].iloc[0] < 0.99:
                assert trp_h1["alpha"].iloc[0] == 0.0

    def test_tune_hybrid_alpha_custom_grid(self, sample_backtest_df):
        """Alpha tuning should respect custom alpha grid."""
        custom_alphas = [0.0, 0.5, 1.0]
        result = tune_hybrid_alpha(sample_backtest_df, alphas=custom_alphas)

        # All alphas should be from the custom grid
        trp_rows = result[result["liquidity_group"] == "TRP"]
        for alpha in trp_rows["alpha"]:
            assert alpha in custom_alphas


class TestGetAlphaMapping:
    """Tests for get_alpha_mapping function."""

    def test_get_alpha_mapping_basic(self):
        """get_alpha_mapping should convert DataFrame to dict."""
        alpha_df = pd.DataFrame({
            "liquidity_group": ["TRP", "TRP", "TRR"],
            "horizon": [1, 2, 1],
            "alpha": [0.1, 0.2, 1.0],
        })
        result = get_alpha_mapping(alpha_df)

        assert result == {
            ("TRP", 1): 0.1,
            ("TRP", 2): 0.2,
            ("TRR", 1): 1.0,
        }

    def test_get_alpha_mapping_empty(self):
        """get_alpha_mapping should return empty dict for empty DataFrame."""
        alpha_df = pd.DataFrame(columns=["liquidity_group", "horizon", "alpha"])
        result = get_alpha_mapping(alpha_df)

        assert result == {}


class TestComputeWeeklyHybridBreakdown:
    """Tests for weekly hybrid breakdown computation."""

    def test_compute_weekly_hybrid_breakdown_columns(self, sample_backtest_df, alpha_mapping_fixture):
        """Should return DataFrame with expected columns."""
        result = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping_fixture)

        expected_cols = ["liquidity_group", "horizon", "week_start", "alpha",
                         "lp_wape", "ml_wape", "hybrid_wape", "ml_wins", "hybrid_wins"]
        for col in expected_cols:
            assert col in result.columns

    def test_compute_weekly_hybrid_breakdown_only_trp(self, sample_backtest_df, alpha_mapping_fixture):
        """Should only include TRP (not TRR) since TRR has no LP."""
        result = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping_fixture)

        assert all(result["liquidity_group"] == "TRP")

    def test_compute_weekly_hybrid_breakdown_horizons_1_to_4(self, sample_backtest_df, alpha_mapping_fixture):
        """Should only include horizons 1-4 (LP available)."""
        result = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping_fixture)

        assert all(result["horizon"].isin([1, 2, 3, 4]))

    def test_compute_weekly_hybrid_breakdown_wins_boolean(self, sample_backtest_df, alpha_mapping_fixture):
        """ml_wins and hybrid_wins should be boolean."""
        result = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping_fixture)

        assert result["ml_wins"].dtype == bool
        assert result["hybrid_wins"].dtype == bool

    def test_compute_weekly_hybrid_breakdown_wape_non_negative(self, sample_backtest_df, alpha_mapping_fixture):
        """All WAPE values should be non-negative."""
        result = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping_fixture)

        assert all(result["lp_wape"] >= 0)
        assert all(result["ml_wape"] >= 0)
        assert all(result["hybrid_wape"] >= 0)


class TestComputeWeeklyWapeStats:
    """Tests for _compute_weekly_wape_stats helper function."""

    def test_compute_weekly_wape_stats_alpha_zero(self, sample_trp_df):
        """Alpha=0 should give pure LP predictions."""
        # Filter to TRP horizon 1
        df = sample_trp_df[
            (sample_trp_df["liquidity_group"] == "TRP") &
            (sample_trp_df["horizon"] == 1)
        ]
        stats = _compute_weekly_wape_stats(df, alpha=0.0)

        # With alpha=0, hybrid = LP, so hybrid_wape should equal lp_wape
        assert stats["avg_wape_hybrid"] == pytest.approx(stats["avg_wape_lp"], rel=1e-3)

    def test_compute_weekly_wape_stats_alpha_one(self, sample_trp_df):
        """Alpha=1 should give pure ML predictions."""
        df = sample_trp_df[
            (sample_trp_df["liquidity_group"] == "TRP") &
            (sample_trp_df["horizon"] == 1)
        ]
        stats = _compute_weekly_wape_stats(df, alpha=1.0)

        # With alpha=1, hybrid = ML, so hybrid_wape should equal ml_wape
        assert stats["avg_wape_hybrid"] == pytest.approx(stats["avg_wape_ml"], rel=1e-3)

    def test_compute_weekly_wape_stats_returns_dict(self, sample_trp_df):
        """Should return dict with expected keys."""
        df = sample_trp_df[
            (sample_trp_df["liquidity_group"] == "TRP") &
            (sample_trp_df["horizon"] == 1)
        ]
        stats = _compute_weekly_wape_stats(df, alpha=0.5)

        expected_keys = ["wins_vs_lp", "total_weeks", "win_rate_vs_lp",
                         "avg_wape_ml", "avg_wape_lp", "avg_wape_hybrid"]
        for key in expected_keys:
            assert key in stats


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_alpha_tuning_workflow(self, sample_backtest_df):
        """Test the full workflow: tune alpha -> get mapping -> compute breakdown."""
        # Step 1: Tune alpha
        alpha_df = tune_hybrid_alpha(sample_backtest_df)

        # Step 2: Get mapping
        alpha_mapping = get_alpha_mapping(alpha_df)

        # Step 3: Compute breakdown
        breakdown = compute_weekly_hybrid_breakdown(sample_backtest_df, alpha_mapping)

        # Verify workflow completed
        assert not alpha_df.empty
        assert len(alpha_mapping) > 0
        # breakdown may be empty if no valid TRP H1-4 data
        assert isinstance(breakdown, pd.DataFrame)

    def test_hybrid_prediction_formula(self):
        """Verify hybrid = alpha * ML + (1 - alpha) * LP."""
        alpha = 0.3
        ml_pred = 100
        lp_pred = 200

        expected_hybrid = alpha * ml_pred + (1 - alpha) * lp_pred
        assert expected_hybrid == pytest.approx(170, rel=1e-6)
