"""
Page 2 - Performance Dashboard (Backtest View)

Shows:
- ML vs LP performance comparison
- Score cards: weeks ML outperforms LP, best week, etc.
- WAPE trends with NET option
- Time machine: select historical week to see H1-H8 retrospective

Design: Warm cream palette, card-based layout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.service import (
    load_latest_backtest_results,
    get_hybrid_performance_summary,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Performance Dashboard - HubbleAI",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Page CSS
# ---------------------------------------------------------------------------

PAGE_CSS = """
<style>
/* Root variables */
:root {
    --bg-cream: #F5F5F0;
    --bg-cream-light: #FAFAF7;
    --bg-white: #FFFFFF;
    --text-dark: #2D3436;
    --text-muted: #636E72;
    --text-light: #95A5A6;
    --accent-green: #4CAF50;
    --accent-green-light: #81C784;
    --accent-green-dark: #388E3C;
    --accent-orange: #FF9800;
    --accent-red: #E53935;
    --accent-blue: #2196F3;
    --border-light: rgba(0, 0, 0, 0.06);
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

.stApp { background-color: var(--bg-cream); }

/* Card styling */
.card {
    background: var(--bg-white);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    margin-bottom: 1rem;
}

.card-sm { padding: 1rem; border-radius: var(--radius-md); }

.card-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.2;
}

.card-value-sm { font-size: 1.25rem; }

.card-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Status colors */
.text-green { color: var(--accent-green); }
.text-orange { color: var(--accent-orange); }
.text-red { color: var(--accent-red); }
.text-muted { color: var(--text-muted); }

/* Score card specific */
.score-card {
    background: var(--bg-white);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
    text-align: center;
}

.score-card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
    color: white;
}

.score-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}

.score-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
    opacity: 0.9;
}

.score-detail {
    font-size: 0.8rem;
    margin-top: 0.5rem;
    opacity: 0.8;
}

/* Typography */
h1, h2, h3, h4 { color: var(--text-dark); font-weight: 600; }
h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 1rem;
}

.page-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* Interpretation box */
.interpretation-box {
    background: linear-gradient(135deg, #E8F5E9 0%, #F1F8E9 100%);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    border-left: 4px solid var(--accent-green);
}
.interpretation-box h4 {
    color: var(--accent-green-dark);
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
}
.interpretation-box p {
    color: var(--text-dark);
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-weight: 500;
    background: var(--bg-white);
    border: 1px solid var(--border-light);
}
.stTabs [aria-selected="true"] {
    background: var(--accent-green) !important;
    color: white !important;
    border-color: var(--accent-green) !important;
}

/* Hide branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


def format_pct(val: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    if pd.isna(val):
        return "-"
    return f"{val * 100:.{decimals}f}%"


def format_pct_value(val: float, decimals: int = 1) -> str:
    """Format already-percentage value."""
    if pd.isna(val):
        return "-"
    return f"{val:.{decimals}f}%"


def render_score_card(value: str, label: str, detail: str = "", accent: bool = False) -> str:
    """Render a score card."""
    card_class = "score-card score-card-accent" if accent else "score-card"
    return f"""
    <div class="{card_class}">
        <div class="score-value">{value}</div>
        <div class="score-label">{label}</div>
        <div class="score-detail">{detail}</div>
    </div>
    """


def compute_net_metrics(metrics_lg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NET metrics by aggregating TRR and TRP.
    Uses aggregate-then-error approach.
    """
    if metrics_lg is None or metrics_lg.empty:
        return pd.DataFrame()

    # Group by week_start and horizon, sum the actuals and predictions
    net_rows = []

    for (week, horizon), grp in metrics_lg.groupby(["week_start", "horizon"], observed=True):
        actual_sum = grp["actual_sum"].sum()
        ml_sum = grp["ml_pred_sum"].sum()
        lp_sum = grp["lp_pred_sum"].sum() if "lp_pred_sum" in grp.columns else np.nan

        eps = 1e-6
        ml_wape = abs(actual_sum - ml_sum) / (abs(actual_sum) + eps) if pd.notna(actual_sum) else np.nan
        lp_wape = abs(actual_sum - lp_sum) / (abs(actual_sum) + eps) if pd.notna(lp_sum) else np.nan

        net_rows.append({
            "week_start": week,
            "horizon": horizon,
            "liquidity_group": "NET",
            "actual_sum": actual_sum,
            "ml_pred_sum": ml_sum,
            "lp_pred_sum": lp_sum,
            "ml_wape": ml_wape,
            "lp_wape": lp_wape,
        })

    return pd.DataFrame(net_rows)


def main():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <h1>Performance Dashboard</h1>
    <p class="page-subtitle">ML vs LP accuracy comparison from backtest results</p>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------------------------

    backtest = load_latest_backtest_results()

    if backtest is None:
        st.warning("No backtest results available.")
        st.info("Run a backtest first: Click 'Run Backtest' on the Latest Forecast page.")
        return

    metrics = backtest.metrics
    diagnostics = backtest.diagnostics

    # Use clean metrics (Tier-1 only) by default
    metrics_lg = metrics.get("metrics_by_lg_clean", metrics.get("metrics_by_lg"))
    metrics_net = metrics.get("metrics_net_clean", metrics.get("metrics_net"))

    # ---------------------------------------------------------------------------
    # Score Cards - ML vs LP Summary
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)

    if metrics_lg is not None and not metrics_lg.empty:
        # Calculate score card metrics
        total_weeks = metrics_lg["week_start"].nunique()

        # Count weeks where ML beats LP (lower WAPE)
        weekly_comparison = metrics_lg.groupby("week_start").agg({
            "ml_wape": "mean",
            "lp_wape": "mean"
        }).dropna()

        ml_wins = (weekly_comparison["ml_wape"] < weekly_comparison["lp_wape"]).sum()
        ml_win_rate = ml_wins / len(weekly_comparison) if len(weekly_comparison) > 0 else 0

        # Best week (lowest ML WAPE)
        best_week_idx = weekly_comparison["ml_wape"].idxmin() if not weekly_comparison.empty else None
        best_week_wape = weekly_comparison.loc[best_week_idx, "ml_wape"] if best_week_idx else np.nan

        # Average WAPE
        avg_ml_wape = metrics_lg["ml_wape"].mean()
        avg_lp_wape = metrics_lg["lp_wape"].mean()
        improvement = (avg_lp_wape - avg_ml_wape) * 100  # In percentage points

        # Display score cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(render_score_card(
                value=f"{ml_wins}/{len(weekly_comparison)}",
                label="Weeks ML Beats LP",
                detail=f"{ml_win_rate:.0%} win rate",
                accent=ml_win_rate > 0.5
            ), unsafe_allow_html=True)

        with col2:
            st.markdown(render_score_card(
                value=format_pct(avg_ml_wape),
                label="Avg ML WAPE",
                detail=f"LP: {format_pct(avg_lp_wape)}"
            ), unsafe_allow_html=True)

        with col3:
            sign = "+" if improvement > 0 else ""
            color = "text-green" if improvement > 0 else "text-red"
            st.markdown(render_score_card(
                value=f"{sign}{improvement:.1f}pp",
                label="WAPE Improvement",
                detail="vs LP baseline",
                accent=improvement > 0
            ), unsafe_allow_html=True)

        with col4:
            best_week_str = str(best_week_idx)[:10] if best_week_idx else "-"
            st.markdown(render_score_card(
                value=format_pct(best_week_wape),
                label="Best Week WAPE",
                detail=best_week_str
            ), unsafe_allow_html=True)
    else:
        st.info("Metrics data not available for score cards.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # WAPE Trends by Liquidity Group (with NET option)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">WAPE Trends by Liquidity Group</div>', unsafe_allow_html=True)

    if metrics_lg is not None and not metrics_lg.empty:
        # Add NET to the data
        net_metrics = compute_net_metrics(metrics_lg)

        # Combine LG + NET
        all_metrics = pd.concat([metrics_lg, net_metrics], ignore_index=True)

        # Filters
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            lg_options = ["TRR", "TRP", "NET"]
            selected_lg = st.selectbox("Liquidity Group", lg_options, key="trend_lg")

        with filter_col2:
            horizon_options = sorted(all_metrics["horizon"].unique())
            selected_h = st.selectbox("Horizon", horizon_options, format_func=lambda x: f"H{x}", key="trend_h")

        # Filter data
        filtered = all_metrics[
            (all_metrics["liquidity_group"] == selected_lg) &
            (all_metrics["horizon"] == selected_h)
        ].copy()

        if not filtered.empty:
            # Prepare chart data
            filtered["week_start"] = pd.to_datetime(filtered["week_start"])
            filtered = filtered.sort_values("week_start")

            # Format for display table
            display_df = filtered[["week_start", "ml_wape", "lp_wape"]].copy()
            display_df = display_df.sort_values("week_start", ascending=False)
            display_df["week_start"] = display_df["week_start"].dt.strftime("%Y-%m-%d")
            display_df["ML WAPE"] = display_df["ml_wape"].apply(format_pct)
            display_df["LP WAPE"] = display_df["lp_wape"].apply(format_pct)
            display_df["Winner"] = display_df.apply(
                lambda r: "ML" if r["ml_wape"] < r["lp_wape"] else ("LP" if r["lp_wape"] < r["ml_wape"] else "Tie"),
                axis=1
            )

            # Two columns: chart + table
            chart_col, table_col = st.columns([2, 1])

            with chart_col:
                # Line chart
                chart_data = filtered.set_index("week_start")[["ml_wape", "lp_wape"]].copy()
                chart_data = chart_data * 100  # Convert to percentage
                chart_data.columns = ["ML WAPE %", "LP WAPE %"]
                st.line_chart(chart_data, height=350)
                st.caption("Lower WAPE = Better accuracy")

            with table_col:
                st.dataframe(
                    display_df[["week_start", "ML WAPE", "LP WAPE", "Winner"]].rename(columns={"week_start": "Week"}),
                    hide_index=True,
                    height=350,
                    use_container_width=True
                )

            # Summary for this selection
            ml_avg = filtered["ml_wape"].mean()
            lp_avg = filtered["lp_wape"].mean()
            ml_wins_filtered = (filtered["ml_wape"] < filtered["lp_wape"]).sum()

            st.markdown(f"""
            <div class="interpretation-box">
                <h4>{selected_lg} H{selected_h} Summary</h4>
                <p>ML wins {ml_wins_filtered} of {len(filtered)} weeks ({ml_wins_filtered/len(filtered):.0%}).
                Average ML WAPE: {format_pct(ml_avg)}, LP WAPE: {format_pct(lp_avg)}.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"No data for {selected_lg} H{selected_h}")
    else:
        st.warning("Metrics by LG not available.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Time Machine: Historical Week Retrospective
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Time Machine: Historical Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="interpretation-box">
        <h4>How to Use</h4>
        <p>Select a historical week from the test set to see what the 8-week forecast (H1-H8) would have looked like.
        This helps understand how Treasury could have made different decisions with this intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    if backtest.backtest_df is not None and not backtest.backtest_df.empty:
        bt_df = backtest.backtest_df.copy()

        # Get available weeks
        if "week_start" in bt_df.columns:
            available_weeks = sorted(bt_df["week_start"].unique())

            if len(available_weeks) > 0:
                # Week selector
                week_col, lg_col = st.columns([2, 1])

                with week_col:
                    selected_week = st.selectbox(
                        "Select Reference Week",
                        available_weeks,
                        format_func=lambda x: str(x)[:10],
                        key="timemachine_week"
                    )

                with lg_col:
                    tm_lg_options = ["NET", "TRR", "TRP"]
                    selected_tm_lg = st.selectbox("View", tm_lg_options, key="timemachine_lg")

                # Filter to selected week
                week_data = bt_df[bt_df["week_start"] == selected_week].copy()

                if not week_data.empty:
                    # Build H1-H8 summary
                    tm_summary = []
                    for h in range(1, 9):
                        h_data = week_data[week_data["horizon"] == h]

                        if selected_tm_lg == "NET":
                            # Sum across LGs
                            subset = h_data
                        else:
                            subset = h_data[h_data["liquidity_group"] == selected_tm_lg]

                        if subset.empty:
                            continue

                        actual = subset["actual_value"].sum() if "actual_value" in subset.columns else np.nan
                        ml_pred = subset["y_pred_point"].sum() if "y_pred_point" in subset.columns else np.nan
                        lp_pred = subset["lp_baseline_point"].sum() if "lp_baseline_point" in subset.columns else np.nan

                        # Get target week
                        target = subset["target_week_start"].iloc[0] if "target_week_start" in subset.columns else None

                        # Compute WAPE
                        eps = 1e-6
                        ml_wape = abs(actual - ml_pred) / (abs(actual) + eps) if pd.notna(actual) and pd.notna(ml_pred) else np.nan
                        lp_wape = abs(actual - lp_pred) / (abs(actual) + eps) if pd.notna(actual) and pd.notna(lp_pred) else np.nan

                        tm_summary.append({
                            "Horizon": f"H{h}",
                            "Target Week": str(target)[:10] if pd.notna(target) else "-",
                            "Actual": actual,
                            "ML Pred": ml_pred,
                            "LP Pred": lp_pred,
                            "ML WAPE": ml_wape,
                            "LP WAPE": lp_wape,
                            "Winner": "ML" if ml_wape < lp_wape else ("LP" if lp_wape < ml_wape else "Tie")
                        })

                    if tm_summary:
                        tm_df = pd.DataFrame(tm_summary)

                        # Format for display
                        display_tm = tm_df.copy()
                        for col in ["Actual", "ML Pred", "LP Pred"]:
                            if col in display_tm.columns:
                                display_tm[col] = display_tm[col].apply(
                                    lambda x: f"{x/1e6:.2f}M" if pd.notna(x) else "-"
                                )
                        display_tm["ML WAPE"] = tm_df["ML WAPE"].apply(format_pct)
                        display_tm["LP WAPE"] = tm_df["LP WAPE"].apply(format_pct)

                        st.dataframe(display_tm, hide_index=True, use_container_width=True)

                        # Chart: Actual vs ML vs LP
                        chart_tm = tm_df[["Horizon", "Actual", "ML Pred", "LP Pred"]].copy()
                        chart_tm = chart_tm.set_index("Horizon")
                        chart_tm = chart_tm / 1e6  # Convert to millions
                        st.bar_chart(chart_tm, height=300)
                        st.caption("Values in millions EUR")

                        # Summary
                        ml_wins_tm = (tm_df["ML WAPE"] < tm_df["LP WAPE"]).sum()
                        st.markdown(f"""
                        <div class="interpretation-box">
                            <h4>Week {str(selected_week)[:10]} Summary ({selected_tm_lg})</h4>
                            <p>ML would have outperformed LP in {ml_wins_tm} of {len(tm_df)} horizons.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No data for selected week/LG combination.")
                else:
                    st.info("No data for selected week.")
            else:
                st.info("No weeks available in backtest data.")
        else:
            st.info("week_start column not found in backtest data.")
    else:
        st.info("Backtest predictions not available for time machine view.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # WAPE by Horizon Overview
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">WAPE by Horizon (All Test Weeks)</div>', unsafe_allow_html=True)

    if metrics_net is not None and not metrics_net.empty:
        # Aggregate by horizon
        horizon_summary = metrics_net.groupby("horizon").agg({
            "ml_wape": "mean",
            "lp_wape": "mean",
        }).reset_index()

        # Format for display
        display_horizon = horizon_summary.copy()
        display_horizon["Horizon"] = display_horizon["horizon"].apply(lambda x: f"H{x}")
        display_horizon["ML WAPE"] = display_horizon["ml_wape"].apply(format_pct)
        display_horizon["LP WAPE"] = display_horizon["lp_wape"].apply(format_pct)
        display_horizon["ML Better"] = display_horizon.apply(
            lambda r: "Yes" if r["ml_wape"] < r["lp_wape"] else "No",
            axis=1
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                display_horizon[["Horizon", "ML WAPE", "LP WAPE", "ML Better"]],
                hide_index=True,
                use_container_width=True
            )

        with col2:
            # Bar chart
            chart_horizon = horizon_summary.set_index("horizon")[["ml_wape", "lp_wape"]] * 100
            chart_horizon.columns = ["ML WAPE %", "LP WAPE %"]
            st.bar_chart(chart_horizon, height=300)
    else:
        st.info("Net metrics not available.")

    # ---------------------------------------------------------------------------
    # Quantile Coverage (if available)
    # ---------------------------------------------------------------------------

    quant_coverage = diagnostics.get("quantile_coverage_by_horizon")
    if quant_coverage is not None and not quant_coverage.empty:
        st.markdown("---")
        st.markdown('<div class="section-title">Quantile Calibration (P10/P50/P90)</div>', unsafe_allow_html=True)

        coverage_display = quant_coverage.copy()
        coverage_display["Horizon"] = coverage_display["horizon"].apply(lambda x: f"H{x}")

        for col in ["prob_below_p10", "prob_between_p10_p90", "prob_above_p90"]:
            if col in coverage_display.columns:
                coverage_display[col] = coverage_display[col].apply(lambda x: f"{x*100:.1f}%")

        coverage_display = coverage_display.rename(columns={
            "prob_below_p10": "Below P10",
            "prob_between_p10_p90": "P10-P90 Band",
            "prob_above_p90": "Above P90",
            "n": "N Samples"
        })

        display_cols = ["Horizon", "N Samples", "Below P10", "P10-P90 Band", "Above P90"]
        display_cols = [c for c in display_cols if c in coverage_display.columns]

        st.dataframe(coverage_display[display_cols], hide_index=True, use_container_width=True)
        st.caption("Ideal calibration: ~10% below P10, ~80% in P10-P90, ~10% above P90")

    # Footer
    st.markdown("---")
    st.caption(f"Backtest reference: {backtest.ref_week_start}")


if __name__ == "__main__":
    main()
