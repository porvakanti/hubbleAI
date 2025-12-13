"""
Page 2 - Performance Dashboard (Analytics View)

Shows:
- ML vs LP performance comparison
- Weekly WAPE trends
- Backtest week explorer

Design: Modern, warm cream palette using shared UI components.

Business logic:
- UI labels model as "ML" (not hybrid)
- For TRP H1-H4: ML uses y_pred_hybrid if available, else y_pred_point
- For TRP H5-H8 and TRR: ML uses y_pred_point
- LP baseline available only for H1-H4
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ui_components import (
    set_global_style,
    render_sidebar,
    render_metric_card,
    render_score_card,
    render_interpretation_box,
    format_pct,
    APP_VERSION,
    WAPE_EPS_THRESHOLD,
)
from hubbleAI.service import (
    load_latest_backtest_results,
    get_last_run_by_mode,
    load_backtest_predictions,
    load_metrics_artifact,
    compute_weekly_wape_by_lg,
    compute_net_weekly_wape,
    get_performance_summary,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Performance Dashboard - Hubble.AI",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styles
set_global_style()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

forward_status = get_last_run_by_mode("forward")
ref_info = None
if forward_status:
    ref_info = {
        "ref_week_start": forward_status.get("ref_week_start", "-"),
        "run_id": forward_status.get("run_id", "-"),
    }

render_sidebar(active_page="Analytics", ref_info=ref_info)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_ml_data_for_view(df: pd.DataFrame, lg: str, horizon: int) -> pd.DataFrame:
    """
    Get ML predictions with proper column selection based on business rules.

    Rules:
    - TRR: y_pred_point
    - TRP H1-H4: y_pred_hybrid if available, else y_pred_point
    - TRP H5-H8: y_pred_point
    - NET: sum of TRR + TRP with above rules
    """
    if df.empty:
        return pd.DataFrame()

    filtered = df[df["horizon"] == horizon].copy()

    if lg == "NET":
        return filtered
    else:
        return filtered[filtered["liquidity_group"] == lg]


def create_wape_line_chart(df_wape: pd.DataFrame, title: str) -> go.Figure:
    """Create a line chart comparing ML vs LP WAPE over time."""
    if df_wape.empty:
        return go.Figure()

    df = df_wape.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    fig = go.Figure()

    # ML line
    fig.add_trace(go.Scatter(
        x=df["week_start"],
        y=df["ml_wape"] * 100,
        mode="lines+markers",
        name="ML",
        line=dict(color="#2E7D32", width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>ML WAPE: %{y:.1f}%<extra></extra>"
    ))

    # LP line (only where available)
    lp_data = df[~df["lp_wape"].isna()]
    if not lp_data.empty:
        fig.add_trace(go.Scatter(
            x=lp_data["week_start"],
            y=lp_data["lp_wape"] * 100,
            mode="lines+markers",
            name="LP",
            line=dict(color="#F57C00", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>LP WAPE: %{y:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Week",
        yaxis_title="WAPE (%)",
        height=400,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified"
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

    return fig


def create_weekly_comparison_table(df_wape: pd.DataFrame) -> pd.DataFrame:
    """Create formatted table for weekly comparison."""
    if df_wape.empty:
        return pd.DataFrame()

    df = df_wape.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start", ascending=False)

    display_df = pd.DataFrame({
        "Week": df["week_start"].dt.strftime("%Y-%m-%d"),
        "ML WAPE": df["ml_wape"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"),
        "LP WAPE": df["lp_wape"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"),
        "Winner": df["winner"],
    })

    return display_df


def create_week_explorer_chart(week_data: pd.DataFrame, selected_view: str) -> go.Figure:
    """Create grouped bar chart for week explorer."""
    if week_data.empty:
        return go.Figure()

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Bar(
        x=week_data["Horizon"],
        y=week_data["Actual (M)"],
        name="Actual",
        marker_color="#2D3436",
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.2f}M EUR<extra></extra>"
    ))

    # ML
    fig.add_trace(go.Bar(
        x=week_data["Horizon"],
        y=week_data["ML Pred (M)"],
        name="ML",
        marker_color="#2E7D32",
        hovertemplate="<b>%{x}</b><br>ML: %{y:.2f}M EUR<extra></extra>"
    ))

    # LP (if available)
    if "LP Pred (M)" in week_data.columns:
        lp_vals = week_data["LP Pred (M)"].dropna()
        if not lp_vals.empty:
            fig.add_trace(go.Bar(
                x=week_data["Horizon"],
                y=week_data["LP Pred (M)"],
                name="LP",
                marker_color="#F57C00",
                hovertemplate="<b>%{x}</b><br>LP: %{y:.2f}M EUR<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text=f"{selected_view} by Horizon", font=dict(size=14)),
        xaxis_title="Horizon",
        yaxis_title="EUR (Millions)",
        barmode="group",
        height=350,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)")

    return fig


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Page Header
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem; font-size: 1.75rem; font-weight: 700; color: #2D3436;">
        Performance Dashboard
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        ML vs LP accuracy comparison from backtest results
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

backtest = load_latest_backtest_results()

if backtest is None:
    st.warning("No backtest results available.")
    st.info("Run a backtest first: Go to **Latest Forecast** page and click 'Run Backtest'.")
    st.stop()

bt_df = backtest.backtest_df.copy() if backtest.backtest_df is not None else pd.DataFrame()
metrics = backtest.metrics
diagnostics = backtest.diagnostics

if bt_df.empty:
    st.warning("Backtest predictions not available.")
    st.stop()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Performance Filters</div>', unsafe_allow_html=True)

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    view_options = ["NET", "TRR", "TRP"]
    selected_view = st.selectbox("View (Liquidity Group)", view_options, key="perf_view")

with filter_col2:
    horizon_options = list(range(1, 9))
    selected_horizon = st.selectbox(
        "Horizon",
        horizon_options,
        format_func=lambda x: f"H{x}",
        key="perf_horizon"
    )

with filter_col3:
    tier1_only = st.toggle("Tier-1 Only (exclude pass-through)", value=True, key="perf_tier1")

# Show LP availability note
if selected_horizon > 4:
    st.info("LP baseline is not available for H5-H8. Only ML performance will be shown.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Compute Weekly WAPE Data
# ---------------------------------------------------------------------------

if selected_view == "NET":
    df_wape = compute_net_weekly_wape(bt_df, selected_horizon, tier1_only)
else:
    df_wape = compute_weekly_wape_by_lg(bt_df, selected_view, selected_horizon, tier1_only)

# Get performance summary
summary = get_performance_summary(df_wape)

# ---------------------------------------------------------------------------
# Score Cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)

card_col1, card_col2, card_col3, card_col4 = st.columns(4)

with card_col1:
    ml_wins = summary["ml_wins"]
    lp_wins = summary["lp_wins"]
    total_valid = ml_wins + lp_wins + summary["ties"]
    win_rate = summary["win_rate"]

    st.markdown(render_score_card(
        value=f"{ml_wins}/{total_valid}",
        label="Weeks ML Beats LP",
        detail=f"{win_rate:.0%} win rate",
        accent=win_rate > 0.5
    ), unsafe_allow_html=True)

with card_col2:
    avg_ml = summary["avg_ml_wape"]
    avg_lp = summary["avg_lp_wape"]
    ml_display = f"{avg_ml*100:.1f}%" if pd.notna(avg_ml) else "-"
    lp_display = f"{avg_lp*100:.1f}%" if pd.notna(avg_lp) else "-"

    st.markdown(render_score_card(
        value=ml_display,
        label="Avg ML WAPE",
        detail=f"LP: {lp_display}"
    ), unsafe_allow_html=True)

with card_col3:
    improvement = summary["improvement_pp"]
    sign = "+" if improvement > 0 else ""

    st.markdown(render_score_card(
        value=f"{sign}{improvement:.1f}pp",
        label="WAPE Improvement",
        detail="vs LP baseline",
        accent=improvement > 0
    ), unsafe_allow_html=True)

with card_col4:
    # Best week
    if not df_wape.empty:
        valid_weeks = df_wape[~df_wape["ml_wape"].isna()]
        if not valid_weeks.empty:
            best_idx = valid_weeks["ml_wape"].idxmin()
            best_wape = valid_weeks.loc[best_idx, "ml_wape"]
            best_week = valid_weeks.loc[best_idx, "week_start"]
            best_week_str = str(best_week)[:10]
        else:
            best_wape = float("nan")
            best_week_str = "-"
    else:
        best_wape = float("nan")
        best_week_str = "-"

    st.markdown(render_score_card(
        value=f"{best_wape*100:.1f}%" if pd.notna(best_wape) else "-",
        label="Best Week WAPE",
        detail=best_week_str
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main WAPE Chart
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Weekly WAPE Comparison</div>', unsafe_allow_html=True)

if df_wape.empty:
    st.info(f"No data available for {selected_view} H{selected_horizon}.")
else:
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        chart_title = f"{selected_view} H{selected_horizon} - ML vs LP WAPE Over Time"
        fig = create_wape_line_chart(df_wape, chart_title)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lower WAPE = Better accuracy. Dashed line = LP baseline.")

    with table_col:
        display_table = create_weekly_comparison_table(df_wape)
        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    # Summary interpretation
    if total_valid > 0:
        st.markdown(f"""
        <div class="interpretation-box">
            <h4>{selected_view} H{selected_horizon} Summary</h4>
            <p>ML outperforms LP in <strong>{ml_wins}</strong> of <strong>{total_valid}</strong> weeks ({win_rate:.0%}).
            Average ML WAPE: <strong>{ml_display}</strong>, LP WAPE: <strong>{lp_display}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Backtest Week Explorer
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Backtest Week Explorer</div>', unsafe_allow_html=True)

st.markdown(render_interpretation_box(
    title="How to Use",
    content="Select a historical week from the test set to see what the 8-week forecast (H1-H8) would have looked like. Compare Actual vs ML vs LP predictions for each horizon."
), unsafe_allow_html=True)

if "week_start" in bt_df.columns:
    available_weeks = sorted(bt_df["week_start"].unique())

    if len(available_weeks) > 0:
        explorer_col1, explorer_col2 = st.columns([2, 1])

        with explorer_col1:
            selected_week = st.selectbox(
                "Select Reference Week",
                available_weeks,
                format_func=lambda x: str(x)[:10],
                key="explorer_week"
            )

        with explorer_col2:
            explorer_view_options = ["NET", "TRR", "TRP"]
            explorer_view = st.selectbox("View", explorer_view_options, key="explorer_view")

        # Filter to selected week
        week_data = bt_df[bt_df["week_start"] == selected_week].copy()

        if tier1_only and "is_pass_through" in week_data.columns:
            week_data = week_data[week_data["is_pass_through"] == False]

        if not week_data.empty:
            # Build H1-H8 summary
            explorer_rows = []
            for h in range(1, 9):
                h_data = week_data[week_data["horizon"] == h]

                if explorer_view == "NET":
                    subset = h_data
                else:
                    subset = h_data[h_data["liquidity_group"] == explorer_view]

                if subset.empty:
                    continue

                # Sum values
                actual = subset["actual_value"].sum() if "actual_value" in subset.columns else 0

                # ML prediction: use hybrid for TRP H1-4, point otherwise
                if explorer_view == "TRP" and h <= 4 and "y_pred_hybrid" in subset.columns:
                    ml_pred = subset["y_pred_hybrid"].sum()
                    if pd.isna(ml_pred):
                        ml_pred = subset["y_pred_point"].sum()
                elif explorer_view == "NET":
                    # For NET: sum TRR point + TRP hybrid/point
                    trr_data = subset[subset["liquidity_group"] == "TRR"]
                    trp_data = subset[subset["liquidity_group"] == "TRP"]

                    trr_ml = trr_data["y_pred_point"].sum() if "y_pred_point" in trr_data.columns else 0
                    if h <= 4 and "y_pred_hybrid" in trp_data.columns:
                        trp_ml = trp_data["y_pred_hybrid"].sum()
                        if pd.isna(trp_ml):
                            trp_ml = trp_data["y_pred_point"].sum()
                    else:
                        trp_ml = trp_data["y_pred_point"].sum() if "y_pred_point" in trp_data.columns else 0

                    ml_pred = trr_ml + trp_ml
                else:
                    ml_pred = subset["y_pred_point"].sum() if "y_pred_point" in subset.columns else 0

                lp_pred = subset["lp_baseline_point"].sum() if "lp_baseline_point" in subset.columns else float("nan")

                # Get target week
                target = subset["target_week_start"].iloc[0] if "target_week_start" in subset.columns else None

                # Compute WAPE (with guardrail)
                eps = WAPE_EPS_THRESHOLD
                if abs(actual) >= eps:
                    ml_wape = abs(actual - ml_pred) / abs(actual)
                    lp_wape = abs(actual - lp_pred) / abs(actual) if pd.notna(lp_pred) and h <= 4 else float("nan")
                else:
                    ml_wape = float("nan")
                    lp_wape = float("nan")

                # Determine winner
                if pd.isna(ml_wape):
                    winner = "N/A"
                elif pd.isna(lp_wape):
                    winner = "ML"
                elif abs(ml_wape - lp_wape) < 0.001:
                    winner = "Tie"
                elif ml_wape < lp_wape:
                    winner = "ML"
                else:
                    winner = "LP"

                explorer_rows.append({
                    "Horizon": f"H{h}",
                    "Target Week": str(target)[:10] if pd.notna(target) else "-",
                    "Actual (M)": actual / 1e6,
                    "ML Pred (M)": ml_pred / 1e6,
                    "LP Pred (M)": lp_pred / 1e6 if pd.notna(lp_pred) else None,
                    "ML WAPE": ml_wape,
                    "LP WAPE": lp_wape,
                    "Winner": winner,
                })

            if explorer_rows:
                explorer_df = pd.DataFrame(explorer_rows)

                # Display table and chart
                exp_table_col, exp_chart_col = st.columns([1, 1.5])

                with exp_table_col:
                    # Format for display
                    display_exp = explorer_df.copy()
                    display_exp["Actual (M)"] = display_exp["Actual (M)"].apply(lambda x: f"{x:.2f}")
                    display_exp["ML Pred (M)"] = display_exp["ML Pred (M)"].apply(lambda x: f"{x:.2f}")
                    display_exp["LP Pred (M)"] = display_exp["LP Pred (M)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    display_exp["ML WAPE"] = display_exp["ML WAPE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
                    display_exp["LP WAPE"] = display_exp["LP WAPE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

                    st.dataframe(
                        display_exp,
                        use_container_width=True,
                        hide_index=True,
                        height=350,
                    )

                with exp_chart_col:
                    fig = create_week_explorer_chart(explorer_df, explorer_view)
                    st.plotly_chart(fig, use_container_width=True)

                # Summary
                ml_wins_exp = (explorer_df["Winner"] == "ML").sum()
                lp_wins_exp = (explorer_df["Winner"] == "LP").sum()
                valid_exp = len(explorer_df[explorer_df["Winner"] != "N/A"])

                st.markdown(f"""
                <div class="interpretation-box">
                    <h4>Week {str(selected_week)[:10]} Summary ({explorer_view})</h4>
                    <p>ML outperformed LP in <strong>{ml_wins_exp}</strong> of <strong>{valid_exp}</strong> horizons.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No data for selected week/view combination.")
        else:
            st.info("No data for selected week.")
    else:
        st.info("No weeks available in backtest data.")
else:
    st.info("week_start column not found in backtest data.")

# ---------------------------------------------------------------------------
# Quantile Calibration (Secondary, only if available)
# ---------------------------------------------------------------------------

quant_coverage = diagnostics.get("quantile_coverage_by_horizon")
if quant_coverage is not None and not quant_coverage.empty:
    st.markdown("---")

    with st.expander("Quantile Calibration (P10/P50/P90)", expanded=False):
        st.markdown("""
        <p style="color: #5A6169; font-size: 0.85rem; margin-bottom: 1rem;">
        Ideal calibration: ~10% below P10, ~80% in P10-P90 band, ~10% above P90.
        </p>
        """, unsafe_allow_html=True)

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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Backtest reference: {backtest.ref_week_start} | Hubble.AI v{APP_VERSION}")
