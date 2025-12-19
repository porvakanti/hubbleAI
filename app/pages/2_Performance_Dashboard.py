"""
Page 2 - Performance Dashboard (Analytics View)

Shows:
- ML vs LP performance comparison
- Weekly WAPE trends with target lines
- Quantile calibration metrics

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


# WAPE targets by horizon (TRR/TRP/NET)
WAPE_TARGETS = {
    1: 5.0,
    2: 7.5,
    3: 10.0,
    4: 12.5,
    5: 15.0,
    6: 17.5,
    7: 20.0,
    8: 22.5,
}


def create_wape_line_chart(df_wape: pd.DataFrame, title: str, horizon: int = 1) -> go.Figure:
    """Create a line chart comparing ML vs LP WAPE over time with target line.

    Highlights weeks where ML WAPE is at or below the target with gold markers.
    """
    if df_wape.empty:
        return go.Figure()

    df = df_wape.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    fig = go.Figure()

    # Target line (horizontal) - Nogaro blue
    target_wape = WAPE_TARGETS.get(horizon, 10.0)
    fig.add_trace(go.Scatter(
        x=[df["week_start"].min(), df["week_start"].max()],
        y=[target_wape, target_wape],
        mode="lines",
        name=f"Target ({target_wape}%)",
        line=dict(color="#2C5AA0", width=2, dash="dot"),
        hovertemplate=f"Target: {target_wape}%<extra></extra>"
    ))

    # Separate weeks meeting target vs not
    df["ml_wape_pct"] = df["ml_wape"] * 100
    meets_target = df[df["ml_wape_pct"] <= target_wape]
    misses_target = df[df["ml_wape_pct"] > target_wape]

    # ML line (continuous) - no hover, markers handle tooltips
    fig.add_trace(go.Scatter(
        x=df["week_start"],
        y=df["ml_wape_pct"],
        mode="lines",
        name="ML",
        line=dict(color="#2E7D32", width=2),
        hoverinfo="skip"
    ))

    # ML markers - weeks NOT meeting target (regular green)
    if not misses_target.empty:
        fig.add_trace(go.Scatter(
            x=misses_target["week_start"],
            y=misses_target["ml_wape_pct"],
            mode="markers",
            name="ML (above target)",
            marker=dict(size=8, color="#2E7D32"),
            showlegend=False,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>ML WAPE: %{y:.1f}%<extra></extra>"
        ))

    # ML markers - weeks MEETING target (gold star markers)
    if not meets_target.empty:
        fig.add_trace(go.Scatter(
            x=meets_target["week_start"],
            y=meets_target["ml_wape_pct"],
            mode="markers",
            name="Target Met",
            marker=dict(size=12, color="#D4AF37", symbol="star", line=dict(width=1, color="#8B7500")),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>ML WAPE: %{y:.1f}% ★ Target Met!<extra></extra>"
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


def format_currency(val: float) -> str:
    """Format value as currency in millions."""
    if pd.isna(val) or val == 0:
        return "-"
    return f"${val/1e6:.1f}M"


def create_weekly_comparison_table(df_wape: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create formatted table for weekly comparison with extended columns."""
    if df_wape.empty:
        return pd.DataFrame()

    df = df_wape.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start", ascending=False)

    # Replace N/A winner with more descriptive text
    def format_winner(row):
        if row["winner"] == "N/A":
            return "Pending"  # Actuals not yet available
        return row["winner"]

    # Build the display dataframe with extended columns
    display_df = pd.DataFrame({
        "Week": df["week_start"].dt.strftime("%Y-%m-%d"),
        "Actuals": df["actual_sum"].apply(format_currency),
        "ML Pred": df["ml_pred_sum"].apply(format_currency),
        "P10": df["p10_sum"].apply(format_currency) if "p10_sum" in df.columns else "-",
        "P50": df["p50_sum"].apply(format_currency) if "p50_sum" in df.columns else "-",
        "P90": df["p90_sum"].apply(format_currency) if "p90_sum" in df.columns else "-",
        "LP Pred": df["lp_pred_sum"].apply(format_currency) if horizon <= 4 else "-",
        "ML WAPE": df["ml_wape"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"),
        "LP WAPE": df["lp_wape"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-") if horizon <= 4 else "-",
        "Winner": df.apply(format_winner, axis=1),
    })

    # For H5-H8, LP columns should show "-"
    if horizon > 4:
        display_df["LP Pred"] = "-"
        display_df["LP WAPE"] = "-"

    return display_df


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
    # Full-width WAPE line chart
    chart_title = f"{selected_view} H{selected_horizon} - ML vs LP WAPE Over Time"
    fig = create_wape_line_chart(df_wape, chart_title, horizon=selected_horizon)
    st.plotly_chart(fig, use_container_width=True)
    target_pct = WAPE_TARGETS.get(selected_horizon, 10.0)
    st.caption(f"Lower WAPE = Better accuracy. Target: {target_pct}% (dotted blue). ★ Gold stars = Target met. LP baseline: dashed orange.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Full-width detailed table
    st.markdown('<div class="section-title">Weekly Details</div>', unsafe_allow_html=True)
    display_table = create_weekly_comparison_table(df_wape, horizon=selected_horizon)
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=400,
    )
    st.caption("**Pending** = Actuals not yet available for WAPE calculation. Values in millions (M).")

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
# Quantile Calibration
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Quantile Calibration</div>', unsafe_allow_html=True)

st.markdown(render_interpretation_box(
    title="What is Quantile Calibration?",
    content="""This checks if the predicted confidence intervals (P10/P50/P90) are accurate.
    <strong>Ideal calibration:</strong> ~10% of actuals below P10, ~80% within P10-P90 band, ~10% above P90.
    If calibration is off, it means the model's uncertainty estimates may not be trustworthy."""
), unsafe_allow_html=True)

quant_coverage = diagnostics.get("quantile_coverage_by_horizon") if diagnostics else None

# If pre-computed file not available, compute on-the-fly from backtest data
if (quant_coverage is None or quant_coverage.empty) and not bt_df.empty:
    required_cols = ["horizon", "actual_value", "y_pred_p10", "y_pred_p90"]
    if all(col in bt_df.columns for col in required_cols):
        # Compute quantile calibration by horizon
        calib_rows = []
        for h in sorted(bt_df["horizon"].dropna().unique()):
            h_data = bt_df[bt_df["horizon"] == h].dropna(subset=["actual_value", "y_pred_p10", "y_pred_p90"])
            n = len(h_data)
            if n > 0:
                below_p10 = (h_data["actual_value"] < h_data["y_pred_p10"]).sum()
                above_p90 = (h_data["actual_value"] > h_data["y_pred_p90"]).sum()
                in_band = n - below_p10 - above_p90
                calib_rows.append({
                    "horizon": int(h),
                    "n": n,
                    "prob_below_p10": below_p10 / n,
                    "prob_between_p10_p90": in_band / n,
                    "prob_above_p90": above_p90 / n,
                })
        if calib_rows:
            quant_coverage = pd.DataFrame(calib_rows)

if quant_coverage is not None and not quant_coverage.empty:
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

    st.dataframe(coverage_display[display_cols], hide_index=True, use_container_width=True, height=340)
else:
    st.warning("Quantile calibration data is not available.")
    st.markdown("""
    <div class="interpretation-box" style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); border-left-color: #F57C00;">
        <h4 style="color: #E65100;">How to Generate This Data</h4>
        <p>Quantile calibration requires P10/P90 prediction columns in the backtest data.
        If these columns exist but no data is shown, there may be too many missing values.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Backtest reference: {backtest.ref_week_start} | Hubble.AI v{APP_VERSION}")
