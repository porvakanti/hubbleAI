"""
Page 3 - Backtest Explorer

Shows:
- Week-by-week drill-down into backtest predictions
- Compare Actual vs ML vs LP for each horizon (H1-H8)
- P10/P50/P90 quantiles on ML predictions

Design: Modern, warm cream palette using shared UI components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ui_components import (
    set_global_style,
    render_sidebar,
    render_interpretation_box,
    APP_VERSION,
    WAPE_EPS_THRESHOLD,
)
from hubbleAI.service import (
    load_latest_backtest_results,
    get_last_run_by_mode,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Backtest Explorer - Hubble.AI",
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

render_sidebar(active_page="Backtest Explorer", ref_info=ref_info)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def create_week_explorer_chart(week_data: pd.DataFrame, selected_view: str) -> go.Figure:
    """Create grouped bar chart for week explorer with P10/P90 error bars on ML."""
    if week_data.empty:
        return go.Figure()

    fig = go.Figure()
    range_color = "#2C5AA0"  # Nogaro blue for error bars

    # Check if we have quantile data
    has_quantiles = ("ML P10 (M)" in week_data.columns and "ML P90 (M)" in week_data.columns
                     and week_data["ML P10 (M)"].notna().any() and week_data["ML P90 (M)"].notna().any())

    # Actual
    fig.add_trace(go.Bar(
        x=week_data["Horizon"],
        y=week_data["Actual (M)"],
        name="Actual",
        marker_color="#2D3436",
        hovertemplate="<b>%{x}</b><br>Actual: %{y:.2f}M EUR<extra></extra>"
    ))

    # ML with error bars if quantiles available
    if has_quantiles:
        ml_pred = week_data["ML Pred (M)"].values
        p10_vals = week_data["ML P10 (M)"].values
        p90_vals = week_data["ML P90 (M)"].values

        # Handle NaN by using prediction as fallback
        p10_vals = np.where(np.isnan(p10_vals), ml_pred, p10_vals)
        p90_vals = np.where(np.isnan(p90_vals), ml_pred, p90_vals)

        error_minus = ml_pred - p10_vals
        error_plus = p90_vals - ml_pred

        fig.add_trace(go.Bar(
            x=week_data["Horizon"],
            y=ml_pred,
            name="ML",
            marker_color="#2E7D32",
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color=range_color,
                thickness=2,
                width=6,
            ),
            hovertemplate="<b>%{x}</b><br>ML: %{y:.2f}M<br>P10: %{customdata[0]:.2f}M<br>P90: %{customdata[1]:.2f}M<extra></extra>",
            customdata=list(zip(p10_vals, p90_vals))
        ))
    else:
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
        height=400,
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
        Backtest Explorer
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        Drill down into historical backtest predictions week by week
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

backtest = load_latest_backtest_results()

if backtest is None:
    st.warning("No backtest results available.")
    st.info("Run a backtest first: Go to **Cash Flows** page and click 'Run Backtest'.")
    st.stop()

bt_df = backtest.backtest_df.copy() if backtest.backtest_df is not None else pd.DataFrame()

if bt_df.empty:
    st.warning("Backtest predictions not available.")
    st.stop()

# ---------------------------------------------------------------------------
# Interpretation Guide
# ---------------------------------------------------------------------------

st.markdown(render_interpretation_box(
    title="How to Use This Explorer",
    content="""Select a historical week from the test set to see what the 8-week forecast (H1-H8) would have looked like.
    Compare <strong>Actual</strong> values against <strong>ML</strong> and <strong>LP</strong> predictions for each horizon.
    The blue error bars show the P10-P90 confidence range for ML predictions."""
), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Explorer Filters</div>', unsafe_allow_html=True)

if "week_start" not in bt_df.columns:
    st.error("week_start column not found in backtest data.")
    st.stop()

available_weeks = sorted(bt_df["week_start"].unique())

if len(available_weeks) == 0:
    st.info("No weeks available in backtest data.")
    st.stop()

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_week = st.selectbox(
        "Reference Week",
        available_weeks,
        format_func=lambda x: str(x)[:10],
        key="explorer_week"
    )

with filter_col2:
    explorer_view_options = ["NET", "TRR", "TRP"]
    explorer_view = st.selectbox("View (Liquidity Group)", explorer_view_options, key="explorer_view")

with filter_col3:
    tier1_only = st.toggle("Tier-1 Only (exclude pass-through)", value=True, key="explorer_tier1")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Build Explorer Data
# ---------------------------------------------------------------------------

# Filter to selected week
week_data = bt_df[bt_df["week_start"] == selected_week].copy()

if tier1_only and "is_pass_through" in week_data.columns:
    week_data = week_data[week_data["is_pass_through"] == False]

if week_data.empty:
    st.info("No data for selected week.")
    st.stop()

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
    # Also extract P10/P90 quantiles
    if explorer_view == "TRP" and h <= 4 and "y_pred_hybrid" in subset.columns:
        ml_pred = subset["y_pred_hybrid"].sum()
        if pd.isna(ml_pred):
            ml_pred = subset["y_pred_point"].sum()
        ml_p10 = subset["y_pred_p10"].sum() if "y_pred_p10" in subset.columns and subset["y_pred_p10"].notna().any() else None
        ml_p50 = subset["y_pred_p50"].sum() if "y_pred_p50" in subset.columns and subset["y_pred_p50"].notna().any() else None
        ml_p90 = subset["y_pred_p90"].sum() if "y_pred_p90" in subset.columns and subset["y_pred_p90"].notna().any() else None
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

        # Quantiles for NET (sum of both LGs)
        trr_p10 = trr_data["y_pred_p10"].sum() if "y_pred_p10" in trr_data.columns and trr_data["y_pred_p10"].notna().any() else 0
        trr_p50 = trr_data["y_pred_p50"].sum() if "y_pred_p50" in trr_data.columns and trr_data["y_pred_p50"].notna().any() else 0
        trr_p90 = trr_data["y_pred_p90"].sum() if "y_pred_p90" in trr_data.columns and trr_data["y_pred_p90"].notna().any() else 0
        trp_p10 = trp_data["y_pred_p10"].sum() if "y_pred_p10" in trp_data.columns and trp_data["y_pred_p10"].notna().any() else 0
        trp_p50 = trp_data["y_pred_p50"].sum() if "y_pred_p50" in trp_data.columns and trp_data["y_pred_p50"].notna().any() else 0
        trp_p90 = trp_data["y_pred_p90"].sum() if "y_pred_p90" in trp_data.columns and trp_data["y_pred_p90"].notna().any() else 0
        ml_p10 = (trr_p10 + trp_p10) if (trr_p10 != 0 or trp_p10 != 0) else None
        ml_p50 = (trr_p50 + trp_p50) if (trr_p50 != 0 or trp_p50 != 0) else None
        ml_p90 = (trr_p90 + trp_p90) if (trr_p90 != 0 or trp_p90 != 0) else None
    else:
        ml_pred = subset["y_pred_point"].sum() if "y_pred_point" in subset.columns else 0
        ml_p10 = subset["y_pred_p10"].sum() if "y_pred_p10" in subset.columns and subset["y_pred_p10"].notna().any() else None
        ml_p50 = subset["y_pred_p50"].sum() if "y_pred_p50" in subset.columns and subset["y_pred_p50"].notna().any() else None
        ml_p90 = subset["y_pred_p90"].sum() if "y_pred_p90" in subset.columns and subset["y_pred_p90"].notna().any() else None

    # LP baseline is only available for H1-H4
    if h <= 4 and "lp_baseline_point" in subset.columns:
        lp_sum = subset["lp_baseline_point"].sum()
        # Treat 0.00 as N/A since LP baseline should have actual values when available
        lp_pred = lp_sum if pd.notna(lp_sum) and abs(lp_sum) > 0.01 else float("nan")
    else:
        lp_pred = float("nan")

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
        "ML P10 (M)": ml_p10 / 1e6 if ml_p10 is not None else None,
        "ML P50 (M)": ml_p50 / 1e6 if ml_p50 is not None else None,
        "ML P90 (M)": ml_p90 / 1e6 if ml_p90 is not None else None,
        "LP Pred (M)": lp_pred / 1e6 if pd.notna(lp_pred) else None,
        "ML WAPE": ml_wape,
        "LP WAPE": lp_wape,
        "Winner": winner,
    })

if not explorer_rows:
    st.info("No data for selected week/view combination.")
    st.stop()

explorer_df = pd.DataFrame(explorer_rows)

# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------

st.markdown(f'<div class="section-title">Week {str(selected_week)[:10]} - {explorer_view} Forecast</div>', unsafe_allow_html=True)

# Format table for display (full width)
display_exp = explorer_df.copy()
display_exp["Actual (M)"] = display_exp["Actual (M)"].apply(lambda x: f"{x:.2f}")
display_exp["ML Pred (M)"] = display_exp["ML Pred (M)"].apply(lambda x: f"{x:.2f}")
display_exp["ML P10 (M)"] = display_exp["ML P10 (M)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
display_exp["ML P50 (M)"] = display_exp["ML P50 (M)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
display_exp["ML P90 (M)"] = display_exp["ML P90 (M)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
display_exp["LP Pred (M)"] = display_exp["LP Pred (M)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
display_exp["ML WAPE"] = display_exp["ML WAPE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")
display_exp["LP WAPE"] = display_exp["LP WAPE"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

# Reorder columns for better readability
col_order = ["Horizon", "Target Week", "Actual (M)", "ML Pred (M)", "ML P10 (M)", "ML P50 (M)", "ML P90 (M)", "LP Pred (M)", "ML WAPE", "LP WAPE", "Winner"]
col_order = [c for c in col_order if c in display_exp.columns]
display_exp = display_exp[col_order]

st.dataframe(
    display_exp,
    use_container_width=True,
    hide_index=True,
    height=320,
)
st.caption("**LP Pred:** Only available for H1-H4. **Winner:** Based on lower WAPE (closer to actual).")

# Chart (full width below table)
fig = create_week_explorer_chart(explorer_df, explorer_view)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ml_wins_exp = (explorer_df["Winner"] == "ML").sum()
lp_wins_exp = (explorer_df["Winner"] == "LP").sum()
valid_exp = len(explorer_df[explorer_df["Winner"] != "N/A"])

# Calculate totals
total_actual = explorer_df["Actual (M)"].sum()
total_ml = explorer_df["ML Pred (M)"].sum()
total_lp = explorer_df["LP Pred (M)"].sum() if explorer_df["LP Pred (M)"].notna().any() else None

summary_content = f"""<p>ML outperformed LP in <strong>{ml_wins_exp}</strong> of <strong>{valid_exp}</strong> horizons for this week.</p>
<p style="margin-top: 0.5rem;"><strong>8-Week Totals:</strong> Actual: <strong>{total_actual:.1f}M</strong> | ML: <strong>{total_ml:.1f}M</strong>"""

if total_lp is not None:
    summary_content += f" | LP: <strong>{total_lp:.1f}M</strong>"
summary_content += "</p>"

st.markdown(f"""
<div class="interpretation-box">
    <h4>Week {str(selected_week)[:10]} Summary ({explorer_view})</h4>
    {summary_content}
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Backtest reference: {backtest.ref_week_start} | Hubble.AI v{APP_VERSION}")
