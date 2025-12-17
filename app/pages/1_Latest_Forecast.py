"""
Page 1 - Cash Flows

Shows:
- 8-week forecast horizon with P10/P50/P90
- Separate views for TRR (Inflows), TRP (Outflows), and NET
- NET summary with risk assessment
- Treasury interpretation guidance

For system administration (run forecasts, upload data), use the Admin page.

Design: Modern, warm cream palette using shared UI components.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ui_components import (
    set_global_style,
    render_sidebar,
    render_interpretation_box,
    APP_VERSION,
)
from hubbleAI.service import (
    load_latest_forward_forecast,
    get_last_run_by_mode,
    validate_forward_predictions,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cash Flows - Hubble.AI",
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

render_sidebar(active_page="Cash Flows", ref_info=ref_info)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def format_val(val: float, in_millions: bool = True) -> str:
    """Format value for display."""
    if pd.isna(val):
        return "-"
    if in_millions:
        return f"{val:.2f}M"
    return f"{val/1e6:.2f}M"


def create_horizon_table(df: pd.DataFrame, lg: str) -> pd.DataFrame:
    """Create horizon summary table for a liquidity group."""
    if df.empty:
        return pd.DataFrame()

    if lg == "NET":
        # Sum across both LGs
        lg_df = df
    else:
        lg_df = df[df["liquidity_group"] == lg]

    if lg_df.empty:
        return pd.DataFrame()

    rows = []
    for h in range(1, 9):
        h_data = lg_df[lg_df["horizon"] == h]
        if h_data.empty:
            continue

        target_week = h_data["target_week_start"].iloc[0] if "target_week_start" in h_data.columns else None
        if pd.notna(target_week):
            if isinstance(target_week, str):
                target_week = target_week[:10]
            else:
                target_week = str(target_week)[:10]
        else:
            target_week = "-"

        row = {
            "Horizon": f"H{h}",
            "Target Week": target_week,
            "Prediction (M)": h_data["y_pred_point"].sum() / 1e6,
        }

        # Quantiles - check if values are all NaN before summing
        if "y_pred_p10" in h_data.columns and h_data["y_pred_p10"].notna().any():
            row["P10 (M)"] = h_data["y_pred_p10"].sum() / 1e6
        else:
            row["P10 (M)"] = None
        if "y_pred_p50" in h_data.columns and h_data["y_pred_p50"].notna().any():
            row["P50 (M)"] = h_data["y_pred_p50"].sum() / 1e6
        else:
            row["P50 (M)"] = None
        if "y_pred_p90" in h_data.columns and h_data["y_pred_p90"].notna().any():
            row["P90 (M)"] = h_data["y_pred_p90"].sum() / 1e6
        else:
            row["P90 (M)"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def create_forecast_chart(table_df: pd.DataFrame, title: str, color: str = "#2E7D32") -> go.Figure:
    """Create a bar chart with P10/P90 range shown as error bars."""
    if table_df.empty:
        return go.Figure()

    fig = go.Figure()

    predictions = table_df["Prediction (M)"].values
    horizons = table_df["Horizon"].tolist()

    # Contrasting blue color for P10/P90 range (Nogaro blue metallic)
    range_color = "#2C5AA0"

    # Check if we have P10/P90 data
    has_quantiles = ("P10 (M)" in table_df.columns and "P90 (M)" in table_df.columns
                     and table_df["P10 (M)"].notna().any() and table_df["P90 (M)"].notna().any())

    if has_quantiles:
        p10_raw = table_df["P10 (M)"].values
        p90_raw = table_df["P90 (M)"].values
        p10_vals = np.where(np.isnan(p10_raw), predictions, p10_raw)
        p90_vals = np.where(np.isnan(p90_raw), predictions, p90_raw)

        # Calculate error bar distances
        error_minus = predictions - p10_vals  # Distance down to P10
        error_plus = p90_vals - predictions   # Distance up to P90

        # ML Prediction bars with error bars showing P10/P90 range
        fig.add_trace(go.Bar(
            x=horizons,
            y=predictions,
            name="Prediction",
            marker_color=color,
            text=[f"{v:.1f}M" for v in predictions],
            textposition="outside",
            error_y=dict(
                type="data",
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                color=range_color,
                thickness=2,
                width=8,
            ),
            hovertemplate="<b>%{x}</b><br>Prediction: %{y:.2f}M<br>P10: %{customdata[0]:.2f}M<br>P90: %{customdata[1]:.2f}M<extra></extra>",
            customdata=list(zip(p10_vals, p90_vals))
        ))
    else:
        # ML Prediction bars without error bars
        fig.add_trace(go.Bar(
            x=horizons,
            y=predictions,
            name="Prediction",
            marker_color=color,
            text=[f"{v:.1f}M" for v in predictions],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Prediction: %{y:.2f}M EUR<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Horizon",
        yaxis_title="EUR (Millions)",
        height=340,
        margin=dict(t=40, b=40, l=50, r=20),
        showlegend=False,
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
        Cash Flows
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        8-week cashflow predictions with confidence intervals
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

forecast_view = load_latest_forward_forecast()

# ---------------------------------------------------------------------------
# 8-Week Forecast Section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">8-Week Forecast by Liquidity Group</div>', unsafe_allow_html=True)

if forecast_view is None:
    st.warning("No forecast data available.")
    st.info("Go to **Admin** page to run a forward forecast or upload new data files.")
    st.stop()

df = forecast_view.forecasts_df.copy()

# Validate predictions
validation = validate_forward_predictions(df)
if not validation["ok"]:
    # Check if the issue is about missing quantile predictions
    has_quantile_issue = any("y_pred_p10" in issue or "y_pred_p50" in issue or "y_pred_p90" in issue for issue in validation["issues"])
    if has_quantile_issue:
        st.warning("**Note:** Confidence intervals (P10/P50/P90) are not available for this forecast. Only point estimates are shown.")

# Check required columns
required_cols = ["horizon", "liquidity_group", "y_pred_point"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ---------------------------------------------------------------------------
# NET Summary Score Cards (computed early for display)
# ---------------------------------------------------------------------------

# Compute NET totals for score cards
net_summary_table = create_horizon_table(df, "NET")
if not net_summary_table.empty:
    total_prediction = net_summary_table["Prediction (M)"].sum()
    total_p10 = net_summary_table["P10 (M)"].sum() if "P10 (M)" in net_summary_table.columns and net_summary_table["P10 (M)"].notna().any() else None
    total_p90 = net_summary_table["P90 (M)"].sum() if "P90 (M)" in net_summary_table.columns and net_summary_table["P90 (M)"].notna().any() else None

    # Determine risk level with justification
    if total_prediction < 0:
        risk_level = "High"
        risk_color = "#D32F2F"  # Red
        risk_reason = "Deficit expected"
    elif total_p10 is not None and total_p10 < 0:
        risk_level = "Moderate"
        risk_color = "#F57C00"  # Orange
        risk_reason = "Downside risk to deficit"
    else:
        risk_level = "Low"
        risk_color = "#2E7D32"  # Green
        risk_reason = "Surplus with limited downside"

    # CSS for score card icons
    icon_style = '''
    <style>
    .score-icon {
        width: 40px; height: 40px; border-radius: 10px; display: flex;
        align-items: center; justify-content: center; margin: 0 auto 0.5rem auto;
    }
    .score-icon svg { stroke: white; }
    .score-icon.green { background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); }
    .score-icon.red { background: linear-gradient(135deg, #C62828 0%, #EF5350 100%); }
    .score-icon.blue { background: linear-gradient(135deg, #1565C0 0%, #42A5F5 100%); }
    .score-icon.orange { background: linear-gradient(135deg, #E65100 0%, #FF9800 100%); }
    </style>
    '''
    st.markdown(icon_style, unsafe_allow_html=True)

    # Score cards row
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        pred_color = "#2E7D32" if total_prediction >= 0 else "#D32F2F"
        pred_label = "Surplus" if total_prediction >= 0 else "Deficit"
        icon_class = "green" if total_prediction >= 0 else "red"
        st.markdown(f'''<div class="hubble-card" style="text-align: center; padding: 1rem;">
            <div class="score-icon {icon_class}">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 7.5C16.5 5.5 14.5 4 12 4c-4 0-7 3.5-7 8s3 8 7 8c2.5 0 4.5-1.5 5.5-3.5"/><line x1="3" y1="10" x2="13" y2="10"/><line x1="3" y1="14" x2="13" y2="14"/></svg>
            </div>
            <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">NET Outlook</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {pred_color};">{total_prediction:.0f}M</div>
            <div style="font-size: 0.75rem; color: {pred_color};">{pred_label}</div>
        </div>''', unsafe_allow_html=True)

    with card_col2:
        if total_p10 is not None:
            p10_color = "#D32F2F" if total_p10 < 0 else "#2E7D32"
            st.markdown(f'''<div class="hubble-card" style="text-align: center; padding: 1rem;">
                <div class="score-icon red">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M21 9l-9 9-4-4-5 5"/></svg>
                </div>
                <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">Downside (P10)</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: {p10_color};">{total_p10:.0f}M</div>
                <div style="font-size: 0.75rem; color: #5A6169;">Conservative</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="hubble-card" style="text-align: center; padding: 1rem;">
                <div class="score-icon red">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M21 9l-9 9-4-4-5 5"/></svg>
                </div>
                <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">Downside (P10)</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #9E9E9E;">N/A</div>
                <div style="font-size: 0.75rem; color: #5A6169;">Not available</div>
            </div>''', unsafe_allow_html=True)

    with card_col3:
        if total_p90 is not None:
            st.markdown(f'''<div class="hubble-card" style="text-align: center; padding: 1rem;">
                <div class="score-icon green">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M21 15l-9-9-4 4-5-5"/></svg>
                </div>
                <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">Upside (P90)</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #2E7D32;">{total_p90:.0f}M</div>
                <div style="font-size: 0.75rem; color: #5A6169;">Optimistic</div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('''<div class="hubble-card" style="text-align: center; padding: 1rem;">
                <div class="score-icon green">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M21 15l-9-9-4 4-5-5"/></svg>
                </div>
                <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">Upside (P90)</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #9E9E9E;">N/A</div>
                <div style="font-size: 0.75rem; color: #5A6169;">Not available</div>
            </div>''', unsafe_allow_html=True)

    with card_col4:
        icon_class = "green" if risk_level == "Low" else ("orange" if risk_level == "Moderate" else "red")
        st.markdown(f'''<div class="hubble-card" style="text-align: center; padding: 1rem;">
            <div class="score-icon {icon_class}">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div style="font-size: 0.7rem; color: #5A6169; text-transform: uppercase; letter-spacing: 0.5px;">Risk Level</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {risk_color};">{risk_level}</div>
            <div style="font-size: 0.75rem; color: {risk_color};">{risk_reason}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# Section 1: Treasury Definitions
st.markdown(render_interpretation_box(
    title="How to Interpret These Forecasts",
    content="""
    <ul style="margin: 0; padding-left: 1.25rem;">
        <li><strong>TRR (Treasury Receipts):</strong> Expected cash inflows from operations</li>
        <li><strong>TRP (Treasury Payments):</strong> Expected cash outflows for payments</li>
        <li><strong>NET:</strong> Net cash position (TRR + TRP). Positive = surplus, Negative = deficit</li>
    </ul>
    """
), unsafe_allow_html=True)

# Section 2: Understanding the Numbers
st.markdown(render_interpretation_box(
    title="Understanding the Numbers",
    content="""
    <ul style="margin: 0; padding-left: 1.25rem;">
        <li><strong>Prediction</strong> — Our best estimate (most likely outcome)</li>
        <li><strong>P10</strong> — "Bad case" scenario (only 10% chance of being worse)</li>
        <li><strong>P90</strong> — "Good case" scenario (only 10% chance of being better)</li>
        <li><strong>Spread (P90 − P10)</strong> — Wide = high uncertainty, Narrow = high confidence</li>
    </ul>
    <p style="margin-top: 0.5rem;"><strong>Quick Guide:</strong> If Prediction is positive but P10 is negative,
    you expect a surplus but have downside risk. Keep liquidity buffers while planning for the surplus.</p>
    """
), unsafe_allow_html=True)

# Create tabs for TRR, TRP, NET
tab_trr, tab_trp, tab_net = st.tabs(["TRR (Inflows)", "TRP (Outflows)", "NET (Position)"])

with tab_trr:
    st.markdown("**TRR: Treasury Receipts (Cash Inflows)**")
    trr_table = create_horizon_table(df, "TRR")

    if not trr_table.empty:
        col_table, col_chart = st.columns([1, 1.5])

        with col_table:
            # Format for display
            display_df = trr_table.copy()
            for col in display_df.columns:
                if "(M)" in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=340,
            )

        with col_chart:
            fig = create_forecast_chart(trr_table, "TRR Predictions with P10/P90 Range", "#2E7D32")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No TRR data available")

with tab_trp:
    st.markdown("**TRP: Treasury Payments (Cash Outflows)**")
    trp_table = create_horizon_table(df, "TRP")

    if not trp_table.empty:
        col_table, col_chart = st.columns([1, 1.5])

        with col_table:
            display_df = trp_table.copy()
            for col in display_df.columns:
                if "(M)" in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=340,
            )

        with col_chart:
            fig = create_forecast_chart(trp_table, "TRP Predictions with P10/P90 Range", "#D32F2F")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No TRP data available")

with tab_net:
    st.markdown("**NET: Net Cash Position (TRR + TRP)**")
    net_table = create_horizon_table(df, "NET")

    if not net_table.empty:
        col_table, col_chart = st.columns([1, 1.5])

        with col_table:
            display_df = net_table.copy()
            for col in display_df.columns:
                if "(M)" in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=340,
            )

        with col_chart:
            # NET chart with colors for positive/negative and P10/P90 error bars
            predictions = net_table["Prediction (M)"].values
            horizons = net_table["Horizon"].tolist()
            colors = ["#2E7D32" if v >= 0 else "#D32F2F" for v in predictions]
            range_color = "#2C5AA0"  # Nogaro blue metallic for error bars
            fig = go.Figure()

            # Check if we have P10/P90 data
            has_quantiles = ("P10 (M)" in net_table.columns and "P90 (M)" in net_table.columns
                             and net_table["P10 (M)"].notna().any() and net_table["P90 (M)"].notna().any())

            if has_quantiles:
                p10_raw = net_table["P10 (M)"].values
                p90_raw = net_table["P90 (M)"].values
                p10_vals = np.where(np.isnan(p10_raw), predictions, p10_raw)
                p90_vals = np.where(np.isnan(p90_raw), predictions, p90_raw)

                # Calculate error bar distances
                error_minus = predictions - p10_vals  # Distance down to P10
                error_plus = p90_vals - predictions   # Distance up to P90

                # NET Prediction bars with error bars showing P10/P90 range
                fig.add_trace(go.Bar(
                    x=horizons,
                    y=predictions,
                    name="Prediction",
                    marker_color=colors,
                    text=[f"{v:.1f}M" for v in predictions],
                    textposition="outside",
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=error_plus,
                        arrayminus=error_minus,
                        color=range_color,
                        thickness=2,
                        width=8,
                    ),
                    hovertemplate="<b>%{x}</b><br>NET: %{y:.2f}M<br>P10: %{customdata[0]:.2f}M<br>P90: %{customdata[1]:.2f}M<extra></extra>",
                    customdata=list(zip(p10_vals, p90_vals))
                ))
            else:
                # NET Prediction bars without error bars
                fig.add_trace(go.Bar(
                    x=horizons,
                    y=predictions,
                    name="Prediction",
                    marker_color=colors,
                    text=[f"{v:.1f}M" for v in predictions],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>NET: %{y:.2f}M EUR<extra></extra>"
                ))

            fig.update_layout(
                title=dict(text="NET Position by Horizon", font=dict(size=14)),
                xaxis_title="Horizon",
                yaxis_title="EUR (Millions)",
                height=340,
                margin=dict(t=40, b=40, l=50, r=20),
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)", zeroline=True, zerolinecolor="rgba(0,0,0,0.2)")

            st.plotly_chart(fig, use_container_width=True)

        # Section 3: Dynamic Outlook Summary
        total_prediction = net_table["Prediction (M)"].sum()
        total_p10 = net_table["P10 (M)"].sum() if "P10 (M)" in net_table.columns and net_table["P10 (M)"].notna().any() else None
        total_p90 = net_table["P90 (M)"].sum() if "P90 (M)" in net_table.columns and net_table["P90 (M)"].notna().any() else None

        # Determine outlook type and recommendation
        if total_prediction >= 0:
            outlook_word = "surplus"
            outlook_color = "#2E7D32"
        else:
            outlook_word = "deficit"
            outlook_color = "#D32F2F"

        # Build dynamic recommendation
        if total_p10 is not None and total_p10 < 0 and total_prediction > 0:
            recommendation = f"Maintain liquidity buffer of ~{abs(total_p10):.0f}M while planning for the expected surplus."
        elif total_prediction < 0:
            recommendation = "Consider arranging short-term financing to cover the expected deficit."
        else:
            recommendation = "Position looks healthy. Continue monitoring weekly forecasts."

        # Build the summary HTML
        summary_lines = [f'<p><strong>Expected 8-week {outlook_word}:</strong> <span style="color: {outlook_color}; font-weight: 600;">{total_prediction:.2f}M EUR</span></p>']

        if total_p10 is not None:
            p10_color = "#D32F2F" if total_p10 < 0 else "#2E7D32"
            p10_label = "potential deficit" if total_p10 < 0 else "surplus"
            summary_lines.append(f'<p><strong>Downside risk (P10):</strong> <span style="color: {p10_color};">{total_p10:.2f}M EUR</span> {p10_label}</p>')

        if total_p90 is not None:
            summary_lines.append(f'<p><strong>Upside potential (P90):</strong> <span style="color: #2E7D32;">{total_p90:.2f}M EUR</span> surplus</p>')

        summary_lines.append(f'<p style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(0,0,0,0.1);"><strong>→ Recommendation:</strong> {recommendation}</p>')

        st.markdown(f"""
        <div class="interpretation-box">
            <h4>Your Outlook Summary</h4>
            {''.join(summary_lines)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No NET data available")

st.markdown("---")

# ---------------------------------------------------------------------------
# Detailed Data View
# ---------------------------------------------------------------------------

with st.expander("View Detailed Forecast Data", expanded=False):
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        lg_options = ["All"] + sorted(df["liquidity_group"].unique().tolist())
        selected_lg = st.selectbox("Liquidity Group", lg_options, key="detail_lg")

    with filter_col2:
        horizon_options = ["All"] + [f"H{h}" for h in sorted(df["horizon"].unique())]
        selected_horizon = st.selectbox("Horizon", horizon_options, key="detail_h")

    with filter_col3:
        entity_list = sorted(df["entity"].unique().tolist()) if "entity" in df.columns else []
        entity_options = ["All"] + entity_list[:50]
        selected_entity = st.selectbox("Entity", entity_options, key="detail_entity")

    # Apply filters
    filtered_df = df.copy()
    if selected_lg != "All":
        filtered_df = filtered_df[filtered_df["liquidity_group"] == selected_lg]
    if selected_horizon != "All":
        h = int(selected_horizon.replace("H", ""))
        filtered_df = filtered_df[filtered_df["horizon"] == h]
    if selected_entity != "All":
        filtered_df = filtered_df[filtered_df["entity"] == selected_entity]

    # Display columns
    display_cols = [
        "entity", "liquidity_group", "horizon", "week_start", "target_week_start",
        "y_pred_point", "y_pred_p10", "y_pred_p50", "y_pred_p90",
        "model_type", "is_pass_through"
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    # Format data for display
    display_filtered = filtered_df[display_cols].copy()

    # Format numeric columns
    for col in ["y_pred_point", "y_pred_p10", "y_pred_p50", "y_pred_p90"]:
        if col in display_filtered.columns:
            display_filtered[col] = display_filtered[col].apply(
                lambda x: f"{x/1e6:.2f}M" if pd.notna(x) else "-"
            )

    # Format datetime columns - show only date part
    for col in ["week_start", "target_week_start"]:
        if col in display_filtered.columns:
            display_filtered[col] = display_filtered[col].apply(
                lambda x: str(x)[:10] if pd.notna(x) else "-"
            )

    # Format is_pass_through as Yes/No
    if "is_pass_through" in display_filtered.columns:
        display_filtered["is_pass_through"] = display_filtered["is_pass_through"].apply(
            lambda x: "Yes" if x else "No" if pd.notna(x) else "-"
        )

    # Rename columns to human-readable names
    column_renames = {
        "entity": "Entity",
        "liquidity_group": "Group",
        "horizon": "Horizon",
        "week_start": "Week Start",
        "target_week_start": "Target Week",
        "y_pred_point": "ML Prediction",
        "y_pred_p10": "P10 (Low)",
        "y_pred_p50": "P50 (Median)",
        "y_pred_p90": "P90 (High)",
        "model_type": "Model",
        "is_pass_through": "LP Input"
    }
    display_filtered = display_filtered.rename(columns=column_renames)

    st.dataframe(
        display_filtered.head(500),
        use_container_width=True,
        height=400,
        hide_index=True,
    )
    st.caption(f"Showing {min(500, len(filtered_df))} of {len(filtered_df)} rows. Values in millions EUR.")

# Footer
st.markdown("---")
if forecast_view.extra:
    st.caption(f"Data source: {forecast_view.extra.get('source', 'unknown')} | Hubble.AI v{APP_VERSION}")
else:
    st.caption(f"Hubble.AI v{APP_VERSION}")
