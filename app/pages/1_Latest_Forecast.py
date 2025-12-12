"""
Page 1 - Latest Forecast (Operations View)

Shows:
- Data health status
- Latest forward run status
- 8-week forecast horizon with P10/P50/P90
- Separate views for TRR, TRP, and NET
- Treasury interpretation guidance

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
    render_metric_card,
    render_interpretation_box,
    format_millions,
    format_currency_millions,
    APP_VERSION,
)
from hubbleAI.service import (
    get_data_health_summary,
    load_latest_forward_forecast,
    get_last_run_by_mode,
    validate_forward_predictions,
    prepare_forecast_views,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Latest Forecast - HubbleAI",
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

render_sidebar(active_page="Latest Forecast", ref_info=ref_info)

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
            "Point (M)": h_data["y_pred_point"].sum() / 1e6,
        }

        # Quantiles
        if "y_pred_p10" in h_data.columns:
            p10_sum = h_data["y_pred_p10"].sum()
            row["P10 (M)"] = p10_sum / 1e6 if pd.notna(p10_sum) else None
        if "y_pred_p50" in h_data.columns:
            p50_sum = h_data["y_pred_p50"].sum()
            row["P50 (M)"] = p50_sum / 1e6 if pd.notna(p50_sum) else None
        if "y_pred_p90" in h_data.columns:
            p90_sum = h_data["y_pred_p90"].sum()
            row["P90 (M)"] = p90_sum / 1e6 if pd.notna(p90_sum) else None

        rows.append(row)

    return pd.DataFrame(rows)


def create_forecast_chart(table_df: pd.DataFrame, title: str, color: str = "#2E7D32") -> go.Figure:
    """Create a bar chart with error bars for forecast visualization."""
    if table_df.empty:
        return go.Figure()

    fig = go.Figure()

    # Point forecast bars
    fig.add_trace(go.Bar(
        x=table_df["Horizon"],
        y=table_df["Point (M)"],
        name="Point Forecast",
        marker_color=color,
        text=[f"{v:.1f}M" for v in table_df["Point (M)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Point: %{y:.2f}M EUR<extra></extra>"
    ))

    # Add P10/P90 range if available
    if "P10 (M)" in table_df.columns and "P90 (M)" in table_df.columns:
        p10_vals = table_df["P10 (M)"].fillna(table_df["Point (M)"])
        p90_vals = table_df["P90 (M)"].fillna(table_df["Point (M)"])

        fig.add_trace(go.Scatter(
            x=table_df["Horizon"],
            y=p90_vals,
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color=color, opacity=0.5),
            name="P90 (High)",
            hovertemplate="P90: %{y:.2f}M EUR<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=table_df["Horizon"],
            y=p10_vals,
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color=color, opacity=0.5),
            name="P10 (Low)",
            hovertemplate="P10: %{y:.2f}M EUR<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Horizon",
        yaxis_title="EUR (Millions)",
        height=350,
        margin=dict(t=50, b=50, l=60, r=20),
        showlegend=True,
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
        Latest Forecast
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        8-week cashflow predictions with confidence intervals
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

health = get_data_health_summary()
forecast_view = load_latest_forward_forecast()

# ---------------------------------------------------------------------------
# Top KPI Cards
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    if forward_status:
        status_val = forward_status.get("status", "unknown").upper()
        status_date = forward_status.get("ref_week_start", "-")
        if isinstance(status_date, str) and len(status_date) > 10:
            status_date = status_date[:10]
    else:
        status_val = "NO RUNS"
        status_date = "-"

    st.markdown(render_metric_card(
        header="Run Status",
        value=status_val,
        subtitle=f"Ref: {status_date}",
        accent=status_val == "SUCCESS"
    ), unsafe_allow_html=True)

with col2:
    ready_text = "READY" if health["is_ready"] else "INCOMPLETE"
    missing_count = len(health.get("missing_inputs", []))
    subtitle = "All files present" if health["is_ready"] else f"{missing_count} file(s) missing"

    st.markdown(render_metric_card(
        header="Data Health",
        value=ready_text,
        subtitle=subtitle,
        accent=health["is_ready"]
    ), unsafe_allow_html=True)

with col3:
    if forecast_view:
        n_rows = len(forecast_view.forecasts_df)
        n_entities = forecast_view.forecasts_df["entity"].nunique() if "entity" in forecast_view.forecasts_df.columns else 0
    else:
        n_rows = 0
        n_entities = 0

    st.markdown(render_metric_card(
        header="Forecast Rows",
        value=f"{n_rows:,}",
        subtitle=f"{n_entities} entities"
    ), unsafe_allow_html=True)

with col4:
    st.markdown(render_metric_card(
        header="Schedule",
        value="WEEKLY",
        subtitle="Every Tuesday"
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Actions & Details Row
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

    if st.button("Run Forward Forecast", type="primary", use_container_width=True):
        try:
            from hubbleAI.pipeline import run_forecast
            with st.spinner("Running forecast... this may take a few minutes"):
                result = run_forecast(mode="forward", trigger_source="manual")
            if result.status == "success":
                st.success("Forecast completed!")
            else:
                st.error(f"Failed: {result.message}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Run Backtest", use_container_width=True):
        try:
            from hubbleAI.pipeline import run_forecast
            with st.spinner("Running backtest..."):
                result = run_forecast(mode="backtest", trigger_source="manual")
            if result.status == "success":
                st.success("Backtest completed!")
            else:
                st.error(f"Failed: {result.message}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.caption("Production runs are automated weekly.")

    # Data health details
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Health</div>', unsafe_allow_html=True)

    details = health.get("details", {})
    for name, info in details.items():
        if info.get("exists"):
            age = info.get("age_days", "?")
            status_icon = "✓" if info.get("healthy") else "!"
            status_color = "#2E7D32" if info.get("healthy") else "#F57C00"
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(0,0,0,0.06);">
                <span style="color: {status_color}; font-weight: 600; width: 16px;">{status_icon}</span>
                <span style="flex: 1; font-weight: 500;">{name.upper()}</span>
                <span style="color: #8B95A1; font-size: 0.85rem;">{age}d old</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(0,0,0,0.06);">
                <span style="color: #D32F2F; font-weight: 600; width: 16px;">✗</span>
                <span style="flex: 1; font-weight: 500;">{name.upper()}</span>
                <span style="color: #8B95A1; font-size: 0.85rem;">Missing</span>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">Last Run Details</div>', unsafe_allow_html=True)

    if forward_status:
        st.markdown('<div class="hubble-card hubble-card-sm">', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            run_id = forward_status.get('run_id', 'N/A')
            if len(str(run_id)) > 25:
                run_id = str(run_id)[:25] + "..."
            st.markdown(f"**Run ID:** `{run_id}`")
            st.markdown(f"**Mode:** {forward_status.get('mode', 'N/A')}")
            st.markdown(f"**Trigger:** {forward_status.get('trigger_source', 'N/A')}")
        with d2:
            created = forward_status.get('created_at', 'N/A')
            if isinstance(created, str) and len(created) > 19:
                created = created[:19].replace('T', ' ')
            st.markdown(f"**Status:** {forward_status.get('status', 'N/A').upper()}")
            st.markdown(f"**As-of Date:** {forward_status.get('as_of_date', 'N/A')}")
            st.markdown(f"**Created:** {created}")
        if forward_status.get("message"):
            st.markdown(f"**Message:** {forward_status.get('message')}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No forward forecast run found yet. Click 'Run Forward Forecast' to generate one.")

st.markdown("---")

# ---------------------------------------------------------------------------
# 8-Week Forecast Section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">8-Week Forecast by Liquidity Group</div>', unsafe_allow_html=True)

if forecast_view is None:
    st.warning("No forecast data available. Run a forward forecast first.")
    st.stop()

df = forecast_view.forecasts_df.copy()

# Validate predictions
validation = validate_forward_predictions(df)
if not validation["ok"]:
    st.warning("Forecast data has some issues:")
    for issue in validation["issues"]:
        st.markdown(f"- {issue}")

# Check required columns
required_cols = ["horizon", "liquidity_group", "y_pred_point"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# Treasury Interpretation Box
st.markdown(render_interpretation_box(
    title="How to Interpret These Forecasts",
    content="""
    <ul style="margin: 0; padding-left: 1.25rem;">
        <li><strong>TRR (Treasury Receipts):</strong> Expected cash inflows from operations</li>
        <li><strong>TRP (Treasury Payments):</strong> Expected cash outflows for payments</li>
        <li><strong>NET:</strong> Net cash position (TRR + TRP). Positive = surplus, Negative = deficit</li>
        <li><strong>P10/P90:</strong> 80% confidence interval. Actual is expected to fall between these values 80% of the time</li>
    </ul>
    <p style="margin-top: 0.5rem;"><strong>Action:</strong> If NET P10 is negative for upcoming weeks, consider arranging short-term financing.
    If NET P90 is significantly positive, consider short-term investments.</p>
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
            fig = create_forecast_chart(trr_table, "TRR Point Forecasts with P10/P90 Range", "#2E7D32")
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
            fig = create_forecast_chart(trp_table, "TRP Point Forecasts with P10/P90 Range", "#D32F2F")
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
            # NET chart with colors for positive/negative
            colors = ["#2E7D32" if v >= 0 else "#D32F2F" for v in net_table["Point (M)"]]
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=net_table["Horizon"],
                y=net_table["Point (M)"],
                marker_color=colors,
                text=[f"{v:.1f}M" for v in net_table["Point (M)"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>NET: %{y:.2f}M EUR<extra></extra>"
            ))

            fig.update_layout(
                title=dict(text="NET Position by Horizon", font=dict(size=14)),
                xaxis_title="Horizon",
                yaxis_title="EUR (Millions)",
                height=350,
                margin=dict(t=50, b=50, l=60, r=20),
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.05)", zeroline=True, zerolinecolor="rgba(0,0,0,0.2)")

            st.plotly_chart(fig, use_container_width=True)

        # NET interpretation summary
        total_net = net_table["Point (M)"].sum()
        direction = "surplus" if total_net > 0 else "deficit"
        color = "#2E7D32" if total_net > 0 else "#D32F2F"

        st.markdown(f"""
        <div class="interpretation-box">
            <h4>NET Position Summary</h4>
            <p>Total projected NET over 8 weeks: <strong style="color: {color};">{total_net:.2f}M EUR</strong> ({direction})</p>
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

    # Format numeric columns for display
    display_filtered = filtered_df[display_cols].copy()
    for col in ["y_pred_point", "y_pred_p10", "y_pred_p50", "y_pred_p90"]:
        if col in display_filtered.columns:
            display_filtered[col] = display_filtered[col].apply(
                lambda x: f"{x/1e6:.2f}M" if pd.notna(x) else "-"
            )

    st.dataframe(
        display_filtered.head(500),
        use_container_width=True,
        height=400,
    )
    st.caption(f"Showing {min(500, len(filtered_df))} of {len(filtered_df)} rows. Values in millions EUR.")

# Footer
st.markdown("---")
if forecast_view.extra:
    st.caption(f"Data source: {forecast_view.extra.get('source', 'unknown')} | HubbleAI v{APP_VERSION}")
else:
    st.caption(f"HubbleAI v{APP_VERSION}")
