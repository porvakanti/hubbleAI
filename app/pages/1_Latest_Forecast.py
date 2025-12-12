"""
Page 1 - Latest Forecast (Operations View)

Shows:
- Data health status
- Latest forward run status
- 8-week forecast horizon with P10/P50/P90
- Separate views for TRR, TRP, and NET
- Treasury interpretation guidance

Design: Warm cream palette, card-based layout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.service import (
    get_data_health_summary,
    load_latest_forward_forecast,
    get_last_run_by_mode,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Latest Forecast - HubbleAI",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Page CSS (extends global CSS)
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

.card-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Accent cards */
.card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
}
.card-accent .card-header, .card-accent .card-value, .card-accent .card-subtitle { color: white; }
.card-accent .card-header { opacity: 0.9; }

/* Status */
.status-success { color: var(--accent-green); font-weight: 600; }
.status-warning { color: var(--accent-orange); font-weight: 600; }
.status-error { color: var(--accent-red); font-weight: 600; }
.text-muted { color: var(--text-muted); }

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

/* Health indicators */
.health-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-light);
}
.health-item:last-child { border-bottom: none; }

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
.interpretation-box p, .interpretation-box ul {
    color: var(--text-dark);
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.interpretation-box ul { padding-left: 1.25rem; margin-top: 0.5rem; }
.interpretation-box li { margin-bottom: 0.25rem; }

/* Forecast table styling */
.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.forecast-table th {
    background: var(--bg-cream-light);
    padding: 0.75rem;
    text-align: right;
    font-weight: 600;
    color: var(--text-muted);
    border-bottom: 2px solid var(--border-light);
}
.forecast-table th:first-child { text-align: left; }
.forecast-table td {
    padding: 0.75rem;
    text-align: right;
    border-bottom: 1px solid var(--border-light);
}
.forecast-table td:first-child { text-align: left; font-weight: 500; }
.forecast-table tr:hover { background: var(--bg-cream-light); }

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


def format_millions(val: float) -> str:
    """Format value in millions with 2 decimals."""
    if pd.isna(val):
        return "-"
    return f"{val/1e6:.2f}M"


def format_currency(val: float) -> str:
    """Format value as currency in millions."""
    if pd.isna(val):
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{val/1e6:.2f}M EUR"


def render_card(header: str, value: str, subtitle: str = "", accent: bool = False) -> str:
    """Render a metric card."""
    card_class = "card card-accent" if accent else "card"
    return f"""
    <div class="{card_class}" style="height: 100%;">
        <div class="card-header">{header}</div>
        <div class="card-value">{value}</div>
        <div class="card-subtitle">{subtitle}</div>
    </div>
    """


def main():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <h1>Latest Forecast</h1>
    <p class="page-subtitle">8-week cashflow predictions with confidence intervals</p>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------------------------

    health = get_data_health_summary()
    forward_status = get_last_run_by_mode("forward")
    forecast_view = load_latest_forward_forecast()

    # ---------------------------------------------------------------------------
    # Top KPI Cards Row
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

        st.markdown(render_card(
            header="Run Status",
            value=status_val,
            subtitle=f"Ref: {status_date}"
        ), unsafe_allow_html=True)

    with col2:
        ready_text = "READY" if health["is_ready"] else "INCOMPLETE"
        missing_count = len(health.get("missing_inputs", []))
        subtitle = "All files present" if health["is_ready"] else f"{missing_count} file(s) missing"

        st.markdown(render_card(
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

        st.markdown(render_card(
            header="Forecast Rows",
            value=f"{n_rows:,}",
            subtitle=f"{n_entities} entities"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(render_card(
            header="Schedule",
            value="WEEKLY",
            subtitle="Every Tuesday"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Two-column layout: Actions + Data Health
    # ---------------------------------------------------------------------------

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

        if st.button("Run Forward Forecast", type="primary", use_container_width=True):
            st.info("Triggering forecast run...")
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
                status = "OK" if info.get("healthy") else "!"
                color = "var(--accent-green)" if info.get("healthy") else "var(--accent-orange)"
                st.markdown(f"""
                <div class="health-item">
                    <span style="color: {color}; font-weight: 600;">{status}</span>
                    <span style="flex: 1;">{name.upper()}</span>
                    <span style="color: var(--text-muted); font-size: 0.85rem;">{age}d old</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="health-item">
                    <span style="color: var(--accent-red); font-weight: 600;">X</span>
                    <span style="flex: 1;">{name.upper()}</span>
                    <span style="color: var(--text-muted); font-size: 0.85rem;">Missing</span>
                </div>
                """, unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-title">Last Run Details</div>', unsafe_allow_html=True)

        if forward_status:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                run_id = forward_status.get('run_id', 'N/A')
                if len(run_id) > 25:
                    run_id = run_id[:25] + "..."
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
        return

    df = forecast_view.forecasts_df.copy()

    # Ensure required columns exist
    required_cols = ["horizon", "liquidity_group", "y_pred_point"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return

    # Treasury Interpretation Box
    st.markdown("""
    <div class="interpretation-box">
        <h4>How to Interpret These Forecasts</h4>
        <p>These 8-week predictions help Treasury plan liquidity positions:</p>
        <ul>
            <li><strong>TRR (Treasury Receipts):</strong> Expected cash inflows from operations</li>
            <li><strong>TRP (Treasury Payments):</strong> Expected cash outflows for payments</li>
            <li><strong>NET:</strong> Net cash position (TRR + TRP). Positive = surplus, Negative = deficit</li>
            <li><strong>P10/P90:</strong> 80% confidence interval. Actual is expected to fall between these values 80% of the time</li>
        </ul>
        <p><strong>Action:</strong> If NET P10 is negative for upcoming weeks, consider arranging short-term financing.
        If NET P90 is significantly positive, consider short-term investments.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for TRR, TRP, NET
    tab_trr, tab_trp, tab_net = st.tabs(["TRR (Inflows)", "TRP (Outflows)", "NET (Position)"])

    def create_horizon_summary(df_subset: pd.DataFrame, show_lp: bool = False) -> pd.DataFrame:
        """Create horizon summary with point estimate and intervals."""
        summary = []
        for h in range(1, 9):
            h_data = df_subset[df_subset["horizon"] == h]
            if h_data.empty:
                continue

            row = {
                "Horizon": f"H{h}",
                "Target Week": "",
                "Point Forecast": h_data["y_pred_point"].sum(),
            }

            # Get target week if available
            if "target_week_start" in h_data.columns:
                target = h_data["target_week_start"].iloc[0]
                if pd.notna(target):
                    if isinstance(target, str):
                        row["Target Week"] = target[:10]
                    else:
                        row["Target Week"] = str(target)[:10]

            # Quantile predictions
            if "y_pred_p10" in h_data.columns:
                row["P10 (Low)"] = h_data["y_pred_p10"].sum()
            if "y_pred_p50" in h_data.columns:
                row["P50 (Median)"] = h_data["y_pred_p50"].sum()
            if "y_pred_p90" in h_data.columns:
                row["P90 (High)"] = h_data["y_pred_p90"].sum()

            summary.append(row)

        return pd.DataFrame(summary)

    def display_forecast_table(summary_df: pd.DataFrame, lg_name: str):
        """Display formatted forecast table."""
        if summary_df.empty:
            st.info(f"No data available for {lg_name}")
            return

        # Format for display
        display_df = summary_df.copy()
        for col in display_df.columns:
            if col not in ["Horizon", "Target Week"]:
                display_df[col] = display_df[col].apply(format_millions)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Horizon": st.column_config.TextColumn("Horizon", width="small"),
                "Target Week": st.column_config.TextColumn("Target Week", width="medium"),
                "Point Forecast": st.column_config.TextColumn("Point (EUR)", width="medium"),
                "P10 (Low)": st.column_config.TextColumn("P10 Low", width="medium"),
                "P50 (Median)": st.column_config.TextColumn("P50 Median", width="medium"),
                "P90 (High)": st.column_config.TextColumn("P90 High", width="medium"),
            }
        )

        # Simple bar chart of point forecasts
        if len(summary_df) > 0 and "Point Forecast" in summary_df.columns:
            chart_data = summary_df[["Horizon", "Point Forecast"]].copy()
            chart_data = chart_data.set_index("Horizon")
            chart_data["Point Forecast"] = chart_data["Point Forecast"] / 1e6  # Convert to millions
            st.bar_chart(chart_data, height=250)
            st.caption("Values in millions EUR")

    with tab_trr:
        trr_df = df[df["liquidity_group"] == "TRR"]
        trr_summary = create_horizon_summary(trr_df)
        st.markdown("**TRR: Treasury Receipts (Cash Inflows)**")
        display_forecast_table(trr_summary, "TRR")

    with tab_trp:
        trp_df = df[df["liquidity_group"] == "TRP"]
        trp_summary = create_horizon_summary(trp_df)
        st.markdown("**TRP: Treasury Payments (Cash Outflows)**")
        display_forecast_table(trp_summary, "TRP")

    with tab_net:
        st.markdown("**NET: Net Cash Position (TRR + TRP)**")

        # Compute NET by summing TRR and TRP per horizon
        net_summary = []
        for h in range(1, 9):
            h_data = df[df["horizon"] == h]
            if h_data.empty:
                continue

            row = {
                "Horizon": f"H{h}",
                "Target Week": "",
                "Point Forecast": h_data["y_pred_point"].sum(),
            }

            # Get target week
            if "target_week_start" in h_data.columns:
                target = h_data["target_week_start"].iloc[0]
                if pd.notna(target):
                    row["Target Week"] = str(target)[:10]

            # Sum quantiles across both LGs
            if "y_pred_p10" in h_data.columns:
                row["P10 (Low)"] = h_data["y_pred_p10"].sum()
            if "y_pred_p50" in h_data.columns:
                row["P50 (Median)"] = h_data["y_pred_p50"].sum()
            if "y_pred_p90" in h_data.columns:
                row["P90 (High)"] = h_data["y_pred_p90"].sum()

            net_summary.append(row)

        net_df = pd.DataFrame(net_summary)
        display_forecast_table(net_df, "NET")

        # NET interpretation
        if not net_df.empty and "Point Forecast" in net_df.columns:
            total_net = net_df["Point Forecast"].sum()
            direction = "surplus" if total_net > 0 else "deficit"
            st.markdown(f"""
            <div class="interpretation-box">
                <h4>NET Position Summary</h4>
                <p>Total projected NET over 8 weeks: <strong>{format_currency(total_net)}</strong> ({direction})</p>
            </div>
            """, unsafe_allow_html=True)

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

        st.dataframe(
            filtered_df[display_cols].head(500),
            use_container_width=True,
            height=400,
        )
        st.caption(f"Showing {min(500, len(filtered_df))} of {len(filtered_df)} rows")

    # Footer
    if forecast_view.extra:
        st.caption(f"Data source: {forecast_view.extra.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
