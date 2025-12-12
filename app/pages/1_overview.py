"""
Page 1 – Latest Forecast / Operations Overview

Design inspired by modern card-based dashboard UI with:
- Top KPI row with 4 metric cards
- Main content area with forecast preview
- Data health status
- Clean, light aesthetic with green accents
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.services import (
    load_latest_run_status,
    load_latest_forward_forecast,
    load_latest_backtest_metrics,
    get_health_summary,
    get_hybrid_performance_summary,
)

# ---------------------------------------------------------------------------
# Custom CSS for card-based design
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Light background */
.stApp {
    background-color: #f5f3ef;
}

/* Card styling */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    height: 100%;
}

.metric-card-green {
    background: linear-gradient(135deg, #4a5d4a 0%, #5c6b5c 100%);
    border-radius: 16px;
    padding: 20px;
    color: white;
}

.metric-label {
    font-size: 14px;
    color: #666;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #1a1a1a;
}

.metric-value-green {
    font-size: 28px;
    font-weight: 600;
    color: white;
}

.status-healthy {
    color: #4a7c4a;
    font-weight: 500;
}

.status-warning {
    color: #c9a227;
    font-weight: 500;
}

.status-error {
    color: #c94a4a;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 16px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def render_metric_card(label: str, value: str, subtitle: str = "", is_green: bool = False):
    """Render a styled metric card."""
    if is_green:
        return f"""
        <div class="metric-card-green">
            <div style="font-size: 14px; opacity: 0.9;">{label}</div>
            <div class="metric-value-green">{value}</div>
            <div style="font-size: 12px; opacity: 0.8;">{subtitle}</div>
        </div>
        """
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="font-size: 12px; color: #888;">{subtitle}</div>
    </div>
    """


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("## 👋 Welcome to HubbleAI")
    st.markdown("Treasury Cashflow Forecasting Dashboard")

    st.markdown("---")

    # Load data
    run_status = load_latest_run_status()
    health = get_health_summary()
    performance = get_hybrid_performance_summary()

    # Calculate average win rate
    avg_win_rate = "N/A"
    if performance is not None and not performance.empty:
        avg_win_rate = f"{performance['win_rate_vs_lp'].mean() * 100:.0f}%"

    # Top KPI Row - 4 cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_text = run_status.status if run_status else "No runs"
        status_date = run_status.as_of_date if run_status else "-"
        st.markdown(
            render_metric_card("Latest Run", status_text, f"As of {status_date}"),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            render_metric_card("Hybrid Win Rate", avg_win_rate, "Avg across H1-H4", is_green=True),
            unsafe_allow_html=True
        )

    with col3:
        health_status = "✓ Healthy" if health.all_healthy else "⚠ Check Required"
        st.markdown(
            render_metric_card("Data Health", health_status, "Raw inputs status"),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            render_metric_card("Next Run", "Tuesday", "Weekly schedule"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main content - two columns
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("### 📊 Hybrid Performance by Horizon")

        if performance is not None and not performance.empty:
            # Create a nice bar chart for win rates
            chart_data = performance.copy()
            chart_data['Horizon'] = 'H' + chart_data['horizon'].astype(str)
            chart_data['Win Rate %'] = chart_data['win_rate_vs_lp'] * 100

            # Display as horizontal metrics
            h_cols = st.columns(4)
            for i, (_, row) in enumerate(chart_data.iterrows()):
                with h_cols[i]:
                    wins = int(row['weekly_wins_vs_lp'])
                    total = int(row['total_weeks'])
                    pct = row['win_rate_vs_lp'] * 100
                    st.metric(
                        label=f"Horizon {int(row['horizon'])}",
                        value=f"{pct:.0f}%",
                        delta=f"{wins}/{total} weeks",
                    )

            # Bar chart
            st.bar_chart(
                chart_data.set_index('Horizon')['Win Rate %'],
                color="#5c6b5c",
                height=200,
            )
        else:
            st.info("No performance data available. Run backtest first.")

    with right_col:
        st.markdown("### 🏥 Data Health")

        if health.checks.get("data"):
            for name, check in health.checks["data"].items():
                if check.get("exists"):
                    icon = "✅" if check.get("healthy") else "⚠️"
                    age = check.get("age_days", "?")
                    st.markdown(f"{icon} **{name.upper()}**: {age} days old")
                else:
                    st.markdown(f"❌ **{name.upper()}**: Missing")

        st.markdown("---")

        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Run Forecast Now", use_container_width=True):
            st.info("This will trigger `run_forecast(mode='forward')`")
            # TODO: Wire to actual pipeline

        if st.button("📈 Run Backtest", use_container_width=True):
            st.info("This will trigger `run_forecast(mode='backtest')`")
            # TODO: Wire to actual pipeline

    st.markdown("---")

    # Forecast Preview Section
    st.markdown("### 📋 Latest Forecast Preview")

    forecast_result = load_latest_forward_forecast()

    if forecast_result.success and forecast_result.data is not None:
        df = forecast_result.data

        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            lg_options = df['liquidity_group'].unique().tolist()
            selected_lg = st.selectbox("Liquidity Group", ["All"] + lg_options)

        with filter_col2:
            horizon_options = sorted(df['horizon'].unique().tolist())
            selected_horizon = st.selectbox("Horizon", ["All"] + [f"H{h}" for h in horizon_options])

        with filter_col3:
            entity_options = df['entity'].unique().tolist()
            selected_entity = st.selectbox("Entity", ["All"] + entity_options[:20])

        # Apply filters
        filtered_df = df.copy()
        if selected_lg != "All":
            filtered_df = filtered_df[filtered_df['liquidity_group'] == selected_lg]
        if selected_horizon != "All":
            h = int(selected_horizon.replace("H", ""))
            filtered_df = filtered_df[filtered_df['horizon'] == h]
        if selected_entity != "All":
            filtered_df = filtered_df[filtered_df['entity'] == selected_entity]

        # Display columns
        display_cols = ['entity', 'liquidity_group', 'horizon', 'week_start',
                        'y_pred_point', 'y_pred_hybrid', 'y_pred_p10', 'y_pred_p90']
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            height=400,
        )

        st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df)} rows")

    else:
        st.warning("No forecast data available.")
        if forecast_result.error:
            st.error(f"Error: {forecast_result.error}")
        st.info("Run the pipeline first: `run_forecast(mode='forward')`")


if __name__ == "__main__":
    main()
