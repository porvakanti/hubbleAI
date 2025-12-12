"""
Page 1 - Latest Forecast (Operations View)

Shows:
- Data health status
- Latest forward run status
- Manual run button
- Forecast summary with filters

Design: Modern card-based dashboard with polished styling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import streamlit as st
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.service import (
    get_data_health_summary,
    load_latest_forward_forecast,
    get_last_run_by_mode,
)

# ---------------------------------------------------------------------------
# Page CSS (extends global CSS from main app)
# ---------------------------------------------------------------------------

PAGE_CSS = """
<style>
/* Main background */
.stApp {
    background-color: #FAF9F6;
}

/* Card styling */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
    height: 100%;
}

.card-sm {
    padding: 1rem;
}

.card-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: #95A5A6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2C3E50;
}

.card-value-lg {
    font-size: 2rem;
}

.card-subtitle {
    font-size: 0.8rem;
    color: #95A5A6;
    margin-top: 0.25rem;
}

/* Status colors */
.text-success { color: #27AE60; }
.text-warning { color: #F39C12; }
.text-error { color: #E74C3C; }
.text-muted { color: #95A5A6; }

/* Accent card */
.card-accent {
    background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
}
.card-accent .card-header,
.card-accent .card-value,
.card-accent .card-subtitle {
    color: white;
}
.card-accent .card-header {
    opacity: 0.9;
}
.card-accent .card-subtitle {
    opacity: 0.8;
}

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Health indicator */
.health-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #F0F0F0;
}
.health-item:last-child {
    border-bottom: none;
}
.health-icon {
    font-size: 1.2rem;
}
.health-label {
    font-weight: 500;
    color: #2C3E50;
}
.health-detail {
    font-size: 0.8rem;
    color: #95A5A6;
    margin-left: auto;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


def render_card(header: str, value: str, subtitle: str = "", accent: bool = False, size: str = "normal") -> str:
    """Render a metric card."""
    card_class = "card card-accent" if accent else "card"
    value_class = "card-value card-value-lg" if size == "large" else "card-value"
    return f"""
    <div class="{card_class}">
        <div class="card-header">{header}</div>
        <div class="{value_class}">{value}</div>
        <div class="card-subtitle">{subtitle}</div>
    </div>
    """


def render_health_item(icon: str, label: str, detail: str) -> str:
    """Render a health status item."""
    return f"""
    <div class="health-item">
        <span class="health-icon">{icon}</span>
        <span class="health-label">{label}</span>
        <span class="health-detail">{detail}</span>
    </div>
    """


def main():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.25rem; color: #2C3E50;">Latest Forecast</h2>
        <p style="color: #95A5A6;">Operational view of forward forecasts and data health</p>
    </div>
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
    # Two-column layout: Data Health + Run Status
    # ---------------------------------------------------------------------------

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="section-title">Data Health Details</div>', unsafe_allow_html=True)

        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)

        # Health items
        details = health.get("details", {})

        for name, info in details.items():
            if info.get("exists"):
                age = info.get("age_days", "?")
                icon = "checkmark" if info.get("healthy") else "warning"
                if icon == "checkmark":
                    icon_html = '<span style="color: #27AE60;">OK</span>'
                else:
                    icon_html = '<span style="color: #F39C12;">!</span>'
                detail = f"{age}d old"
            else:
                icon_html = '<span style="color: #E74C3C;">X</span>'
                detail = "Missing"

            st.markdown(f"""
            <div class="health-item">
                <span class="health-icon">{icon_html}</span>
                <span class="health-label">{name.upper()}</span>
                <span class="health-detail">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

        if st.button("Run Forward Forecast", type="primary", use_container_width=True):
            st.info("Triggering `run_forecast(mode='forward')`...")
            try:
                from hubbleAI.pipeline import run_forecast
                with st.spinner("Running forecast... this may take a few minutes"):
                    result = run_forecast(mode="forward", trigger_source="manual")
                if result.status == "success":
                    st.success("Forecast completed successfully!")
                else:
                    st.error(f"Forecast failed: {result.message}")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Run Backtest", use_container_width=True):
            st.info("Triggering `run_forecast(mode='backtest')`...")
            try:
                from hubbleAI.pipeline import run_forecast
                with st.spinner("Running backtest... this may take several minutes"):
                    result = run_forecast(mode="backtest", trigger_source="manual")
                if result.status == "success":
                    st.success("Backtest completed successfully!")
                else:
                    st.error(f"Backtest failed: {result.message}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.caption("Production runs are automated weekly.")

    with right_col:
        st.markdown('<div class="section-title">Last Forward Run Details</div>', unsafe_allow_html=True)

        if forward_status:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Run details grid
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown(f"""
                **Run ID:** `{forward_status.get('run_id', 'N/A')[:20]}...`

                **Mode:** {forward_status.get('mode', 'N/A')}

                **Trigger:** {forward_status.get('trigger_source', 'N/A')}
                """)

            with detail_col2:
                created = forward_status.get('created_at', 'N/A')
                if isinstance(created, str) and len(created) > 19:
                    created = created[:19].replace('T', ' ')

                st.markdown(f"""
                **Status:** {forward_status.get('status', 'N/A').upper()}

                **As-of Date:** {forward_status.get('as_of_date', 'N/A')}

                **Created:** {created}
                """)

            if forward_status.get("message"):
                st.markdown(f"**Message:** {forward_status.get('message')}")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("No forward forecast run found yet. Click 'Run Forward Forecast' to generate one.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Forecast Preview Section
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Forecast Data Preview</div>', unsafe_allow_html=True)

    if forecast_view is None:
        st.warning("No forecast data available. Run a forward forecast first.")
        return

    df = forecast_view.forecasts_df.copy()

    # Filters row
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        lg_options = ["All"] + sorted(df["liquidity_group"].unique().tolist()) if "liquidity_group" in df.columns else ["All"]
        selected_lg = st.selectbox("Liquidity Group", lg_options)

    with filter_col2:
        horizon_options = ["All"] + [f"H{h}" for h in sorted(df["horizon"].unique().tolist())] if "horizon" in df.columns else ["All"]
        selected_horizon = st.selectbox("Horizon", horizon_options)

    with filter_col3:
        if "entity" in df.columns:
            entity_list = sorted(df["entity"].unique().tolist())
            entity_options = ["All"] + entity_list[:50]  # Limit for performance
        else:
            entity_options = ["All"]
        selected_entity = st.selectbox("Entity", entity_options)

    # Apply filters
    filtered_df = df.copy()

    if selected_lg != "All" and "liquidity_group" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["liquidity_group"] == selected_lg]

    if selected_horizon != "All" and "horizon" in filtered_df.columns:
        h = int(selected_horizon.replace("H", ""))
        filtered_df = filtered_df[filtered_df["horizon"] == h]

    if selected_entity != "All" and "entity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["entity"] == selected_entity]

    # Net summary by horizon
    st.markdown("**Net Forecast by Horizon (TRR + TRP)**")

    if "horizon" in filtered_df.columns and "y_pred_point" in filtered_df.columns:
        agg_cols = {"y_pred_point": "sum"}
        if "y_pred_hybrid" in filtered_df.columns:
            agg_cols["y_pred_hybrid"] = "sum"

        net_summary = (
            filtered_df
            .groupby(["horizon"], as_index=False)
            .agg(agg_cols)
        )
        net_summary.columns = ["Horizon", "ML Forecast"] + (["Hybrid Forecast"] if "y_pred_hybrid" in agg_cols else [])

        # Format as millions
        for col in net_summary.columns[1:]:
            net_summary[col] = net_summary[col].apply(lambda x: f"{x/1e6:.2f}M" if pd.notna(x) else "-")

        st.dataframe(net_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Cannot compute net summary - missing required columns.")

    # Detailed data
    with st.expander("View Detailed Forecast Data", expanded=False):
        display_cols = [
            "entity", "liquidity_group", "horizon", "week_start", "target_week_start",
            "y_pred_point", "y_pred_hybrid", "y_pred_p10", "y_pred_p90",
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
