"""
HubbleAI - Treasury Cashflow Forecasting Platform

Main entry point for the Streamlit app (Home page).
Design: Modern, warm cream palette with clean navigation.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui_components import (
    set_global_style,
    render_sidebar,
    render_metric_card,
    APP_VERSION,
)
from hubbleAI.service import (
    get_data_health_summary,
    get_last_run_by_mode,
    load_latest_forward_forecast,
    load_latest_backtest_results,
)

# ---------------------------------------------------------------------------
# Page Config - must be first Streamlit command
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HubbleAI - Treasury Forecasts",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styles
set_global_style()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# Get reference info for sidebar
forward_status = get_last_run_by_mode("forward")
ref_info = None
if forward_status:
    ref_info = {
        "ref_week_start": forward_status.get("ref_week_start", "-"),
        "run_id": forward_status.get("run_id", "-"),
    }

render_sidebar(active_page="Overview", ref_info=ref_info)

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem; font-size: 1.75rem; font-weight: 700; color: #2D3436;">
        Welcome to HubbleAI
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        Explore your Treasury cashflow forecasts and performance analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick Stats Row
# ---------------------------------------------------------------------------

# Load data for stats
health = get_data_health_summary()
forecast_view = load_latest_forward_forecast()
backtest_view = load_latest_backtest_results()

col1, col2, col3, col4 = st.columns(4)

with col1:
    data_status = "Ready" if health["is_ready"] else "Incomplete"
    data_subtitle = "All inputs present" if health["is_ready"] else f"{len(health.get('missing_inputs', []))} missing"
    st.markdown(render_metric_card(
        header="Data Status",
        value=data_status,
        subtitle=data_subtitle,
        accent=health["is_ready"]
    ), unsafe_allow_html=True)

with col2:
    if forward_status:
        run_status = forward_status.get("status", "unknown").upper()
        ref_date = forward_status.get("ref_week_start", "-")
        if isinstance(ref_date, str) and len(ref_date) > 10:
            ref_date = ref_date[:10]
    else:
        run_status = "No Runs"
        ref_date = "-"
    st.markdown(render_metric_card(
        header="Last Forecast",
        value=run_status,
        subtitle=f"Ref: {ref_date}"
    ), unsafe_allow_html=True)

with col3:
    if forecast_view:
        n_entities = forecast_view.forecasts_df["entity"].nunique() if "entity" in forecast_view.forecasts_df.columns else 0
        n_rows = len(forecast_view.forecasts_df)
    else:
        n_entities = 0
        n_rows = 0
    st.markdown(render_metric_card(
        header="Active Entities",
        value=str(n_entities),
        subtitle=f"{n_rows:,} forecast rows"
    ), unsafe_allow_html=True)

with col4:
    if backtest_view:
        test_weeks = backtest_view.backtest_df["week_start"].nunique() if not backtest_view.backtest_df.empty and "week_start" in backtest_view.backtest_df.columns else 0
    else:
        test_weeks = 0
    st.markdown(render_metric_card(
        header="Backtest",
        value=f"{test_weeks} weeks",
        subtitle="Test set coverage"
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Cards (Clickable)
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Quick Navigation</div>', unsafe_allow_html=True)

# Add CSS for clickable nav cards
st.markdown("""
<style>
.clickable-nav-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.06);
    cursor: pointer;
    transition: all 0.2s ease;
    height: 160px;
    display: flex;
    flex-direction: column;
    position: relative;
}
.clickable-nav-card:hover {
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.nav-card-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.nav-card-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.nav-card-icon.green { background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); }
.nav-card-icon.blue { background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%); }
.nav-card-icon svg { stroke: white; }
.nav-card-body h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #2D3436;
}
.nav-card-body p {
    margin: 0;
    color: #5A6169;
    font-size: 0.82rem;
    line-height: 1.5;
}
.nav-card-footer {
    margin-top: auto;
    padding-top: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(0,0,0,0.06);
}
.nav-card-footer span:first-child {
    font-size: 0.68rem;
    color: #8B95A1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.nav-card-footer .arrow { font-weight: 600; font-size: 1.1rem; }
.arrow.green { color: #2E7D32; }
.arrow.blue { color: #1976D2; }

/* Style page_link to integrate with cards */
[data-testid="stPageLink-NavLink"] {
    margin-top: -8px !important;
    border-radius: 0 0 16px 16px !important;
    background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%) !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 0.75rem 1rem !important;
    text-align: center !important;
    text-decoration: none !important;
    transition: all 0.2s ease !important;
}
[data-testid="stPageLink-NavLink"]:hover {
    opacity: 0.9 !important;
}
</style>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""
    <div class="clickable-nav-card">
        <div class="nav-card-header">
            <div class="nav-card-icon green">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
            </div>
            <div class="nav-card-body">
                <h3>Latest Forecast</h3>
                <p>View 8-week ML predictions with P10/P50/P90 intervals for TRR, TRP, and NET.</p>
            </div>
        </div>
        <div class="nav-card-footer">
            <span>Operations View</span>
            <span class="arrow green">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Latest_Forecast.py", label="Open Forecast", use_container_width=True)

with col_right:
    st.markdown("""
    <div class="clickable-nav-card">
        <div class="nav-card-header">
            <div class="nav-card-icon blue">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <div class="nav-card-body">
                <h3>Performance Dashboard</h3>
                <p>Compare ML vs LP accuracy, view historical performance, and explore backtest results.</p>
            </div>
        </div>
        <div class="nav-card-footer">
            <span>Analytics View</span>
            <span class="arrow blue">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Performance_Dashboard.py", label="Open Dashboard", use_container_width=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# System Status Section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="hubble-card hubble-card-flat">', unsafe_allow_html=True)

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.markdown("**Data Inputs**")
        details = health.get("details", {})
        for name, info in details.items():
            if info.get("exists"):
                status_icon = "✓" if info.get("healthy") else "!"
                status_color = "#2E7D32" if info.get("healthy") else "#F57C00"
                age = info.get("age_days", "?")
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0;">
                    <span style="color: {status_color}; font-weight: 600;">{status_icon}</span>
                    <span style="flex: 1;">{name.upper()}</span>
                    <span style="color: #8B95A1; font-size: 0.8rem;">{age}d</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0;">
                    <span style="color: #D32F2F; font-weight: 600;">✗</span>
                    <span style="flex: 1;">{name.upper()}</span>
                    <span style="color: #8B95A1; font-size: 0.8rem;">Missing</span>
                </div>
                """, unsafe_allow_html=True)

    with status_col2:
        st.markdown("**Forecast Schedule**")
        st.markdown("""
        <div style="padding: 0.25rem 0;">
            <span style="color: #2E7D32; font-weight: 600;">●</span> Weekly runs every Tuesday
        </div>
        <div style="padding: 0.25rem 0;">
            <span style="color: #8B95A1;">8-week rolling horizon</span>
        </div>
        <div style="padding: 0.25rem 0;">
            <span style="color: #8B95A1;">TRR + TRP liquidity groups</span>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        st.markdown("**Model Info**")
        st.markdown("""
        <div style="padding: 0.25rem 0;">
            <span style="font-weight: 500;">Engine:</span> LightGBM
        </div>
        <div style="padding: 0.25rem 0;">
            <span style="font-weight: 500;">Quantiles:</span> P10/P50/P90
        </div>
        <div style="padding: 0.25rem 0;">
            <span style="font-weight: 500;">TRP H1-4:</span> Hybrid ML+LP
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #8B95A1; font-size: 0.8rem;">
    HubbleAI Treasury Forecasting Platform · v{APP_VERSION}
</div>
""", unsafe_allow_html=True)
