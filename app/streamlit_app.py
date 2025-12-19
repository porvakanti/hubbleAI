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
    page_title="Hubble.AI - Treasury Forecasts",
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
        Welcome to Hubble.AI
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        Bringing long-range financial foresight for short-term liquidity through intelligent modeling and analytics
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Quick Guide (Collapsible)
# ---------------------------------------------------------------------------

# CSS for the Quick Guide section - aligned with app's hubble-card patterns
st.markdown("""
<style>
/* Expander styling for Quick Guide */
[data-testid="stExpander"] details summary {
    background: linear-gradient(135deg, #FFFBF5 0%, #FFF8E7 100%);
    border: 1px solid #E8DCC8;
    border-radius: 12px;
    padding: 0.75rem 1rem;
}
[data-testid="stExpander"] details summary:hover {
    background: linear-gradient(135deg, #FFF8E7 0%, #FFEFCC 100%);
    border-color: #D4AF37;
}
[data-testid="stExpander"] details[open] summary {
    border-radius: 12px 12px 0 0;
    border-bottom: none;
}
[data-testid="stExpander"] details > div {
    background: linear-gradient(135deg, #FFFBF5 0%, #FFF8E7 100%);
    border: 1px solid #E8DCC8;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1rem;
}

.quick-guide-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E8DCC8;
}
.quick-guide-logo {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
}
.quick-guide-tagline {
    font-size: 0.9rem;
    color: #5A6169;
    line-height: 1.5;
}
.quick-guide-tagline strong {
    color: #2D3436;
}
.quick-guide-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
@media (max-width: 768px) {
    .quick-guide-grid {
        grid-template-columns: 1fr;
    }
}
.quick-guide-item {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.06);
    transition: all 0.2s ease;
    cursor: default;
}
.quick-guide-item:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    transform: translateY(-2px);
    border-color: rgba(212,175,55,0.3);
}
.quick-guide-item-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.quick-guide-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.quick-guide-icon.gold { background: linear-gradient(135deg, #D4AF37 0%, #F4D03F 100%); }
.quick-guide-icon.green { background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%); }
.quick-guide-icon.blue { background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%); }
.quick-guide-icon.purple { background: linear-gradient(135deg, #5E35B1 0%, #7E57C2 100%); }
.quick-guide-icon svg { stroke: white; }
.quick-guide-item h4 {
    margin: 0;
    font-size: 0.85rem;
    color: #2D3436;
    font-weight: 600;
}
.quick-guide-item p {
    margin: 0;
    font-size: 0.8rem;
    color: #5A6169;
    line-height: 1.5;
}
.quick-guide-item code {
    background: linear-gradient(135deg, #FFFBF5 0%, #FFF8E7 100%);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #5D4E37;
    border: 1px solid #E8DCC8;
}
.quick-guide-intro {
    font-size: 0.85rem;
    color: #5A6169;
    margin-bottom: 1rem;
    line-height: 1.6;
}
.quick-guide-intro strong {
    color: #2D3436;
}
</style>
""", unsafe_allow_html=True)

with st.expander("📘 Quick Guide — New to Hubble.AI?", expanded=False):
    st.markdown("""
    <div class="quick-guide-header">
        <svg class="quick-guide-logo" viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="200" height="200" rx="40" fill="#1a1a1a"/>
            <circle cx="100" cy="85" r="50" stroke="#D4AF37" stroke-width="12" fill="none"/>
            <circle cx="100" cy="85" r="18" fill="#D4AF37"/>
            <line x1="100" y1="135" x2="100" y2="170" stroke="#D4AF37" stroke-width="12" stroke-linecap="round"/>
            <line x1="65" y1="168" x2="135" y2="168" stroke="#D4AF37" stroke-width="10" stroke-linecap="round"/>
        </svg>
        <div class="quick-guide-tagline">
            Just as the <strong>Hubble Space Telescope</strong> revolutionized astronomy by enabling us to see deeper into the universe,
            <strong>Hubble.AI</strong> empowers Treasury to see further into the future — extending cash flow visibility from 4 weeks to 8 weeks with improved accuracy.
        </div>
    </div>
    <div class="quick-guide-intro">
        <strong>The Challenge:</strong> Traditional liquidity plan (LP) forecasts faced accuracy limitations,
        and visibility was capped at 4 weeks — making long-term liquidity planning difficult.
    </div>
    <div class="quick-guide-grid">
        <div class="quick-guide-item">
            <div class="quick-guide-item-header">
                <div class="quick-guide-icon gold">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                </div>
                <h4>WAPE</h4>
            </div>
            <p>
                Weighted Absolute Percentage Error — our primary accuracy metric. <strong>Lower = better.</strong><br/>
                Formula: <code>|Σ Actuals − Σ Predictions| ÷ |Σ Actuals|</code><br/>
                A 5% WAPE means €5 error for every €100 of actual cash flow.
            </p>
        </div>
        <div class="quick-guide-item">
            <div class="quick-guide-item-header">
                <div class="quick-guide-icon green">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 10 10H12V2z"/><path d="M12 2a10 10 0 0 1 10 10"/><circle cx="12" cy="12" r="3"/></svg>
                </div>
                <h4>ML vs LP</h4>
            </div>
            <p>
                <strong>ML</strong> = Our AI model trained on historical patterns.<br/>
                <strong>LP</strong> = Treasury's baseline Liquidity Plan forecasts.<br/>
                We compare both to show where ML adds value.
            </p>
        </div>
        <div class="quick-guide-item">
            <div class="quick-guide-item-header">
                <div class="quick-guide-icon blue">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                </div>
                <h4>P10 / P50 / P90</h4>
            </div>
            <p>
                Confidence intervals for uncertainty quantification.<br/>
                <strong>P10</strong> = Conservative (10% chance of worse)<br/>
                <strong>P50</strong> = Most likely · <strong>P90</strong> = Optimistic (10% chance of better)
            </p>
        </div>
        <div class="quick-guide-item">
            <div class="quick-guide-item-header">
                <div class="quick-guide-icon purple">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                </div>
                <h4>Target Accuracy</h4>
            </div>
            <p>
                WAPE goals by horizon: <strong>H1: 5%</strong>, H2: 7.5%, H3: 10%, H4: 12.5%,
                H5: 15%, H6: 17.5%, H7: 20%, H8: 22.5%.<br/>
                Longer horizons naturally have higher targets.
            </p>
        </div>
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
.arrow.purple { color: #5E35B1; }
.arrow.gray { color: #455A64; }
.link-text.purple { color: #5E35B1; }
.link-text.gray { color: #455A64; }

/* Style page_link as footer link (not a button) */
.nav-card-link {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 0.75rem 1.5rem;
    border-top: 1px solid rgba(0,0,0,0.06);
    display: flex;
    justify-content: space-between;
    align-items: center;
    text-decoration: none;
    transition: all 0.2s ease;
    border-radius: 0 0 16px 16px;
    pointer-events: none;
}
.link-text {
    font-size: 0.68rem;
    color: #8B95A1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
    transition: color 0.2s ease;
}
.nav-card-link .arrow {
    font-weight: 600;
    font-size: 1.1rem;
    transition: transform 0.2s ease;
}

/* Hover effects triggered from parent card - more prominent */
.clickable-nav-card:hover .nav-card-link.green-link {
    background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
    border-top: 1px solid #A5D6A7;
}
.clickable-nav-card:hover .nav-card-link.blue-link {
    background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
    border-top: 1px solid #90CAF9;
}
.clickable-nav-card:hover .link-text.green {
    color: #1B5E20;
    font-weight: 600;
}
.clickable-nav-card:hover .link-text.blue {
    color: #0D47A1;
    font-weight: 600;
}
.clickable-nav-card:hover .nav-card-link .arrow.green {
    transform: translateX(4px);
    color: #1B5E20;
}
.clickable-nav-card:hover .nav-card-link .arrow.blue {
    transform: translateX(4px);
    color: #0D47A1;
}
.clickable-nav-card:hover .nav-card-link.purple-link {
    background: linear-gradient(135deg, #EDE7F6 0%, #D1C4E9 100%);
    border-top: 1px solid #B39DDB;
}
.clickable-nav-card:hover .nav-card-link.gray-link {
    background: linear-gradient(135deg, #ECEFF1 0%, #CFD8DC 100%);
    border-top: 1px solid #B0BEC5;
}
.clickable-nav-card:hover .link-text.purple {
    color: #4527A0;
    font-weight: 600;
}
.clickable-nav-card:hover .link-text.gray {
    color: #37474F;
    font-weight: 600;
}
.clickable-nav-card:hover .nav-card-link .arrow.purple {
    transform: translateX(4px);
    color: #4527A0;
}
.clickable-nav-card:hover .nav-card-link .arrow.gray {
    transform: translateX(4px);
    color: #37474F;
}

/* Hide default page_link styling, make it cover the footer */
[data-testid="stPageLink-NavLink"] {
    position: absolute !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 48px !important;
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 10 !important;
    border-radius: 0 0 16px 16px !important;
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
        <div class="nav-card-link green-link">
            <span class="link-text green">Operations View</span>
            <span class="arrow green">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Latest_Forecast.py", label=" ", use_container_width=True)

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
        <div class="nav-card-link blue-link">
            <span class="link-text blue">Analytics View</span>
            <span class="arrow blue">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Performance_Dashboard.py", label=" ", use_container_width=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

# Second row of navigation cards
col_left2, col_right2 = st.columns(2, gap="large")

with col_left2:
    st.markdown("""
    <div class="clickable-nav-card">
        <div class="nav-card-header">
            <div class="nav-card-icon" style="background: linear-gradient(135deg, #5E35B1 0%, #7E57C2 100%);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
            </div>
            <div class="nav-card-body">
                <h3>Backtest Explorer</h3>
                <p>Drill down into historical predictions week by week. Compare actual vs forecasted values.</p>
            </div>
        </div>
        <div class="nav-card-link purple-link">
            <span class="link-text purple">Historical Analysis</span>
            <span class="arrow purple">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Backtest_Explorer.py", label=" ", use_container_width=True)

with col_right2:
    st.markdown("""
    <div class="clickable-nav-card">
        <div class="nav-card-header">
            <div class="nav-card-icon" style="background: linear-gradient(135deg, #455A64 0%, #78909C 100%);">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            </div>
            <div class="nav-card-body">
                <h3>Admin</h3>
                <p>Run forecasts, upload data files, and manage system settings.</p>
            </div>
        </div>
        <div class="nav-card-link gray-link">
            <span class="link-text gray">System Settings</span>
            <span class="arrow gray">→</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/9_Admin.py", label=" ", use_container_width=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# System Status Section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

# Build System Status content as a single HTML block to avoid Streamlit container issues
details = health.get("details", {})
data_inputs_html = ""
for name, info in details.items():
    if info.get("exists"):
        status_icon = "✓" if info.get("healthy") else "!"
        status_color = "#2E7D32" if info.get("healthy") else "#F57C00"
        data_inputs_html += f'<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0;"><span style="color: {status_color}; font-weight: 600;">{status_icon}</span><span>{name.upper()}</span></div>'
    else:
        data_inputs_html += f'<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0;"><span style="color: #D32F2F; font-weight: 600;">✗</span><span>{name.upper()}</span></div>'

st.markdown(f"""<div class="hubble-card hubble-card-flat"><div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem;"><div><div style="font-weight: 600; margin-bottom: 0.75rem; color: #2D3436;">Data Inputs</div>{data_inputs_html}</div><div><div style="font-weight: 600; margin-bottom: 0.75rem; color: #2D3436;">Forecast Schedule</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> Weekly runs every Tuesday</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> 8-week rolling horizon</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> TRR + TRP liquidity groups</div></div><div><div style="font-weight: 600; margin-bottom: 0.75rem; color: #2D3436;">Model Info</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> Engine: LightGBM</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> Quantiles: P10/P50/P90</div><div style="padding: 0.25rem 0;"><span style="color: #2E7D32; font-weight: 600;">●</span> TRP H1-4: Hybrid ML+LP</div></div></div></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #8B95A1; font-size: 0.8rem;">
    Hubble.AI Treasury Forecasting Platform · v{APP_VERSION}
</div>
""", unsafe_allow_html=True)
