"""
HubbleAI - Treasury Cashflow Forecasting Platform

Main entry point for the Streamlit app.
Design inspired by modern dashboard UI (Dribbble reference).

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="HubbleAI - Treasury Forecasts",
    page_icon="./assets/hubble_icon.png" if False else "H",  # Use custom icon if available
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS - Dribbble-inspired modern dashboard design
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
/* ============================================
   GLOBAL STYLES - Light, modern dashboard
   ============================================ */

/* Main background - warm off-white */
.stApp {
    background-color: #FAF9F6;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ============================================
   SIDEBAR STYLING
   ============================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1C2833 0%, #2C3E50 100%);
    border-right: none;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #ECF0F1;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #BDC3C7 !important;
}

/* Sidebar navigation links */
[data-testid="stSidebar"] a {
    color: #ECF0F1 !important;
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: background-color 0.2s;
}

[data-testid="stSidebar"] a:hover {
    background-color: rgba(255, 255, 255, 0.1);
}

/* ============================================
   CARD STYLES
   ============================================ */

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}

.card-header {
    font-size: 0.85rem;
    font-weight: 500;
    color: #7F8C8D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2C3E50;
    line-height: 1.2;
}

.card-subtitle {
    font-size: 0.8rem;
    color: #95A5A6;
    margin-top: 0.25rem;
}

/* Accent card - green gradient */
.card-accent {
    background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
    color: white;
}

.card-accent .card-header {
    color: rgba(255, 255, 255, 0.85);
}

.card-accent .card-value {
    color: white;
}

.card-accent .card-subtitle {
    color: rgba(255, 255, 255, 0.75);
}

/* Status indicators */
.status-success {
    color: #27AE60;
    font-weight: 600;
}

.status-warning {
    color: #F39C12;
    font-weight: 600;
}

.status-error {
    color: #E74C3C;
    font-weight: 600;
}

/* ============================================
   TYPOGRAPHY
   ============================================ */

h1, h2, h3 {
    color: #2C3E50;
    font-weight: 600;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ============================================
   BUTTONS
   ============================================ */

.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
    border: none;
}

/* ============================================
   TABLES & DATAFRAMES
   ============================================ */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ============================================
   METRICS
   ============================================ */

[data-testid="stMetric"] {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    color: #7F8C8D;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
    color: #2C3E50;
}

/* ============================================
   HIDE STREAMLIT BRANDING
   ============================================ */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============================================
   CUSTOM LOGO AREA
   ============================================ */

.logo-container {
    padding: 1.5rem 1rem;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 1.5rem;
}

.logo-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.logo-subtitle {
    font-size: 0.75rem;
    color: #BDC3C7;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Navigation items */
.nav-section {
    padding: 0 1rem;
    margin-bottom: 1.5rem;
}

.nav-label {
    font-size: 0.7rem;
    color: #7F8C8D;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.75rem;
    padding-left: 0.5rem;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar - Logo and Branding
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-icon">H</div>
    <div class="logo-text">HubbleAI</div>
    <div class="logo-subtitle">Treasury Forecasts</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="nav-section">
    <div class="nav-label">Navigation</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("Select a page from the menu above")

st.sidebar.markdown("---")

# Version info at bottom
st.sidebar.markdown("""
<div style="position: fixed; bottom: 1rem; left: 1rem; color: #7F8C8D; font-size: 0.7rem;">
    v0.2.0 | Powered by ML
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Content - Welcome Page
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.25rem;">Welcome to HubbleAI</h1>
    <p style="color: #7F8C8D; font-size: 1.1rem;">Treasury Cashflow Forecasting Platform for Aperam</p>
</div>
""", unsafe_allow_html=True)

# Quick navigation cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="section-title">Latest Forecast</div>
        <p style="color: #7F8C8D; margin-bottom: 1rem;">
            View the most recent forward forecast, check data health,
            and trigger new forecast runs.
        </p>
        <div style="color: #27AE60; font-weight: 500;">Go to Latest Forecast &rarr;</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="section-title">Performance Dashboard</div>
        <p style="color: #7F8C8D; margin-bottom: 1rem;">
            Compare ML vs LP vs Hybrid accuracy. View win rates,
            WAPE trends, and detailed metrics.
        </p>
        <div style="color: #27AE60; font-weight: 500;">Go to Performance &rarr;</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature highlights
st.markdown("""
<div class="section-title">Platform Capabilities</div>
""", unsafe_allow_html=True)

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="card" style="text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">ML</div>
        <div style="font-weight: 600; color: #2C3E50; margin-bottom: 0.5rem;">Machine Learning</div>
        <div style="color: #7F8C8D; font-size: 0.9rem;">LightGBM models trained on historical patterns</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="card" style="text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">LP</div>
        <div style="font-weight: 600; color: #2C3E50; margin-bottom: 0.5rem;">LP Baseline</div>
        <div style="color: #7F8C8D; font-size: 0.9rem;">Established benchmark from treasury planning</div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="card card-accent" style="text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">H</div>
        <div style="font-weight: 600; margin-bottom: 0.5rem;">Hybrid Model</div>
        <div style="font-size: 0.9rem;">Best of both - beats LP 57-80% of weeks</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick stats
st.markdown("""
<div class="section-title">Current Performance (TRP Horizons)</div>
""", unsafe_allow_html=True)

perf_cols = st.columns(4)
horizons_data = [
    ("H1", "57%", "16/28 weeks"),
    ("H2", "74%", "20/27 weeks"),
    ("H3", "69%", "18/26 weeks"),
    ("H4", "80%", "20/25 weeks"),
]

for i, (horizon, rate, detail) in enumerate(horizons_data):
    with perf_cols[i]:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div class="card-header">{horizon}</div>
            <div class="card-value" style="color: #27AE60;">{rate}</div>
            <div class="card-subtitle">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95A5A6; font-size: 0.85rem;">
    HubbleAI v0.2.0 | Treasury Cashflow Forecasting | Built with Streamlit
</div>
""", unsafe_allow_html=True)
