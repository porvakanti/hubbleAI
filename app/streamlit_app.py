"""
HubbleAI - Treasury Cashflow Forecasting Platform

Main entry point for the Streamlit app.
Design: Warm cream tones, icon-only sidebar with hover expand.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="HubbleAI - Treasury Forecasts",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start collapsed for icon-only view
)

# ---------------------------------------------------------------------------
# Global CSS - Warm cream palette, icon-only sidebar
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
/* ============================================
   ROOT VARIABLES - Warm Cream Palette
   ============================================ */

:root {
    --bg-cream: #F5F5F0;
    --bg-cream-light: #FAFAF7;
    --bg-white: #FFFFFF;
    --sidebar-bg: #F7F7F5;
    --sidebar-hover: #EEEEE8;
    --text-dark: #2D3436;
    --text-muted: #636E72;
    --text-light: #95A5A6;
    --accent-green: #4CAF50;
    --accent-green-light: #81C784;
    --accent-green-dark: #388E3C;
    --accent-orange: #FF9800;
    --accent-red: #E53935;
    --border-light: rgba(0, 0, 0, 0.06);
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* ============================================
   GLOBAL STYLES - Warm cream background
   ============================================ */

.stApp {
    background-color: var(--bg-cream);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ============================================
   SIDEBAR - Light cream, icon-only with hover
   ============================================ */

[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border-light);
    min-width: 70px !important;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"]:hover {
    min-width: 240px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 0.75rem;
}

/* Sidebar text colors */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-dark);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-muted);
}

/* Nav items in sidebar */
[data-testid="stSidebarNav"] {
    padding: 0.5rem 0;
}

[data-testid="stSidebarNav"] a {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    margin: 0.25rem 0.5rem;
    border-radius: var(--radius-sm);
    color: var(--text-dark) !important;
    text-decoration: none !important;
    transition: background 0.2s ease;
    font-weight: 500;
}

[data-testid="stSidebarNav"] a:hover {
    background: var(--sidebar-hover);
}

[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: var(--accent-green);
    color: white !important;
}

/* ============================================
   CARD STYLES - Rounded, subtle shadows
   ============================================ */

.card {
    background: var(--bg-white);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    margin-bottom: 1rem;
}

.card-sm {
    padding: 1rem;
    border-radius: var(--radius-md);
}

.card-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.2;
}

.card-value-sm {
    font-size: 1.5rem;
}

.card-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* Accent card - green */
.card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
}

.card-accent .card-header,
.card-accent .card-value,
.card-accent .card-subtitle {
    color: white;
}

.card-accent .card-header {
    opacity: 0.9;
}

/* ============================================
   STATUS COLORS
   ============================================ */

.status-success { color: var(--accent-green); font-weight: 600; }
.status-warning { color: var(--accent-orange); font-weight: 600; }
.status-error { color: var(--accent-red); font-weight: 600; }
.text-muted { color: var(--text-muted); }
.text-green { color: var(--accent-green); }
.text-orange { color: var(--accent-orange); }
.text-red { color: var(--accent-red); }

/* ============================================
   TYPOGRAPHY
   ============================================ */

h1, h2, h3, h4 {
    color: var(--text-dark);
    font-weight: 600;
}

h1 { font-size: 2rem; margin-bottom: 0.5rem; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.page-subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* ============================================
   BUTTONS
   ============================================ */

.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    transition: all 0.2s ease;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.stButton > button[kind="primary"] {
    background: var(--accent-green);
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: var(--accent-green-dark);
}

/* ============================================
   TABLES & DATAFRAMES
   ============================================ */

[data-testid="stDataFrame"] {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-light);
}

[data-testid="stDataFrame"] table {
    font-size: 0.9rem;
}

/* ============================================
   METRICS
   ============================================ */

[data-testid="stMetric"] {
    background: var(--bg-white);
    padding: 1rem;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    color: var(--text-muted);
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-dark);
}

/* ============================================
   SELECTBOX & INPUTS
   ============================================ */

.stSelectbox > div > div {
    border-radius: var(--radius-sm);
    border-color: var(--border-light);
}

.stSelectbox label {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* ============================================
   EXPANDER
   ============================================ */

.streamlit-expanderHeader {
    font-weight: 500;
    color: var(--text-dark);
    background: var(--bg-cream-light);
    border-radius: var(--radius-sm);
}

/* ============================================
   TABS
   ============================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-green);
    color: white;
}

/* ============================================
   CHARTS
   ============================================ */

[data-testid="stArrowVegaLiteChart"] {
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ============================================
   INFO/WARNING/SUCCESS BOXES
   ============================================ */

.stAlert {
    border-radius: var(--radius-md);
    border: none;
}

/* ============================================
   HIDE STREAMLIT BRANDING
   ============================================ */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============================================
   LOGO & BRANDING IN SIDEBAR
   ============================================ */

.sidebar-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem 0.5rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 1rem;
}

.sidebar-logo-icon {
    width: 40px;
    height: 40px;
    background: var(--accent-green);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
}

.sidebar-logo-text {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-top: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.sidebar-logo-subtitle {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Hide logo text when collapsed */
[data-testid="stSidebar"]:not(:hover) .sidebar-logo-text,
[data-testid="stSidebar"]:not(:hover) .sidebar-logo-subtitle {
    display: none;
}

/* ============================================
   NAV ICONS
   ============================================ */

.nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem;
    margin: 0.25rem 0;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background 0.2s ease;
    color: var(--text-dark);
    text-decoration: none;
}

.nav-item:hover {
    background: var(--sidebar-hover);
}

.nav-item.active {
    background: var(--accent-green);
    color: white;
}

.nav-icon {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.nav-label {
    font-weight: 500;
    font-size: 0.9rem;
    white-space: nowrap;
    overflow: hidden;
}

/* Hide nav labels when collapsed */
[data-testid="stSidebar"]:not(:hover) .nav-label {
    display: none;
}

/* ============================================
   INTERPRETATION BOX
   ============================================ */

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

.interpretation-box p {
    color: var(--text-dark);
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ============================================
   METRIC CARDS ROW
   ============================================ */

.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    flex: 1;
    background: var(--bg-white);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}

.metric-card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
    color: white;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.8;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}

.metric-delta {
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.metric-delta.positive { color: var(--accent-green); }
.metric-delta.negative { color: var(--accent-red); }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar - Logo and Branding
# ---------------------------------------------------------------------------

st.sidebar.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">H</div>
    <div class="sidebar-logo-text">HubbleAI</div>
    <div class="sidebar-logo-subtitle">Treasury Forecasts</div>
</div>
""", unsafe_allow_html=True)

# Version at bottom
st.sidebar.markdown("---")
st.sidebar.caption("v0.3.0")

# ---------------------------------------------------------------------------
# Main Content - Minimal Home (redirects focus to Page 1)
# ---------------------------------------------------------------------------

st.markdown("""
<h1>HubbleAI</h1>
<p class="page-subtitle">Treasury Cashflow Forecasting Platform</p>
""", unsafe_allow_html=True)

st.info("Use the sidebar to navigate to **Latest Forecast** or **Performance Dashboard**.")

# Quick navigation
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="section-title">Latest Forecast</div>
        <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
            View 8-week ML predictions with P10/P50/P90 intervals for TRR, TRP, and NET cashflows.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Forecast", key="nav_forecast"):
        st.switch_page("pages/1_Latest_Forecast.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="section-title">Performance Dashboard</div>
        <p style="color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem;">
            Compare ML vs LP accuracy, view historical performance, and explore backtest results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Performance", key="nav_perf"):
        st.switch_page("pages/2_Performance_Dashboard.py")
