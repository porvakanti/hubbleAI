"""
Shared UI Components for HubbleAI Streamlit App.

Provides:
- set_global_style(): Inject consistent CSS across all pages
- render_sidebar(): Render persistent navigation sidebar
- Helper functions for formatting values

Design: Modern, warm cream palette inspired by financial dashboards.
Light sidebar, green accents, clean typography.
"""

from __future__ import annotations

import streamlit as st
from typing import Optional, Dict, Any
from datetime import date
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_VERSION = "0.4.0"
APP_NAME = "HubbleAI"
APP_SUBTITLE = "Treasury Forecasts"

# WAPE threshold for near-zero denominators (in EUR)
WAPE_EPS_THRESHOLD = 500_000  # 0.5 million EUR


# ---------------------------------------------------------------------------
# Global CSS Styles
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
/* ============================================
   ROOT VARIABLES - Warm Cream Palette
   ============================================ */

:root {
    --bg-main: #F5F2ED;
    --bg-light: #FAF9F7;
    --bg-white: #FFFFFF;
    --bg-sidebar: #FAFAF8;
    --sidebar-hover: #F0EEE8;
    --sidebar-active: #E8F5E9;
    --text-dark: #2D3436;
    --text-secondary: #5A6169;
    --text-muted: #8B95A1;
    --text-light: #A0AEC0;
    --accent-green: #2E7D32;
    --accent-green-light: #4CAF50;
    --accent-green-lighter: #81C784;
    --accent-green-bg: #E8F5E9;
    --accent-orange: #F57C00;
    --accent-red: #D32F2F;
    --accent-blue: #1976D2;
    --border-light: rgba(0, 0, 0, 0.06);
    --border-medium: rgba(0, 0, 0, 0.1);
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

/* ============================================
   GLOBAL STYLES
   ============================================ */

.stApp {
    background-color: var(--bg-main);
}

.block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ============================================
   SIDEBAR - Light, Warm Design
   ============================================ */

[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-light);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 0;
}

/* Sidebar text colors */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text-dark);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

/* Native sidebar nav styling */
[data-testid="stSidebarNav"] {
    padding: 0;
}

[data-testid="stSidebarNav"] a {
    padding: 0.75rem 1rem;
    border-radius: var(--radius-sm);
    margin: 0.25rem 0.75rem;
    color: var(--text-dark) !important;
    font-weight: 500;
    transition: all 0.2s ease;
}

[data-testid="stSidebarNav"] a:hover {
    background: var(--sidebar-hover);
}

[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: var(--accent-green);
    color: white !important;
}

/* ============================================
   CARD STYLES
   ============================================ */

.hubble-card {
    background: var(--bg-white);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    margin-bottom: 1rem;
    transition: all 0.2s ease;
}

.hubble-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.hubble-card-clickable {
    cursor: pointer;
}

.hubble-card-sm {
    padding: 1rem;
    border-radius: var(--radius-md);
}

.hubble-card-flat {
    box-shadow: none;
    border: 1px solid var(--border-medium);
}

.hubble-card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
}

.hubble-card-accent .card-header,
.hubble-card-accent .card-value,
.hubble-card-accent .card-subtitle {
    color: white;
}

.card-header {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
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

.card-value-lg {
    font-size: 2.25rem;
}

.card-value-sm {
    font-size: 1.25rem;
}

.card-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* ============================================
   METRIC CARDS
   ============================================ */

.metric-card {
    background: var(--bg-white);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-dark);
}

.metric-delta {
    font-size: 0.8rem;
    font-weight: 500;
}

.metric-delta.positive { color: var(--accent-green); }
.metric-delta.negative { color: var(--accent-red); }
.metric-delta.neutral { color: var(--text-muted); }

/* ============================================
   SCORE CARDS (centered display)
   ============================================ */

.score-card {
    background: var(--bg-white);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
    text-align: center;
}

.score-card-accent {
    background: linear-gradient(135deg, var(--accent-green) 0%, var(--accent-green-light) 100%);
    color: white;
    border: none;
}

.score-value {
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.2;
}

.score-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
    opacity: 0.9;
}

.score-detail {
    font-size: 0.75rem;
    margin-top: 0.5rem;
    opacity: 0.8;
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
.text-blue { color: var(--accent-blue); }

/* ============================================
   TYPOGRAPHY
   ============================================ */

h1, h2, h3, h4 {
    color: var(--text-dark);
    font-weight: 600;
}

h1 { font-size: 1.75rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1rem; }

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-dark);
    margin-bottom: 0.25rem;
}

.page-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
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
    color: var(--accent-green);
    margin: 0 0 0.5rem 0;
    font-size: 0.9rem;
    font-weight: 600;
}

.interpretation-box p, .interpretation-box ul {
    color: var(--text-dark);
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.6;
}

.interpretation-box ul { padding-left: 1.25rem; margin-top: 0.5rem; }
.interpretation-box li { margin-bottom: 0.25rem; }

/* ============================================
   BUTTONS
   ============================================ */

.stButton > button {
    border-radius: var(--radius-sm);
    font-weight: 500;
    padding: 0.5rem 1.25rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent-green) !important;
    color: white !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: var(--accent-green-light) !important;
}

/* Sidebar nav buttons - ensure white text on primary */
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    background: var(--accent-green) !important;
    color: white !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] span,
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--text-dark) !important;
    border: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-secondary"]:hover {
    background: var(--sidebar-hover) !important;
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
    font-size: 0.85rem;
}

.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.forecast-table th {
    background: var(--bg-light);
    padding: 0.75rem;
    text-align: right;
    font-weight: 600;
    color: var(--text-secondary);
    border-bottom: 2px solid var(--border-light);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.forecast-table th:first-child { text-align: left; }

.forecast-table td {
    padding: 0.75rem;
    text-align: right;
    border-bottom: 1px solid var(--border-light);
}

.forecast-table td:first-child { text-align: left; font-weight: 500; }

.forecast-table tr:hover { background: var(--bg-light); }

/* ============================================
   TABS
   ============================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}

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

/* ============================================
   SELECTBOX & INPUTS
   ============================================ */

.stSelectbox > div > div {
    border-radius: var(--radius-sm);
    border-color: var(--border-light);
}

.stSelectbox label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ============================================
   METRICS (native Streamlit)
   ============================================ */

[data-testid="stMetric"] {
    background: var(--bg-white);
    padding: 1rem;
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-light);
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    color: var(--text-muted);
}

[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-dark);
}

/* ============================================
   EXPANDER
   ============================================ */

.streamlit-expanderHeader {
    font-weight: 500;
    color: var(--text-dark);
    background: var(--bg-light);
    border-radius: var(--radius-sm);
}

/* ============================================
   CHARTS
   ============================================ */

[data-testid="stArrowVegaLiteChart"] {
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ============================================
   ALERTS
   ============================================ */

.stAlert {
    border-radius: var(--radius-md);
    border: none;
}

/* ============================================
   SIDEBAR LOGO & BRANDING
   ============================================ */

.sidebar-brand {
    padding: 1.25rem 1rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 0.5rem;
}

.sidebar-brand-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.sidebar-logo-icon {
    width: 36px;
    height: 36px;
    background: var(--accent-green);
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
}

.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-dark);
    line-height: 1.2;
}

.sidebar-logo-subtitle {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ============================================
   SIDEBAR NAVIGATION
   ============================================ */

.sidebar-nav {
    padding: 0.5rem 0.75rem;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--text-dark);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.9rem;
}

.sidebar-nav-item:hover {
    background: var(--sidebar-hover);
}

.sidebar-nav-item.active {
    background: var(--accent-green);
    color: white;
}

.sidebar-nav-icon {
    width: 20px;
    text-align: center;
    font-size: 1rem;
}

/* ============================================
   SIDEBAR FOOTER
   ============================================ */

.sidebar-footer {
    padding: 1rem;
    border-top: 1px solid var(--border-light);
    margin-top: auto;
}

.sidebar-footer-text {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-align: center;
}

/* ============================================
   HIDE STREAMLIT BRANDING & NATIVE NAV
   ============================================ */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hide native Streamlit sidebar navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* ============================================
   NAV CARD ALIGNMENT FIX
   ============================================ */

.nav-card-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: stretch;
}

.nav-card {
    background: var(--bg-white);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-light);
    transition: all 0.2s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 200px;
}

.nav-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

.nav-card-content {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    flex: 1;
}

.nav-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.nav-card-icon.green {
    background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
    color: white;
}

.nav-card-icon.blue {
    background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%);
    color: white;
}

.nav-card-text {
    flex: 1;
}

.nav-card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin: 0 0 0.5rem 0;
}

.nav-card-desc {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
    margin: 0;
}

.nav-card-footer {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-light);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-card-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.nav-card-arrow {
    font-weight: 500;
}

/* ============================================
   HEALTH INDICATORS
   ============================================ */

.health-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-light);
}

.health-item:last-child { border-bottom: none; }

.health-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.health-dot.ok { background: var(--accent-green); }
.health-dot.warning { background: var(--accent-orange); }
.health-dot.error { background: var(--accent-red); }

/* ============================================
   WINNER BADGE
   ============================================ */

.winner-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
}

.winner-badge.ml {
    background: var(--accent-green-bg);
    color: var(--accent-green);
}

.winner-badge.lp {
    background: #FFF3E0;
    color: var(--accent-orange);
}

.winner-badge.tie {
    background: #E3F2FD;
    color: var(--accent-blue);
}

.winner-badge.na {
    background: var(--bg-light);
    color: var(--text-muted);
}
</style>
"""


# ---------------------------------------------------------------------------
# Core UI Functions
# ---------------------------------------------------------------------------


def set_global_style():
    """Inject global CSS styles into the page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar(active_page: str = "Overview", ref_info: Optional[Dict[str, Any]] = None):
    """
    Render the persistent sidebar with navigation.

    Args:
        active_page: Name of the current page ("Overview", "Latest Forecast", "Performance Dashboard")
        ref_info: Optional dict with ref_week_start, run_id, etc.
    """
    with st.sidebar:
        # Brand header with modern telescope logo
        # Minimal design: rounded square with abstract telescope/lens shape + gold accent
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-logo">
                <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="36" height="36" rx="10" fill="#1a1a1a"/>
                    <circle cx="18" cy="16" r="8" stroke="#D4AF37" stroke-width="2.5" fill="none"/>
                    <circle cx="18" cy="16" r="3" fill="#D4AF37"/>
                    <line x1="18" y1="24" x2="18" y2="31" stroke="#D4AF37" stroke-width="2.5" stroke-linecap="round"/>
                    <line x1="13" y1="29" x2="23" y2="29" stroke="#D4AF37" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <div>
                    <div class="sidebar-logo-text">{APP_NAME}</div>
                    <div class="sidebar-logo-subtitle">{APP_SUBTITLE}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation - add inline CSS for button text color fix
        st.markdown("""
        <style>
        /* Force white text on active/primary buttons in sidebar */
        section[data-testid="stSidebar"] button[kind="primary"],
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background-color: #2E7D32 !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"] p,
        section[data-testid="stSidebar"] button[kind="primary"] span,
        section[data-testid="stSidebar"] button[kind="primary"] div,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] p,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] span,
        section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] div {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

        nav_items = [
            ("Overview", "streamlit_app.py"),
            ("Latest Forecast", "pages/1_Latest_Forecast.py"),
            ("Performance Dashboard", "pages/2_Performance_Dashboard.py"),
        ]

        for label, page_path in nav_items:
            is_active = label == active_page
            if st.button(
                label,
                key=f"nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    st.switch_page(page_path)

        st.markdown('</div>', unsafe_allow_html=True)

        # Reference info (if available)
        if ref_info:
            st.markdown("---")
            ref_week = ref_info.get("ref_week_start", "-")
            run_id = ref_info.get("run_id", "-")
            if isinstance(ref_week, date):
                ref_week = ref_week.isoformat()
            if len(str(run_id)) > 12:
                run_id = str(run_id)[:12] + "..."
            st.caption(f"Data: {ref_week}")
            st.caption(f"Run: {run_id}")

        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div class="sidebar-footer-text">
            v{APP_VERSION}<br>
            HubbleAI Treasury Platform
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------


def format_millions(val: float, decimals: int = 2, suffix: str = "M") -> str:
    """Format value in millions with specified decimals."""
    if pd.isna(val):
        return "-"
    return f"{val / 1e6:.{decimals}f}{suffix}"


def format_currency_millions(val: float, decimals: int = 2) -> str:
    """Format value as currency in millions EUR."""
    if pd.isna(val):
        return "-"
    sign = "+" if val > 0 else ""
    return f"{sign}{val / 1e6:.{decimals}f}M EUR"


def format_pct(val: float, decimals: int = 1) -> str:
    """Format value as percentage (val is a fraction, e.g., 0.12 -> 12.0%)."""
    if pd.isna(val):
        return "-"
    return f"{val * 100:.{decimals}f}%"


def format_pct_direct(val: float, decimals: int = 1) -> str:
    """Format value that's already a percentage (e.g., 12.5 -> 12.5%)."""
    if pd.isna(val):
        return "-"
    return f"{val:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Card Renderers
# ---------------------------------------------------------------------------


def render_metric_card(
    header: str,
    value: str,
    subtitle: str = "",
    accent: bool = False,
    clickable: bool = False
) -> str:
    """Render a metric/KPI card."""
    classes = ["hubble-card"]
    if accent:
        classes.append("hubble-card-accent")
    if clickable:
        classes.append("hubble-card-clickable")

    class_str = " ".join(classes)

    return f"""
    <div class="{class_str}" style="height: 100%;">
        <div class="card-header">{header}</div>
        <div class="card-value">{value}</div>
        <div class="card-subtitle">{subtitle}</div>
    </div>
    """


def render_score_card(
    value: str,
    label: str,
    detail: str = "",
    accent: bool = False
) -> str:
    """Render a centered score card."""
    card_class = "score-card score-card-accent" if accent else "score-card"
    return f"""
    <div class="{card_class}">
        <div class="score-value">{value}</div>
        <div class="score-label">{label}</div>
        <div class="score-detail">{detail}</div>
    </div>
    """


def render_interpretation_box(title: str, content: str) -> str:
    """Render an interpretation/guidance box."""
    return f"""
    <div class="interpretation-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """


def render_winner_badge(winner: str) -> str:
    """Render a winner badge (ML, LP, Tie, or NA)."""
    winner_lower = winner.lower() if winner else "na"
    if winner_lower in ["ml", "lp", "tie"]:
        badge_class = winner_lower
    else:
        badge_class = "na"
    return f'<span class="winner-badge {badge_class}">{winner}</span>'


# ---------------------------------------------------------------------------
# WAPE Computation Helpers
# ---------------------------------------------------------------------------


def compute_wape_safe(actual: float, predicted: float, eps: float = WAPE_EPS_THRESHOLD) -> tuple[float, bool]:
    """
    Compute WAPE with near-zero denominator handling.

    Args:
        actual: Actual value
        predicted: Predicted value
        eps: Threshold below which denominator is considered too small

    Returns:
        (wape, is_undefined): WAPE value and whether it's undefined due to small denominator
    """
    if pd.isna(actual) or pd.isna(predicted):
        return float('nan'), True

    abs_actual = abs(actual)
    if abs_actual < eps:
        return float('nan'), True

    wape = abs(actual - predicted) / abs_actual
    return wape, False


def determine_winner(ml_wape: float, lp_wape: float) -> str:
    """
    Determine winner between ML and LP based on WAPE.

    Returns: "ML", "LP", "Tie", or "N/A"
    """
    if pd.isna(ml_wape) or pd.isna(lp_wape):
        if pd.isna(lp_wape) and not pd.isna(ml_wape):
            return "ML"  # LP not available
        return "N/A"

    if abs(ml_wape - lp_wape) < 0.001:  # Within 0.1% is a tie
        return "Tie"
    elif ml_wape < lp_wape:
        return "ML"
    else:
        return "LP"


# ---------------------------------------------------------------------------
# Data Display Helpers
# ---------------------------------------------------------------------------


def create_forecast_table_html(df: pd.DataFrame, columns: list[str]) -> str:
    """Create an HTML table for forecast display."""
    if df.empty:
        return "<p>No data available</p>"

    # Build table HTML
    html = '<table class="forecast-table">'

    # Header
    html += '<thead><tr>'
    for col in columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead>'

    # Body
    html += '<tbody>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in columns:
            val = row.get(col, "")
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody>'

    html += '</table>'
    return html
