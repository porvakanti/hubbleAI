"""
HubbleAI - Treasury Cashflow Forecasting

Main entry point for the Streamlit app.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="HubbleAI – Treasury Cashflow Forecasts",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for consistent styling
CUSTOM_CSS = """
<style>
/* Light background */
.stApp {
    background-color: #f5f3ef;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 14px;
}

/* Hide default hamburger menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Logo area */
.logo-container {
    padding: 20px;
    text-align: center;
    border-bottom: 1px solid #eee;
    margin-bottom: 20px;
}

.logo-text {
    font-size: 24px;
    font-weight: 700;
    color: #4a5d4a;
    letter-spacing: -0.5px;
}

.logo-subtitle {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-text">🔭 HubbleAI</div>
    <div class="logo-subtitle">Treasury Forecasting</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Navigate using the pages above")

# Main content for home page
st.markdown("""
## 🔭 Welcome to HubbleAI

**Treasury Cashflow Forecasting for Aperam**

HubbleAI uses machine learning to forecast treasury cashflows, combining:
- **ML Models** (LightGBM) for pattern recognition
- **LP Baseline** for established benchmarks
- **Hybrid Approach** that beats LP in 57-80% of weeks

---

### 📚 Quick Navigation

| Page | Description |
|------|-------------|
| **Overview** | Latest forecast status, data health, quick actions |
| **Performance** | ML vs LP comparison, win rates, WAPE charts |
| **Scenarios** | What-if analysis (coming soon) |
| **Insights** | AI-powered Q&A (coming soon) |

---

### 🎯 Current Performance (TRP Horizons)

| Horizon | Hybrid Win Rate |
|---------|-----------------|
| H1 | 57% |
| H2 | 74% |
| H3 | 69% |
| H4 | 80% |

*Hybrid model beats LP in most weeks across all horizons.*

---

### 🚀 Getting Started

1. Check **Overview** page for latest run status
2. View **Performance** dashboard for detailed metrics
3. Use filters to drill down by horizon, entity, or date

""")

# Footer
st.markdown("---")
st.caption("HubbleAI v0.1 | Built with Streamlit")
