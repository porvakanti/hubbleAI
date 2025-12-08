"""
Main entry point for the hubbleAI Streamlit app.

Run with:
    streamlit run app/streamlit_app.py

This file sets up global config; the actual pages live under app/pages/.
"""

import streamlit as st

st.set_page_config(
    page_title="hubbleAI – Treasury Cashflow Forecasts",
    layout="wide",
)

st.sidebar.title("hubbleAI")
st.sidebar.caption("Treasury Cashflow Forecasting")

st.write("# Welcome to hubbleAI")
st.write(
    "Use the left navigation to view the latest forecasts, performance, "
    "scenarios, and insights."
)
