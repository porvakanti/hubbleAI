"""Page 3 – Scenario Planner (LP vs ML vs Hybrid)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
FORECASTS_DIR = REPO_ROOT / "data" / "processed" / "forecasts"


def main():
    st.title("Scenario Planner – LP vs ML vs Hybrid")
    st.write(
        "This page will simulate how decisions / positions might differ under "
        "different forecasting strategies (LP-only, ML-only, hybrid)."
    )

    st.markdown("### Scenario Configuration")
    strategy = st.selectbox(
        "Strategy",
        ["LP only", "ML only", "Hybrid (TRR=ML, TRP=LP)"],
    )
    st.date_input("Historical period for simulation", [])

    st.info(
        "Once historical forecasts and actuals are available in a consistent schema, "
        "this page will use them to compute scenario comparisons (e.g. error, bias, "
        "indicative impact on buffers)."
    )

    # Placeholder for future charts / tables
    st.placeholder()


if __name__ == "__main__":
    main()
