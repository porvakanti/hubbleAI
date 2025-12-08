"""Page 2 – Performance Dashboard (ML vs LP)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# In the future this page will load precomputed metrics (WAPE, MAE, RMSE,
# direction accuracy) and visualize ML vs LP across time, LG, entity, and horizon.
# For now, we only provide a structural skeleton for Claude / developers to fill in.

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = REPO_ROOT / "data" / "processed" / "metrics"


def main():
    st.title("Performance Dashboard – ML vs LP")
    st.write(
        "This page will show historical performance of the ML model versus the "
        "Liquidity Plan, with filters for date range, entity, liquidity group, and horizon."
    )

    # Placeholder controls
    with st.expander("Filters"):
        st.date_input("Date range", [])
        st.multiselect("Liquidity Group", ["TRR", "TRP"], default=["TRR", "TRP"])
        st.text_input("Entity (optional)")
        st.multiselect("Horizons", [f"H{i}" for i in range(1, 9)], default=["H1", "H2", "H3", "H4"])

    st.info(
        "Once metrics are computed and stored under data/processed/metrics, "
        "this page will load and visualize them."
    )

    # Here we will add charts such as:
    # - WAPE over time (ML vs LP)
    # - Horizon vs error heatmap
    # - Direction accuracy over time
    # The exact plotting code will depend on the final metrics schema.


if __name__ == "__main__":
    main()
