"""Page 4 – Insights & Conversational Assistant (future)."""

from __future__ import annotations

import streamlit as st


def main():
    st.title("Insights & Assistant")
    st.write(
        "This page is reserved for more agentic / conversational features, such as "
        "explaining forecasts, highlighting anomalies, and answering natural-language "
        "questions about model performance and drivers."
    )

    st.info(
        "For the first version, this page can host pre-defined analyses and simple "
        "textual summaries. Later, it can be upgraded to use LLMs (e.g. Azure OpenAI) "
        "over forecasts, metrics, and SHAP outputs."
    )

    # Basic placeholder UI
    st.markdown("### Example questions (to be implemented)")
    st.markdown("- How has the model performed vs LP in the last 12 weeks for TRR?")
    st.markdown("- Why is this week's forecast for a given entity very different from LP?")
    st.markdown("- Which entities show the largest performance improvements over LP?")


if __name__ == "__main__":
    main()
