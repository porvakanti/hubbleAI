"""Page 1 – Latest Forecast / Operations Overview."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# NOTE: this path assumes the Streamlit app lives under app/ and the repo root
# contains data/processed/run_status. Adjust as needed if your structure differs.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_STATUS_DIR = REPO_ROOT / "data" / "processed" / "run_status"


@dataclass
class SimpleRunStatus:
    run_id: str
    as_of_date: str
    trigger_source: str
    status: str
    created_at: str
    message: str
    missing_inputs: List[str]
    output_paths: Dict[str, str]


def load_latest_run_status() -> Optional[SimpleRunStatus]:
    """Load the latest run status written by the pipeline.

    This expects that pipeline.run_forecast has written:
    - run_status_<run_id>.json, and
    - latest_run_status.json containing the filename of the latest status.
    """
    try:
        latest_pointer = RUN_STATUS_DIR / "latest_run_status.json"
        if not latest_pointer.exists():
            return None

        latest_filename = latest_pointer.read_text(encoding="utf-8").strip()
        status_path = RUN_STATUS_DIR / latest_filename
        if not status_path.exists():
            return None

        data = json.loads(status_path.read_text(encoding="utf-8"))
        return SimpleRunStatus(
            run_id=data["run_id"],
            as_of_date=data["as_of_date"],
            trigger_source=data["trigger_source"],
            status=data["status"],
            created_at=data["created_at"],
            message=data["message"],
            missing_inputs=data.get("missing_inputs", []),
            output_paths=data.get("output_paths", {}),
        )
    except Exception:
        return None


def main():
    st.title("Latest Forecast – Operations Overview")

    status = load_latest_run_status()

    if status is None:
        st.warning("No forecast run status found yet.")
        st.info("Once the pipeline has run at least once, details will appear here.")
        return

    # Top summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Run ID", status.run_id)
    col2.metric("As-of Date", status.as_of_date)
    col3.metric("Status", status.status)
    # We don't yet know the next scheduled run programmatically, so display a placeholder.
    col4.metric("Next Scheduled Run", "Tuesday (configurable)")

    st.markdown("---")  # separator

    # Detailed status
    st.subheader("Run Details")
    st.write(status.message)
    if status.missing_inputs:
        st.error("Missing inputs: " + ", ".join(status.missing_inputs))

    # Manual trigger button – placeholder for now.
    st.markdown("### Actions")
    if st.button("Run Forecast Now"):
        st.info(
            "This button will be wired to `hubbleAI.pipeline.run_forecast` "
            "once backend integration is implemented."
        )

    st.markdown("---")  # separator

    st.markdown("### Latest Forecast Preview")
    forecasts_path = status.output_paths.get("forecasts")
    if forecasts_path:
        try:
            import pandas as pd

            df = pd.read_parquet(forecasts_path)
            # Show a small sample; filtering / charts will live on other pages.
            st.dataframe(df.head(50))
        except Exception as exc:  # display-only
            st.error(f"Could not load forecasts from {forecasts_path}: {exc}")
    else:
        st.info("No forecasts path found in run status.")


if __name__ == "__main__":
    main()
