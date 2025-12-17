"""
Page 9 - Admin

System administration and data management:
- Run status and controls
- Data health monitoring
- File upload for actuals, LP, and FX data

Design: Modern, warm cream palette using shared UI components.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

# Add paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ui_components import (
    set_global_style,
    render_sidebar,
    render_metric_card,
    render_interpretation_box,
    APP_VERSION,
)
from hubbleAI.service import (
    get_data_health_summary,
    get_last_run_by_mode,
)
from hubbleAI.config import (
    DATA_RAW_DIR,
    ACTUALS_FILENAME,
    LP_FILENAME,
    FX_FILENAME,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Admin - Hubble.AI",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global styles
set_global_style()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

forward_status = get_last_run_by_mode("forward")
ref_info = None
if forward_status:
    ref_info = {
        "ref_week_start": forward_status.get("ref_week_start", "-"),
        "run_id": forward_status.get("run_id", "-"),
    }

render_sidebar(active_page="Admin", ref_info=ref_info)

# ---------------------------------------------------------------------------
# Required Columns for Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "actuals": [
        "Entity",
        "Value Date",
        "Amount Functional Currency",
        "Liquidity Group",
        "Counterpart",
        "Status",
        "ISO Country Code",
    ],
    "lp": [
        "Entity",
        "Entity Name",
        "Liquidity Group/Super Liquidity Group",
        "Year Title",
        "Item's Date",
        "Amount",
        "Currency",
        "Plan Currency",
        "Amount in plan currency",
        "Rate",
        "Comment",
    ],
    "fx": [
        "Date",
        "USD",
        "CHF",
    ],
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def validate_file_columns(df: pd.DataFrame, file_type: str) -> tuple[bool, list[str]]:
    """
    Validate that uploaded file has required columns.

    Returns:
        (is_valid, missing_columns)
    """
    required = REQUIRED_COLUMNS.get(file_type, [])
    # Case-insensitive column matching
    df_cols_lower = [c.lower().strip() for c in df.columns]
    missing = []
    for col in required:
        if col.lower().strip() not in df_cols_lower:
            missing.append(col)
    return len(missing) == 0, missing


def backup_and_save_file(uploaded_file, target_path: Path, file_type: str) -> tuple[bool, str]:
    """
    Validate, backup existing file, and save uploaded file.

    Returns:
        (success, message)
    """
    try:
        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return False, "Unsupported file format. Please upload CSV or Excel file."

        # Validate columns
        is_valid, missing = validate_file_columns(df, file_type)
        if not is_valid:
            return False, f"Missing required columns: {', '.join(missing)}"

        # Create backup if file exists
        if target_path.exists():
            backup_path = target_path.with_suffix(target_path.suffix + '.backup')
            shutil.copy2(target_path, backup_path)

        # Save new file
        df.to_csv(target_path, index=False)

        return True, f"File uploaded successfully. {len(df):,} rows."

    except Exception as e:
        return False, f"Error processing file: {str(e)}"


def get_file_info(file_path: Path) -> dict:
    """Get file information including last modified date."""
    if not file_path.exists():
        return {"exists": False}

    stat = file_path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime)
    age_days = (datetime.now() - modified).days

    return {
        "exists": True,
        "filename": file_path.name,
        "modified": modified.strftime("%Y-%m-%d %H:%M"),
        "age_days": age_days,
        "size_kb": stat.st_size / 1024,
    }


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Page Header
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem; font-size: 1.75rem; font-weight: 700; color: #2D3436;">
        Admin
    </h1>
    <p style="color: #5A6169; font-size: 0.95rem;">
        System administration and data management
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

health = get_data_health_summary()

# ---------------------------------------------------------------------------
# Run Status & Data Health Cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if forward_status:
        status_val = forward_status.get("status", "unknown").upper()
        status_date = forward_status.get("ref_week_start", "-")
        if isinstance(status_date, str) and len(status_date) > 10:
            status_date = status_date[:10]
    else:
        status_val = "NO RUNS"
        status_date = "-"

    st.markdown(render_metric_card(
        header="Run Status",
        value=status_val,
        subtitle=f"Ref: {status_date}",
        accent=status_val == "SUCCESS"
    ), unsafe_allow_html=True)

with col2:
    ready_text = "READY" if health["is_ready"] else "INCOMPLETE"
    missing_count = len(health.get("missing_inputs", []))
    subtitle = "All files present" if health["is_ready"] else f"{missing_count} file(s) missing"

    st.markdown(render_metric_card(
        header="Data Health",
        value=ready_text,
        subtitle=subtitle,
        accent=health["is_ready"]
    ), unsafe_allow_html=True)

with col3:
    backtest_status = get_last_run_by_mode("backtest")
    if backtest_status:
        bt_status_val = backtest_status.get("status", "unknown").upper()
        bt_date = backtest_status.get("ref_week_start", "-")
        if isinstance(bt_date, str) and len(bt_date) > 10:
            bt_date = bt_date[:10]
    else:
        bt_status_val = "NO RUNS"
        bt_date = "-"

    st.markdown(render_metric_card(
        header="Backtest Status",
        value=bt_status_val,
        subtitle=f"Ref: {bt_date}",
        accent=bt_status_val == "SUCCESS"
    ), unsafe_allow_html=True)

with col4:
    st.markdown(render_metric_card(
        header="Schedule",
        value="WEEKLY",
        subtitle="Every Tuesday"
    ), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Actions & Last Run Details
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="section-title">Actions</div>', unsafe_allow_html=True)

    if st.button("Run Forward Forecast", type="primary", use_container_width=True):
        try:
            from hubbleAI.pipeline import run_forecast
            with st.spinner("Running forecast... this may take a few minutes"):
                result = run_forecast(mode="forward", trigger_source="manual")
            if result.status == "success":
                st.success("Forecast completed!")
            else:
                st.error(f"Failed: {result.message}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("Run Backtest", use_container_width=True):
        try:
            from hubbleAI.pipeline import run_forecast
            with st.spinner("Running backtest..."):
                result = run_forecast(mode="backtest", trigger_source="manual")
            if result.status == "success":
                st.success("Backtest completed!")
            else:
                st.error(f"Failed: {result.message}")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.caption("Production runs are automated weekly.")

    # Data health details
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Health Details</div>', unsafe_allow_html=True)

    details = health.get("details", {})
    for name, info in details.items():
        if info.get("exists"):
            status_icon = "✓" if info.get("healthy") else "!"
            status_color = "#2E7D32" if info.get("healthy") else "#F57C00"
            age_text = f"{info.get('age_days', '?')}d ago" if info.get("age_days") is not None else ""
            st.markdown(f'''<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(0,0,0,0.06);">
                <span style="color: {status_color}; font-weight: 600; width: 16px;">{status_icon}</span>
                <span style="font-weight: 500; flex: 1;">{name.upper()}</span>
                <span style="font-size: 0.75rem; color: #5A6169;">{age_text}</span>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(0,0,0,0.06);">
                <span style="color: #D32F2F; font-weight: 600; width: 16px;">✗</span>
                <span style="font-weight: 500;">{name.upper()}</span>
            </div>''', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">Last Run Details</div>', unsafe_allow_html=True)

    if forward_status:
        run_id = forward_status.get('run_id', 'N/A')
        mode = forward_status.get('mode', 'N/A')
        trigger = forward_status.get('trigger_source', 'N/A')
        status = forward_status.get('status', 'N/A').upper()
        as_of = forward_status.get('as_of_date', 'N/A')
        created = forward_status.get('created_at', 'N/A')
        if isinstance(created, str) and len(created) > 19:
            created = created[:19].replace('T', ' ')
        message = forward_status.get('message', '')

        message_html = f'<div style="grid-column: 1 / -1; padding-top: 0.5rem; border-top: 1px solid rgba(0,0,0,0.06);"><strong>Message:</strong> {message}</div>' if message else ''

        st.markdown(f'''<div class="hubble-card hubble-card-sm">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div><strong>Run ID:</strong> <code style="font-size: 0.85rem;">{run_id}</code></div>
                <div><strong>Status:</strong> {status}</div>
                <div><strong>Mode:</strong> {mode}</div>
                <div><strong>As-of Date:</strong> {as_of}</div>
                <div><strong>Trigger:</strong> {trigger}</div>
                <div><strong>Created:</strong> {created}</div>
                {message_html}
            </div>
        </div>''', unsafe_allow_html=True)
    else:
        st.info("No forward forecast run found yet. Click 'Run Forward Forecast' to generate one.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Data Inputs / File Upload Section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-title">Data Inputs</div>', unsafe_allow_html=True)

st.markdown(render_interpretation_box(
    title="Upload Data Files",
    content="""Upload new data files from your Treasury Management System. Files will be validated for required columns before saving.
    A backup of the previous file will be kept automatically."""
), unsafe_allow_html=True)

# File upload cards
file_configs = [
    {
        "name": "Actuals",
        "type": "actuals",
        "filename": ACTUALS_FILENAME,
        "path": DATA_RAW_DIR / ACTUALS_FILENAME,
        "required_cols": REQUIRED_COLUMNS["actuals"],
    },
    {
        "name": "Liquidity Plan (LP)",
        "type": "lp",
        "filename": LP_FILENAME,
        "path": DATA_RAW_DIR / LP_FILENAME,
        "required_cols": REQUIRED_COLUMNS["lp"],
    },
    {
        "name": "FX Rates",
        "type": "fx",
        "filename": FX_FILENAME,
        "path": DATA_RAW_DIR / FX_FILENAME,
        "required_cols": REQUIRED_COLUMNS["fx"],
    },
]

for config in file_configs:
    file_info = get_file_info(config["path"])

    st.markdown(f'''<div class="hubble-card hubble-card-sm" style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
            <div>
                <div style="font-weight: 600; font-size: 1rem; color: #2D3436;">{config["name"]}</div>
                <div style="font-size: 0.8rem; color: #5A6169;">
                    {"Current: " + file_info["filename"] if file_info["exists"] else "No file found"}
                </div>
            </div>
            <div style="text-align: right;">
                {"<span style='color: #2E7D32; font-weight: 500;'>✓ Available</span>" if file_info["exists"] else "<span style='color: #D32F2F; font-weight: 500;'>✗ Missing</span>"}
            </div>
        </div>
        {"<div style='font-size: 0.75rem; color: #5A6169; margin-bottom: 0.5rem;'>Last modified: " + file_info["modified"] + " (" + str(file_info["age_days"]) + " days ago)</div>" if file_info["exists"] else ""}
        <div style="font-size: 0.7rem; color: #8B95A1;">Required columns: {", ".join(config["required_cols"][:5])}{"..." if len(config["required_cols"]) > 5 else ""}</div>
    </div>''', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        f"Upload new {config['name']} file",
        type=["csv", "xlsx", "xls"],
        key=f"upload_{config['type']}",
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Show preview
        try:
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)

            # Reset file position for later saving
            uploaded_file.seek(0)

            # Validate columns
            is_valid, missing = validate_file_columns(preview_df, config["type"])

            if is_valid:
                st.success(f"✓ File validated: {len(preview_df):,} rows, all required columns present")

                with st.expander("Preview uploaded data"):
                    st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)

                if st.button(f"Save {config['name']} File", key=f"save_{config['type']}", type="primary"):
                    uploaded_file.seek(0)
                    success, message = backup_and_save_file(uploaded_file, config["path"], config["type"])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.error(f"✗ Missing required columns: {', '.join(missing)}")
                with st.expander("Show uploaded file columns"):
                    st.write(list(preview_df.columns))

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Restore Backups Section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown('<div class="section-title">Restore Previous Files</div>', unsafe_allow_html=True)

backup_files = []
for config in file_configs:
    backup_path = config["path"].with_suffix(config["path"].suffix + '.backup')
    if backup_path.exists():
        backup_info = get_file_info(backup_path)
        backup_files.append({
            "name": config["name"],
            "path": config["path"],
            "backup_path": backup_path,
            "info": backup_info,
        })

if backup_files:
    st.markdown(render_interpretation_box(
        title="Available Backups",
        content="Click to restore the previous version of a data file."
    ), unsafe_allow_html=True)

    for backup in backup_files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{backup['name']}** backup from {backup['info']['modified']}")
        with col2:
            if st.button(f"Restore", key=f"restore_{backup['name']}", use_container_width=True):
                try:
                    shutil.copy2(backup["backup_path"], backup["path"])
                    st.success(f"{backup['name']} restored from backup!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error restoring: {str(e)}")
else:
    st.info("No backup files available. Backups are created automatically when you upload new files.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(f"Hubble.AI v{APP_VERSION}")
