"""
Page 2 - Performance Dashboard

Shows backtest-based performance comparison:
- ML vs LP vs Hybrid accuracy
- Win rates by horizon
- Quantile coverage (calibration)
- Drilldown by liquidity group

Design: Modern card-based dashboard with polished styling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.service import (
    load_latest_backtest_results,
    get_hybrid_performance_summary,
)

# ---------------------------------------------------------------------------
# Page CSS
# ---------------------------------------------------------------------------

PAGE_CSS = """
<style>
/* Main background */
.stApp {
    background-color: #FAF9F6;
}

/* Card styling */
.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    border: 1px solid rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}

.card-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #95A5A6;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.card-value {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1;
}

.card-subtitle {
    font-size: 0.8rem;
    color: #95A5A6;
    margin-top: 0.5rem;
}

/* Win rate colors */
.win-high { color: #27AE60; }
.win-medium { color: #F39C12; }
.win-low { color: #E74C3C; }

/* Section title */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Winner badge */
.winner-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
.winner-hybrid {
    background: #D5F5E3;
    color: #27AE60;
}
.winner-lp {
    background: #FADBD8;
    color: #E74C3C;
}
.winner-ml {
    background: #D6EAF8;
    color: #2980B9;
}

/* Tier toggle */
.tier-toggle {
    background: #F0F0F0;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


def get_win_color(rate: float) -> str:
    """Get CSS class for win rate."""
    if rate >= 0.65:
        return "win-high"
    elif rate >= 0.50:
        return "win-medium"
    return "win-low"


def render_win_card(horizon: int, wins: int, total: int, win_rate: float) -> str:
    """Render a win rate card."""
    pct = win_rate * 100
    color_class = get_win_color(win_rate)
    return f"""
    <div class="card" style="text-align: center;">
        <div class="card-header">Horizon {horizon}</div>
        <div class="card-value {color_class}">{pct:.0f}%</div>
        <div class="card-subtitle">{wins}/{total} weeks</div>
    </div>
    """


def main():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="margin-bottom: 0.25rem; color: #2C3E50;">Performance Dashboard</h2>
        <p style="color: #95A5A6;">Backtest comparison: ML vs LP vs Hybrid</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Load Data
    # ---------------------------------------------------------------------------

    backtest = load_latest_backtest_results()

    if backtest is None:
        st.warning("No backtest results available.")
        st.info("Run a backtest first: `run_forecast(mode='backtest')`")
        return

    metrics = backtest.metrics
    diagnostics = backtest.diagnostics

    # ---------------------------------------------------------------------------
    # Tier Toggle (Clean vs Full)
    # ---------------------------------------------------------------------------

    use_clean = st.checkbox(
        "Show Tier-1 only (exclude passthrough entities)",
        value=True,
        help="Clean metrics exclude Tier-2 entities that use LP passthrough instead of ML predictions."
    )

    metrics_net_key = "metrics_net_clean" if use_clean else "metrics_net"
    metrics_lg_key = "metrics_by_lg_clean" if use_clean else "metrics_by_lg"

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 1: Hybrid Win Rates (TRP H1-4)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Hybrid Win Rates vs LP (TRP)</div>', unsafe_allow_html=True)

    alpha_table = metrics.get("alpha_by_lg_horizon")

    if alpha_table is not None and not alpha_table.empty:
        # Filter to TRP horizons 1-4
        trp_data = alpha_table[
            (alpha_table["liquidity_group"] == "TRP") &
            (alpha_table["horizon"] <= 4)
        ].copy()

        if not trp_data.empty:
            # Win rate cards
            cols = st.columns(4)
            for i, (_, row) in enumerate(trp_data.iterrows()):
                with cols[i]:
                    st.markdown(
                        render_win_card(
                            horizon=int(row["horizon"]),
                            wins=int(row["weekly_wins_vs_lp"]),
                            total=int(row["total_weeks"]),
                            win_rate=row["win_rate_vs_lp"],
                        ),
                        unsafe_allow_html=True
                    )

            # Summary insight
            avg_win = trp_data["win_rate_vs_lp"].mean() * 100
            best_row = trp_data.loc[trp_data["win_rate_vs_lp"].idxmax()]
            st.success(
                f"**Average Win Rate: {avg_win:.0f}%** | "
                f"Best: H{int(best_row['horizon'])} ({best_row['win_rate_vs_lp']*100:.0f}%)"
            )
        else:
            st.info("No TRP data available in alpha table.")
    else:
        st.warning("Alpha tuning table not available.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 2: Net WAPE by Horizon (ML vs LP vs Hybrid)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Net Accuracy by Horizon (TRR + TRP)</div>', unsafe_allow_html=True)

    metrics_net = metrics.get(metrics_net_key)
    weekly_hybrid = metrics.get("weekly_hybrid_breakdown")

    if metrics_net is not None and not metrics_net.empty:
        # Aggregate by horizon
        net_by_h = (
            metrics_net
            .groupby("horizon", as_index=False)
            .agg({
                "ml_wape": "mean",
                "lp_wape": "mean",
            })
        )

        # Add hybrid from weekly breakdown if available
        if weekly_hybrid is not None and not weekly_hybrid.empty:
            hybrid_by_h = (
                weekly_hybrid
                .groupby("horizon", as_index=False)
                .agg({"hybrid_wape": "mean"})
            )
            net_by_h = net_by_h.merge(hybrid_by_h, on="horizon", how="left")

        # Display table
        display_df = net_by_h.copy()
        display_df.columns = ["Horizon"] + [c.replace("_", " ").title() for c in display_df.columns[1:]]

        # Format as percentages
        for col in display_df.columns[1:]:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart
        chart_df = net_by_h.set_index("horizon")
        chart_cols = [c for c in ["ml_wape", "lp_wape", "hybrid_wape"] if c in chart_df.columns]

        if chart_cols:
            st.bar_chart(chart_df[chart_cols] * 100, height=300)
            st.caption("Lower WAPE = Better accuracy")
    else:
        st.warning(f"Net metrics ({metrics_net_key}) not available.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 3: Quantile Forecast Calibration
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Quantile Forecast Calibration</div>', unsafe_allow_html=True)

    # Use quantile_coverage_by_horizon (NOT metrics_horizon_profiles!)
    quant_coverage = diagnostics.get("quantile_coverage_by_horizon")

    if quant_coverage is not None and not quant_coverage.empty:
        coverage_cols = ["horizon", "n", "prob_below_p10", "prob_between_p10_p90", "prob_above_p90"]
        coverage_cols = [c for c in coverage_cols if c in quant_coverage.columns]

        display_quant = quant_coverage[coverage_cols].copy()

        # Format percentages
        for col in display_quant.columns:
            if col.startswith("prob_"):
                display_quant[col] = display_quant[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

        display_quant.columns = ["Horizon", "N", "Below P10", "Between P10-P90", "Above P90"]
        st.dataframe(display_quant, use_container_width=True, hide_index=True)

        # Line chart for coverage
        chart_quant = quant_coverage[["horizon", "prob_below_p10", "prob_between_p10_p90", "prob_above_p90"]].copy()
        chart_quant = chart_quant.set_index("horizon")
        st.line_chart(chart_quant * 100, height=250)

        st.caption(
            "**Ideal calibration:** ~10% below P10, ~80% between P10-P90, ~10% above P90. "
            "Deviations indicate intervals are too narrow or wide."
        )
    else:
        st.info("Quantile coverage diagnostics not available.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 4: Drilldown by Liquidity Group
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Drilldown by Liquidity Group</div>', unsafe_allow_html=True)

    metrics_lg = metrics.get(metrics_lg_key)

    if metrics_lg is not None and not metrics_lg.empty:
        col1, col2 = st.columns(2)

        with col1:
            lg_options = sorted(metrics_lg["liquidity_group"].unique())
            selected_lg = st.selectbox("Liquidity Group", lg_options)

        with col2:
            horizon_options = sorted(metrics_lg["horizon"].unique())
            selected_h = st.selectbox("Horizon", horizon_options, format_func=lambda x: f"H{x}")

        # Filter data
        lg_subset = metrics_lg[
            (metrics_lg["liquidity_group"] == selected_lg) &
            (metrics_lg["horizon"] == selected_h)
        ].copy()

        # Try to add hybrid from weekly breakdown
        if weekly_hybrid is not None and not weekly_hybrid.empty:
            hybrid_subset = weekly_hybrid[
                (weekly_hybrid["liquidity_group"] == selected_lg) &
                (weekly_hybrid["horizon"] == selected_h)
            ][["week_start", "hybrid_wape"]].copy()

            if not hybrid_subset.empty:
                lg_subset = lg_subset.merge(hybrid_subset, on="week_start", how="left")

        st.markdown(f"**Weekly WAPE for {selected_lg}, H{selected_h}:**")

        if not lg_subset.empty:
            # Display table
            display_cols = ["week_start", "ml_wape", "lp_wape"]
            if "hybrid_wape" in lg_subset.columns:
                display_cols.append("hybrid_wape")

            display_lg = lg_subset[display_cols].copy()
            display_lg = display_lg.sort_values("week_start", ascending=False)

            # Format
            display_lg["week_start"] = pd.to_datetime(display_lg["week_start"]).dt.strftime("%Y-%m-%d")
            for col in ["ml_wape", "lp_wape", "hybrid_wape"]:
                if col in display_lg.columns:
                    display_lg[col] = display_lg[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")

            display_lg.columns = ["Week", "ML WAPE", "LP WAPE"] + (["Hybrid WAPE"] if "hybrid_wape" in lg_subset.columns else [])

            st.dataframe(display_lg, use_container_width=True, height=300, hide_index=True)

            # Line chart
            chart_lg = lg_subset.copy()
            chart_lg["week_start"] = pd.to_datetime(chart_lg["week_start"])
            chart_lg = chart_lg.sort_values("week_start")
            chart_lg = chart_lg.set_index("week_start")

            chart_cols_lg = [c for c in ["ml_wape", "lp_wape", "hybrid_wape"] if c in chart_lg.columns]
            if chart_cols_lg:
                st.line_chart(chart_lg[chart_cols_lg] * 100, height=250)
        else:
            st.info("No data for selected combination.")
    else:
        st.warning(f"LG-level metrics ({metrics_lg_key}) not available.")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Section 5: Weekly Hybrid Breakdown (TRP H1-4)
    # ---------------------------------------------------------------------------

    st.markdown('<div class="section-title">Weekly Hybrid Breakdown</div>', unsafe_allow_html=True)

    if weekly_hybrid is not None and not weekly_hybrid.empty:
        # Add winner column
        weekly_display = weekly_hybrid.copy()

        def determine_winner(row):
            if pd.isna(row.get("hybrid_wape")) or pd.isna(row.get("lp_wape")):
                return "-"
            if row["hybrid_wape"] < row["lp_wape"] and row["hybrid_wape"] < row.get("ml_wape", float("inf")):
                return "Hybrid"
            elif row["ml_wape"] < row["lp_wape"]:
                return "ML"
            return "LP"

        weekly_display["winner"] = weekly_display.apply(determine_winner, axis=1)

        # Format for display
        weekly_display["week_start"] = pd.to_datetime(weekly_display["week_start"]).dt.strftime("%Y-%m-%d")
        for col in ["ml_wape", "lp_wape", "hybrid_wape"]:
            if col in weekly_display.columns:
                weekly_display[col] = weekly_display[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")

        display_cols_weekly = [
            "liquidity_group", "horizon", "week_start",
            "ml_wape", "lp_wape", "hybrid_wape", "winner"
        ]
        display_cols_weekly = [c for c in display_cols_weekly if c in weekly_display.columns]

        weekly_display = weekly_display[display_cols_weekly]
        weekly_display.columns = ["LG", "H", "Week", "ML WAPE", "LP WAPE", "Hybrid WAPE", "Winner"]

        st.dataframe(
            weekly_display.sort_values(["LG", "H", "Week"], ascending=[True, True, False]),
            use_container_width=True,
            height=400,
            hide_index=True,
        )

        # Summary stats
        if "winner" in weekly_display.columns:
            winner_counts = weekly_display["Winner"].value_counts()
            st.markdown(f"""
            **Winner Distribution:**
            Hybrid: {winner_counts.get('Hybrid', 0)} |
            ML: {winner_counts.get('ML', 0)} |
            LP: {winner_counts.get('LP', 0)}
            """)
    else:
        st.info("Weekly hybrid breakdown not available.")

    # Footer
    st.markdown("---")
    st.caption(f"Backtest reference: {backtest.ref_week_start}")


if __name__ == "__main__":
    main()
