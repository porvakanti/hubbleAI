"""
Page 2 – Performance Dashboard (ML vs LP vs Hybrid)

Design inspired by modern card-based dashboard UI with:
- Top summary cards with win rates
- WAPE over time line chart
- Win rate by horizon visualization
- Weekly breakdown table
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hubbleAI.services import (
    load_latest_backtest_metrics,
    get_hybrid_performance_summary,
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Light background */
.stApp {
    background-color: #f5f3ef;
}

/* Card styling */
.perf-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}

.perf-card-green {
    background: linear-gradient(135deg, #4a5d4a 0%, #5c6b5c 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
}

.win-rate-large {
    font-size: 48px;
    font-weight: 700;
    color: #4a5d4a;
}

.win-rate-label {
    font-size: 14px;
    color: #666;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


def render_win_rate_card(horizon: int, wins: int, total: int, win_rate: float):
    """Render a win rate card for a horizon."""
    pct = win_rate * 100
    color = "#4a5d4a" if pct >= 60 else "#c9a227" if pct >= 50 else "#c94a4a"
    return f"""
    <div class="perf-card" style="text-align: center;">
        <div style="font-size: 14px; color: #666;">Horizon {horizon}</div>
        <div style="font-size: 36px; font-weight: 700; color: {color};">{pct:.0f}%</div>
        <div style="font-size: 12px; color: #888;">{wins}/{total} weeks</div>
    </div>
    """


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("## 📈 Performance Dashboard")
    st.markdown("ML vs LP vs Hybrid Comparison")

    st.markdown("---")

    # Load metrics
    metrics = load_latest_backtest_metrics()
    performance = get_hybrid_performance_summary()

    if not metrics.success:
        st.warning("No backtest metrics available.")
        st.info("Run backtest first: `run_forecast(mode='backtest')`")
        if metrics.error:
            st.error(f"Error: {metrics.error}")
        return

    # Top Summary Row
    st.markdown("### 🏆 Hybrid Win Rates vs LP (TRP)")

    if performance is not None and not performance.empty:
        cols = st.columns(4)
        for i, (_, row) in enumerate(performance.iterrows()):
            with cols[i]:
                st.markdown(
                    render_win_rate_card(
                        horizon=int(row['horizon']),
                        wins=int(row['weekly_wins_vs_lp']),
                        total=int(row['total_weeks']),
                        win_rate=row['win_rate_vs_lp'],
                    ),
                    unsafe_allow_html=True
                )

        # Average win rate
        avg_rate = performance['win_rate_vs_lp'].mean() * 100
        st.success(f"**Average Win Rate: {avg_rate:.0f}%** — Hybrid beats LP in {avg_rate:.0f}% of weeks on average")
    else:
        st.info("No TRP performance data available.")

    st.markdown("---")

    # Main Charts Section
    chart_col, table_col = st.columns([3, 2])

    with chart_col:
        st.markdown("### 📊 WAPE Over Time")

        if metrics.weekly_breakdown is not None and not metrics.weekly_breakdown.empty:
            # Filters
            filter_cols = st.columns(2)
            with filter_cols[0]:
                horizons = sorted(metrics.weekly_breakdown['horizon'].unique())
                selected_horizon = st.selectbox(
                    "Select Horizon",
                    horizons,
                    format_func=lambda x: f"Horizon {x}"
                )

            # Filter data
            df_chart = metrics.weekly_breakdown[
                metrics.weekly_breakdown['horizon'] == selected_horizon
            ].copy()
            df_chart = df_chart.sort_values('week_start')

            if not df_chart.empty:
                # Prepare chart data
                chart_data = pd.DataFrame({
                    'Week': df_chart['week_start'],
                    'LP WAPE': df_chart['lp_wape'] * 100,
                    'ML WAPE': df_chart['ml_wape'] * 100,
                    'Hybrid WAPE': df_chart['hybrid_wape'] * 100,
                })
                chart_data = chart_data.set_index('Week')

                # Line chart
                st.line_chart(
                    chart_data,
                    color=["#999999", "#4a7c9c", "#4a5d4a"],
                    height=300,
                )

                # Win summary for selected horizon
                hybrid_wins = df_chart['hybrid_wins'].sum()
                total_weeks = len(df_chart)
                win_pct = (hybrid_wins / total_weeks) * 100 if total_weeks > 0 else 0

                st.markdown(f"""
                **Horizon {selected_horizon} Summary:**
                - Hybrid wins: **{hybrid_wins}/{total_weeks} weeks ({win_pct:.0f}%)**
                - Alpha used: **{df_chart['alpha'].iloc[0]:.1f}**
                """)
            else:
                st.info("No data for selected horizon.")
        else:
            st.info("Weekly breakdown not available.")

    with table_col:
        st.markdown("### 📋 Weekly Breakdown")

        if metrics.weekly_breakdown is not None and not metrics.weekly_breakdown.empty:
            # Filter to selected horizon
            df_table = metrics.weekly_breakdown[
                metrics.weekly_breakdown['horizon'] == selected_horizon
            ].copy()
            df_table = df_table.sort_values('week_start', ascending=False)

            if not df_table.empty:
                # Format for display
                display_df = pd.DataFrame({
                    'Week': df_table['week_start'].dt.strftime('%Y-%m-%d'),
                    'LP': (df_table['lp_wape'] * 100).round(1).astype(str) + '%',
                    'ML': (df_table['ml_wape'] * 100).round(1).astype(str) + '%',
                    'Hybrid': (df_table['hybrid_wape'] * 100).round(1).astype(str) + '%',
                    'Winner': df_table['hybrid_wins'].map({True: '✅ Hybrid', False: '❌ LP'}),
                })

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=350,
                    hide_index=True,
                )
            else:
                st.info("No data for selected horizon.")
        else:
            st.info("Weekly breakdown not available.")

    st.markdown("---")

    # Comparison Section
    st.markdown("### 🔬 Detailed Metrics Comparison")

    if metrics.alpha_table is not None and not metrics.alpha_table.empty:
        # Show full alpha table with metrics
        alpha_display = metrics.alpha_table.copy()

        # Format columns
        alpha_display['Win Rate'] = (alpha_display['win_rate_vs_lp'] * 100).round(1).astype(str) + '%'
        alpha_display['Avg ML WAPE'] = (alpha_display['avg_wape_ml'] * 100).round(2).astype(str) + '%'
        alpha_display['Avg LP WAPE'] = (alpha_display['avg_wape_lp'] * 100).round(2).astype(str) + '%'
        alpha_display['Avg Hybrid WAPE'] = (alpha_display['avg_wape_hybrid'] * 100).round(2).astype(str) + '%'

        display_cols = ['liquidity_group', 'horizon', 'alpha', 'weekly_wins_vs_lp',
                        'total_weeks', 'Win Rate', 'Avg ML WAPE', 'Avg LP WAPE', 'Avg Hybrid WAPE']
        display_cols = [c for c in display_cols if c in alpha_display.columns]

        st.dataframe(
            alpha_display[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # Insights
        st.markdown("#### 💡 Key Insights")

        trp_data = alpha_display[alpha_display['liquidity_group'] == 'TRP']
        if not trp_data.empty:
            best_horizon = trp_data.loc[trp_data['win_rate_vs_lp'].idxmax()]
            worst_horizon = trp_data.loc[trp_data['win_rate_vs_lp'].idxmin()]

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"""
                **Best Performing Horizon:** H{int(best_horizon['horizon'])}
                - Win Rate: {best_horizon['win_rate_vs_lp']*100:.0f}%
                - Alpha: {best_horizon['alpha']}
                """)
            with col2:
                st.warning(f"""
                **Needs Improvement:** H{int(worst_horizon['horizon'])}
                - Win Rate: {worst_horizon['win_rate_vs_lp']*100:.0f}%
                - Alpha: {worst_horizon['alpha']}
                """)

    else:
        st.info("Alpha table not available.")

    # Footer with data source info
    st.markdown("---")
    if metrics.ref_date:
        st.caption(f"📅 Data from backtest run: {metrics.ref_date}")


if __name__ == "__main__":
    main()
