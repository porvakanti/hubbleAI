"""
Data preparation module for hubbleAI.

This module contains all data loading, filtering, and preparation logic.
All I/O file access is centralized here so that later we can swap CSVs
for Denodo / Databricks / Reval without changing callers.
"""

from hubbleAI.data_prep.load_data import (
    load_actuals,
    load_liquidity_plan,
    load_fx_rates,
)
from hubbleAI.data_prep.aggregation import (
    aggregate_actuals_weekly,
    build_trp_weekly_features,
)
from hubbleAI.data_prep.fx_conversion import convert_lp_to_eur
from hubbleAI.data_prep.prepare import prepare_weekly_data

__all__ = [
    "load_actuals",
    "load_liquidity_plan",
    "load_fx_rates",
    "aggregate_actuals_weekly",
    "build_trp_weekly_features",
    "convert_lp_to_eur",
    "prepare_weekly_data",
]
