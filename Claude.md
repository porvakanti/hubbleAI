
# Claude Instructions for hubbleAI

You are helping build **hubbleAI** – a Treasury Cashflow Forecasting application for Aperam.

This file is the **source of truth** for how you should work in this repo.  
Always read it before proposing or applying changes.

---

## 1. Repository Overview

Target structure (can be adjusted slightly as we evolve, but keep the spirit):

```text
hubbleAI/
├─ data/
│  ├─ raw/           # Original CSV files (actuals, LP, FX, etc.)
│  └─ processed/     # Curated / intermediate datasets
├─ src/
│  ├─ data_prep/
│  │  ├─ load_data.py        # Load actuals, LP, FX
│  │  ├─ fx_conversion.py    # EUR conversion (if needed)
│  │  └─ aggregation.py      # Weekly aggregation
│  ├─ features/
│  │  ├─ lag_features.py
│  │  ├─ rolling_features.py
│  │  ├─ calendar_features.py
│  │  └─ lp_features.py      # Horizon-specific LP injection
│  ├─ models/
│  │  ├─ base_model.py       # Abstract base class / interface
│  │  ├─ lightgbm_model.py   # MAIN model (current focus)
│  │  └─ xgboost_model.py    # FUTURE: optional
│  ├─ evaluation/
│  │  ├─ metrics.py          # WAPE, MAE, RMSE, direction accuracy
│  │  └─ reporting.py        # Aggregated reports (LG, Entity+LG, TRR+TRP)
│  ├─ tuning/
│  │  └─ hyperparameter_search.py  # Per LG + horizon tuning
│  └─ pipeline.py            # End-to-end orchestration (training + inference)
├─ app/
│  ├─ streamlit_app.py       # Treasury UI (required)
│  └─ pages/
│     ├─ overview.py         # Latest forecast status
│     ├─ performance.py      # Metrics & dashboard
│     └─ scenarios.py        # Scenario planner & insights (optional / later)
├─ notebooks/
│  └─ TCF_V2.ipynb           # MAIN notebook for exploration (do not break)
├─ config/
│  └─ model_config.yaml      # Hyperparameters, feature lists, entity sets
├─ requirements.txt
└─ Claude.md                 # This file
```

Important:

- The Python package name is **`hubbleAI`** (camel case), not `hubble_ai`.
- The **truth for current modeling logic is `notebooks/TCF_V2.ipynb`**.  
  Do **not** ignore or overwrite it.

---

## 2. Business & Modeling Rules (must NOT be changed silently)

### 2.1 Entities: Tier‑1 vs Tier‑2

- There are Tier‑1 and Tier‑2 entities.
- Tier‑2 entities are **excluded from ML forecasting** due to poor/unreliable history.
- For Tier‑2, the final forecast simply **passes through the Liquidity Plan (LP) values**.
- For horizons **5–8**, there is **no LP**, so Tier‑2 **can have no forecast** for those horizons. This is acceptable.

The list of Tier‑2 entities / (Entity, Liquidity Group) combinations should be read from the existing notebook or config, not hard‑coded in multiple places.

### 2.2 Forecast Horizon & Strategy

- Horizon: **1 to 8 weeks ahead**.
- We **do NOT use a recursive strategy**.
- Instead, we train **separate models for each (Liquidity Group × Horizon) combination**  
  (e.g. `TRR_H1`, `TRR_H2`, `TRP_H1`, …).
- Do not switch to recursive or multi‑horizon models unless explicitly requested.

### 2.3 Models / Algorithms

- Primary model: **LightGBM**.
- Secondary model: **XGBoost**, after the core LightGBM pipeline works.
- LSTM, SARIMAX and other model types are **future ideas** – do **not** add them unless explicitly requested.

### 2.4 Liquidity Plan (LP) Features – horizon‑specific

- LP has columns: `W1_Forecast`, `W2_Forecast`, `W3_Forecast`, `W4_Forecast`.
- Use exactly one LP column per horizon:

  | Horizon | LP feature to include |
  |--------:|-----------------------|
  | 1       | `W1_Forecast`         |
  | 2       | `W2_Forecast`         |
  | 3       | `W3_Forecast`         |
  | 4       | `W4_Forecast`         |
  | 5–8     | **no LP feature**     |

- Base `feature_cols` should **not** contain any LP columns.
- LP feature(s) must be injected **dynamically** per horizon, via a helper in `lp_features.py`
  (e.g. `get_feature_cols_for_horizon(horizon, base_feature_cols)`).
- Ensure all training / inference code paths use this helper so that we never accidentally pass all four LP columns together.

### 2.5 Prediction Types – Point + Quantiles

For each (Entity, Liquidity Group, Week, Horizon) we ultimately want **four predictions**:

- **Point prediction** (standard regression model output)
- **P10**
- **P50**
- **P90**

Key points:

- The **point prediction is not automatically equal to P50** – do **not** assume this.
- Acceptable implementation strategies include:
  - Point model (e.g. standard LGBM) plus separate LightGBM quantile models for P10, P50, P90.
  - Or a unified quantile approach where point prediction is one of the quantiles.
- Before making a big architectural choice here, **propose an approach and wait for approval**.

The output schema for forecasts should include at least:

- `entity`
- `liquidity_group`
- `week_start`
- `target_week_start`
- `horizon`
- `actual_value` (Backtest mode: actual observed weekly amount for the target_week_start; Forward mode: NaN placeholder (future actual unknown))
- `y_pred_point`
- `y_pred_p10`
- `y_pred_p50`
- `y_pred_p90`
- `model_name` / `model_type`
- `is_pass_through` (for Tier‑2)

### 2.6 Weekly Aggregation & Forecast Modes

#### Monday-Based Weeks

All weekly aggregation in hubbleAI uses **Monday-based weeks**:
- `week_start` is ALWAYS the Monday of that week
- `target_week_start` is ALWAYS the Monday of the target week (`week_start + 7 * horizon` days)

This ensures consistency across actuals, LP data, and all forecast outputs.

#### Forecast Modes

The `run_forecast()` function supports two modes:

**Forward Mode** (`mode="forward"`):
- Normal operational forecast for production use
- Uses the latest Monday in the data as `ref_week_start`
- Outputs ONLY forecasts for the next 8 weeks after `ref_week_start`
- Output schema: 8 horizons × (#entities × #LG)
- `actual_value` = NaN (future not yet observed)
- Saved to: `data/processed/forecasts/{ref_week_start}/forecasts.parquet`

**Backtest Mode** (`mode="backtest"`):
- Evaluation mode for model performance assessment
- Uses 85/10/5 chronological split:
  - First 85% of weeks → training
  - Next 10% of weeks → validation
  - Last 5% of weeks → test (predictions generated here)
- Outputs predictions ONLY for the last 5% (test split) weeks
- `actual_value` = observed amount for that target_week_start
- `lp_baseline_point` = LP forecast for horizons 1-4, NaN for horizons 5-8
- Saved to: `data/processed/backtests/{ref_week_start}/backtest_predictions.parquet`
- Metrics saved to: `data/processed/metrics/backtests/{ref_week_start}/`

### Backtest Outputs

When running in backtest mode, the pipeline produces:

#### 1. Predictions DataFrame
Saved to: `data/processed/backtests/{ref_week_start}/backtest_predictions.parquet`

Includes all standard forecast columns plus:
- `lp_baseline_point`: LP forecast value for comparison
  - Horizons 1-4: Uses W1_Forecast, W2_Forecast, W3_Forecast, W4_Forecast respectively
  - Horizons 5-8: NaN (no LP baseline exists)

#### 2. Metrics Files
Saved to: `data/processed/metrics/backtests/{ref_week_start}/`

Two types of metrics are computed:
- **Full metrics**: Include all predictions (Tier-1 ML + Tier-2 LP passthroughs) - represents Treasury's total cash view
- **Clean metrics**: Tier-1 only, excluding LP passthroughs - represents true ML model performance

| File | Grouping | Description |
|------|----------|-------------|
| `metrics_by_lg.parquet` | week_start, liquidity_group, horizon | LG-level metrics (full) |
| `metrics_by_lg_clean.parquet` | week_start, liquidity_group, horizon | LG-level metrics (Tier-1 only) |
| `metrics_by_entity.parquet` | week_start, entity, liquidity_group, horizon | Entity-level metrics (includes is_pass_through flag) |
| `metrics_net.parquet` | week_start, horizon | Net TRR+TRP summed metrics (full) |
| `metrics_net_clean.parquet` | week_start, horizon | Net TRR+TRP summed metrics (Tier-1 only) |
| `metrics_net_entity.parquet` | week_start, entity, horizon | Net TRR+TRP per entity metrics |

#### Metrics Computed

**WAPE (Aggregate-then-Error)** - Treasury-aligned formula:
- `WAPE = |sum(actual) - sum(pred)| / |sum(actual)|`
- This allows over/under predictions to cancel out when aggregated
- Reflects Treasury's view of total cash position accuracy

**MAE** = mean(|actual - pred|)

**Directional Accuracy** = fraction where sign(pred - prev_actual) = sign(actual - prev_actual)

Each metric is computed for both ML predictions and LP baseline.

#### Programmatic Access with include_passthrough Parameter

The metric functions support filtering to exclude LP passthroughs:

```python
from hubbleAI.evaluation.metrics import compute_metrics_by_lg, compute_metrics_net

# Full metrics (includes Tier-2 passthroughs) - Treasury view
metrics_full = compute_metrics_by_lg(forecasts_df, include_passthrough=True)

# Clean metrics (Tier-1 only) - true ML performance
metrics_clean = compute_metrics_by_lg(forecasts_df, include_passthrough=False)
```

Usage examples:
```python
from hubbleAI.pipeline import run_forecast
import pandas as pd

# Forward mode (operational)
status = run_forecast(mode="forward", trigger_source="manual")
forecasts_df = pd.read_parquet(status.output_paths["forecasts"])

# Backtest mode (evaluation)
status = run_forecast(mode="backtest", trigger_source="notebook")
backtest_df = pd.read_parquet(status.output_paths["backtest"])

# Access backtest metrics (full - includes passthroughs)
metrics_lg = pd.read_parquet(status.metrics_paths["metrics_by_lg"])
metrics_entity = pd.read_parquet(status.metrics_paths["metrics_by_entity"])
metrics_net = pd.read_parquet(status.metrics_paths["metrics_net"])
metrics_net_entity = pd.read_parquet(status.metrics_paths["metrics_net_entity"])

# Access clean metrics (Tier-1 only - true ML performance)
metrics_lg_clean = pd.read_parquet(status.metrics_paths["metrics_by_lg_clean"])
metrics_net_clean = pd.read_parquet(status.metrics_paths["metrics_net_clean"])

# Access diagnostics
horizon_profiles = pd.read_parquet(status.metrics_paths["metrics_horizon_profiles"])
residuals = pd.read_parquet(status.metrics_paths["residual_diagnostics"])
stability = pd.read_parquet(status.metrics_paths["entity_stability"])
wins = pd.read_parquet(status.metrics_paths["model_vs_lp_wins"])
```

#### 3. Error Diagnostics (Backtest Only)
Saved to: `data/processed/metrics/backtests/{ref_week_start}/diagnostics/`

Diagnostics provide deeper analysis of model performance. They always use the **full dataset** (includes Tier-2 passthroughs).

| File | Grouping | Description |
|------|----------|-------------|
| `metrics_horizon_profiles.parquet` | horizon | ML/LP WAPE, MAE, MSE, RMSE per horizon |
| `residual_diagnostics.parquet` | liquidity_group, horizon | Residual distribution (mean, median, std, percentiles) |
| `entity_stability.parquet` | entity, liquidity_group, horizon | Error volatility over time per entity |
| `model_vs_lp_wins.parquet` | liquidity_group, horizon | Win-loss counts (ML vs LP) |

**Horizon Profiles Schema:**
```
horizon, ml_wape, ml_mae, ml_mse, ml_rmse,
lp_wape, lp_mae, lp_mse, lp_rmse, n_obs
```

**Residual Diagnostics Schema:**
```
liquidity_group, horizon, count, mean_residual, median_residual,
std_residual, p10_residual, p25_residual, p75_residual, p90_residual
```
- Residual = actual_value − y_pred_point

**Entity Stability Schema:**
```
entity, liquidity_group, horizon, mean_abs_error, median_abs_error,
std_abs_error, mean_4w_volatility, max_4w_volatility, n_weeks
```
- `mean_4w_volatility`: Rolling 4-week std of absolute errors (identifies unstable entities)

**Model vs LP Wins Schema:**
```
liquidity_group, horizon, ml_better_count, lp_better_count,
tie_count, ml_win_rate, lp_win_rate, total
```
- Win determined by comparing |ml_error| vs |lp_error| per observation
- Horizons 5-8 have no LP comparison (LP baseline is NaN)

### 2.7 Data Sources & I/O Strategy (Current vs Future)

  Current phase (what you should implement **now**):

    - All inputs are **local files** under `data/raw`:
      - Actuals (e.g. `actuals.csv`)
      - Liquidity Plan (e.g. `liquidity_plan.csv`)
      - FX rates (e.g. `fx_rates.csv`)
    - All outputs are **local files** under `data/processed`:
      - Forecasts under `data/processed/forecasts/{ref_week_start}/`
      - Backtests under `data/processed/backtests/{ref_week_start}/`
      - Run status under `data/processed/run_status/`
      - Metrics under `data/processed/metrics/`

  Future phase (what you should be prepared for, but **not** implement yet):

    - Treasury data (actuals and LP) may come from:
      - Denodo views
      - Direct connection to Reval
      - Databricks tables
    - FX data may come from:
      - An internal reference system
      - A central rates table on Databricks

  DESIGN RULE:

    - Keep all external I/O and data-source-specific logic inside **well-isolated modules**, especially under `src/hubbleAI/data_prep`.
    - Do **not** tightly couple the rest of the pipeline (features, models, evaluation, UI) to local CSV files.
    - For example, implement a function like `load_and_prepare_data(as_of_date)` that currently reads CSVs, but could later read from Denodo/Databricks/Reval without changing calling code.
    - Do **not** attempt to add Denodo / Databricks / Reval connectors now; just design the interfaces so they can be swapped in later.

---

## 3. Evaluation Requirements

We need weekly accuracy at **three aggregation levels**:

1. **Liquidity Group (TRR, TRP)** – *primary KPI.*
2. **Entity + Liquidity Group** – to detect entity‑specific issues.
3. **Net TRR + TRP (combined)** per week – overall treasury view.

Metrics to support in `evaluation/metrics.py`:

- WAPE
- MAE
- RMSE
- Direction accuracy (correct sign of change vs actual)

`evaluation/reporting.py` should provide helpers to:

- Compute and store these metrics over time.
- Produce tables/frames ready for the UI (Streamlit dashboards).

---

## 4. Refactoring & Experimentation Guidelines

Your job is to **extract and improve** code from `notebooks/TCF_V2.ipynb`, **not to discard it**.

1. Move stable logic into `src/hubbleAI/` (under the structure described above).
2. Keep the notebook for:
   - Exploration
   - Visualisation
   - Experiments, model tweaks, and backtesting

### 4.1 Experiments & Development Flow

- The primary place to run experiments, backtests, and model changes is the **notebook(s)** (e.g. `TCF_V2.ipynb`).
- Do **not** try to implement a full “development playground” page inside Streamlit for now.
- Once the core pipeline and UI are stable, we may later add a dedicated **Dev / Sandbox** page, but only after explicit agreement.
- When an experiment matures into a stable approach, refactor it from the notebook into reusable functions/modules under `src/hubbleAI` (e.g. `features/`, `models/`, `evaluation/`).

### 4.2 Refactoring Principles

When moving logic out of the notebook:

- Use **small, pure functions** with clear inputs/outputs.
- Add **docstrings** and basic **type hints** for public functions.
- Centralise constants and configuration in `config/` (e.g. entity lists, LP column names, horizon list).
- Avoid introducing heavy dependencies unless necessary.
- Where you see opportunities to simplify or speed up things:
  - First **describe the proposed change** (what, why, impact).
  - Apply it only after it’s been agreed.

The goal is to keep:

- **Notebooks** → for ideas, experimentation, and analysis.
- **`src/hubbleAI`** → for hardened, reusable, testable code that the pipeline and UI depend on.

----------

## 5. Streamlit UI & Scheduling

Streamlit UI is **required**. It is the main interface for the treasury team.

There will be at least **two core pages**, plus optional advanced pages.

### 5.1 Page 1 – “Latest Forecast” / Operations Overview

Purpose: operational view of the most recent forecast run.

Core behaviour:

- A **background scheduler** (cron / APS / Azure Function / similar) runs the forecast automatically every **Tuesday at an agreed time**.
- This scheduler uses the hubbleAI pipeline to:
  - Load the latest actuals, LP, FX.
  - Validate data freshness and completeness.
  - Run the forecasting models (using saved model artifacts, e.g. `pkl` files).
  - Store results and run status in a suitable location (DB / files).

Streamlit’s **“Latest Forecast”** page must:

1. Display **last run status**:
   - Last run timestamp
   - As‑of date used
   - Whether all inputs were available
   - High‑level summary (e.g. non‑zero warnings)
2. Display **next scheduled run** time.
3. Show a clear message if **latest data was missing**, such as:
   - “LP not yet available for week X”
   - “FX file missing”
   - “Actuals incomplete up to date Y”
4. In case of missing or stale data, the pipeline should:
   - Avoid running a broken forecast.
   - Record the problem in status.
   - Trigger an **email notification** to a small configurable list of 3–5 users.

#### “Run Forecast Now” Button

We support an **on‑demand** run with **constraints**:

- The button **does NOT allow arbitrary historical dates**.
- It simply triggers the pipeline for the **latest valid as‑of date** (e.g. last fully completed week) using the most recent data.
- Use cases:
  - Treasury / IT fixed an input data issue and want to re‑run.
  - Leadership wants a fresh run after late LP arrival.

Implementation details:

- The button calls a backend function in `pipeline.py` that:
  - Re‑checks data availability.
  - Runs the full forecast.
  - Updates stored outputs and run status.
- Backtesting for older dates should **not** depend on this button; it should use a separate backtest process or pre‑computed results.

This page **does not need accuracy metrics**, as it focuses on the latest run (often for a future horizon where actuals are not yet known).

### 5.2 Page 2 – “Performance Dashboard”

Purpose: make the benefit vs LP visible and compelling.

Core features:

- Time‑series dashboard showing **historical performance** of:
  - ML model vs LP
  - Per Liquidity Group
  - Optionally per Entity + Liquidity Group
  - Net TRR+TRP

- Typical visuals (Interactive wherever possible, refer to the notebook for some sample visuals, and reuse wherever makes sense):
  - Line charts of error metrics (WAPE, MAE, RMSE) by week.
  - Bars or heatmaps by horizon (H1–H8).
  - Direction accuracy plots.

- Filters:
  - Date range picker
  - Liquidity Group (TRR / TRP)
  - Entity
  - Horizon

The goal is for treasury to quickly see:

- “How much better are we than LP overall?”
- “Where does the model struggle?”
- “Which entities / horizons are problematic?”

This page is a **key selling point**.

### 5.3 Page 3 – Scenario Planner (Recommended Advanced Page)

Optional but highly recommended to increase the “wow” factor.

Idea: **simulate decisions under different forecasting strategies**, such as:

- **LP‑only** scenario
- **ML‑only** scenario
- **Hybrid** (e.g. TRR from ML, TRP from LP, or confidence‑weighted blends)

For each scenario:

- Show how cash positions, buffer requirements or other KPIs would differ.
- Where possible, compute simple impact metrics (e.g. indicative improvement in liquidity buffer sizing or missed opportunities).

This page can start simple, using backtest data and basic rules, and expand over time.

---

## 6. Agentic & Conversational Features (Aspirational but within scope)

To make hubbleAI feel more like an **assistant** than a static tool, we plan for some agent‑like features. Do not implement all of this at once; instead, treat this as a roadmap and discuss before implementing.

### 6.1 Data Health & Readiness Agent

- Checks input data daily or before a run:
  - Are all expected files present?
  - Are there suspicious spikes / drops in volumes?
  - Are there missing days or repeated dates?
- Generates a short summary:
  - “All good”
  - Or a list of issues to review
- Feeds into:
  - Status on Page 1
  - Email notifications

### 6.2 Forecast QA Agent

After each run:

- Highlights **where the model and LP diverge significantly** (e.g. > X% difference).
- Flags **low‑confidence forecasts** where:
  - Prediction intervals are very wide (P10–P90 range).
  - The model is operating in regimes with little historical data.
- Can generate a short natural language summary, e.g.:
  > “This week, TRR for entity 11G5 has a much lower forecast than LP and wide uncertainty; consider manual review.”

### 6.3 Scenario & What‑If Assistant

Under the Scenario Planner page:

- Allow simple what‑if questions like:
  - “What happens if FX moves by +5%?”
  - “What if we applied ML only to TRR but kept TRP as LP?”
- Uses pre‑computed sensitivities or simplified recalculations rather than full retraining.

### 6.4 Conversational “Treasury Copilot”

A conversational feature accessible from the UI (e.g. a chat panel) that can answer questions such as:

- “How has the model performed vs LP in the last 12 weeks for TRR?”
- “Why is the forecast for entity 14C1 TRR so different from LP this week?”
- “Which features were most important for the latest prediction for entity 17C7 TRR?”
- “Show me entities where the model consistently under‑forecasts.”

Under the hood, this could combine:

- Query access to:
  - Forecast outputs
  - Historical metrics
  - Feature importance / SHAP summaries
- Simple natural language templates
- Later: a proper RAG‑based assistant over documentation and metrics.

Start simple (e.g. canned queries + explanations) and only introduce true LLM integration once the core pipeline is stable.

---

## 7. Collaboration & Safety Rules

These are **very important**:

1. **Respect existing logic**
   - Do not change the core business rules, Tier‑2 handling, or LG×horizon strategy without explicit approval.

2. **Propose before applying**
   - For any large refactor, new model type, quantile approach, or change in evaluation logic:
     - First summarise what you want to change and why.
     - Wait for user approval.
     - Only then implement.

3. **Optimisations welcome, but controlled**
   - You may suggest:
     - Removing unused functions
     - Simplifying pipelines
     - Speeding up feature engineering or training
   - Do not remove or rewrite major components silently.

4. **Keep changes focused**
   - When asked for a specific change (e.g. “add horizon‑specific LP feature logic”), limit your edits to that scope, unless otherwise agreed.

5. **Use this file as the contract**
   - If something in the repo seems to contradict `Claude.md`, ask for clarification instead of guessing.
   - If we collectively decide to evolve the design, update `Claude.md` as part of that change.

---

## 8. Where to Look First

When starting work in this repo:

1. Read this `Claude.md` fully.
2. Read `notebooks/TCF_V2.ipynb` to understand the current pipeline.
3. Inspect relevant `src/` files (if already created).
4. For new tasks, the user will reference this file, e.g.:

> “As per Claude.md, implement horizon‑specific LP feature injection in the modeling pipeline and adjust training functions accordingly.”

Always align your changes with the rules and context in this document.
