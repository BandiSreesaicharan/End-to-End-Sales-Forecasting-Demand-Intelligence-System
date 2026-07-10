# End-to-End Sales Forecasting & Demand Intelligence System

An end-to-end retail analytics project covering time series forecasting, anomaly detection, product demand segmentation, and an interactive dashboard — built on 4 years of Superstore sales data.

## Live Dashboard

https://bandisreesaicharan-end-to-end-sales-forecasting-dema-app-oo4hxf.streamlit.app/

## Project Overview

This project answers a core retail question: how much of each product will sell next month, and where should stocking priority go? It covers Exploratory Data Analysis, Time Series Decomposition, three-way Forecast Model Comparison, Category/Region-Level Forecasting, Anomaly Detection, Product Demand Segmentation, an Interactive Streamlit Dashboard, and a non-technical Executive Business Report.

---

## Planning & Approach

The project was scoped around 8 tasks (data exploration, decomposition, forecasting, segment forecasting, anomaly detection, clustering, dashboard, executive report), built entirely in Kaggle Notebooks. The plan was to build each phase sequentially, verify outputs against real numbers at every step rather than assuming code correctness from its logic alone, and only move to the next phase once the current one produced evidence-backed results.

A recurring principle throughout: every claim in this README and in the notebook's markdown observations is backed by an actual computed number, re-verified against raw data where a first pass turned out to be wrong (see the Seasonality and Clustering sections below for two cases where this caught real mistakes).

---

## Dataset & Exploratory Analysis

**Primary dataset:** Superstore Sales (`train.csv`) — 9,800 rows, January 2015 to December 2018, covering Sales, Order/Ship Dates, Region, Category, Sub-Category.

**Secondary dataset:** Global Video Game Sales (`vgsales.csv`) — merged on Year as a structural multi-source exercise. The merge revealed a genuine data coverage gap rather than any real business relationship: `vgsales` data density collapses after 2016 (614 titles in 2015 down to 3 in 2017, none in 2018), so no correlational claim was drawn from it.

**Key EDA findings:**
- Technology is the highest-revenue category (₹827,456), narrowly ahead of Furniture (₹728,659) and Office Supplies (₹705,422).
- East is the most consistent region for year-over-year growth (lowest variance, 0.018), rising every year with no dip. West has the highest average growth (0.257 variance) but far less predictably.
- Order-to-ship delay averages 3.96 days with negligible regional variation (under 0.16 days spread).
- **Seasonality — corrected mid-project:** an early pass concluded "November is the strongest month every year." Re-verifying directly against the raw pivot table showed this was wrong — September led in 2015 (81,624 vs November's 77,908) and December led in 2017 (95,739 vs November's 79,066). The corrected finding: September, November, and December are consistently the three strongest months, but the single peak month varies by year.

---

## Time Series Decomposition & Stationarity

Monthly sales were decomposed into trend, seasonal, and residual components. Trend rises steadily from ~₹40,000 (2015) to ~₹60,000 (late 2018). Seasonal swing is roughly ±25,000 — strong relative to trend level. The Augmented Dickey-Fuller test on the raw monthly series returned a statistic of -4.42 (p = 0.00028), confirming the series is already stationary — no differencing was statistically required at this stage, a fact that became relevant (and briefly contradicted) during SARIMA model selection below.

---

## Model Training & Comparison

Three forecasting approaches were built and evaluated on the same held-out 3-month window:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| SARIMA | 19,244.49 | 19,950.07 | 20.53% |
| **Prophet** | **18,059.43** | **18,558.52** | 19.60% |
| XGBoost | 18,274.19 | 20,720.17 | **18.53%** |

**Prophet was selected for production** on lowest RMSE/MAE, and separately confirmed a strong weekly seasonality (Thursday dip, Tuesday/weekend peaks) and yearly seasonality (April/May bump, a large September-November peak, late-February trough) consistent with the EDA findings above.

**SARIMA order selection — a real methodological catch:** the ADF test confirmed the raw series was already stationary, yet the chosen SARIMA order used `d=1` (first-order differencing) — an apparent contradiction. Rather than silently picking one or the other, both alternatives were tested empirically: `d=0` with no trend term diverged into an unusable model (RMSE ~10^16), and `d=0` with an explicit trend term produced RMSE 199,270 — roughly 10x worse than `d=1`'s 19,950. `d=1` was kept because it empirically outperformed both alternatives on real held-out error, with that reasoning documented directly in the notebook rather than left implicit.

**Honest limitation stated throughout:** all three models show RMSE in the 18,000-21,000 range against an average month of ~₹45,000 — roughly 40-45% typical forecast error, a direct consequence of forecasting from under 4 years of monthly history (45 data points).

---

## Category & Region-Level Forecasting

Prophet (the confirmed best model) was re-run individually for Furniture, Technology, Office Supplies, West, and East.

| Segment | Typical Error | 3-Month Growth |
|---|---|---|
| Office Supplies | 6.5% | +17.3% |
| Furniture | 14.7% | +8.0% |
| West | 31.9% | +65.0% |
| Technology | 40.5% | +81.2% |
| East | 55.3% | +1.5% |

Segment-level reliability varies far more than the overall model — Office Supplies and Furniture are reasonably tight, while Technology and especially East carry substantial error and should be treated as directional signals rather than firm planning numbers. This distinction is surfaced live in the dashboard (see below), not just noted in text.

---

## Anomaly Detection

Two independent methods were applied to weekly sales: Isolation Forest (flags absolute extremes regardless of local context) and a rolling Z-score method (flags deviations from a local recent window).

**A real bug caught mid-build:** the Z-score method initially used a 4-week rolling window and flagged zero anomalies — the window was too small and too volatile to ever cross a 2-sigma threshold. Widening it to 8 weeks fixed this, after which the two methods showed genuine, explainable disagreement: only 1 of 16 total flagged weeks was caught by both. This divergence is itself a finding — Isolation Forest surfaces overall extremes, Z-score surfaces local, context-dependent deviations, and both answer different business questions.

**Top 3 anomalies:**
1. March 22, 2015 (₹37,704) — largest spike in the dataset, no seasonal explanation, likely a single bulk order.
2. December 2, 2018 (₹35,999) — second-largest, falls inside the confirmed November-December seasonal peak.
3. September 13, 2015 (₹29,959) — third-largest, again no seasonal explanation.

Not every anomaly fits a "festive spike" story — two of the three biggest spikes have no seasonal explanation at all, and that was stated directly rather than forced into a single tidy narrative.

---

## Product Demand Segmentation

K-Means clustering was applied to 17 sub-categories using total sales, growth rate, volatility, and average order value.

**A genuine multi-step correction:** the initial GrowthRate feature (averaged year-over-year % change) inflated Supplies to a 192.8% "growth rate" — traced back to one anomalously low sales year (2016) rather than real growth. Recomputing GrowthRate as CAGR fixed this (Supplies dropped to a realistic 3.6%). This in turn changed which cluster count (k) was actually optimal — re-running silhouette analysis across k=2 through k=6 after the fix shifted the best score from k=4 to k=3.

Even after that fix, k=4 was tested and re-tested against k=3 multiple times, because each run kept producing single-item "clusters" (first Supplies, then Machines, then Copiers) that inertia alone couldn't distinguish from real structure. Silhouette score, plus a direct check of cluster-level standard deviation, confirmed k=3 was the more defensible choice — one genuine singleton (Copiers, isolated in every k tested from 3-6) is a more honest result than forcing a second marginal cluster split.

**Final segments:**
- **Stable Core Demand** (11 sub-categories) — standard reorder-point inventory.
- **Niche High-Value, High-Growth** (Copiers only) — highest growth (79.6%) and highest average order value (₹2,216) of any product; prioritize availability, consider pre-order/consignment.
- **High-Value, Variable Demand** (Binders, Chairs, Machines, Phones, Storage, Tables) — maintain safety stock; Machines is a partial fit (declining -11.1% while the rest of the group grows) and is flagged individually rather than assumed to follow the group average.

---

## Dashboard (Task 7)

Built in Streamlit — a deliberate shift from the author's prior familiarity with Dash. The core adjustment: Streamlit has no callback model; the entire script reruns on every interaction, so anything expensive (model/CSV loading) is wrapped in `@st.cache_data`/`@st.cache_resource` rather than assumed to persist.

**Four pages, each independently verified against real data, not assumed from code correctness:**
1. **Sales Overview** — yearly bar chart, monthly trend line, region/category filterable table.
2. **Forecast Explorer** — two tabs: overall model comparison (SARIMA/Prophet/XGBoost, with a *dynamically computed* "best model" banner driven directly off `comparison_df.RMSE.idxmin()`, not a hardcoded claim), and per-segment forecasts with a live confidence flag (>30% relative error triggers a "Low confidence" warning, verified against both a low-error segment and East's 55.3% high-error case).
3. **Anomaly Report** — the anomaly chart plus a full table of all 11 flagged weeks.
4. **Product Segments** — the cluster scatter plot plus the full 17-row cluster assignment table.

Every page was checked from an actual fresh GitHub clone (clean virtual environment, dependencies installed only from `requirements.txt`) — not just the original development folder — to catch hidden environment dependencies before deployment.

---

## Deployment

Deployment to Streamlit Community Cloud surfaced two real, non-code issues:

1. **A documented platform bug** where Streamlit Cloud provisioned Python 3.14 regardless of the version explicitly selected in Advanced Settings, breaking source builds for pinned dependency versions with no available prebuilt wheels for that Python version.
2. **An unused dependency** (`scikit-learn`) sitting in `requirements.txt` from the original notebook pipeline but never actually imported by `app.py` (the dashboard never loads the KMeans/Isolation Forest/PCA pickles directly) — removing it, along with unused `matplotlib`/`seaborn`, eliminated the specific package that was failing to build.

After a full delete-and-redeploy with the trimmed dependency list, the build succeeded on Python 3.11.15 with all 58 resolved packages installing cleanly, and all four dashboard pages were re-verified live on the deployed URL.

---

## Executive Business Report (Task 8)

A 2-page `Summary.docx`, verified by actual PDF-page-count conversion rather than assumed, written for a non-technical audience (Head of Supply Chain / CFO). Covers the executive summary, EDA/forecasting findings, a 3-month forecast with an explicitly-disclosed approximate range (built from RMSE, stated plainly as *not* a formal statistical confidence interval since exact model confidence intervals were never captured), top 3 anomalies, segmentation and stocking strategy, a model performance comparison table in plain language, 3 data-backed recommendations, and a stated risk/limitation section covering the segment-level reliability variance.

---

## Repository Structure

```
├── analysis.ipynb               # Full analysis notebook (Tasks 1-6)
├── app.py                       # Streamlit dashboard (Task 7)
├── requirements.txt             # Python dependencies (trimmed to what app.py actually uses)
├── Summary.docx                 # Executive business report (Task 8)
├── Dataset/
│   ├── train.csv                # Superstore sales dataset
│   └── vgsales.csv              # Secondary dataset (multi-source merge exercise)
├── models/                      # Saved model artifacts (SARIMA, Prophet, XGBoost, segment models, clustering)
├── charts/                      # Saved chart images
├── model_comparison.csv
├── anomaly_report.csv
├── cluster_assignments.csv
└── segment_model_metrics.csv
```

## Running the Dashboard Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Limitations

The company-wide forecast carries roughly 40-45% typical error, a direct consequence of forecasting from under 4 years of monthly history. Segment-level forecasts vary far more in reliability than the overall model — Office Supplies and Furniture are reasonably tight (6.5%/14.7% typical error), while West, Technology, and especially East (31.9%/40.5%/55.3%) should be treated as directional signals, not firm planning figures. This is stated explicitly in both the dashboard (live confidence flag) and the executive report, rather than left implicit.
