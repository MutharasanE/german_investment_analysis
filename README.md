# Causal Explainability for AI-Driven Investment Decisions in German Banks

Master Thesis | Frankfurt School of Finance & Management | 2026

## Overview

This project implements a causal XAI framework that explains black-box AI investment decisions using **causal discovery** and **counterfactual probabilities** (LEWIS scores). It adapts and extends the methodology from [Takahashi et al. (2024)](https://arxiv.org/abs/2402.02678) to the German/EU banking regulatory context.

### What it does

```
Investment Data → Black-box Model (CatBoost) → Causal Graph (DAG) → LEWIS Scores → Explanation
```

1. **Trains a classifier** (CatBoost) to approve/reject investments
2. **Discovers causal structure** between features using PC, DirectLiNGAM, NOTEARS algorithms
3. **Computes counterfactual explanations** via LEWIS scores:
   - **Nesuf** — overall feature importance (causal)
   - **Nec** — "if this feature decreased, would the decision flip?"
   - **Suf** — "if this feature increased, would the decision flip?"
4. **Compares** causal explanations vs SHAP (correlation-only baseline)

### Why it matters

- EU AI Act (Aug 2026) requires causal justification for high-risk AI
- SHAP only shows correlation — regulators need causal "why"
- Counterfactuals give actionable advice: "change X by Y to flip the decision"

## Project Structure

```
├── src/
│   ├── data_generation.py    # SCM synthetic data (3-var and 8-var graphs)
│   ├── discretization.py     # Equal-width / equal-frequency binning
│   ├── causal_discovery.py   # PC, DirectLiNGAM, RESIT, LiM, NOTEARS wrappers
│   ├── backdoor.py           # P(Y|do(X)) via backdoor adjustment
│   ├── lewis.py              # LEWIS scores: Nec, Suf, Nesuf
│   ├── evaluation.py         # MAE + Spearman rank correlation
│   ├── visualization.py      # Causal graph + bar chart plots
│   ├── data_loader.py        # DAX + macro data (ECB, VIX) via yfinance & ECB API
│   └── pipeline.py           # Main experiment orchestration
├── run_experiments.py        # Reproduce Takahashi et al. paper results
├── run_investment.py         # Run on real DAX investment data
├── run_shap_comparison.py    # LEWIS vs SHAP comparison (uses saved artifacts)
├── app.py                    # Streamlit demo + expert survey
├── models/                   # Saved CatBoost model, adj matrix, metadata
├── data/                     # Downloaded raw data + survey database
│   └── survey.db             # SQLite — expert votes (created by Streamlit app)
├── results/
│   ├── experiments/          # Paper validation (synthetic data)
│   └── investment/           # Real DAX results (plots, scores)
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
├── docs/                     # Thesis documents
└── pyproject.toml            # Dependencies
```

## Local Setup (Start to End)

### Prerequisites

```bash
# macOS
brew install python@3.11
brew install libomp        # Required for XGBoost/CatBoost

# Windows (PowerShell)
winget install Python.Python.3.11
```

### Step 1: Create virtual environment and install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev,data]"
pip install streamlit shap gspread google-auth
```

### Step 2: Patch causal-learn BIC score bug

`causal-learn==0.1.4.5` has a bug in `LocalScoreFunction.py` where `float()` fails on numpy matrix results, breaking GES, NOTEARS, and NOTEARS-MLP. Apply this one-line fix:

```bash
sed -i '' 's/sigma = float(cov\[i, i\] - yX @ XX_inv @ yX.T)/sigma = np.asarray(cov[i, i] - yX @ XX_inv @ yX.T).item()/g' \
  venv/lib/python3.11/site-packages/causallearn/score/LocalScoreFunction.py
```

Note: NOTEARS-MLP (nonlinear) module doesn't exist in causal-learn 0.1.4.x, so `run_notears_mlp()` uses GES with BIC scoring instead.

### Step 3: Run experiments (validates methodology on synthetic data)

```bash
# Quick test (10 trials, ~2 min)
python run_experiments.py --experiment 3var --trials 10

# Full experiments (100 trials, ~3-5 hours)
python run_experiments.py --experiment all --trials 100 --parallel

# Adjust workers if needed (default: 4)
python run_experiments.py --experiment all --trials 100 --parallel --workers 2
```

Output: `results/experiments/` (6 CSV/JSON files)

### Step 4: Run investment pipeline (real DAX data)

Downloads stock + macro data, trains model, computes LEWIS scores, saves all artifacts.

```bash
python run_investment.py
```

Output: `data/` (raw CSVs), `models/` (CatBoost model, adjacency matrix), `results/investment/` (plots, scores)

**Steps 3 and 4 can run in parallel** (separate terminal tabs) — they are independent.

### Step 5: Run SHAP comparison

Requires step 4. Loads saved model, computes SHAP values, generates side-by-side plots.

```bash
python run_shap_comparison.py
```

Output: `results/investment/lewis_vs_shap_*.png`, `results/investment/shap_scores_*.csv`

### Step 5b: Run holdout evaluation (recommended)

Evaluates generalization on unseen data split (time-based by default).

```bash
python run_investment_split_eval.py
# Optional random split
python run_investment_split_eval.py --split random --test-size 0.2 --random-state 42
```

Output: `results/investment/holdout_eval_*.json`, `results/investment/holdout_eval_*.csv`

### Step 6: Launch Streamlit demo

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Experts can view explanations and vote on LEWIS vs SHAP. Votes are saved to `data/survey.db` (SQLite).

### Step 7: Run tests

```bash
pytest -q
```

### Quick Start (TL;DR)

```bash
# One-time setup
python3.11 -m venv venv && source venv/bin/activate
pip install -e ".[dev,data]" && pip install streamlit shap
# Apply causal-learn patch (see Step 2)

# Run everything
python run_experiments.py --experiment all --trials 100 --parallel  # Tab 1 (~3-5 hrs)
python run_investment.py                                             # Tab 2 (~2 min)
python run_shap_comparison.py                                        # After investment
streamlit run app.py                                                 # Launch demo
```

### 5. Output

```
data/
├── SAP_DE.csv, SIE_DE.csv, ...    # Raw stock price data per ticker
└── investment_dataset.csv          # Full dataset (9 features + target, 960 rows)

models/                             # Saved artifacts for Streamlit & SHAP
├── catboost_*.cbm                  # Trained CatBoost classifier
├── adj_matrix_*.npy                # Discovered causal graph (adjacency matrix)
└── metadata_*.json                 # Feature list, accuracy, run config

results/
├── experiments/                    # Paper validation (synthetic data)
│   ├── 3var_linear.json            # Tables II-III reproduction
│   ├── 3var_nonlinear.json
│   └── 8var_*.csv                  # Tables V-X reproduction
└── investment/                     # Real DAX results
    ├── causal_graph_*.png          # Discovered DAG visualizations
    ├── nesuf_comparison_*.png      # Causal vs No-graph importance
    ├── reversal_*.png              # Nec/Suf reversal probability charts
    ├── lewis_scores_*.csv          # LEWIS Nesuf/Nec/Suf per feature
    ├── lewis_scores_no_graph_*.csv # Baseline (no causal graph) scores
    └── reversal_scores_*.csv       # Reversal probability scores
```

### 6. Saved Artifacts (for SHAP & Streamlit)

`run_investment.py` saves all artifacts needed for downstream tasks:

| Artifact | Path | Used by |
|----------|------|---------|
| CatBoost model | `models/catboost_*.cbm` | SHAP computation, Streamlit demo |
| Adjacency matrix | `models/adj_matrix_*.npy` | Streamlit causal graph display |
| Metadata | `models/metadata_*.json` | Feature names, accuracy, config |
| LEWIS scores | `results/investment/lewis_scores_*.csv` | SHAP vs LEWIS comparison |
| Dataset | `data/investment_dataset.csv` | SHAP computation, Streamlit demo |

To load in Python:
```python
from catboost import CatBoostClassifier
import numpy as np, json

model = CatBoostClassifier().load_model("models/catboost_DirectLiNGAM(b).cbm")
adj = np.load("models/adj_matrix_DirectLiNGAM(b).npy")
meta = json.load(open("models/metadata_DirectLiNGAM(b).json"))
```

## Data Sources

### Stock Data
- **Source**: Yahoo Finance (yfinance) — 20 DAX companies, 5 years monthly OHLCV
- **Features**: volatility, momentum, volume_avg, return_1y, max_drawdown

### Macroeconomic Data (all free, no API key)
| Feature | Source | Description |
|---------|--------|-------------|
| `ecb_rate` | ECB Data API | ECB main refinancing rate (%) |
| `eur_usd` | ECB Data API | EUR/USD exchange rate |
| `de_inflation` | ECB Data API | German HICP year-over-year (%) |
| `vix` | Yahoo Finance | CBOE VIX index (VSTOXX proxy) |

### Target Variable
- `investment_decision` (binary): 1 = APPROVE, 0 = REJECT
- Defined as: risk-adjusted return (return_1y / volatility) > 0.5

## Key Methods

| Method | Type | Assumptions |
|--------|------|-------------|
| PC | Constraint-based | Conditional independence |
| DirectLiNGAM | Score-based | Linear, non-Gaussian errors |
| RESIT | Score-based | Nonlinear, additive noise |
| LiM | Score-based | Linear, mixed data |
| NOTEARS | Optimization | Continuous, equal variance |
| NOTEARS-MLP | Optimization | Nonlinear (uses GES with BIC) |

## Prior Information on Causal Structure

| Prior | Description | Effect |
|-------|-------------|--------|
| (0) | No prior | Causal discovery runs blind |
| (a) | All features → target | Forces direct edges to target |
| (b) | Target is sink | Prevents reverse causation |

## Known Issues

- **numpy "Mean of empty slice" warnings**: Suppressed in `pipeline.py`. Occurs when rare feature-value combinations have zero observations during backdoor probability computation. Results are unaffected (NaN values are skipped).
- **RESIT requires regressor**: Uses `GradientBoostingRegressor` as the nonlinear regressor (causal-learn 0.1.4.x requires this argument explicitly).

## 8-Week Timeline

| Week | Phase | What to run |
|------|-------|-------------|
| 1-2 | Model Development | `run_experiments.py` — reproduce paper |
| 3 | Investment Data | `run_investment.py` |
| 4 | SHAP Comparison | Add SHAP baseline, generate LEWIS vs SHAP comparison plots |
| 5 | User Validation | Streamlit demo with A/B survey (LEWIS vs SHAP) |
| 6 | Literature Review | Write chapters 1-3 |
| 7 | Thesis Writing | Write chapters 4-6 |
| 8 | Finalize | Polish, submit |

## SHAP Comparison (Week 4)

SHAP serves as the **correlational baseline** against LEWIS (causal). Both run on the same CatBoost model, same data — the only difference is whether causal structure is used.

| Method | Type | How it ranks features |
|--------|------|----------------------|
| **LEWIS** | Causal | Uses discovered DAG + backdoor adjustment for P(Y\|do(X)) |
| **SHAP** | Correlational | Uses Shapley values — marginal contribution regardless of causation |

### What to expect
- SHAP will rank features by **predictive power** (correlation with outcome)
- LEWIS will rank features by **causal influence** (would changing this feature change the decision?)
- Features like `ecb_rate` may rank high in LEWIS (upstream cause) but lower in SHAP (indirect predictor)
- Features like `momentum` may rank high in SHAP (strong predictor) but lower in LEWIS (no direct causal path)

### Implementation
Uses saved artifacts from `run_investment.py` — no need to retrain:
```python
import shap
model = CatBoostClassifier().load_model("models/catboost_DirectLiNGAM(b).cbm")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

Output: side-by-side LEWIS vs SHAP ranking bar chart saved to `results/investment/lewis_vs_shap_*.png`

## Streamlit Demo & Expert Survey (Week 5)

Interactive web app (`streamlit run app.py`) for thesis defense and expert validation.

### App Features
1. **Select a DAX company** — see the AI's approve/reject decision
2. **View causal graph** — discovered DAG between features
3. **LEWIS explanations** — causal feature importance with Nesuf, Nec, Suf scores
4. **SHAP explanations** — correlational feature importance (baseline)
5. **Side-by-side comparison** — LEWIS vs SHAP rankings on the same decision

### Expert Voting System
The app includes a built-in survey with a **SQLite database** (`data/survey.db`) to collect and persist expert preferences:

- **Per-decision vote**: For each investment decision, the expert selects "LEWIS" or "SHAP" as the more trustworthy explanation
- **Reasoning**: Free-text field — "why did you prefer this explanation?"
- **Expert profile**: Role (e.g., risk analyst, portfolio manager, compliance officer), years of experience
- **Aggregated dashboard**: 
  - Total votes: LEWIS vs SHAP (bar chart + percentage)
  - Votes by expert role (do compliance officers prefer causal explanations?)
  - Votes by company/decision (are causal explanations preferred more for certain types of decisions?)
  - Export to CSV for thesis analysis

### Survey Database Schema
```
survey.db
└── votes (id, expert_name, expert_role, experience_years, ticker,
           decision, preference, comment, timestamp)
```

### Why This Matters for the Thesis
- Provides **qualitative evidence** from domain experts (Chapter 5 — User Study)
- Bridges the gap between technical metrics (MAE, Spearman) and practical trust
- Key hypothesis: experts in regulated roles (compliance, risk) prefer causal (LEWIS) explanations because they align with EU AI Act requirements for causal justification
- Target: 10-15 expert responses (sufficient for exploratory qualitative study in a management thesis)

## References

- Takahashi et al. (2024). *Counterfactual Explanations of Black-box ML Models using Causal Discovery*. arXiv:2402.02678
- Galhotra et al. (2021). *LEWIS: Explaining black-box algorithms using probabilistic contrastive counterfactuals*. SIGMOD 2021
- Pearl (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press
