# Causal Explainability for ML-Driven Investment Decisions on the German Stock Market

Master's Thesis | Frankfurt School of Finance & Management | Class of 2026

Live demo: https://germaninvestmentanalysis.streamlit.app
Repository: https://github.com/MutharasanE/german_investment_analysis

## What this project does

The project ports the counterfactual-explanation framework of Takahashi et al. (2024) from synthetic benchmarks to a live German equity-market setting. It trains a black-box CatBoost classifier on twenty DAX constituents, recovers a directed acyclic graph (DAG) over the features and the decision, computes Pearl-style counterfactual probabilities (LEWIS scores) on that DAG, and benchmarks them against SHAP. A Streamlit dashboard then exposes both explanation styles to domain experts and collects A/B trust ratings, which are analysed with an Ordinary Least Squares (OLS) regression.

```
Investment data -> CatBoost classifier -> Causal DAG (DirectLiNGAM, prior b)
                                    -> LEWIS scores (Nec, Suf, maxNesuf)
                                    -> SHAP scores (TreeExplainer)
                                    -> Streamlit A/B survey (n = 25 experts)
                                    -> OLS regression (trust drivers)
```

## Why the project matters

The EU AI Act, fully applicable in August 2026, classifies investment-decision support as high-risk AI and requires causal justification of model outputs. SHAP, the dominant explainability tool in finance, is correlational by design and cannot answer the regulator's "why did this happen?" question. LEWIS provides exactly that justification by quantifying necessity and sufficiency of each feature on a discovered causal graph. The pipeline implemented here produces, for every served decision, a counterfactual artefact suitable for an EU AI Act audit log.

## Project structure

```
.
|-- src/
|   |-- data_loader.py           # DAX + macro download, feature engineering
|   |-- discretization.py        # Equal-frequency / equal-width binning
|   |-- causal_discovery.py      # PC, DirectLiNGAM, RESIT, LiM, NOTEARS wrappers
|   |-- backdoor.py              # P(Y|do(X)) via backdoor adjustment
|   |-- lewis.py                 # Nec, Suf, maxNesuf score computations
|   |-- evaluation.py            # MAE + Spearman rank correlation
|   |-- visualization.py         # DAG and bar-chart rendering
|   |-- data_generation.py       # SCM synthetic-data generators (3-var, 8-var)
|   `-- pipeline.py              # Experiment orchestration
|-- pipeline_scripts/            # Numbered, executable pipeline stages (01-10)
|-- run_experiments.py           # Reproduce Takahashi et al. paper results
|-- run_investment.py            # Run on real DAX investment data
|-- run_shap_comparison.py       # LEWIS vs SHAP comparison
|-- run_investment_split_eval.py # Time- and random-split holdout evaluation
|-- app.py                       # Streamlit demo + A/B expert survey
|-- ols_regression.py            # Primary OLS (Python, statsmodels)
|-- ols_minimal.R                # OLS replication in R (lm + stargazer)
|-- ols_hierarchical.R           # Hierarchical M1 / M2 / M3 nested models
|-- survey_analytics.py          # Descriptive analytics on survey responses
|-- fetch_survey.py              # Pull survey data from MongoDB Atlas
|-- upload_synthetic.py          # Upload synthetic responses for pipeline tests
|-- models/                      # Saved CatBoost model + adjacency matrix
|-- data/                        # Raw DAX CSVs + investment_dataset.csv + survey.db
|-- results/
|   |-- experiments/             # Synthetic validation results
|   |-- investment/              # Real DAX outputs (LEWIS, SHAP, plots)
|   |-- plots/                   # Diagnostic plots (confusion, ROC, sectors, regimes)
|   `-- tables/                  # CSV reports (collinearity, stationarity, etc.)
|-- tests/                       # Unit tests
|-- pyproject.toml               # Dependencies
`-- requirements.txt             # Pinned dependency list
```

## Setup

### Prerequisites

```bash
# macOS
brew install python@3.11
brew install libomp        # required by CatBoost on Apple Silicon

# Windows (PowerShell)
winget install Python.Python.3.11
```

### Step 1 - virtual environment + dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate            # Windows
pip install -e ".[dev,data]"
pip install streamlit shap pymongo[srv] dnspython statsmodels
```

### Step 2 - patch causal-learn BIC bug

`causal-learn==0.1.4.5` calls `float()` on a `numpy.matrix` in `LocalScoreFunction.py`, which breaks GES, NOTEARS and the GES-based NOTEARS-MLP fallback. Apply this one-line patch:

```bash
sed -i '' 's/sigma = float(cov\[i, i\] - yX @ XX_inv @ yX.T)/sigma = np.asarray(cov[i, i] - yX @ XX_inv @ yX.T).item()/g' \
  venv/lib/python3.11/site-packages/causallearn/score/LocalScoreFunction.py
```

NOTEARS-MLP (non-linear) is not implemented in `causal-learn` 0.1.4.x, so `run_notears_mlp()` falls back to GES with BIC scoring.

### Step 3 - synthetic-data validation (optional, replicates the paper)

```bash
# Quick smoke-test
python run_experiments.py --experiment 3var --trials 10

# Full replication (~3-5 hours on a laptop)
python run_experiments.py --experiment all --trials 100 --parallel --workers 4
```

Outputs land in `results/experiments/`.

### Step 4 - real DAX pipeline

Downloads stock + macro data, trains CatBoost, runs DirectLiNGAM (prior b), computes LEWIS scores, saves all artefacts.

```bash
python run_investment.py
```

Outputs:
- `data/{TICKER}.csv` (raw monthly OHLCV)
- `data/investment_dataset.csv` (960 ticker-month rows, 9 features + target)
- `models/catboost_DirectLiNGAM(b).cbm` and adjacency matrix
- `results/investment/lewis_scores_DirectLiNGAM(b).csv`
- `results/investment/causal_graph_DirectLiNGAM(b).png`
- `results/investment/nesuf_comparison_DirectLiNGAM(b).png`

Steps 3 and 4 are independent and can run in parallel terminal tabs.

### Step 5 - SHAP comparison

Loads the saved CatBoost model, computes SHAP values on the same dataset and produces a side-by-side LEWIS-vs-SHAP ranking.

```bash
python run_shap_comparison.py
```

Outputs:
- `results/investment/shap_scores_DirectLiNGAM(b).csv`
- `results/investment/lewis_vs_shap_DirectLiNGAM(b).{csv,png}`

### Step 5b - holdout evaluation

```bash
python run_investment_split_eval.py                                  # time split (default)
python run_investment_split_eval.py --split random --random-state 42 # random split
```

Outputs:
- `results/investment/holdout_eval_time.{json,csv}`
- `results/investment/holdout_eval_random.{json,csv}`

Time-split test accuracy reaches 0.974 (precision 1.000, recall 0.959); random-split test accuracy reaches 0.990. The classifier is therefore a strong baseline against which the explanation methodology - not predictive accuracy - is the salient variable.

### Step 6 - Streamlit dashboard + expert survey

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Features:

1. DAX ticker selector with the AI's BUY / HOLD / SELL recommendation.
2. Discovered causal DAG.
3. LEWIS panel (Nec, Suf, maxNesuf, counterfactual examples).
4. SHAP panel (mean absolute Shapley values).
5. Side-by-side ranking comparison.
6. A/B preference questionnaire (trust 1-10, confidence 1-10, perceived accuracy, free-text).

Survey responses are persisted to MongoDB Atlas (`thesis_survey.votes` collection); a local SQLite fallback at `data/survey.db` is used when `MONGO_URI` is not set.

A live deployment is available at https://germaninvestmentanalysis.streamlit.app.

### Step 7 - regression on collected expert ratings

```bash
python ols_regression.py        # Primary OLS, JSON + plot exports
Rscript ols_minimal.R           # R replication
Rscript ols_hierarchical.R      # Nested M1 / M2 / M3 models
```

Inputs are pulled from MongoDB Atlas via `MONGO_URI` (see `.env.example`). Outputs land in the project root: `regression_results.txt`, `regression_summary.json`, `trust_drivers_plot.png`, `ols_coefficients.csv`, `ols_regression_table.html`.

### Step 8 - tests

```bash
pytest -q
```

## Data sources

### Equity universe (Yahoo Finance via `yfinance`)

Twenty DAX constituents, 5 years monthly OHLCV, period 2019-2024 (covers COVID and the ECB tightening cycle):

```
SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, MBG.DE, BMW.DE, MUV2.DE,
AIR.DE, IFX.DE, ADS.DE, DB1.DE, RWE.DE, HEN3.DE, VOW3.DE, SY1.DE,
BEI.DE, EOAN.DE, FRE.DE, MTX.DE
```

### Engineered features

| Feature | Definition | Window |
|---------|------------|--------|
| volatility | Rolling std of monthly returns | 12 m |
| momentum | 6-month price change | 6 m |
| volume_avg | Rolling mean of monthly volume | 6 m |
| return_1y | 12-month price change | 12 m |
| max_drawdown | Peak-to-trough decline | 12 m |

### Macroeconomic series (free, no API key)

| Feature | Source | Description |
|---------|--------|-------------|
| ecb_rate | ECB Data API | ECB main refinancing rate (%) |
| eur_usd | ECB Data API | EUR/USD exchange rate |
| de_inflation | ECB Data API | German HICP year-over-year (%) |
| vix | Yahoo Finance | CBOE VIX index (VSTOXX proxy) |

### Target variable

- `investment_decision` (binary): 1 = APPROVE, 0 = REJECT
- Defined as: risk-adjusted return (return_1y / volatility) > 0.5 (Sharpe-style threshold)

## Methods implemented

| Method | Family | Core assumption |
|--------|--------|-----------------|
| PC | Constraint-based | Conditional independence + faithfulness |
| DirectLiNGAM | Score-based | Linear, non-Gaussian errors |
| RESIT | Score-based | Non-linear additive noise (uses `GradientBoostingRegressor`) |
| LiM | Score-based | Linear, mixed (discrete + continuous) |
| NOTEARS | Continuous-optimisation | Continuous, equal-variance |
| NOTEARS-MLP | Continuous-optimisation | Non-linear (uses GES + BIC fallback) |

### Prior regimes

| Prior | Meaning |
|-------|---------|
| (0) | No prior - causal discovery runs blind |
| (a) | All features -> target (forces direct edges) |
| (b) | Target is a sink (no reverse causation - primary specification) |

DirectLiNGAM under prior (b) is the primary specification used downstream: it combines non-Gaussian identifiability with the no-reverse-causation axiom of a decision node.

## Headline empirical results

| Result | Value |
|--------|-------|
| CatBoost training accuracy (full panel) | 1.000 |
| Time-split holdout accuracy (192 rows) | 0.974 |
| Random-split holdout accuracy (192 rows) | 0.990 |
| LEWIS top-ranked feature | return_1y (maxNesuf = 1.000) |
| SHAP top-ranked feature | return_1y (mean abs SHAP = 5.39) |
| LEWIS vs SHAP rank correlation (Spearman) | 0.137 (p = 0.599) |
| Expert survey panel size | 25 evaluations |
| Counterfactual preference among experts who chose | 8 / 11 (73%) |
| OLS R^2 on trust score | 0.419 |
| F(5, 19) | 2.743 (p = 0.0499) |
| Significant predictor | confidence_score (beta = 0.216, p = 0.047) |

## Output artefact map

| Artefact | Path | Used by |
|----------|------|---------|
| CatBoost model | `models/catboost_DirectLiNGAM(b).cbm` | SHAP, Streamlit |
| Adjacency matrix | `models/adj_matrix_DirectLiNGAM(b).npy` | Streamlit causal-graph display |
| Metadata | `models/metadata_DirectLiNGAM(b).json` | Feature names, accuracy, config |
| LEWIS scores | `results/investment/lewis_scores_DirectLiNGAM(b).csv` | LEWIS-vs-SHAP comparison |
| SHAP scores | `results/investment/shap_scores_DirectLiNGAM(b).csv` | LEWIS-vs-SHAP comparison |
| Comparison table | `results/investment/lewis_vs_shap_DirectLiNGAM(b).csv` | Thesis Section 4 |
| Causal graph | `results/investment/causal_graph_DirectLiNGAM(b).png` | Thesis Section 4 |
| Reversal scores | `results/investment/reversal_scores_DirectLiNGAM(b).csv` | Counterfactual feasibility |

To load the saved model in Python:

```python
from catboost import CatBoostClassifier
import numpy as np, json

model = CatBoostClassifier().load_model("models/catboost_DirectLiNGAM(b).cbm")
adj = np.load("models/adj_matrix_DirectLiNGAM(b).npy")
meta = json.load(open("models/metadata_DirectLiNGAM(b).json"))
```

## Known issues

- numpy "Mean of empty slice" warnings are suppressed in `pipeline.py`. They occur when rare feature-value combinations have zero observations during backdoor probability computation; NaN values are skipped, so results are unaffected.
- `causal-learn` 0.1.4.x ships a non-linear NOTEARS-MLP module that is not actually wired in. The wrapper falls back to GES with BIC scoring and the limitation is documented in the thesis.
- Survey panel is small (n = 25). The OLS F-test is borderline (p = 0.0499); coefficient inference should be read with caution and bootstrap CIs are recommended for the final write-up.

## References

- Takahashi, R. et al. (2024). *Counterfactual Explanations of Black-Box ML Models using Causal Discovery*. arXiv:2402.02678.
- Galhotra, S., Pradhan, R., Salimi, B. (2021). *LEWIS: Explaining black-box algorithms using probabilistic contrastive counterfactuals*. SIGMOD.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
- Shimizu, S. et al. (2011). *DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model*. JMLR.
- Zheng, X. et al. (2018). *DAGs with NO TEARS: Continuous Optimization for Structure Learning*. NeurIPS.
- Lundberg, S., Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- European Union (2024). *Regulation laying down harmonised rules on Artificial Intelligence (EU AI Act)*.
