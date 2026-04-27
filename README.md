# Causal Explainability for AI-Driven Investment Decisions in German Banks

Master Thesis | Frankfurt School of Finance & Management | 2026

## Overview

This project implements a causal XAI framework that explains black-box AI investment decisions using **causal discovery** and **counterfactual probabilities** (LEWIS scores). It adapts and extends the methodology from [Takahashi et al. (2024)](https://arxiv.org/abs/2402.02678) to the German/EU banking regulatory context.

### What it does

```
DAX 40 + S&P 500 Daily Data → Logistic Regression (Baseline) + CatBoost (Main)
  → Causal Graph (DAG) → LEWIS Scores → Explanation
```

1. **Downloads real market data** — DAX 40 + S&P 500 (top 50) daily prices + macro indicators (VIX, EUR/USD)
2. **Engineers features** — volatility, momentum, volume, RSI, max drawdown + macro
3. **Labels** Buy/Hold/Sell from real 5-day forward returns
4. **Trains two classifiers** — Logistic Regression (baseline) + CatBoost (main model, 3-class) with TimeSeriesSplit hyperparameter tuning
5. **Discovers causal structure** between features using DirectLiNGAM + PC cross-check
6. **Computes counterfactual explanations** via multi-class LEWIS scores (novel contribution):
   - **Nesuf** — overall feature importance (causal)
   - **Nec** — "if this feature decreased, would the decision flip?"
   - **Suf** — "if this feature increased, would the decision flip?"
7. **Compares** causal explanations (LEWIS) vs correlational baseline (SHAP)
8. **Evaluates** with confusion matrix, AUC-ROC, Cohen's Kappa, counterfactual validity
9. **Maps to regulations** — EU AI Act, BaFin guidelines, MiFID II

### Why it matters

- EU AI Act (Aug 2026) requires causal justification for high-risk AI
- SHAP only shows correlation — regulators need causal "why"
- Counterfactuals give actionable advice: "change X by Y to flip the decision"
- **Novel contribution**: Multi-class extension of LEWIS (original paper is binary only)

## Project Structure

```
├── src/
│   ├── 01_data_download.py      # Step 1: Download DAX 40 + S&P 500 daily + macro data
│   ├── 02_feature_engineering.py # Step 2: Compute 7 features per ticker per date
│   ├── 03_labeling.py           # Step 3: Buy/Hold/Sell from 5-day forward returns
│   ├── 04_data_preparation.py   # Step 4: Cleaning, stationarity, scaling, temporal split
│   ├── 05_baseline_model.py     # Step 5: LogReg baseline + CatBoost + SHAP + confusion matrices
│   ├── 06_causal_discovery.py   # Step 6: DirectLiNGAM + PC cross-check + bootstrap
│   ├── 07_lewis_scores.py       # Step 7: Multi-class LEWIS Nec/Suf/Nesuf scores
│   ├── 08_evaluation.py         # Step 8: Full evaluation framework
│   ├── 09_comparison.py         # Step 9: SHAP vs LEWIS comparison
│   ├── 10_governance.py         # Step 10: Regulatory compliance report
│   ├── pipeline.py              # Master runner: executes all steps in order
│   ├── backdoor.py              # P(Y|do(X)) via backdoor adjustment
│   └── __init__.py              # Package marker
├── app.py                       # Streamlit demo + expert survey
├── data/
│   ├── raw/                     # Downloaded prices + macro CSVs
│   └── processed/               # Train/test splits after preprocessing
├── models/                      # Saved CatBoost model, adj matrix, scaler, metadata
├── results/
│   ├── plots/                   # All visualizations (causal graph, confusion matrix, etc.)
│   ├── tables/                  # CSV exports of all metrics and scores
│   └── reports/                 # Regulatory compliance report, disagreement analysis
├── notebooks/                   # Jupyter notebooks for exploration
├── requirements.txt             # Dependencies
└── runtime.txt                  # Python version
```

## Local Setup

### Prerequisites

```bash
# macOS
brew install python@3.11
brew install libomp        # Required for CatBoost
```

### Step 1: Create virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install streamlit
```

### Step 2: Patch causal-learn BIC score bug

```bash
sed -i '' 's/sigma = float(cov\[i, i\] - yX @ XX_inv @ yX.T)/sigma = np.asarray(cov[i, i] - yX @ XX_inv @ yX.T).item()/g' \
  venv/lib/python3.11/site-packages/causallearn/score/LocalScoreFunction.py
```

### Step 3: Run the full pipeline

```bash
# Run all 10 steps (~15-25 min total)
python src/pipeline.py --steps all

# Or run specific steps
python src/pipeline.py --steps 1,2,3      # Data download + features + labeling
python src/pipeline.py --steps 4,5         # Prep + model training
python src/pipeline.py --steps 6,7,8,9,10  # Causal discovery + LEWIS + evaluation
```

### Step 4: Launch Streamlit demo

```bash
streamlit run app.py
```

## Pipeline Steps & Time Estimates

| Step | Module | What it does | Est. time |
|------|--------|-------------|-----------|
| 1 | `01_data_download.py` | Download DAX 40 + S&P 500 daily OHLCV + VIX, EUR/USD | ~3 min |
| 2 | `02_feature_engineering.py` | Compute volatility, momentum, volume, RSI, drawdown | ~10 sec |
| 3 | `03_labeling.py` | Buy/Hold/Sell from 5-day forward returns | ~5 sec |
| 4 | `04_data_preparation.py` | ADF stationarity, winsorization, scaling, temporal split | ~10 sec |
| 5 | `05_baseline_model.py` | Logistic Regression baseline + CatBoost grid search (108 combos x 5-fold) + SHAP | ~15-20 min |
| 6 | `06_causal_discovery.py` | DirectLiNGAM + PC + 30-run bootstrap stability | ~1 min |
| 7 | `07_lewis_scores.py` | Multi-class LEWIS (pairwise decomposition) | ~2-3 min |
| 8 | `08_evaluation.py` | ML metrics, AUC-ROC/PR, counterfactual validity | ~30 sec |
| 9 | `09_comparison.py` | SHAP vs LEWIS ranking comparison + Spearman | ~10 sec |
| 10 | `10_governance.py` | EU AI Act / BaFin / MiFID II compliance report | ~5 sec |

**Total: ~20-25 min**

## Data

### Stock Data
- **Source**: Yahoo Finance (yfinance) — DAX 40 + S&P 500 (top 50 by market cap), daily OHLCV
- **Period**: ~4 months effective (with lookback buffer for rolling features)
- **Split**: 3 months train (Dec 2025 -- Feb 2026), 1 month test (Mar 2026)

### Features (7 total)

| Feature | Type | Description |
|---------|------|-------------|
| `volatility` | Stock | 20-day rolling std of log returns (annualized) |
| `momentum` | Stock | 21-day return |
| `volume_avg` | Stock | 20-day average volume (log-scaled) |
| `rsi_14` | Stock | 14-day Relative Strength Index |
| `max_drawdown` | Stock | 20-day max peak-to-trough drop |
| `vix` | Macro | CBOE VIX index (market fear) |
| `eur_usd` | Macro | EUR/USD exchange rate |

### Target Variable
- `label` (3-class): Buy=2, Hold=1, Sell=0
- Derived from 5-day forward return with distribution-based thresholds (mean ± 0.5 std)

## Key Methods

| Method | Type | Role |
|--------|------|------|
| Logistic Regression | ML Classifier | Baseline model |
| CatBoost | ML Classifier | Main model — 3-class Buy/Hold/Sell prediction |
| DirectLiNGAM | Causal Discovery | Primary method (linear, non-Gaussian) |
| PC Algorithm | Causal Discovery | Cross-check / validation |
| LEWIS (multi-class) | Counterfactual XAI | Causal feature importance (Nec, Suf, Nesuf) |
| SHAP | Correlational XAI | Baseline comparison (TreeExplainer) |

## Output Artifacts

```
data/
├── raw/prices_{ticker}.csv         # Raw daily OHLCV per ticker
├── raw/macro_data.csv              # VIX + EUR/USD daily
├── raw/labeled_dataset.csv         # Full dataset with labels
└── processed/train.csv, test.csv   # Preprocessed train/test splits

models/
├── logistic_baseline.pkl            # Trained Logistic Regression baseline
├── catboost_best.cbm               # Trained CatBoost model (best hyperparams)
├── adj_matrix_directlingam.npy     # Discovered causal graph
├── scaler.pkl                      # StandardScaler fitted on train data
└── metadata.json                   # Feature names, accuracy, config, model comparison

results/
├── plots/
│   ├── confusion_matrix.png        # CatBoost confusion matrix heatmap
│   ├── confusion_matrix_baseline.png  # Logistic Regression confusion matrix
│   ├── causal_graph_directlingam.png
│   ├── shap_summary_plot.png       # SHAP beeswarm plot
│   ├── shap_bar_plot.png           # SHAP mean importance
│   ├── shap_vs_lewis_comparison.png
│   ├── reversal_probabilities.png  # Nec vs Suf per feature
│   ├── dag_stability_heatmap.png   # Bootstrap edge stability
│   ├── auc_roc.png                 # ROC curves per class
│   └── auc_pr.png                  # Precision-recall curves
├── tables/
│   ├── label_distribution.csv
│   ├── stationarity_report.csv
│   ├── classification_report.csv        # CatBoost
│   ├── classification_report_baseline.csv  # Logistic Regression
│   ├── model_comparison.csv            # Side-by-side metric comparison
│   ├── hyperparameter_tuning_results.csv
│   ├── shap_scores.csv
│   ├── lewis_scores_causal.csv     # LEWIS with causal graph
│   ├── lewis_scores_no_graph.csv   # LEWIS without graph (baseline)
│   ├── reversal_scores.csv
│   ├── causal_method_agreement.csv # DirectLiNGAM vs PC edge agreement
│   ├── dag_stability_scores.csv
│   ├── dag_evaluation_metrics.csv
│   ├── shap_vs_lewis_comparison.csv
│   ├── ml_metrics.csv
│   └── counterfactual_validity_report.csv
└── reports/
    ├── regulatory_compliance_report.json
    ├── regulatory_compliance_report.md
    └── key_disagreements.txt
```

## Known Limitations (document in thesis)

1. **DirectLiNGAM assumes linear non-Gaussian relationships** — financial data may violate this in nonlinear regimes
2. **DAG may be unstable during structural breaks** — bootstrap stability analysis documents which edges are reliable
3. **Multi-class LEWIS extension is novel** — no peer-reviewed validation exists yet (this IS the thesis contribution)
4. **Short data window (4 months daily)** — limits regime diversity; justified by keeping training time manageable
5. **Expert survey sample size** — document minimum required for statistical significance

## References

- Takahashi et al. (2024). *Counterfactual Explanations of Black-box ML Models using Causal Discovery*. arXiv:2402.02678
- Galhotra et al. (2021). *LEWIS: Explaining black-box algorithms using probabilistic contrastive counterfactuals*. SIGMOD 2021
- Pearl (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press
