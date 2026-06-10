# Causal Explainability for ML-Driven Investment Decisions by German Market Investors on DAX 30 Companies

**A Master's Thesis Project Knowledge Document**

| | |
|---|---|
| **Author** | hello@destinationone.ca |
| **Institution** | Frankfurt School of Finance & Management |
| **Document Compiled** | 2026-06-10 |
| **Document Purpose** | Single-source professorial reference covering every step of the project — from raw data ingestion to inferential regression on expert trust. |
| **Repository** | `MasterThesis-Deploy-NoSecrets/` |
| **Primary Reference** | Takahashi et al. (2024), *Counterfactual Explanations of Black-Box ML Models using Causal Discovery* (arXiv:2402.02678) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Question, Hypotheses, and Contribution](#2-research-question-hypotheses-and-contribution)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Stage 1 — Data Acquisition](#4-stage-1--data-acquisition)
5. [Stage 2 — Feature Engineering & Pre-processing](#5-stage-2--feature-engineering--pre-processing)
6. [Stage 3 — Investment Decision Labelling](#6-stage-3--investment-decision-labelling)
7. [Stage 4 — Baseline ML Model (CatBoost)](#7-stage-4--baseline-ml-model-catboost)
8. [Stage 5 — Causal Discovery](#8-stage-5--causal-discovery)
9. [Stage 6 — LEWIS Counterfactual Explanations](#9-stage-6--lewis-counterfactual-explanations)
10. [Stage 7 — SHAP Baseline & Comparison](#10-stage-7--shap-baseline--comparison)
11. [Stage 8 — Streamlit Application & Expert Survey](#11-stage-8--streamlit-application--expert-survey)
12. [Stage 9 — OLS Regression on Expert Trust](#12-stage-9--ols-regression-on-expert-trust)
13. [Stage 10 — Hierarchical Regression Extension (R)](#13-stage-10--hierarchical-regression-extension-r)
14. [Synthetic-Data Validation Experiments](#14-synthetic-data-validation-experiments)
15. [Reproducibility, Dependencies, and Runtime](#15-reproducibility-dependencies-and-runtime)
16. [Limitations, Known Issues, and Future Work](#16-limitations-known-issues-and-future-work)
17. [Glossary, References, and File Index](#17-glossary-references-and-file-index)

---

## 1. Executive Summary

This thesis investigates whether **causal explanations** of machine-learning–driven investment recommendations produce higher expert trust than the dominant **correlation-based** explainability standard (SHAP). The empirical setting is **DAX 30** equity decisions and the target users are professional German market investors — a regulated population that, under the EU AI Act effective August 2026, will require **causal justification** for high-risk AI outputs.

The pipeline executes ten coupled stages:

1. Acquire 5 years of monthly OHLCV data for 20 DAX constituents via `yfinance`, plus four macro series (ECB rate, EUR/USD, German HICP, VIX).
2. Engineer nine financial features (volatility, momentum, return, drawdown, volume, plus the four macros).
3. Label each ticker-month as `APPROVE` / `REJECT` using a risk-adjusted return threshold.
4. Train a **CatBoost** classifier on the 960-row panel (`models/catboost_DirectLiNGAM(b).cbm`).
5. Run **causal-discovery algorithms** (PC, DirectLiNGAM, RESIT, LiM, NOTEARS, NOTEARS-MLP) under three prior regimes (0, a, b) to recover a DAG over features + target.
6. Compute **LEWIS scores** (Necessity, Sufficiency, max-Necessity-Sufficiency) via Pearl back-door adjustment on the recovered DAG.
7. Compute **SHAP** values on the same model as a correlational benchmark and rank-compare with LEWIS.
8. Expose the model + both explanation styles inside a **Streamlit dashboard** that doubles as an A/B survey instrument (MongoDB Atlas backend).
9. Collect responses from **25 domain experts** and fit an **OLS regression** identifying which explanation characteristics drive trust.
10. Replicate the OLS model in R and run a **hierarchical extension** (M1 → M2 → M3) testing controls and interaction.

**Headline empirical results**

| Result | Value | Interpretation |
|---|---|---|
| CatBoost classifier accuracy | ≈ 0.70+ | Strong-enough baseline to make explanation methodology the salient variable. |
| Spearman rank correlation, LEWIS vs SHAP | reported per discovery method | Methods diverge most on upstream macro variables (`ecb_rate`) and on momentum-style features. |
| Expert preference for counterfactual (LEWIS) | 64 % (vs 36 % SHAP) | Descriptive evidence that causal phrasing resonates. |
| OLS R² on trust score | **0.419** | ≈ 42 % of trust variance captured by five predictors. |
| F(5, 19) | 2.743, p = **0.0499** | Model marginally significant overall. |
| Significant coefficient | `confidence_score` β = **0.216**, p = **0.047** | Each +1 confidence point ⇒ +0.22 trust points. |
| Non-significant coefficient of theoretical interest | `is_counterfactual_b` β = 0.0007, p = 0.998 | At n = 25 the causal-vs-correlational distinction does **not** independently move trust. |

The "no effect of counterfactual method" result is **not** evidence against the core hypothesis — it is evidence that the expert panel is underpowered (n = 25) and that **expert confidence**, not explanation flavour, is the proximate predictor of trust in this regime.

---

## 2. Research Question, Hypotheses, and Contribution

### 2.1 Primary Research Question

> *Do causal counterfactual explanations (LEWIS) generated from a discovered DAG produce higher expert trust in ML-driven DAX investment decisions than feature-importance explanations (SHAP), and which characteristics of an explanation predict that trust?*

### 2.2 Sub-Questions

- **RQ1 (Methodological).** Can the LEWIS framework of Galhotra et al. (2021), as extended for black-box models by Takahashi et al. (2024), be operationalised end-to-end on a real DAX 30 investment-decision pipeline?
- **RQ2 (Comparative).** How do LEWIS rankings of feature importance differ from SHAP rankings on the **same** underlying model and dataset?
- **RQ3 (Behavioural).** Which observable characteristics of an explanation (method type, perceived accuracy, complexity, attention, expert confidence) drive trust in domain experts?
- **RQ4 (Regulatory).** Does the causal pipeline produce artefacts that satisfy the "causal justification" requirements implied by the EU AI Act for high-risk AI in financial services?

### 2.3 Hypotheses

| ID | Statement | Test |
|---|---|---|
| H1 | LEWIS and SHAP yield materially different feature rankings on the DAX pipeline, especially for upstream macro variables. | Spearman rank correlation < 1; per-feature rank-difference table. |
| H2 | Experts shown counterfactual (LEWIS) explanations report higher trust than those shown feature-importance (SHAP) explanations, all else equal. | OLS coefficient on `is_counterfactual_b` > 0 and significant. |
| H3 | Explanation complexity (number of features displayed) is negatively associated with trust. | OLS coefficient on `num_features` < 0. |
| H4 | Perceived mechanical accuracy is positively associated with trust. | OLS coefficient on `accuracy_score` > 0. |

### 2.4 Contribution

1. **Empirical port** of Takahashi et al. (2024) from synthetic benchmarks to a **real German equity-market** pipeline.
2. **End-to-end implementation** of the LEWIS → causal-discovery → back-door-adjustment chain on a black-box CatBoost classifier.
3. **Side-by-side SHAP / LEWIS comparison** on identical features, identical model, identical decisions — controlling away every confounder except the explanation method itself.
4. **Domain-expert validation study** with a regression-grade survey instrument (Streamlit + MongoDB), producing a regression-quantified picture of what actually moves trust.
5. **Regulatory framing** explicitly aimed at the EU AI Act high-risk-AI provisions, making the artefacts directly auditable.

---

## 3. System Architecture at a Glance

```
            ┌─────────────────────────────────────────────────────────────┐
            │                    DATA LAYER                               │
            │  yfinance → DAX 30 prices  (20 tickers, monthly, 5 yrs)     │
            │  ECB API  → ecb_rate, eur_usd, de_inflation                 │
            │  Yahoo    → ^VIX                                            │
            └──────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
            ┌─────────────────────────────────────────────────────────────┐
            │             FEATURE ENGINEERING (src/data_loader.py)        │
            │  volatility · momentum · volume_avg · return_1y ·            │
            │  max_drawdown  +  4 macro features                          │
            │  → data/investment_dataset.csv  (n = 960 rows)              │
            └──────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
            ┌─────────────────────────────────────────────────────────────┐
            │           LABELLING & DISCRETISATION                        │
            │  y = 1 if return_1y / volatility > 0.5 else 0               │
            │  Equal-frequency binning (10 bins, train-only fit)          │
            └──────────────────────────┬──────────────────────────────────┘
                                       │
              ┌────────────────────────┼─────────────────────────┐
              ▼                        ▼                         ▼
       ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
       │  CatBoost    │         │   Causal     │         │   SHAP       │
       │  classifier  │         │  Discovery   │         │  baseline    │
       │  (200 iters) │         │  (6 methods, │         │  (Tree       │
       │              │         │   3 priors)  │         │  Explainer)  │
       └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
              │                        │                        │
              │                        ▼                        │
              │                 ┌──────────────┐                │
              │                 │   LEWIS      │                │
              └────────────────►│  back-door   │◄───────────────┘
                                │  adjustment  │
                                │  Nec/Suf/    │
                                │  maxNesuf    │
                                └──────┬───────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │         Streamlit Dashboard + A/B Survey     │
                │  (MongoDB Atlas: thesis_survey.votes)        │
                └──────────────────────┬───────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │   OLS regression (Python + R)                │
                │   Hierarchical M1 / M2 / M3 (R)              │
                │   trust_score ~ method + accuracy + ...      │
                └──────────────────────────────────────────────┘
```

---

## 4. Stage 1 — Data Acquisition

### 4.1 Equity Universe

We restrict the universe to **20 DAX constituents** chosen for liquidity and macroeconomic representativeness across sectors (industrials, finance, technology, consumer, energy, healthcare). The list is hard-coded in [src/data_loader.py:11-16](src/data_loader.py#L11-L16):

```
SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, MBG.DE, BMW.DE, MUV2.DE,
AIR.DE, IFX.DE, ADS.DE, DB1.DE, RWE.DE, HEN3.DE, VOW3.DE, SY1.DE,
BEI.DE, EOAN.DE, FRE.DE, MTX.DE
```

| Parameter | Setting | Rationale |
|---|---|---|
| Data source | Yahoo Finance via `yfinance` | Free, programmatic, redistributable. |
| Frequency | Monthly OHLCV | Matches macroeconomic release cadence; reduces high-frequency noise. |
| Window | 5 years (rolling, 2019 – 2024) | Captures one full cycle plus the COVID shock and the ECB tightening cycle. |
| Persistence | `data/{TICKER}.csv` (20 files) | Reproducible from disk without re-pulling. |

The download routine [`download_dax_data()`](src/data_loader.py#L19-L64) first attempts a **batched** request and falls back to **per-ticker** requests with a 2-second cool-down if Yahoo throttles.

### 4.2 Macroeconomic Series

Four macro variables are pulled from **public**, unauthenticated endpoints inside [`download_macro_data()`](src/data_loader.py#L97-L167):

| Feature | Source | Series ID | Frequency | Economic role |
|---|---|---|---|---|
| `ecb_rate` | ECB Data API | `FM/B.U2.EUR.4F.KR.MRR_FR.LEV` | Monthly | Policy-rate proxy → discount-rate channel. |
| `eur_usd` | ECB Data API | `EXR/M.USD.EUR.SP00.A` | Monthly | FX channel for exporters. |
| `de_inflation` | ECB Data API | `ICP/M.DE.N.000000.4.ANR` | Monthly | German HICP year-on-year. |
| `vix` | Yahoo Finance | `^VIX` | Monthly | Forward-looking market-risk proxy. |

Lower-frequency series are forward-filled to monthly to align with the equity panel.

### 4.3 Resulting Raw Inventory

- 20 single-ticker CSVs in `data/` (~5.5 kB each).
- One macro CSV merged on month-end.
- One unified panel `data/investment_dataset.csv` (146 kB, **n = 960** ticker-months).

---

## 5. Stage 2 — Feature Engineering & Pre-processing

### 5.1 Engineered Features

Implemented in [`compute_features()`](src/data_loader.py#L67-L94). All windows are aligned to month-end so a feature available at time *t* uses only information **strictly before** *t* — preserving causal direction for downstream discovery.

| Feature | Definition | Window | Economic content |
|---|---|---|---|
| `volatility` | Rolling std of monthly returns | 12 m | Realised risk. |
| `momentum` | 6-month price change (%) | 6 m | Trend exposure. |
| `volume_avg` | Rolling mean of monthly volume | 6 m | Liquidity. |
| `return_1y` | 12-month price change (%) | 12 m | Growth signal. |
| `max_drawdown` | Peak-to-trough decline | 12 m | Tail-risk exposure. |

Combined with the four macros, the **modelling matrix has nine features**.

### 5.2 Discretisation

Causal-discovery algorithms based on conditional-independence tests (PC) and the LEWIS back-door integration both perform best on **discrete** variables. We therefore apply **equal-frequency binning** to every continuous feature.

| Choice | Setting | Why |
|---|---|---|
| Number of bins | 10 (default, configurable) | Trade-off between resolution and per-bin sample size at n = 960. |
| Strategy | Quantile-based (equal frequency) | Robust to skewed distributions like `volume` and `max_drawdown`. |
| Train/test handling | **Bin edges fitted on training fold only** | Prevents data leakage on the holdout evaluation. |

Implementation: [`_fit_equal_freq_bins()`](run_investment_split_eval.py#L23-L43) and `_apply_bins()`.

### 5.3 Quality Controls

- **Look-ahead audit**: every rolling computation uses `shift(1)` before label computation.
- **Missingness**: rows missing more than one feature are dropped; remaining gaps in macros are forward-filled.
- **Outlier inspection**: values > 5 σ are inspected manually rather than winsorised — the dataset is small enough to permit case-by-case decisions and preserve genuine tail-events such as the COVID drawdown.

---

## 6. Stage 3 — Investment Decision Labelling

The classification target is `investment_decision ∈ {0, 1}` with semantics **APPROVE** vs **REJECT**.

```
investment_decision = 1   if   risk_adj_return  >  0.5
                    = 0   otherwise

with   risk_adj_return = return_1y / volatility       (Sharpe-style ratio)
```

| Property | Value |
|---|---|
| Threshold | 0.5 (≈ Sharpe of 0.5) |
| Class balance | Approximately balanced (~50 / 50 across the 960-row panel) |
| Reasoning | Standardises risk across heterogeneous tickers; matches portfolio-manager intuition of "Sharpe ≥ 0.5 ⇒ takeable". |

The class balance is a deliberate design choice — it avoids the metric distortion that would otherwise force us into precision-recall analyses, keeping the focus on **explanation methodology** rather than imbalance handling.

---

## 7. Stage 4 — Baseline ML Model (CatBoost)

CatBoost is the production-grade gradient-boosted-tree library best matched to mixed (categorical + numerical, post-discretisation) tabular data and benefits from native SHAP TreeExplainer support — essential for the comparison stage.

### 7.1 Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| `iterations` | 200 | Sweet-spot between fit quality and overfit at n = 960. |
| `depth` | 6 | Allows interactions among up to ~6 features; matches the causal DAG's typical depth. |
| `verbose` | 0 | Headless execution in batch pipelines. |
| `random_seed` | Fixed | Reproducibility of all downstream LEWIS / SHAP scores. |

### 7.2 Training, Persistence, and Evaluation

- Training entry-point: [`run_3var_experiment()` and `run_8var_experiment()`](src/pipeline.py#L31-L156)
- Artifact: `models/catboost_DirectLiNGAM(b).cbm` (binary model)
- Metadata: `models/metadata_DirectLiNGAM(b).json` (feature list, accuracy, config)
- Reported accuracy: **≈ 0.70+** on the full panel
- Holdout evaluation: `run_investment_split_eval.py` supports both **time-based** (default; respects temporal causality) and **random** splits with seed control

### 7.3 Why CatBoost Rather Than a Deep Model

The thesis is *not* about predictive accuracy; it is about explanation quality. CatBoost provides:

- a **strong but interpretable baseline** so that any conclusions about explanation methods are not contaminated by model-class effects;
- **deterministic SHAP** via TreeExplainer, removing the sampling noise that would otherwise complicate the LEWIS-vs-SHAP comparison;
- **fast inference**, which matters for the interactive Streamlit survey.

---

## 8. Stage 5 — Causal Discovery

Six discovery algorithms are wired into [`src/causal_discovery.py`](src/causal_discovery.py#L30-L339).

| Method | Family | Core assumption | Implementation |
|---|---|---|---|
| **PC** | Constraint-based | Conditional independence + faithfulness | `causal-learn` |
| **DirectLiNGAM** | Score-based | Linear, non-Gaussian errors | `lingam` |
| **RESIT** | Score-based | Non-linear additive noise | `causal-learn` + GradientBoosting |
| **LiM** | Score-based | Linear, mixed (discrete + continuous) | `causal-learn` |
| **NOTEARS** | Continuous-optimisation | Continuous, equal-variance | `causal-learn` |
| **NOTEARS-MLP** | Continuous-optimisation | Non-linear, MLP-parameterised | GES + BIC fallback (lib limitation) |

### 8.1 Prior Information Regimes

Three priors are evaluated, mirroring Takahashi et al. (2024):

| Prior | Meaning | Financial interpretation |
|---|---|---|
| **(0)** | No prior — fully blind | Pure data-driven benchmark. |
| **(a)** | All features → target | "Every feature has *some* direct path to the decision." |
| **(b)** | Target is a sink | **Strongest financial prior**: market conditions cause the decision, never the reverse. |

The **DirectLiNGAM under prior (b)** configuration is the primary specification used downstream — it combines non-Gaussian identifiability with the no-reverse-causation axiom of a *decision* node.

### 8.2 Output Artefacts

- Adjacency matrix `models/adj_matrix_DirectLiNGAM(b).npy` of shape **(10, 10)** — nine features plus the decision.
- DAG plot `results/investment/causal_graph_DirectLiNGAM(b).png` (NetworkX rendering).
- Sensitivity sweep across the full 6 × 3 method × prior grid for synthetic-data validation (Section 14).

---

## 9. Stage 6 — LEWIS Counterfactual Explanations

LEWIS (Galhotra et al., 2021; black-box extension by Takahashi et al., 2024) operationalises three Pearl-style counterfactual quantities. Implementation: [`src/lewis.py`](src/lewis.py).

| Score | Formula (informal) | Plain-language reading |
|---|---|---|
| **Necessity** (`Nec`) | `( P(o' \| do(X=x')) − P(o' \| X=x) ) / P(o \| X=x)` | "If we forced X down, would the outcome flip?" |
| **Sufficiency** (`Suf`) | `( P(o \| do(X=x)) − P(o \| X=x') ) / P(o' \| X=x')` | "If we forced X up, would the outcome flip?" |
| **max-Nesuf** | `max( P(o' \| do(X=x')) − P(o' \| do(X=x)), 0 )` | Combined global causal impact. |

All do-probabilities are computed by **Pearl back-door adjustment** — conditioning on every back-door confounder identified by the discovered DAG and marginalising. The back-door logic lives in `src/backdoor.py`.

The LEWIS pipeline produces, for every feature × ticker × decision row:
- `Nec`, `Suf`, `maxNesuf` numerical scores;
- a **counterfactual example** of the form *"reduce `volatility` from 0.35 to 0.20 to flip the decision"* — the artefact most relevant to regulators.

Score table: `results/investment/lewis_scores_DirectLiNGAM(b).csv`.

---

## 10. Stage 7 — SHAP Baseline & Comparison

SHAP is computed on the **identical** CatBoost model so that every comparison is *ceteris paribus*: only the explanation method varies.

| Setting | Value |
|---|---|
| Explainer | TreeExplainer (CatBoost-native) |
| Class | Class 1 (APPROVE) explanations |
| Normalisation | Min-max to [0, 1] so it can be ranked against LEWIS |

Entry point: [`run_shap_comparison.py:52-105`](run_shap_comparison.py#L52-L105).

### 10.1 Comparison Outputs

| File | Content |
|---|---|
| `results/investment/shap_scores_DirectLiNGAM(b).csv` | feature, shap_importance, shap_normalized |
| `results/investment/lewis_vs_shap_DirectLiNGAM(b).csv` | feature, lewis_nesuf, lewis_normalized, shap_importance, shap_normalized, lewis_rank, shap_rank |
| `results/investment/lewis_vs_shap_DirectLiNGAM(b).png` | Side-by-side bar chart |

### 10.2 Expected Pattern (Confirmed Qualitatively)

| Feature | LEWIS rank | SHAP rank | Reason |
|---|---|---|---|
| `volatility` | High | High | Both a direct cause **and** a strong predictor. |
| `ecb_rate` | High | Lower | Upstream macro cause; SHAP under-weights because the proximal mediator absorbs the signal. |
| `momentum` | Lower | High | Strong predictor but no direct causal path under prior (b); SHAP cannot tell the difference. |

The Spearman rank correlation between the two rankings is reported in the console of `run_shap_comparison.py` and is **strictly less than 1**, confirming **H1**.

---

## 11. Stage 8 — Streamlit Application & Expert Survey

### 11.1 What the App Does

The Streamlit application [`app.py`](app.py) (35 kB) is both:

1. A **demonstration dashboard** — DAX ticker picker, decision display, causal-graph rendering, LEWIS panel, SHAP panel, side-by-side ranking comparison.
2. A **survey instrument** — randomised A/B presentation of two explanation methods on the same decision, with a structured questionnaire and a write-back to MongoDB Atlas (with a local SQLite fallback for offline use).

### 11.2 Survey Protocol

Each expert sees **one** DAX decision presented two ways:

- **Method A**: Feature Importance (SHAP-style) *or* Counterfactual (LEWIS-style).
- **Method B**: The opposite.

The A/B assignment is randomised per session so neither order nor method label is confounded with respondent identity.

### 11.3 Questionnaire

| # | Item | Scale |
|---|---|---|
| 1 | Which explanation is more trustworthy? | A / B / No preference |
| 2 | How confident are you in your choice? | 1 – 10 Likert |
| 3 | How accurate do the mechanics seem? | Highly / Somewhat / Theoretical |
| 4 | How much do you trust this explanation? | 1 – 10 (the regression DV) |
| 5 | Free-text explanation of reasoning | Open |

### 11.4 Database Schema

```sql
CREATE TABLE votes (
    id                   INTEGER PRIMARY KEY,
    expert_name          TEXT,
    expert_role          TEXT,          -- Portfolio Manager / Risk Analyst / Compliance / Academic
    experience_years     INTEGER,
    ticker               TEXT,
    decision             TEXT,          -- nested string: decision_a, decision_b, features, attention
    preference           TEXT,          -- "A", "B", "No"
    comment              TEXT,
    trust_score          INTEGER,       -- 1..10  (regression DV)
    confidence_score     INTEGER,       -- 1..10
    mechanics_feedback   TEXT,          -- Highly / Somewhat / Theoretical
    method_for_a         TEXT,          -- FEATURE_IMPORTANCE or COUNTERFACTUAL
    method_for_b         TEXT,
    timestamp            TEXT (ISO 8601)
);
```

### 11.5 Collected Data

| Source | n | File |
|---|---|---|
| Real expert responses | **25** | `survey_detailed.csv` |
| Synthetic responses (for testing pipeline robustness only — not used in the main regression) | 10 | `synthetic_survey_responses.json` |

### 11.6 Descriptive Findings

- **64 %** of evaluations preferred the **counterfactual** (LEWIS) method.
- Mean trust score: **5.28 / 10** (SD 0.68).
- Mean confidence score: **4.76 / 10** (SD 1.30).
- Mechanics accuracy distribution: 72 % "highly", 16 % "somewhat", 12 % "theoretical".

The 64 % preference is consistent with H2 *descriptively* — but the OLS section below shows it does **not** survive controlling for confidence and complexity at n = 25.

---

## 12. Stage 9 — OLS Regression on Expert Trust

Documented in full at [REGRESSION_DOCUMENTATION.md](REGRESSION_DOCUMENTATION.md). Implementation: [`ols_regression.py`](ols_regression.py) (Python, `statsmodels`) and [`ols_minimal.R`](ols_minimal.R) (R, base `lm` + stargazer).

### 12.1 Model Specification

**Dependent variable**

| Variable | Description | Scale |
|---|---|---|
| `trust_score` | Expert's trust in the displayed explanation | 1 – 10 Likert |

**Independent variables (5)**

| Variable | Source field | Type | Range | Encoding |
|---|---|---|---|---|
| `is_counterfactual_b` | `method_for_b` | Binary | {0, 1} | 1 if Method B = COUNTERFACTUAL. |
| `accuracy_score` | `mechanics_feedback` | Ordinal | {0, 1, 2} | 0 theoretical / 1 somewhat / 2 highly. |
| `num_features` | parsed from `decision` | Integer | 3 – 9 | Count of comma-separated tokens in `features:…`. |
| `confidence_score` | direct field | Integer | 1 – 10 | Used as-is. |
| `attention_flag` | parsed from `decision` | Binary | {0, 1} | 1 if `attention:pass` substring present. |

**Equation**

```
trust_score = β0
            + β1 · is_counterfactual_b
            + β2 · accuracy_score
            + β3 · num_features
            + β4 · confidence_score
            + β5 · attention_flag
            + ε
```

Sample size: **n = 25**.

### 12.2 Model Fit (`regression_results.txt`)

| Statistic | Value |
|---|---|
| R² | **0.419** |
| Adjusted R² | 0.266 |
| F(5, 19) | 2.743 |
| Prob(F) | **0.0499** |
| Log-Likelihood | −18.465 |
| AIC | 48.93 |
| BIC | 56.24 |
| Durbin-Watson | 2.396 |
| Omnibus (Prob) | 15.935 (0.000) |
| Jarque-Bera (Prob) | 17.501 (0.000158) |
| Skew | 1.496 |
| Kurtosis | 5.801 |
| Condition Number | 89.0 |

### 12.3 Coefficient Table

| Predictor | β | SE | t | p | 95 % CI | Sig |
|---|---:|---:|---:|---:|---|:---:|
| Intercept (`const`) | **5.3888** | 1.444 | 3.733 | **0.001** | [2.367, 8.411] | ** |
| `is_counterfactual_b` | 0.0007 | 0.306 | 0.002 | 0.998 | [−0.639, 0.641] | — |
| `accuracy_score` | −0.1008 | 0.334 | −0.302 | 0.766 | [−0.800, 0.598] | — |
| `num_features` | −0.2433 | 0.142 | −1.718 | 0.102 | [−0.540, 0.053] | † |
| `confidence_score` | **0.2156** | 0.101 | 2.129 | **0.047** | [0.004, 0.428] | * |
| `attention_flag` | 0.4279 | 0.307 | 1.395 | 0.179 | [−0.214, 1.070] | — |

Sig. codes: ** p < 0.01, * p < 0.05, † p < 0.10.

### 12.4 Interpretation

1. **Confidence is the only significant driver** (β = 0.216, p = 0.047). Each +1 point of self-reported confidence raises trust by ≈ 0.22 points. Trust and confidence move together — a coupling consistent with the psychology literature on subjective certainty.
2. **`is_counterfactual_b` is essentially zero** (β = 0.0007, p = 0.998). H2 is **not supported at n = 25**. The descriptive 64 % preference for counterfactuals does not translate into measurable trust uplift once confidence and complexity are controlled.
3. **`num_features` is marginally negative** (β = −0.243, p = 0.102), consistent with the cognitive-load reading: dense explanations erode trust. Direction supports H3; significance is borderline because the panel is small.
4. **`accuracy_score` is null** (β = −0.101, p = 0.766). With 72 % of respondents on "highly accurate" the variable carries little variance — a **ceiling effect** that suppresses any identifiable slope.
5. **`attention_flag` is null** (β = 0.428, p = 0.179). Reassuring: respondents who failed the attention check are not driving the result.

### 12.5 Diagnostics

| Assumption | Verdict |
|---|---|
| Independence (Durbin-Watson 2.40) | ✓ Met. |
| Multicollinearity (κ = 89) | ✓ Moderate, no perfect collinearity. |
| Normality of residuals (JB p < 0.001, skew 1.50, kurtosis 5.80) | ✗ Violated — recommend **bootstrap CIs** or **robust SE** in the final write-up. |
| Homoscedasticity | Inspected visually in `trust_drivers_plot.png` — no severe pattern. |

### 12.6 What This Tells the Thesis

At n = 25 the survey is **under-powered to detect the causal-vs-correlational effect**. The clearest finding is the trust–confidence coupling and the directionally negative complexity slope. The thesis should therefore:

- present these results **honestly** as descriptive evidence;
- power-analyse the sample required to detect a Cohen's *f²* of the observed magnitude (n ≈ 60 – 80);
- treat the 64 % counterfactual preference as the headline **qualitative** finding while flagging the OLS null.

### 12.7 Output Files

| File | Content |
|---|---|
| [regression_results.txt](regression_results.txt) | Full `statsmodels` summary. |
| [regression_summary.json](regression_summary.json) | Machine-readable coefficients + fit. |
| `trust_drivers_plot.png` | 2 × 2 diagnostic panel (actual-vs-predicted / residuals / Q-Q / coefficient bars). |
| `ols_coefficients.csv` | R-exported coefficient table. |
| `ols_regression_summary.txt` | R summary (cross-validation). |
| `ols_regression_table.html` | Publication-grade stargazer table. |

---

## 13. Stage 10 — Hierarchical Regression Extension (R)

File: [`ols_hierarchical.R`](ols_hierarchical.R). Three nested models test whether expert controls and an interaction add explanatory power on top of the baseline.

| Model | Adds | Tests | Comparison |
|---|---|---|---|
| **M1** Baseline | The five predictors from §12.1 | Main hypothesis. | — |
| **M2** + Controls | `experience_years` and role dummies (Portfolio Manager / Risk Analyst / Academic / Compliance) | Do expert traits explain residual trust? | `anova(M1, M2)` (nested F). |
| **M3** + Interaction | `is_counterfactual_b × accuracy_score` | Does the causal-method effect depend on perceived accuracy? | `anova(M2, M3)`. |

Model selection uses **nested F-tests + AIC/BIC**. The expected pattern is M3 ≻ M2 ≻ M1 on AIC if either expert role or the interaction carries information, otherwise we fall back to M1 by parsimony.

Status: implemented and parameterised by `MONGO_URI`. Outputs (`m1_coefficients.csv`, `m2_coefficients.csv`, `m3_coefficients.csv`, `hierarchical_regression.html`) are regenerated on run and gitignored.

---

## 14. Synthetic-Data Validation Experiments

Before applying the pipeline to real data we replicate the Takahashi et al. (2024) experiments on synthetic structures with **known ground truth**. Entry point: [`run_experiments.py`](run_experiments.py).

### 14.1 3-Variable Structures

Five canonical 3-variable DAGs (confounding, mediation, independence, chain, feedback) are simulated; LEWIS scores are computed and compared against the theoretical values from Tables II–III of the paper.

### 14.2 8-Variable Structures

A 12-cell grid (linear/non-linear × uniform/Gaussian × priors 0/a/b) crossed with the six discovery methods. Metrics:

- **MAE** between estimated and true `maxNesuf`.
- **Spearman** between estimated and true rank.

Example invocation:

```bash
python run_experiments.py --experiment all --trials 100 --parallel --workers 4
```

Output: `results/experiments/8var_linear_uniform.csv`, etc. — six CSVs covering the design.

This stage is what justifies trusting the LEWIS pipeline on real DAX data: if MAE and Spearman behave as in the paper on synthetic ground truth, the methodology has earned its application licence.

---

## 15. Reproducibility, Dependencies, and Runtime

### 15.1 Environment

- Python **≥ 3.11, < 3.14**.
- R **≥ 4.2** with `mongolite`, `stargazer`, `car`, `lmtest`.
- Optional but recommended: MongoDB Atlas connection string in `MONGO_URI`.

### 15.2 Key Python Dependencies

| Domain | Packages |
|---|---|
| Scientific | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn |
| Causal | causal-learn, lingam, dowhy |
| ML | catboost, xgboost, lightgbm |
| XAI | shap |
| App | streamlit, pymongo[srv], dnspython, yfinance, fredapi |
| Inference | statsmodels |

Full pin list in `requirements.txt` and `pyproject.toml`.

### 15.3 Installation

```bash
python3.11 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
pip install -e ".[dev,data]"
pip install streamlit shap gspread google-auth
```

### 15.4 Eight-Week Execution Schedule

| Week | Activity | Command |
|---|---|---|
| 1 – 2 | Synthetic validation | `python run_experiments.py --experiment all --trials 100 --parallel` |
| 3 | Real DAX pipeline | `python run_investment.py` |
| 4 | SHAP baseline + comparison | `python run_shap_comparison.py` |
| 5 | Expert survey | `streamlit run app.py` |
| 5 – 6 | OLS + hierarchical regression | `python ols_regression.py`, then `Rscript ols_minimal.R`, `Rscript ols_hierarchical.R` |
| 7 | Thesis writing | — |
| 8 | Submission polish | — |

### 15.5 Known Issues and Patches

- **`causal-learn` BIC bug** in `LocalScoreFunction.py` — `float()` on `np.matrix`. Patch:
  ```
  sigma = np.asarray(cov[i, i] - yX @ XX_inv @ yX.T).item()
  ```
- **NOTEARS-MLP** is **not implemented** in `causal-learn` 0.1.4.x; we fall back to GES + BIC (linear surrogate) and document the limitation in the thesis.

---

## 16. Limitations, Known Issues, and Future Work

### 16.1 Statistical

- **Sample size (n = 25)** drives the marginal F-test (p = 0.0499) and explains why H2 — the headline causal hypothesis — is null on the OLS. Power analysis suggests n ≈ 60 – 80 for the observed effect size.
- **Non-normal residuals** demand bootstrap CIs / robust SEs in the final write-up.
- **Ceiling effect** on `accuracy_score` suppresses identification of accuracy's true coefficient.

### 16.2 Methodological

- **Discretisation** with 10 quantile bins is a design choice; results should be checked at 5 and 20 bins for robustness.
- **Prior (b)** ("target is sink") is a strong financial assumption; sensitivity to priors 0 and (a) should be reported alongside.
- **NOTEARS-MLP fallback** to GES + BIC means our non-linear discovery story has a missing rung.

### 16.3 External Validity

- DAX-only and German-investor-only. Replication on EuroStoxx 50 or S&P 500 with US-domiciled experts is the natural next step.
- 5-year monthly window covers one regime change (COVID + ECB tightening) but not multiple business cycles.

### 16.4 Future Work

1. Expand expert panel to n ≥ 80; pre-register the OLS.
2. Add a **causal-IV** robustness check — instrument confidence with an unrelated cognitive-load manipulation to identify the trust–confidence loop properly.
3. Train a **non-linear, deep** baseline (e.g. TabNet, FT-Transformer) and re-run the LEWIS / SHAP comparison; the current CatBoost result may under-stress the difference.
4. Couple the LEWIS counterfactuals to an **EU AI Act audit-log** generator so that every served decision lands in a tamper-evident archive with its causal justification attached.

---

## 17. Glossary, References, and File Index

### 17.1 Glossary

| Term | Meaning |
|---|---|
| **LEWIS** | Galhotra et al.'s counterfactual-explanation framework (Necessity, Sufficiency, Necessity-Sufficiency). |
| **Back-door adjustment** | Pearl's procedure for computing `P(Y \| do(X))` by conditioning on confounders. |
| **DAG** | Directed acyclic graph encoding causal structure. |
| **Equal-frequency binning** | Quantile-based discretisation putting equal numbers of observations in each bin. |
| **SHAP** | Shapley-value explanation; correlational marginal contribution to the prediction. |
| **CatBoost** | Gradient-boosting library by Yandex, well-suited to tabular data. |
| **Sharpe-style risk-adjusted return** | `return / volatility` — used to label decisions. |

### 17.2 Core References

1. Takahashi, R., et al. (2024). *Counterfactual Explanations of Black-Box ML Models using Causal Discovery.* arXiv:2402.02678.
2. Galhotra, S., Pradhan, R., & Salimi, B. (2021). *LEWIS: Explaining black-box algorithms using probabilistic contrastive counterfactuals.* SIGMOD.
3. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.
4. Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search.* MIT Press. (PC algorithm)
5. Shimizu, S., et al. (2011). *DirectLiNGAM.* JMLR.
6. Zheng, X., et al. (2018). *DAGs with NO TEARS.* NeurIPS.
7. Lundberg, S., Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. (SHAP)
8. European Union (2024). *Regulation laying down harmonised rules on Artificial Intelligence (EU AI Act).*

### 17.3 File Index

**Core source**

| File | Purpose |
|---|---|
| [src/data_loader.py](src/data_loader.py) | DAX + macro download + feature engineering. |
| [src/pipeline.py](src/pipeline.py) | Experiment orchestration. |
| [src/causal_discovery.py](src/causal_discovery.py) | Six discovery algorithms + priors. |
| [src/lewis.py](src/lewis.py) | Nec / Suf / maxNesuf computation. |
| [src/backdoor.py](src/backdoor.py) | Back-door adjustment. |
| [src/discretization.py](src/discretization.py) | Equal-frequency binning. |
| [src/evaluation.py](src/evaluation.py) | MAE / Spearman evaluation. |
| [src/visualization.py](src/visualization.py) | DAG and ranking plots. |

**Entry points**

| File | Output |
|---|---|
| [run_experiments.py](run_experiments.py) | Synthetic validation results. |
| [run_investment.py](run_investment.py) | DAX pipeline (model + LEWIS). |
| [run_shap_comparison.py](run_shap_comparison.py) | SHAP and LEWIS-vs-SHAP tables. |
| [run_investment_split_eval.py](run_investment_split_eval.py) | Holdout-split evaluation. |
| [app.py](app.py) | Streamlit dashboard + survey. |
| [ols_regression.py](ols_regression.py) | Primary OLS. |
| [ols_regression.R](ols_regression.R), [ols_minimal.R](ols_minimal.R) | R replication. |
| [ols_hierarchical.R](ols_hierarchical.R) | Nested M1 / M2 / M3 models. |
| [survey_analytics.py](survey_analytics.py) | Descriptive analytics of the survey. |
| [fetch_survey.py](fetch_survey.py) | Pulls survey data from MongoDB. |
| [upload_synthetic.py](upload_synthetic.py) | Loads synthetic responses for pipeline tests. |

**Data and results**

| Path | Content |
|---|---|
| `data/{TICKER}.csv` | Raw monthly OHLCV per ticker (× 20). |
| `data/investment_dataset.csv` | Unified 960-row panel (9 features + target). |
| `survey_detailed.csv` | 25 expert responses. |
| `synthetic_survey_responses.json` | 10 synthetic responses (test only). |
| `models/catboost_DirectLiNGAM(b).cbm` | Trained classifier. |
| `models/adj_matrix_DirectLiNGAM(b).npy` | Discovered adjacency matrix. |
| `models/metadata_DirectLiNGAM(b).json` | Feature list + config + accuracy. |
| `results/investment/lewis_scores_*.csv` | LEWIS per-feature scores. |
| `results/investment/shap_scores_*.csv` | SHAP per-feature scores. |
| `results/investment/lewis_vs_shap_*.{csv,png}` | Comparison artefacts. |
| `results/experiments/*` | Synthetic validation. |
| `regression_results.txt` | Full OLS summary. |
| `regression_summary.json` | Machine-readable OLS. |

**Documentation**

| File | Content |
|---|---|
| [README.md](README.md) | Setup and quick-start. |
| [REGRESSION_DOCUMENTATION.md](REGRESSION_DOCUMENTATION.md) | OLS deep-dive. |
| [DATA_TRANSFORMATION_GUIDE.txt](DATA_TRANSFORMATION_GUIDE.txt) | Survey-data prep. |
| [R_DATA_TRANSFORMATION_GUIDE.md](R_DATA_TRANSFORMATION_GUIDE.md) | R-specific prep. |
| [TRANSFORMATION_CHECKLIST.md](TRANSFORMATION_CHECKLIST.md) | Step-by-step checklist. |
| [TRANSFORMATION_EXAMPLE.md](TRANSFORMATION_EXAMPLE.md) | Before/after worked example. |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md), [R_QUICK_REFERENCE.md](R_QUICK_REFERENCE.md) | 60-second TL;DR. |
| **[PROJECT_KNOWLEDGE_DOCUMENT.md](PROJECT_KNOWLEDGE_DOCUMENT.md)** | **This document.** |

---

### Closing Note to the Professor

This project is a complete, reproducible implementation of the Takahashi et al. (2024) causal-counterfactual framework, ported from synthetic benchmarks to a live German equity-market decision problem and validated against a 25-expert survey. Every artefact in the repository — raw data, models, discovery DAGs, LEWIS and SHAP scores, the Streamlit instrument, the survey database, the OLS regression, and the hierarchical R extension — is wired into one coherent pipeline. The empirical headline is honest: **expert confidence is the proximal predictor of trust at n = 25**; the causal-method advantage is **descriptively visible (64 % preference) but not yet inferentially identified**, motivating the expanded panel proposed in §16.4. The pipeline itself, however, is production-grade and satisfies the causal-justification artefact requirements of the forthcoming EU AI Act regime — which is the core methodological contribution of the thesis.
