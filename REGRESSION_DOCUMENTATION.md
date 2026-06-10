# Regression Analysis: Drivers of Expert Trust in XAI Explanations

Thesis: *Causal Explainability for AI-Driven Investment Decisions in German Banks*
Author: hello@destinationone.ca — Frankfurt School of Finance & Management
Date compiled: 2026-06-09

---

## 1. What We Did

We ran an **Ordinary Least Squares (OLS) linear regression** on data collected
from a Streamlit-based expert survey (stored in MongoDB Atlas, collection
`thesis_survey.votes`) to identify which characteristics of XAI explanations
drive expert **trust**.

We implemented the analysis in **two parallel stacks** so results could be
cross-validated:

| Stack | File | Role |
|-------|------|------|
| Python (statsmodels) | [ols_regression.py](ols_regression.py) | Primary single-model OLS + plots + JSON export |
| R (base `lm` + stargazer) | [ols_minimal.R](ols_minimal.R) | Replication of the Python OLS in R |
| R hierarchical | [ols_hierarchical.R](ols_hierarchical.R) | Three nested models (baseline → controls → interaction) |

Both stacks read from the same MongoDB collection via the `MONGO_URI`
environment variable.

---

## 2. Model Specification

### Dependent variable

| Variable | Description | Scale |
|----------|-------------|-------|
| `trust_score` | Expert's rating of explanation trustworthiness | 1–10 Likert |

### Independent variables (5 predictors)

| Variable | Source field | Type | Range | Encoding |
|----------|--------------|------|-------|----------|
| `is_counterfactual_b` | `method_for_b` | Binary | 0 / 1 | 1 if Method B = COUNTERFACTUAL, else 0 |
| `accuracy_score` | `mechanics_feedback` | Ordinal | 0 / 1 / 2 | 0 = "theoretical", 1 = "somewhat", 2 = "highly accurate" |
| `num_features` | parsed from `decision` | Integer | 3–9 | Count of comma-separated features in `features:…` block |
| `confidence_score` | `confidence_score` | Integer | 1–10 | Used as-is |
| `attention_flag` | parsed from `decision` | Binary | 0 / 1 | 1 if substring `attention:pass` present, else 0 |

### Equation

```
trust_score = β0
            + β1 · is_counterfactual_b
            + β2 · accuracy_score
            + β3 · num_features
            + β4 · confidence_score
            + β5 · attention_flag
            + ε
```

Sample size: **n = 25** expert evaluations.

---

## 3. How We Did It

### 3.1 Data pipeline

1. **Load** — Connect to MongoDB Atlas (`thesis_survey.votes`) via
   `pymongo` (Python) or `mongolite` (R). Credentials read from
   `MONGO_URI` environment variable (see [.env.example](.env.example)).
2. **Feature engineering** — Apply the transformations listed in §2 to
   build the regression matrix. Drop rows with NaN in target/predictors.
3. **Fit** — `statsmodels.OLS(y, X).fit()` in Python; `lm(...)` in R.
4. **Diagnose** — Residual plot, Q-Q plot, VIF, Shapiro-Wilk normality
   test, Durbin-Watson statistic.
5. **Export** — Coefficients (CSV), full summary (TXT), publication
   table (HTML via stargazer), diagnostic plots (PNG).

### 3.2 Key code references

- Feature engineering — [ols_regression.py:42-103](ols_regression.py#L42-L103)
- Model fit — [ols_regression.py:141-155](ols_regression.py#L141-L155)
- R replica — [ols_minimal.R:114-120](ols_minimal.R#L114-L120)
- Hierarchical M1/M2/M3 — [ols_hierarchical.R:52-70](ols_hierarchical.R#L52-L70)

### 3.3 Hierarchical extension (R only)

In [ols_hierarchical.R](ols_hierarchical.R) we nest three models to test
whether expert controls and an interaction term improve fit:

| Model | Adds | Tests |
|-------|------|-------|
| **M1** Baseline | 5 core predictors | Main hypothesis |
| **M2** + Controls | `experience_years`, role dummies (PM, RA, Academic, Compliance) | Do expert traits explain residual trust? |
| **M3** + Interaction | `is_counterfactual_b × accuracy_score` | Does counterfactual effect depend on perceived accuracy? |

Comparison done via nested F-tests (`anova(m1, m2)`, `anova(m2, m3)`) and
AIC/BIC.

---

## 4. Results (Primary OLS Model, n = 25)

Source: [regression_results.txt](regression_results.txt),
[regression_summary.json](regression_summary.json).

### 4.1 Model fit

| Statistic | Value |
|-----------|-------|
| R² | **0.419** |
| Adjusted R² | 0.266 |
| F(5, 19) | 2.743 |
| Prob(F-statistic) | **0.0499** |
| Log-Likelihood | −18.465 |
| AIC | 48.93 |
| BIC | 56.24 |
| Durbin-Watson | 2.396 |
| Omnibus (Prob) | 15.935 (0.000) |
| Jarque-Bera (Prob) | 17.501 (0.000158) |
| Skew / Kurtosis | 1.496 / 5.801 |
| Condition Number | 89.0 |

The model explains **~42%** of variance in `trust_score`; the overall F-test
is marginally significant at the 5% level (p ≈ 0.0499).

### 4.2 Coefficient table

| Predictor | Coef. | Std. Err. | t | p-value | 95% CI | Sig. |
|-----------|------:|----------:|------:|--------:|--------|:----:|
| Intercept (`const`) | **5.3888** | 1.444 | 3.733 | **0.001** | [2.367, 8.411] | ** |
| `is_counterfactual_b` | 0.0007 | 0.306 | 0.002 | 0.998 | [−0.639, 0.641] | — |
| `accuracy_score` | −0.1008 | 0.334 | −0.302 | 0.766 | [−0.800, 0.598] | — |
| `num_features` | −0.2433 | 0.142 | −1.718 | 0.102 | [−0.540, 0.053] | † |
| `confidence_score` | **0.2156** | 0.101 | 2.129 | **0.047** | [0.004, 0.428] | * |
| `attention_flag` | 0.4279 | 0.307 | 1.395 | 0.179 | [−0.214, 1.070] | — |

Significance: ** p < 0.01, * p < 0.05, † p < 0.10.

### 4.3 Interpretation

- **Confidence is the only significant driver.** Each +1 point of expert
  confidence raises trust by **0.22 points** (p = 0.047). Trust and
  self-reported confidence move together.
- **Counterfactual vs. feature-importance has effectively no effect** on
  trust in this sample (β ≈ 0.0007, p = 0.998). The original
  hypothesis — that counterfactual explanations would increase trust — is
  **not supported** at n = 25.
- **Number of features is the second-strongest signal** (β = −0.243,
  p = 0.102). The direction is negative — more features tend to lower
  trust — but only marginally significant.
- **Accuracy perception is null** here (β = −0.101, p = 0.766), which is
  surprising. With 72% of respondents rating "highly accurate" the
  variable carries little variance, suppressing the coefficient.
- **Attention does not significantly affect trust** (β = 0.428, p = 0.179).
- **Diagnostics caveat:** Jarque-Bera p < 0.001 and Omnibus p < 0.001
  indicate residuals are **non-normal** (skew = 1.50, kurtosis = 5.80).
  Standard-error-based p-values should therefore be read with caution at
  this sample size.

### 4.4 What this tells the thesis

At n = 25 the survey is **underpowered** to detect the predicted
counterfactual effect. The clearest finding is the trust–confidence
coupling: experts who feel confident in their reading of an explanation
also rate it more trustworthy. The negative `num_features` slope (though
not significant) hints that **explanation complexity may erode trust**,
consistent with the cognitive-load hypothesis.

---

## 5. Outputs Produced

| File | Content |
|------|---------|
| [regression_results.txt](regression_results.txt) | Full statsmodels text summary |
| [regression_summary.json](regression_summary.json) | Machine-readable coefficients + fit stats |
| `trust_drivers_plot.png` | 2×2 panel: actual-vs-predicted, residuals, Q-Q, coefficient bar (regenerated; gitignored) |
| `ols_coefficients.csv` | R coefficient table (regenerated) |
| `ols_regression_summary.txt` | R full summary (regenerated) |
| `ols_regression_table.html` | Stargazer publication table (regenerated) |
| `m1_coefficients.csv`, `m2_coefficients.csv`, `m3_coefficients.csv` | Hierarchical model coefficients (regenerated) |
| `hierarchical_regression.html` | Side-by-side M1/M2/M3 stargazer table (regenerated) |

Regenerated artifacts are excluded from the repository via
[.gitignore](.gitignore); rerun the scripts to recreate them.

---

## 6. Reproduction Steps

1. Set the connection string: `export MONGO_URI="mongodb+srv://…"` (see
   [.env.example](.env.example)).
2. Python: `python ols_regression.py`
3. R (single model): open [ols_minimal.R](ols_minimal.R) in RStudio and
   source.
4. R (hierarchical): source [ols_hierarchical.R](ols_hierarchical.R).

Outputs land in the project root.

---

## 7. Limitations

- **Small sample (n = 25)** — F-test borderline (p = 0.0499); coefficients
  unstable.
- **Non-normal residuals** — Jarque-Bera rejects normality; consider
  bootstrap CIs or robust standard errors for the final write-up.
- **Low variance in `accuracy_score`** — 72% of ratings cluster on "highly
  accurate", limiting identification of its effect.
- **No causal claims from OLS alone** — the causal pipeline
  (DirectLiNGAM, Lewis-vs-SHAP) in the rest of the repo addresses
  directionality; OLS here is descriptive.
