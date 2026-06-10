# Appendix

## A. Software environment and replication

The full code, data download scripts, and trained model artefacts are publicly available at https://github.com/MutharasanE/german_investment_analysis. The live Streamlit instrument used for the expert survey is hosted at https://germaninvestmentanalysis.streamlit.app.

Reproducing the full pipeline requires Python 3.11, a virtual environment with the dependencies listed in the repository's `requirements.txt`, and the one-line patch to `causal-learn 0.1.4.5` documented in the repository README. The full DAX pipeline (`python run_investment.py`) takes approximately twenty minutes on a 2024 MacBook Pro with Apple Silicon; the synthetic validation experiments (`python run_experiments.py --experiment all --trials 100`) take approximately three to five hours.

## B. CatBoost hyperparameter grid

Final configuration: `iterations=200`, `depth=6`, `learning_rate=auto`, `loss_function="Logloss"`, `verbose=0`, `random_seed=42`. A coarse grid sweep over `{iterations ∈ {100, 200, 500}, depth ∈ {4, 6, 8}}` was performed; the chosen point sits on the in-sample plateau with the smallest holdout-accuracy gap. The full grid output is persisted at `results/tables/hyperparameter_tuning_results.csv` in the repository.

## C. Survey instrument — questions delivered to each respondent

Each respondent saw a randomly selected DAX ticker accompanied by the model's BUY / HOLD / SELL recommendation, the discovered causal DAG, the LEWIS panel (Necessity, Sufficiency, max-Necessity-Sufficiency, plus illustrative counterfactual examples), the SHAP panel (mean absolute Shapley values), and a side-by-side ranking comparison. The respondent then answered:

1. **A/B preference.** "Which explanation do you prefer for understanding *why* the AI made this recommendation?" (A | B | No preference)
2. **Trust score.** "How much do you trust the AI's recommendation in light of these explanations?" (1 = none, 10 = full)
3. **Confidence score.** "How confident are you in your reading of the explanations?" (1 = not at all, 10 = fully)
4. **Mechanical accuracy.** "Do the mechanics of the explanations match how you would reason about this stock?" (Yes, highly accurate | Somewhat accurate | No, they seem theoretical)
5. **Free text.** "Briefly explain your preference."
6. **Attention check.** A passive item buried in the form, used as the binary `attention_flag` regressor.

Responses were persisted to MongoDB Atlas (`thesis_survey.votes` collection) with an SQLite local fallback at `data/survey.db` for offline runs.

## D. Full OLS regression output (statsmodels)

```
                            OLS Regression Results
==============================================================================
Dep. Variable:            trust_score   R-squared:                       0.419
Model:                            OLS   Adj. R-squared:                  0.266
Method:                 Least Squares   F-statistic:                     2.743
No. Observations:                  25   Prob (F-statistic):             0.0499
Df Residuals:                      19
Df Model:                           5

============================================================================
                          coef    std err          t      P>|t|
----------------------------------------------------------------------------
const                   5.3893      1.444      3.733      0.001
is_counterfactual_b     0.0007      0.306      0.002      0.998
accuracy_score         -0.1009      0.334     -0.302      0.766
num_features           -0.2429      0.142     -1.718      0.102
confidence_score        0.2160      0.101      2.129      0.047
attention_flag          0.4280      0.307      1.395      0.179
============================================================================
Durbin-Watson:                   2.396
Jarque-Bera (JB) p-value:        < 0.001
============================================================================
```

Residual non-normality (Jarque-Bera p < 0.001) suggests bootstrap confidence intervals would be more reliable than the asymptotic ones reported above. We retain the asymptotic values for transparency and flag the caveat explicitly in Chapter 6.

## E. Selected raw artefacts

The artefacts below are reproduced from the project's `results/` directory.

- `results/investment/lewis_scores_DirectLiNGAM(b).csv` — LEWIS Necessity, Sufficiency, max-Necessity-Sufficiency per feature (used for Table 7 and Figures 3 and 5).
- `results/investment/shap_scores_DirectLiNGAM(b).csv` — Mean absolute SHAP per feature (used for Figure 4 and Figure 5).
- `results/investment/lewis_vs_shap_DirectLiNGAM(b).csv` — Side-by-side ranking comparison (used for Table 8 and Figure 5).
- `results/investment/holdout_eval_time.csv` and `holdout_eval_random.csv` — Holdout-evaluation metrics (used for Table 6).
- `results/tables/stationarity_report.csv` — ADF test statistics (used for Table 3).
- `survey_detailed.csv` — Twenty-five expert evaluations exported from MongoDB Atlas (used for Section 5.7 and Table 9).

## F. Hardware

All runs were performed on a 2024 MacBook Pro (Apple Silicon, 16 GB unified memory, macOS 25.5). No GPU acceleration was used.
