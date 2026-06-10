# Tables (rendered into the body where the build script anchors them; otherwise placed at end of relevant chapters)

## Table 1. Equity universe — 20 DAX constituents

| # | Ticker | Company | Sector |
|---|---|---|---|
| 1 | SAP.DE | SAP SE | Information Technology |
| 2 | SIE.DE | Siemens AG | Industrials |
| 3 | ALV.DE | Allianz SE | Financials |
| 4 | DTE.DE | Deutsche Telekom AG | Communication Services |
| 5 | BAS.DE | BASF SE | Materials |
| 6 | MBG.DE | Mercedes-Benz Group AG | Consumer Discretionary |
| 7 | BMW.DE | Bayerische Motoren Werke AG | Consumer Discretionary |
| 8 | MUV2.DE | Münchener Rückversicherungs-Gesellschaft AG | Financials |
| 9 | AIR.DE | Airbus SE | Industrials |
| 10 | IFX.DE | Infineon Technologies AG | Information Technology |
| 11 | ADS.DE | adidas AG | Consumer Discretionary |
| 12 | DB1.DE | Deutsche Börse AG | Financials |
| 13 | RWE.DE | RWE AG | Utilities |
| 14 | HEN3.DE | Henkel AG & Co. KGaA | Consumer Staples |
| 15 | VOW3.DE | Volkswagen AG | Consumer Discretionary |
| 16 | SY1.DE | Symrise AG | Materials |
| 17 | BEI.DE | Beiersdorf AG | Consumer Staples |
| 18 | EOAN.DE | E.ON SE | Utilities |
| 19 | FRE.DE | Fresenius SE & Co. KGaA | Health Care |
| 20 | MTX.DE | MTU Aero Engines AG | Industrials |

## Table 2. Engineered features and definitions

| Feature | Definition | Window | Source |
|---|---|---|---|
| volatility | Rolling standard deviation of monthly returns | 12 months | Yahoo Finance |
| momentum | Price change over the rolling window | 6 months | Yahoo Finance |
| volume_avg | Rolling mean monthly trading volume | 6 months | Yahoo Finance |
| return_1y | 12-month price change (%) | 12 months | Yahoo Finance |
| max_drawdown | Peak-to-trough decline | 12 months | Yahoo Finance |
| ecb_rate | ECB main refinancing rate (%) | level (FD applied) | ECB Data API |
| eur_usd | ECB EUR/USD reference rate | level | ECB Data API |
| de_inflation | German HICP year-over-year (%) | level (FD applied) | ECB Data API |
| vix | CBOE Volatility Index | level | Yahoo Finance |

## Table 3. Stationarity test results (Augmented Dickey–Fuller)

| Feature | ADF statistic | p-value | Verdict |
|---|---|---|---|
| volatility | — | < 0.05 | Stationary at level |
| momentum | — | < 0.05 | Stationary at level |
| volume_avg | — | < 0.05 | Stationary at level |
| return_1y | — | < 0.05 | Stationary at level |
| max_drawdown | — | < 0.05 | Stationary at level |
| eur_usd | — | < 0.05 | Stationary at level |
| vix | — | < 0.05 | Stationary at level |
| ecb_rate | — | > 0.05 | First-difference applied |
| de_inflation | — | > 0.05 | First-difference applied |

*ADF statistics and p-values are reproduced from `results/tables/stationarity_report.csv`. Where dashes appear, see the appendix for full numerical values.*

## Table 4. CatBoost hyperparameters

| Hyperparameter | Value | Comment |
|---|---|---|
| iterations | 200 | Tree count |
| depth | 6 | Tree depth |
| loss_function | Logloss | Binary classification |
| learning_rate | auto | CatBoost-selected |
| random_seed | 42 | Fixed for reproducibility |
| verbose | 0 | Silent |
| class_weights | None | Class balance ≈ 50/50 by labelling design |

## Table 5. Causal discovery methods and core assumptions

| Method | Family | Core assumption |
|---|---|---|
| PC | Constraint-based | Conditional independence + faithfulness |
| DirectLiNGAM | Score-based | Linear, non-Gaussian errors |
| RESIT | Score-based | Non-linear additive noise |
| LiM | Score-based | Linear, mixed (discrete + continuous) |
| NOTEARS | Continuous optimisation | Continuous, equal-variance |
| NOTEARS-MLP | Continuous optimisation | Non-linear (causal-learn falls back to GES + BIC) |

Prior regimes: (0) no prior — discovery runs blind; (a) all features → target; (b) target as sink (no reverse causation). DirectLiNGAM under prior (b) is the primary specification.

## Table 6. Holdout evaluation metrics

| Split | n (test) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Time-split (last 12 months) | 192 | 0.974 | 1.000 | 0.959 | 0.979 |
| Random-split (seed 42) | 192 | 0.990 | — | — | — |

*Precision, recall and F1 for the random-split case are reported in `results/investment/holdout_eval_random.csv`.*

## Table 7. LEWIS scores (max-Necessity-Sufficiency, normalised)

| Rank | Feature | LEWIS (norm.) |
|---|---|---|
| 1 | return_1y | 1.000 |
| 2 | volume_avg | 0.281 |
| 3 | eur_usd | 0.144 |
| 4 | ecb_rate | 0.111 |
| 5 | momentum | 0.110 |
| 6 | max_drawdown | 0.057 |
| 7 | volatility | 0.041 |
| 8 | de_inflation | 0.020 |
| 9 | vix | 0.012 |

## Table 8. LEWIS vs SHAP rankings on the nine-feature panel

| Feature | LEWIS rank | SHAP rank | Δ rank |
|---|---|---|---|
| return_1y | 1 | 1 | 0 |
| volume_avg | 2 | 9 | −7 |
| eur_usd | 3 | 7 | −4 |
| ecb_rate | 4 | 6 | −2 |
| momentum | 5 | 4 | +1 |
| max_drawdown | 6 | 2 | +4 |
| volatility | 7 | 3 | +4 |
| de_inflation | 8 | 8 | 0 |
| vix | 9 | 5 | +4 |

Spearman rank correlation between the two rankings: **0.137 (p = 0.599)**.

## Table 9. Survey demographics and descriptive statistics (n = 25)

| Item | Value |
|---|---|
| Respondents | 25 |
| Role | Portfolio Manager (all) |
| Years of experience | 5 (all) |
| A/B preference: Method A | 6 |
| A/B preference: Method B | 5 |
| A/B preference: No preference | 14 |
| chosen_method: COUNTERFACTUAL (LEWIS) | 8 |
| chosen_method: FEATURE_IMPORTANCE (SHAP) | 3 |
| chosen_method: No preference | 14 |
| Mechanics: Yes, highly accurate | 18 |
| Mechanics: Somewhat accurate | 5 |
| Mechanics: No, theoretical | 2 |
| Trust score (mean ± sd) | 5.28 ± 0.68 |
| Confidence score (mean ± sd) | 4.76 ± 1.30 |

## Table 10. OLS coefficient table — drivers of expert trust (n = 25)

Model: `trust_score ~ is_counterfactual_b + accuracy_score + num_features + confidence_score + attention_flag`.

| Predictor | β | SE | t | p |
|---|---|---|---|---|
| Intercept | 5.389 | 1.444 | 3.733 | 0.001 |
| is_counterfactual_b | 0.001 | 0.306 | 0.002 | 0.998 |
| accuracy_score | −0.101 | 0.334 | −0.302 | 0.766 |
| num_features | −0.243 | 0.142 | −1.718 | 0.102 |
| confidence_score | 0.216 | 0.101 | 2.129 | 0.047 |
| attention_flag | 0.428 | 0.307 | 1.395 | 0.179 |

R² = 0.419; adjusted R² = 0.266; F(5, 19) = 2.743, p = 0.0499. Durbin-Watson = 2.396; Jarque-Bera p < 0.001 (residual non-normality — bootstrap CIs recommended). Only `confidence_score` is significant at α = 0.05.
