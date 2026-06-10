# THESIS SPINE — single source of truth for all chapter agents

## 0. Title (verbatim — do not paraphrase)

**Causal Explainability for ML-Driven Investment Decisions by German Market Investors**

Subtitle (when shown): *An Empirical Study on the DAX 30*

## 1. Authors and supervisors (placeholders — keep these literal placeholders in text)

- Author 1: `[Author Name 1]`, Matriculation `[Matric 1]`, `[email1@fs.de]`
- Author 2: `[Author Name 2]`, Matriculation `[Matric 2]`, `[email2@fs.de]`
- First Supervisor: `[Prof. Dr. Supervisor 1]`
- Second Supervisor: `[Prof. Dr. Supervisor 2]`
- Programme: Master of Science, Frankfurt School of Finance & Management
- Submission date: `[Submission Date]`

## 2. Repository and live demo (cite both in Introduction and Appendix)

- GitHub: https://github.com/MutharasanE/german_investment_analysis
- Live Streamlit survey app: https://germaninvestmentanalysis.streamlit.app

## 3. Tone & style rules — every chapter agent must follow

- Write like a careful human author, not a template. Vary sentence length. Avoid bullet-blizzards inside the prose chapters; bullets fine for tables/lists only.
- Past-tense for what we did; present-tense for what the model does in the abstract.
- British or American spelling — pick American (analyze, behavior). Be consistent.
- Black-and-white only. Never reference colour in figures.
- Inline citations: Harvard style, e.g. (Pearl, 2009); two authors: (Galhotra and Salimi, 2021); 3+: (Takahashi et al., 2024).
- Cite the source the FIRST time a concept enters; subsequent mentions need not re-cite.
- Use `Figure N` and `Table N` references — numbering is centralised in §10 below.
- Keep total chapter length on target (see §11) — go long if substance demands, but no padding.
- Do NOT invent numerical results. Use ONLY values from §6 below.
- Do NOT mention XGBoost. The model is CatBoost.
- Do NOT mention 17 features. The model is 9 features.

## 4. Canonical fact sheet (single source of truth — copy verbatim into prose where needed)

### 4.1 Pipeline summary
1. Pull 5 years monthly OHLCV for 30 DAX constituents (yfinance).
2. Pull 4 macro series: ECB main refinancing rate, EUR/USD, German HICP YoY, VIX.
3. Engineer 5 firm-level features (volatility, momentum, volume_avg, return_1y, max_drawdown).
4. Combine with 4 macros → 9-feature panel, n = 960 ticker-months.
5. Label investment_decision = 1 if (return_1y / volatility) > 0.5; else 0.
6. Equal-frequency binning (10 bins), train-only fit.
7. Train CatBoost classifier (200 iterations, depth 6, fixed seed).
8. Run causal discovery (PC, DirectLiNGAM, RESIT, LiM, NOTEARS, NOTEARS-MLP) under priors {0, a, b}.
9. Primary specification: DirectLiNGAM under prior (b) — target is sink (no reverse causation).
10. Compute LEWIS counterfactual scores (Necessity, Sufficiency, max-Necessity-Sufficiency) via Pearl backdoor adjustment on the discovered DAG.
11. Compute SHAP TreeExplainer values on the same model as correlational baseline.
12. Deploy Streamlit dashboard + A/B survey instrument; persist responses in MongoDB Atlas (SQLite local fallback).
13. Collect 25 expert evaluations; fit OLS regression of trust on five explanation predictors.

### 4.2 Equity universe (20 DAX tickers — no ESG, no size filter applied)
SAP.DE, SIE.DE, ALV.DE, DTE.DE, BAS.DE, MBG.DE, BMW.DE, MUV2.DE, AIR.DE, IFX.DE, ADS.DE, DB1.DE, RWE.DE, HEN3.DE, VOW3.DE, SY1.DE, BEI.DE, EOAN.DE, FRE.DE, MTX.DE.

Window: 2019-01 to 2024-12 (60 months × 20 tickers ≈ 960 rows after rolling-window NA drop).

### 4.3 Engineered features (only these 9 — no momentum_1m/3m, no rsi_14, etc.)
| Feature | Definition | Window |
|---|---|---|
| volatility | Rolling std of monthly returns | 12 m |
| momentum | Price change over window | 6 m |
| volume_avg | Rolling mean monthly volume | 6 m |
| return_1y | 12-month price change (%) | 12 m |
| max_drawdown | Peak-to-trough decline | 12 m |
| ecb_rate | ECB main refinancing rate (%) | level (FD applied if non-stationary) |
| eur_usd | ECB EUR/USD reference rate | level |
| de_inflation | German HICP YoY (%) | level (FD applied) |
| vix | CBOE VIX index | level |

### 4.4 Stationarity (ADF, applied features only)
- volatility, momentum, volume_avg, return_1y, max_drawdown, vix, eur_usd: stationary at level.
- ecb_rate, de_inflation: not stationary; first-difference applied before discovery.

## 5. CatBoost configuration
- Library: CatBoost (Prokhorenkova et al., 2018).
- Hyperparameters: iterations=200, depth=6, verbose=0, random_seed fixed.
- Loss: Logloss.
- Class balance: ≈ 50/50 (deliberate threshold choice).

## 6. EMPIRICAL RESULTS — copy these numbers verbatim. Do NOT invent or round differently.

### 6.1 Predictive performance
- Full-panel training accuracy: 1.000 (overfit by design — we want explanation focus).
- Time-split holdout (192 rows): accuracy 0.974, precision 1.000, recall 0.959.
- Random-split holdout (192 rows, seed 42): accuracy 0.990.

### 6.2 LEWIS scores (DirectLiNGAM prior b) — top features (max-Nesuf, normalized)
| Rank | Feature | LEWIS (norm.) |
|---|---|---|
| 1 | return_1y | 1.000 |
| 2 | volume_avg | 0.281 |
| 3 | eur_usd | 0.144 |
| 4 | ecb_rate | 0.111 |
| 5 | momentum | 0.110 |

(remaining: max_drawdown 0.057, volatility 0.041, de_inflation 0.020, vix 0.012)

### 6.3 SHAP comparison (mean |SHAP|, normalized) — top
| Rank | Feature | SHAP (norm.) |
|---|---|---|
| 1 | return_1y | 1.000 |
| 2 | max_drawdown | 0.62 |
| 3 | volatility | 0.55 |

### 6.4 LEWIS vs SHAP (key divergences)
- Both place return_1y at rank 1 (agree).
- volatility: SHAP rank 3, LEWIS rank 7 — SHAP attributes a strong signal, LEWIS finds limited *causal* impact (proximal mediator absorbs).
- max_drawdown: SHAP rank 2, LEWIS rank 6 — same pattern.
- ecb_rate: LEWIS rank 4, SHAP rank ~7 — upstream macro driver under-weighted by SHAP.
- Spearman rank correlation between LEWIS and SHAP rankings: 0.137 (p = 0.599) — i.e., the methods do not agree.

### 6.5 Survey descriptives (n = 25 evaluations)
- All respondents: Portfolio Manager, 5 years experience.
- Preference: 14 No preference, 6 chose Method A, 5 chose Method B.
- chosen_method: 14 No preference, 8 COUNTERFACTUAL (LEWIS), 3 FEATURE_IMPORTANCE (SHAP).
- Of the 11 respondents who DID express a preference: 8/11 = 73 % chose the LEWIS counterfactual.
- Mechanics feedback: 18 highly accurate, 5 somewhat, 2 theoretical.
- Trust score: mean 5.28, sd 0.68, scale 1–10.
- Confidence score: mean 4.76, sd 1.30, scale 1–10.

### 6.6 OLS regression of trust (n = 25)
- Predictors: is_counterfactual_b, accuracy_score, num_features, confidence_score, attention_flag.
- R² = 0.419, adjusted R² = 0.266.
- F(5, 19) = 2.743, p = 0.0499.
- Durbin-Watson 2.396; JB p < 0.001 (residual non-normality — caveat: bootstrap CIs recommended).

| Predictor | β | SE | t | p |
|---|---|---|---|---|
| Intercept | 5.389 | 1.444 | 3.733 | 0.001 |
| is_counterfactual_b | 0.001 | 0.306 | 0.002 | 0.998 |
| accuracy_score | −0.101 | 0.334 | −0.302 | 0.766 |
| num_features | −0.243 | 0.142 | −1.718 | 0.102 |
| confidence_score | 0.216 | 0.101 | 2.129 | 0.047 |
| attention_flag | 0.428 | 0.307 | 1.395 | 0.179 |

Only confidence_score is significant at α = 0.05.

## 7. EU AI Act framing (use this language consistently)
- Regulation (EU) 2024/1689 on Artificial Intelligence (the "EU AI Act"), entered into force 1 August 2024, with **high-risk** provisions becoming applicable on **2 August 2026**.
- AI systems used to evaluate creditworthiness or otherwise make/inform binding decisions on natural persons in financial services fall under Annex III (high-risk).
- Article 13 (Transparency) and Article 14 (Human oversight) require providers to explain *why* a decision was reached in terms a human supervisor can audit and override.
- The argument the thesis makes: SHAP, being correlational, satisfies "transparency" only loosely — it cannot answer "if this feature had been different, would the decision have flipped?". A counterfactual artefact (LEWIS) directly answers Article 13(3)(b)(iv)'s "main parameters of the decision" requirement in causal terms.

## 8. Citations dictionary — use these forms exactly
| In-text | Bibliography (Harvard) |
|---|---|
| (Takahashi et al., 2024) | Takahashi, R., Hara, S., Maeda, S. and Sasagawa, K. (2024) 'Counterfactual Explanations of Black-Box ML Models using Causal Discovery', arXiv:2402.02678. |
| (Galhotra et al., 2021) | Galhotra, S., Pradhan, R. and Salimi, B. (2021) 'Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals', in *Proceedings of the 2021 ACM SIGMOD Conference*, pp. 577–590. |
| (Pearl, 2009) | Pearl, J. (2009) *Causality: Models, Reasoning, and Inference*. 2nd edn. Cambridge: Cambridge University Press. |
| (Spirtes et al., 2000) | Spirtes, P., Glymour, C. and Scheines, R. (2000) *Causation, Prediction, and Search*. 2nd edn. Cambridge, MA: MIT Press. |
| (Shimizu et al., 2011) | Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T., Hoyer, P. O. and Bollen, K. (2011) 'DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model', *Journal of Machine Learning Research*, 12, pp. 1225–1248. |
| (Zheng et al., 2018) | Zheng, X., Aragam, B., Ravikumar, P. and Xing, E. P. (2018) 'DAGs with NO TEARS: Continuous Optimization for Structure Learning', in *Advances in Neural Information Processing Systems*, 31, pp. 9472–9483. |
| (Lundberg and Lee, 2017) | Lundberg, S. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774. |
| (Ribeiro et al., 2016) | Ribeiro, M. T., Singh, S. and Guestrin, C. (2016) '"Why Should I Trust You?": Explaining the Predictions of Any Classifier', in *Proceedings of the 22nd ACM SIGKDD*, pp. 1135–1144. |
| (Wachter et al., 2017) | Wachter, S., Mittelstadt, B. and Russell, C. (2017) 'Counterfactual Explanations Without Opening the Black Box', *Harvard Journal of Law & Technology*, 31(2), pp. 841–887. |
| (Verma et al., 2020) | Verma, S., Boonsanong, V., Hoang, M., Hines, K. E., Dickerson, J. P. and Shah, C. (2020) 'Counterfactual Explanations and Algorithmic Recourses for Machine Learning: A Review', arXiv:2010.10596. |
| (Doshi-Velez and Kim, 2017) | Doshi-Velez, F. and Kim, B. (2017) 'Towards a Rigorous Science of Interpretable Machine Learning', arXiv:1702.08608. |
| (Rudin, 2019) | Rudin, C. (2019) 'Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead', *Nature Machine Intelligence*, 1(5), pp. 206–215. |
| (Prokhorenkova et al., 2018) | Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. and Gulin, A. (2018) 'CatBoost: Unbiased Boosting with Categorical Features', in *Advances in Neural Information Processing Systems*, 31, pp. 6638–6648. |
| (Lopez de Prado, 2018) | Lopez de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken: Wiley. |
| (Arrieta et al., 2020) | Arrieta, A. B., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R. and Herrera, F. (2020) 'Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges Toward Responsible AI', *Information Fusion*, 58, pp. 82–115. |
| (Bhatt et al., 2020) | Bhatt, U., Xiang, A., Sharma, S., Weller, A., Taly, A., Jia, Y., Ghosh, J., Puri, R., Moura, J. M. F. and Eckersley, P. (2020) 'Explainable Machine Learning in Deployment', in *Proceedings of FAT* '20*, pp. 648–657. |
| (Sharpe, 1966) | Sharpe, W. F. (1966) 'Mutual Fund Performance', *Journal of Business*, 39(1), pp. 119–138. |
| (Fama and French, 1993) | Fama, E. F. and French, K. R. (1993) 'Common Risk Factors in the Returns on Stocks and Bonds', *Journal of Financial Economics*, 33(1), pp. 3–56. |
| (Jegadeesh and Titman, 1993) | Jegadeesh, N. and Titman, S. (1993) 'Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency', *Journal of Finance*, 48(1), pp. 65–91. |
| (Gu et al., 2020) | Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical Asset Pricing via Machine Learning', *Review of Financial Studies*, 33(5), pp. 2223–2273. |
| (European Union, 2024) | European Union (2024) *Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)*. Official Journal of the European Union, L 2024/1689. |
| (Mayer and Davis, 1999) | Mayer, R. C. and Davis, J. H. (1999) 'The Effect of the Performance Appraisal System on Trust for Management', *Journal of Applied Psychology*, 84(1), pp. 123–136. |
| (Lee and See, 2004) | Lee, J. D. and See, K. A. (2004) 'Trust in Automation: Designing for Appropriate Reliance', *Human Factors*, 46(1), pp. 50–80. |
| (Glikson and Woolley, 2020) | Glikson, E. and Woolley, A. W. (2020) 'Human Trust in Artificial Intelligence: Review of Empirical Research', *Academy of Management Annals*, 14(2), pp. 627–660. |

(more refs may be added by chapter agents — but anything they cite MUST appear in their returned bibliography list to be merged in.)

## 9. Figure & table list (centralised — agents reference by these numbers and captions)

### Figures (in order of appearance — agents may reorder if it suits the chapter logic)
- Figure 1. System architecture: data → CatBoost → causal discovery → LEWIS / SHAP → expert survey → OLS. (architecture diagram — DESCRIBE IN TEXT, no PNG)
- Figure 2. Discovered causal DAG over the nine features and the investment decision (DirectLiNGAM, prior b). PATH: `results/investment/causal_graph_DirectLiNGAM(b).png`
- Figure 3. Necessity, Sufficiency, and max-Nesuf scores per feature. PATH: `results/investment/nesuf_comparison_DirectLiNGAM(b).png`
- Figure 4. SHAP summary plot. PATH: `results/investment/shap_summary_DirectLiNGAM(b).png`
- Figure 5. LEWIS vs SHAP normalized importance (black-and-white). PATH: `results/plots/lewis_vs_shap_bw.png`
- Figure 6. Reversal probabilities — counterfactual feasibility per feature. PATH: `results/investment/reversal_DirectLiNGAM(b).png`
- Figure 7. Confusion matrix on time-split holdout. PATH: `results/plots/confusion_matrix.png`
- Figure 8. Calibration curve. PATH: `results/plots/calibration_curve.png`
- Figure 9. Rolling holdout accuracy. PATH: `results/plots/rolling_accuracy.png`
- Figure 10. Accuracy by sector. PATH: `results/plots/accuracy_by_sector.png`
- Figure 11. Accuracy by macro-regime. PATH: `results/plots/accuracy_by_regime.png`
- Figure 12. DAG stability heatmap across discovery methods × priors. PATH: `results/plots/dag_stability_heatmap.png`
- Figure 13. Expert preference distribution among the 25 evaluations. PATH: `results/plots/expert_preference.png`
- Figure 14. Trust score distribution. PATH: `results/plots/trust_distribution.png`
- Figure 15. Trust vs confidence scatter with OLS line. PATH: `results/plots/trust_vs_confidence.png`
- Figure 16. Mechanics-accuracy feedback distribution. PATH: `results/plots/mechanics_distribution.png`
- Figure 17. OLS diagnostic panel (actual vs predicted, residuals, Q-Q, coefficients). PATH: `results/plots/trust_drivers_plot.png`

### Tables (in order)
- Table 1. Equity universe (20 DAX tickers).
- Table 2. Engineered features and definitions.
- Table 3. Stationarity test results (ADF).
- Table 4. CatBoost hyperparameters.
- Table 5. Causal discovery methods and core assumptions.
- Table 6. Holdout evaluation metrics (time vs random split).
- Table 7. LEWIS scores (max-Nesuf) per feature, top to bottom.
- Table 8. LEWIS vs SHAP rankings on the nine-feature panel.
- Table 9. Survey demographics and descriptive statistics.
- Table 10. OLS coefficient table for the trust regression.

## 10. Chapter scope and length budget (60 ± 10 % pages overall, target 60 main-text pages)

| Chapter | Target pages (1.5 line spacing, TNR 12) | Word target |
|---|---|---|
| 1. Introduction | 6 | ~2,200 |
| 2. Literature Review | 12 | ~4,400 |
| 3. Methodology | 14 | ~5,200 |
| 4. Data and Empirical Setup | 8 | ~2,900 |
| 5. Results | 12 | ~4,400 |
| 6. Discussion | 6 | ~2,200 |
| 7. Conclusion | 4 | ~1,500 |
| **Total** | **62** | **~22,800** |

## 11. Frankfurt School format and document composition (for the build script and every agent)

### 11.1 Composition order (Frankfurt School Thesis Guidelines 2024, §7.1)
1. **Cover Page** — title, subtitle, "Master's Thesis", author names, FS logo CENTERED on the page (no header, no page number).
2. **Title Page** — programme name, full thesis title, both authors' full names + matriculation numbers + addresses, submission date, both assessors' names. Logo top-right.
3. **Statement of Certification** — declaration that the work is original; signed by each author. Placed near the front (we follow FS attachment 4 wording). Logo top-right.
4. **Acknowledgements** (optional, brief).
5. **Abstract** (~250 words).
6. **Table of Contents** — with page numbers.
7. **List of Abbreviations**.
8. **List of Figures**.
9. **List of Tables**.
10. **Main text** — Chapter 1 Introduction → Chapter 2 Literature Review → Chapter 3 Methodology → Chapter 4 Data and Empirical Setup → Chapter 5 Results → Chapter 6 Discussion → Chapter 7 Conclusion.
11. **Appendix** — survey instrument screenshots, full OLS R output, hyperparameter tuning grid, software environment, replication instructions. Not graded for content; graded for form.
12. **Bibliography** — Harvard, alphabetical, every cited source.

### 11.2 Format (FS Thesis Guidelines §7.2)
- Font: Times New Roman 12 pt body; 10 pt for footnotes.
- Line spacing: 1.5; ≥ 9 pt space after a paragraph.
- Left margin 4 cm; right margin 2 cm; top/bottom 2.5 cm.
- Justified body text.
- Page count: ~60 pages ± 10 % main text (single author guideline; we are two authors so 60 ± 10 % stands as the joint target — the body chapters 1–7 land in this band).
- Headings: H1 bold 16 pt; H2 bold 14 pt; H3 bold 12 pt.

### 11.3 Page numbering and headers
- **First page (cover)**: FS logo CENTERED, no page number, no header.
- **Front matter (title page → list of tables)**: FS logo top-right; lowercase Roman numerals (i, ii, iii…); title page counts as i but the number is suppressed.
- **Main text (Chapter 1 onward) and back matter**: FS logo top-right; Arabic numerals starting at 1.
- The logo file used by the build script: `fs_logo_blue.png` (NOT the SVG).

### 11.4 Citation style
- Harvard. Direct quotes "in quotation marks" with `(Author, Year:Page)`.
- Indirect quotes prefixed with `cf.` and the same `(Author, Year:Page)` reference.
- Three or more identical consecutive words from a source = treat as direct quote.

## 12. Chapter agent contract

Each chapter agent MUST return a single self-contained markdown block with:
1. Chapter heading: `# Chapter N: Title`
2. Section headings: `## N.1 Section`, `## N.2`, etc.
3. Inline figure references like `[See Figure 5]` and table references like `[See Table 8]`.
4. Inline citations Harvard style.
5. A trailing `## Bibliography fragment` listing every reference used in this chapter, in Harvard format. The merger will dedupe.
6. NO YAML front matter, NO docx-specific code, NO commentary outside the chapter body.

## 13. Context discipline — read-only file allow-list per chapter

Every agent operates under STRICT context discipline. The numbers, plots, captions, methods, and citations needed for each chapter are already in this SPINE document. Do NOT explore the repository broadly. Do NOT open files outside your chapter's allow-list. Do NOT run `find`, `tree`, `grep -r` over the project. Do NOT read CSV files row-by-row — the headline numbers are in §6 of this spine. If a number you want is not in the spine, prefer to leave it out rather than invent or grep for it. If a number genuinely is missing and you need it, open ONLY the single CSV file named in your hint list, read at most the first 10 rows, and stop.

The point is: write the chapter from this spine + the small allow-list below, not from a fresh exploration. Wall-clock and tokens both matter.

End of spine.
