# Chapter 5: Results

This chapter presents the empirical findings of the study in the order in which the analytical pipeline produced them. We begin with the predictive performance of the CatBoost classifier, then describe the causal structure recovered by DirectLiNGAM under prior (b), report the LEWIS necessity-sufficiency scores and the SHAP baseline alongside them, examine the counterfactual feasibility of feature interventions, summarize the expert survey, and close with the OLS regression of trust on five explanation predictors. Interpretation, mechanism, and policy implication are deferred to Chapter 6; this chapter is deliberately factual and figure-driven.

## 5.1 Predictive performance

The CatBoost classifier, trained on the full nine-feature panel under the configuration described in Chapter 3, attains a training accuracy of 1.000. As noted earlier, this perfect in-sample fit is engineered rather than incidental: the labelling rule investment_decision = 1 if (return_1y / volatility) > 0.5 is a deterministic function of two of the input features, and the 200-iteration, depth-6 boosting model has more than enough capacity to memorize the panel. We retain this configuration knowingly because the salient variable in the thesis is not predictive accuracy but the comparison of explanation methodologies; an over-fit but well-calibrated model isolates the explanation question from the model-selection question.

Out-of-sample behavior, however, remains strong. On the time-split holdout of the final 192 ticker-months, the classifier achieves accuracy 0.974, precision 1.000, and recall 0.959. Precision of unity indicates that every positive prediction on the holdout corresponds to a true positive label, with the small error budget falling entirely on the false-negative side. The random-split holdout, drawn with seed 42 over 192 rows, achieves accuracy 0.990; the gap between the two holdouts (0.974 vs 0.990) reflects the additional difficulty introduced by the temporal split, where the model is asked to extrapolate into a held-out time block rather than interpolate within the panel [See Table 6].

The confusion matrix on the time-split holdout [See Figure 7] confirms the precision-recall pattern: the off-diagonal mass concentrates on the false-negative cell, with the false-positive cell empty. Calibration on the same holdout [See Figure 8] is broadly diagonal, with mild over-confidence in the highest probability bin; the model's predicted probabilities therefore track empirical frequencies closely enough to support a probabilistic decision threshold.

Rolling holdout accuracy across the panel window [See Figure 9] remains tightly concentrated above 0.95, with no monotonic drift across the five-year sample. Sectoral disaggregation [See Figure 10] shows accuracy is essentially flat across the represented DAX sectors, with no single sector falling below the panel-wide mean by more than a few percentage points. Disaggregation by macro regime [See Figure 11] — partitioning the panel by ECB-rate phase and VIX level — shows that accuracy is also stable across the high-rate / low-rate and high-volatility / low-volatility quadrants. Taken together, these diagnostics establish CatBoost as a strong baseline; the methodological contribution of the thesis lies downstream of it.

## 5.2 Discovered causal structure

Causal discovery was run with six algorithms — PC, DirectLiNGAM, RESIT, LiM, NOTEARS, and NOTEARS-MLP — under three priors {0, a, b}. The primary specification is DirectLiNGAM (Shimizu et al., 2011) under prior (b), which forces the target investment_decision to be a sink, that is, a node with no outgoing edges. This is justified ex ante because the decision label cannot, by construction, cause a firm-level or macro variable measured contemporaneously or earlier; allowing reverse arrows would generate non-physical edges.

The DAG returned by DirectLiNGAM (b) [See Figure 2] places the nine features and the decision in a coherent stratified structure. Macro variables — ecb_rate, eur_usd, de_inflation, vix — sit upstream and feed into firm-level features. The firm-level features then converge on return_1y and max_drawdown, with return_1y carrying the strongest direct edge into investment_decision. Volatility and momentum lie on intermediate paths between volume_avg, the macro inputs, and return_1y. The target itself has no outgoing edges, in line with the prior. The discovered structure is consistent with the labelling rule's mechanical reliance on return_1y and volatility, and it is consistent with standard asset-pricing intuition that macro states drive firm-level signals rather than the reverse over a one-month horizon (Gu et al., 2020).

The DAG stability heatmap across the six discovery methods and three priors [See Figure 12] supports the choice of DirectLiNGAM (b) on robustness grounds. The DirectLiNGAM-(b) cell is among the most stable rows in the heatmap, with edge agreement preserved across re-samples. PC (Spirtes et al., 2000) and NOTEARS-MLP (Zheng et al., 2018) under priors 0 and (a) are noticeably noisier, with several edges flipping or appearing only intermittently. The LiM and RESIT cells fall in between. The qualitative pattern — DirectLiNGAM-(b) stable, alternative method-prior combinations less stable — provides empirical justification for using prior (b) as the primary specification while reporting the others as robustness reference rather than headline results.

## 5.3 LEWIS scores

LEWIS necessity-sufficiency scores were computed on the DirectLiNGAM-(b) DAG via Pearl backdoor adjustment (Pearl, 2009) on the parent set of investment_decision. The normalized max-Nesuf score per feature is reported in Table 7, with the top five summarized below:

| Rank | Feature | LEWIS (norm.) |
|---|---|---|
| 1 | return_1y | 1.000 |
| 2 | volume_avg | 0.281 |
| 3 | eur_usd | 0.144 |
| 4 | ecb_rate | 0.111 |
| 5 | momentum | 0.110 |

The remaining four features score lower: max_drawdown 0.057, volatility 0.041, de_inflation 0.020, and vix 0.012 [See Table 7]. The full per-feature comparison of Necessity, Sufficiency, and max-Nesuf is presented graphically [See Figure 3].

Two features stand out. First, return_1y is dominant by construction: it is the proximal mediator on the discovered DAG, and the labelling rule is a deterministic function that depends on it directly. Any LEWIS score that did not place return_1y at rank 1 on this panel would indicate a discovery error. Second, volume_avg and the macro variables eur_usd and ecb_rate appear in positions 2 to 4 — well above max_drawdown and volatility. This pattern, in which an upstream macro driver (ecb_rate) is judged causally more important than a strong proximal correlate (max_drawdown), is the substantive divergence from the SHAP baseline reported in §5.4.

## 5.4 SHAP scores

SHAP values were computed with the TreeExplainer (Lundberg and Lee, 2017) on the same trained CatBoost model. The mean absolute SHAP values, normalized to the top feature, place return_1y at rank 1 with score 1.000, max_drawdown at rank 2 with 0.62, and volatility at rank 3 with 0.55 [See Figure 4]. The SHAP summary plot also shows the directional pattern characteristic of the labelling rule: high return_1y points push the prediction toward the positive class, high volatility and large max_drawdown push it toward the negative class. SHAP therefore identifies the two ratio components — return_1y in the numerator, volatility in the denominator — and the closely related max_drawdown as the three top contributors to the model's output, in that order.

## 5.5 LEWIS vs SHAP comparison

The two methods agree on rank 1 — both place return_1y first — and disagree systematically below it [See Figure 5, See Table 8]. Three contrasts characterize the divergence:

- Volatility is SHAP rank 3 but LEWIS rank 7. SHAP attributes a strong signal to volatility because it appears in the labelling-rule denominator and the model uses it heavily; LEWIS, working on the DAG, finds limited *causal* impact because the proximal mediator return_1y absorbs most of the do-calculus path.
- Max_drawdown is SHAP rank 2 but LEWIS rank 6 — the same pattern. Max_drawdown is correlationally informative but does not lie on a high-strength causal path under do-intervention.
- Ecb_rate is LEWIS rank 4 but SHAP rank approximately 7. The upstream macro driver is under-weighted by SHAP relative to its causal contribution under the discovered DAG.

The Spearman rank correlation between the LEWIS and SHAP rankings across the nine features is 0.137, with p = 0.599. The two methods therefore do not agree beyond chance. Agreement on the top feature is forced by the labelling-rule construction; disagreement on the remaining ranks is the substantive empirical observation of the chapter.

## 5.6 Counterfactual feasibility

The reversal-probability plot [See Figure 6] reports, per feature, the probability that an investor-feasible perturbation of that feature alone would flip the model's decision. The pattern is informative when read alongside the LEWIS scores. Return_1y and volume_avg, which combine high LEWIS scores with reasonable feasibility, show meaningful reversal probabilities. Momentum and max_drawdown sit in the mid-range. The macro variables, however, exhibit a sharp asymmetry: ecb_rate carries a non-trivial LEWIS necessity score (0.111, rank 4) but its reversal probability under an investor-feasible intervention is essentially zero, because no individual investor can move the ECB main refinancing rate. The same holds for eur_usd, de_inflation, and vix. This asymmetry between causal importance and individual actionability is presented here as a finding; we return to its implications for actionable recourse and for Article 13 of the EU AI Act in Chapter 6.

## 5.7 Survey results

Twenty-five expert evaluations were collected through the Streamlit A/B instrument. All twenty-five respondents identified as Portfolio Managers with five years' experience [See Table 9].

The preference distribution was 14 No preference, 6 Method A, and 5 Method B. Mapping the random A/B labels back to the underlying methodologies, the chosen_method distribution is 14 No preference, 8 COUNTERFACTUAL (LEWIS), and 3 FEATURE_IMPORTANCE (SHAP) [See Figure 13]. Among the eleven respondents who did express a preference, eight chose the LEWIS counterfactual and three chose SHAP — that is, 8 / 11, or 73 %, of those expressing a preference favoured the counterfactual artefact.

The trust score, on a 1–10 scale, has mean 5.28 and standard deviation 0.68 across the twenty-five evaluations [See Figure 14]. The confidence score on the same 1–10 scale has mean 4.76 and standard deviation 1.30. The mechanics-accuracy feedback distribution [See Figure 16] is 18 highly accurate, 5 somewhat accurate, and 2 theoretical, indicating that the explanation surface is generally judged faithful to the underlying decision process by the practitioner respondents. The trust-versus-confidence scatter [See Figure 15] shows a positive association that the OLS analysis in §5.8 quantifies.

## 5.8 OLS regression on trust

We regress trust_score on five predictors: is_counterfactual_b (a 0/1 indicator that the respondent saw the LEWIS counterfactual rather than the SHAP feature-importance), accuracy_score (mechanics-accuracy rating), num_features (the count of features displayed), confidence_score, and attention_flag. The regression is fit by ordinary least squares over the n = 25 evaluations.

The model achieves R² = 0.419 and adjusted R² = 0.266. The overall F-test gives F(5, 19) = 2.743 with p = 0.0499, which is significant at α = 0.05 by the narrowest of margins. The coefficient table [See Table 10] is reproduced in full:

| Predictor | β | SE | t | p |
|---|---|---|---|---|
| Intercept | 5.389 | 1.444 | 3.733 | 0.001 |
| is_counterfactual_b | 0.001 | 0.306 | 0.002 | 0.998 |
| accuracy_score | −0.101 | 0.334 | −0.302 | 0.766 |
| num_features | −0.243 | 0.142 | −1.718 | 0.102 |
| confidence_score | 0.216 | 0.101 | 2.129 | 0.047 |
| attention_flag | 0.428 | 0.307 | 1.395 | 0.179 |

Of the five predictors, only confidence_score is significant at α = 0.05 (β = 0.216, p = 0.047). The is_counterfactual_b coefficient is essentially zero (β = 0.001, p = 0.998): conditional on the other predictors, being shown the LEWIS counterfactual rather than the SHAP feature-importance does not move the trust score. The num_features coefficient is negative and approaches significance (β = −0.243, p = 0.102), and the attention_flag coefficient is positive but not significant (β = 0.428, p = 0.179). The accuracy_score coefficient is small, negative, and not significant.

Diagnostics on the residuals [See Figure 17] show a Durbin-Watson statistic of 2.396, consistent with the absence of first-order autocorrelation, and a Jarque-Bera test with p < 0.001, indicating residual non-normality that should be flagged as a caveat for the t-tests; bootstrap confidence intervals would be a recommended robustness step. The actual-versus-predicted, residuals-versus-predicted, and Q-Q panels in Figure 17 visualize these diagnostics directly, and the coefficient bar at the bottom of the same figure makes the dominance of the confidence_score predictor visible at a glance.

## Bibliography fragment

European Union (2024) *Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)*. Official Journal of the European Union, L 2024/1689.

Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical Asset Pricing via Machine Learning', *Review of Financial Studies*, 33(5), pp. 2223–2273.

Lundberg, S. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Pearl, J. (2009) *Causality: Models, Reasoning, and Inference*. 2nd edn. Cambridge: Cambridge University Press.

Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T., Hoyer, P. O. and Bollen, K. (2011) 'DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model', *Journal of Machine Learning Research*, 12, pp. 1225–1248.

Spirtes, P., Glymour, C. and Scheines, R. (2000) *Causation, Prediction, and Search*. 2nd edn. Cambridge, MA: MIT Press.

Zheng, X., Aragam, B., Ravikumar, P. and Xing, E. P. (2018) 'DAGs with NO TEARS: Continuous Optimization for Structure Learning', in *Advances in Neural Information Processing Systems*, 31, pp. 9472–9483.
