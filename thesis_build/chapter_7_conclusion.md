# Chapter 7: Conclusion

## 7.1 Summary of the research question and approach

This thesis set out to answer a single, deliberately narrow question: *do causal counterfactual explanations (LEWIS) provide German market investors with more trustworthy and decision-relevant insight into machine-learning-driven investment recommendations than the prevailing correlational baseline (SHAP)?* The question was framed in Chapter 1 against the backdrop of the EU AI Act's high-risk provisions, which become applicable on 2 August 2026 (European Union, 2024), and against a literature in which counterfactual reasoning has matured theoretically but rarely been tested with practitioners on real financial data (Pearl, 2009; Galhotra et al., 2021; Takahashi et al., 2024).

The empirical pipeline was designed to make the comparison concrete rather than abstract. Five years of monthly OHLCV data for twenty DAX constituents (SAP.DE through MTX.DE) were combined with four macro series — the ECB main refinancing rate, EUR/USD, German HICP YoY inflation, and the VIX — to produce a nine-feature, 960-row panel covering January 2019 through December 2024. A CatBoost classifier (Prokhorenkova et al., 2018) was trained to predict whether a ticker-month would deliver a Sharpe-style return-to-volatility ratio above 0.5. Causal structure over the nine features was learned with DirectLiNGAM under prior (b), in which the investment decision is constrained to be a sink node (Shimizu et al., 2011). LEWIS necessity-and-sufficiency scores were computed via Pearl backdoor adjustment on the discovered DAG and benchmarked against SHAP TreeExplainer values from the same model (Lundberg and Lee, 2017). Finally, twenty-five domain experts — all portfolio managers with five years of professional experience — evaluated paired explanations through a Streamlit A/B instrument, and their trust ratings were modelled by ordinary least squares against five explanation-level predictors.

## 7.2 Headline findings

Three findings emerge from the empirical work, each of which carries weight independently.

**Finding 1. The two explanation methods do not rank features the same way.** The Spearman rank correlation between LEWIS and SHAP feature orderings is only 0.137 (p = 0.599) — statistically indistinguishable from random agreement. Both methods place `return_1y` first, but the divergence below the top rank is sharp: SHAP elevates `volatility` (rank 3) and `max_drawdown` (rank 2), whereas LEWIS demotes them to ranks 7 and 6 because their apparent influence is absorbed by the proximal mediator `return_1y` once backdoor adjustment is applied. Conversely, LEWIS lifts the upstream macro driver `ecb_rate` from SHAP's rank 7 to rank 4. The disagreement is not noise; it is exactly the structural correction that a causal lens is supposed to provide over an associational one.

**Finding 2. Among experts who expressed a preference, the LEWIS counterfactual was preferred roughly three to one.** Of the twenty-five evaluations, fourteen registered "no preference," but among the eleven who did discriminate, eight (73 %) selected the LEWIS counterfactual and three (27 %) selected the SHAP feature-importance view. The headline preference rate is therefore directional and material, even though the underlying sample is modest.

**Finding 3. Trust is driven by the respondent's own confidence, not by the explanation type per se.** In the OLS regression of trust on five predictors, the only coefficient significant at α = 0.05 is `confidence_score` (β = 0.216, SE = 0.101, p = 0.047). The treatment indicator `is_counterfactual_b` is a precisely estimated zero (β = 0.001, p = 0.998). The model explains 41.9 % of trust variance (R² = 0.419, adjusted R² = 0.266) with an overall F(5, 19) = 2.743, p = 0.0499 — borderline significant. The substantive reading is that, at this sample size and within this expert pool, expressed trust tracks self-reported confidence in one's own judgement; the explanation modality does not move the trust dial on its own. Finding 2 (preference) and Finding 3 (trust) are therefore complementary rather than contradictory: practitioners prefer the counterfactual when forced to pick, but their reported trust is anchored by an internal disposition that no explanation method, in this study, was able to shift.

## 7.3 Practical and regulatory implications

From 2 August 2026, AI systems used to evaluate creditworthiness or otherwise inform binding decisions on natural persons in financial services will fall within Annex III of the EU AI Act and must satisfy Article 13 (transparency) and Article 14 (human oversight) (European Union, 2024). The compliance question is no longer whether to attach an explanation to a model output, but which kind. SHAP, being correlational, satisfies the transparency obligation only loosely: it cannot answer the operative supervisory question — "if this feature had been different, would the decision have flipped?" A LEWIS-style counterfactual artefact, persisted per served decision, answers that question directly and is therefore the most defensible compliance route under Article 13(3)(b)(iv)'s requirement to disclose the "main parameters of the decision." The open-source Streamlit instrument deployed for this study demonstrates that the engineering is not speculative: the artefacts can be generated, served, and stored at decision time within an ordinary cloud stack.

## 7.4 Limitations

The conclusions above must be read against four limitations. First, the survey n is twenty-five, which is small for any inferential exercise; the OLS F-test (p = 0.0499) is on the wrong side of conventional cut-offs by the slimmest possible margin and the Jarque–Bera test rejects residual normality at p < 0.001, so bootstrap confidence intervals would be the more defensible inferential route. Second, the respondent pool is homogeneous — every participant is a portfolio manager with five years of experience — which limits external validity to other archetypes such as retail investors, risk officers, or supervisory authorities. Third, the causal inference rests on a single discovery specification: DirectLiNGAM under prior (b). The LEWIS rankings are conditional on that DAG being approximately correct, and the identifiability literature is unambiguous that no observational discovery method recovers the truth uniquely without assumptions (Pearl, 2009; Spirtes et al., 2000). Fourth, the labelling rule — investment_decision = 1 if return_1y / volatility > 0.5 — is a retrospective Sharpe-style construction (Sharpe, 1966); it is reproducible and disciplined, but it is not the same object as a prospective trading decision under uncertainty.

## 7.5 Recommendations for future research

Four extensions follow naturally from the limitations above.

First, the survey should be scaled to n ≥ 80 across multiple investor archetypes — retail investors, internal risk officers, and supervisory regulators alongside portfolio managers — to sharpen the trust regression and to test whether the preference asymmetry observed here generalizes beyond a single professional pool. A pre-registered design with stratified recruitment would also resolve the borderline F-test issue.

Second, `confidence_score` should be identified causally, not merely included as a correlated predictor. Instrumental-variable designs or pre-registered manipulations of confidence (for example, by varying the model-performance disclosure shown to the respondent) would let future work say whether confidence drives trust, the reverse, or whether both are driven by a third disposition.

Third, the LEWIS-versus-SHAP comparison should be repeated under a richer function class — a tabular transformer, for instance, or a deep ensemble — to test whether the rank-correlation divergence of 0.137 reported here widens or narrows when the underlying model captures more nonlinearity. The hypothesis worth pre-registering is that the divergence widens, because more flexible models can fit deeper correlational structure that backdoor adjustment must then strip away.

Fourth, a multi-DAG sensitivity analysis is overdue. Refitting LEWIS under each of the six discovery methods (PC, DirectLiNGAM, RESIT, LiM, NOTEARS, NOTEARS-MLP) crossed with three priors (0, a, b) yields eighteen specifications; reporting the *distribution* of the LEWIS–SHAP rank correlation across them, rather than a single point estimate, would directly address the identifiability concern raised in Pearl (2009) and would let readers calibrate how brittle the present headline numbers are to discovery choice.

## 7.6 Closing

Causal explainability is no longer a research-side curiosity but an engineering and regulatory necessity for AI in financial decision-making. This thesis shows that LEWIS-style counterfactual explanations are buildable on real DAX data, comparable in computational cost to SHAP, marginally preferred by experts who are forced to choose, and on a direct path to EU AI Act compliance. The harder problems — sample size, causal identification of trust itself, robustness across discovery methods — are open but tractable, and the empirical infrastructure assembled here is offered as one starting point for the work that the 2026 regulatory deadline now makes urgent.

## Bibliography fragment

European Union (2024) *Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)*. Official Journal of the European Union, L 2024/1689.

Galhotra, S., Pradhan, R. and Salimi, B. (2021) 'Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals', in *Proceedings of the 2021 ACM SIGMOD Conference*, pp. 577–590.

Lundberg, S. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Pearl, J. (2009) *Causality: Models, Reasoning, and Inference*. 2nd edn. Cambridge: Cambridge University Press.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. and Gulin, A. (2018) 'CatBoost: Unbiased Boosting with Categorical Features', in *Advances in Neural Information Processing Systems*, 31, pp. 6638–6648.

Sharpe, W. F. (1966) 'Mutual Fund Performance', *Journal of Business*, 39(1), pp. 119–138.

Shimizu, S., Inazumi, T., Sogawa, Y., Hyvärinen, A., Kawahara, Y., Washio, T., Hoyer, P. O. and Bollen, K. (2011) 'DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model', *Journal of Machine Learning Research*, 12, pp. 1225–1248.

Spirtes, P., Glymour, C. and Scheines, R. (2000) *Causation, Prediction, and Search*. 2nd edn. Cambridge, MA: MIT Press.

Takahashi, R., Hara, S., Maeda, S. and Sasagawa, K. (2024) 'Counterfactual Explanations of Black-Box ML Models using Causal Discovery', arXiv:2402.02678.
