# Chapter 1: Introduction

## 1.1 Background and Motivation

Over the past decade, machine learning has moved from a curiosity at the periphery of asset management to a working tool inside many investment processes. Empirical asset pricing studies that compare dozens of predictive models on the same return panel now routinely show that flexible learners — gradient-boosted trees and shallow neural networks in particular — extract risk premia that linear factor models leave on the table (Gu et al., 2020). Practitioner-facing texts have followed the same trajectory, arguing that financial machine learning is no longer optional for serious quantitative shops but is instead a core skill (Lopez de Prado, 2018). The German equity market is part of this shift. The DAX is a relatively concentrated, well-instrumented universe of large-cap firms whose monthly price, volume, and macro covariates are clean enough to feed a classifier and rich enough to make the resulting decisions consequential.

What changed alongside the modelling toolkit is the kind of question portfolio managers are asked. A decade ago, a quantitative signal that beat the benchmark on a backtest was largely self-justifying. Today, an investment committee is far more likely to ask *why* the model has flagged a particular ticker as a buy, and whether the answer would survive a regulator looking over their shoulder. That question is not academic. It is the operational question that decides whether a model gets deployed, whether a trade ticket gets signed, and whether a compliance officer puts the system into production at all. The motivation for this thesis grows out of that gap: we have models that predict well on German equities, and we have a market practice that increasingly demands a defensible account of how each prediction was reached.

## 1.2 The Explainability Gap

The current standard tools for explaining a black-box classifier are SHAP (Lundberg and Lee, 2017) and LIME (Ribeiro et al., 2016). Both are correlational. Each attributes a prediction to features by measuring how the model's output co-varies with those features in a local neighbourhood of the input. That is genuinely useful for sanity-checking a model and for surfacing spurious shortcuts, but it is not what an investor — or, increasingly, a regulator — actually wants to know. The decision-maker wants to know what would have happened if the world had been different: if the past twelve-month return had been five points lower, would the model still recommend buying? If the ECB rate had risen by another twenty-five basis points, would the call flip? These are counterfactual questions, and a method that ranks features by their marginal correlation with the score cannot answer them with any rigour (Wachter et al., 2017).

The literature has been clear on this distinction for some time. Rudin (2019) goes further, arguing that for high-stakes decisions one should prefer interpretable models outright, because post-hoc correlational explanations of black-box models can mislead in subtle ways. The middle path that this thesis takes is to keep the predictive power of a strong learner while delivering explanations that are causal rather than correlational, drawing on the recent line of work on counterfactual explanations grounded in causal discovery (Takahashi et al., 2024).

## 1.3 Regulatory Context

The regulatory backdrop sharpens the problem. Regulation (EU) 2024/1689 on Artificial Intelligence (the "EU AI Act") entered into force on 1 August 2024, with high-risk provisions becoming applicable on 2 August 2026 (European Union, 2024). AI systems used to evaluate creditworthiness or otherwise make or inform binding decisions on natural persons in financial services fall under Annex III and are classified as high-risk. Article 13 (Transparency) and Article 14 (Human oversight) require providers to explain *why* a decision was reached in terms a human supervisor can audit and override.

That language matters for the choice of explanation method. SHAP, being correlational, satisfies "transparency" only loosely — it cannot answer the question "if this feature had been different, would the decision have flipped?". A counterfactual artefact directly answers Article 13(3)(b)(iv)'s "main parameters of the decision" requirement in causal terms. The EU AI Act therefore does not just create a deadline; it changes the kind of explanation that counts. From August 2026 onward, an investment system that uses ML to inform decisions about retail clients in Germany will need transparency that an auditor can interrogate, and the bar is more demanding than a feature-importance bar chart alone can clear.

## 1.4 Research Question and Contribution

This thesis sits at the intersection of those two pressures — the practical demand for "why" inside an investment committee and the regulatory demand for auditable transparency in high-risk AI. It frames a single primary research question:

*Can a counterfactual, causally-grounded explanation method (LEWIS) deliver more useful and trustworthy explanations of an ML investment decision than a correlational baseline (SHAP), in the eyes of expert German-market investors and against EU AI Act transparency requirements?*

The contribution is fourfold. First, the thesis operationalises LEWIS — a counterfactual explanation framework that combines causal discovery with Pearl's backdoor adjustment (Takahashi et al., 2024) — on a real DAX panel rather than a synthetic benchmark. Second, it carries out a head-to-head comparison of LEWIS against the SHAP baseline on the same trained model, so that any divergence in the rankings can be attributed to the explanation method rather than to the underlying classifier. Third, it embeds both explanation styles in a live A/B survey instrument deployed to expert evaluators, generating primary data on which kind of explanation practitioners actually find more trustworthy. Fourth, it audits the survey responses with a regression of trust on five candidate predictors, isolating which features of an explanation actually move the needle for German-market investors.

## 1.5 Approach and Scope

The empirical setup is deliberately compact so that the analytical machinery, not the data engineering, is the centre of attention. The equity universe is a 20-ticker subset of the DAX (no ESG or size filter), observed monthly from January 2019 to December 2024, yielding a panel of 960 ticker-months. Five firm-level features (volatility, momentum, average volume, twelve-month return, and maximum drawdown) are combined with four macro covariates (the ECB main refinancing rate, the EUR/USD reference rate, German HICP year-on-year inflation, and the VIX). The label is a simple risk-adjusted return threshold: investment_decision = 1 when the ratio of twelve-month return to volatility exceeds 0.5, and 0 otherwise.

The classifier is CatBoost, with two hundred iterations, depth six, and a fixed random seed. The causal layer runs six discovery algorithms — PC, DirectLiNGAM, RESIT, LiM, NOTEARS, and NOTEARS-MLP — under three prior specifications, but the primary specification fixes DirectLiNGAM under prior (b), which forces the investment decision to be a sink in the graph and rules out reverse causation from the label back into the features. LEWIS necessity, sufficiency, and max-Nesuf scores are then computed via Pearl backdoor adjustment on the discovered DAG, and SHAP TreeExplainer values are computed on the same fitted model as a like-for-like correlational baseline.

The expert evaluation is run through a Streamlit application that randomly assigns each respondent to either the LEWIS or the SHAP explanation for a given ticker, collects trust and confidence ratings, and stores the responses. The analysis in this thesis is based on twenty-five expert evaluations. The application is publicly accessible at `https://germaninvestmentanalysis.streamlit.app`, and the underlying code, replication data, and configuration are available at `https://github.com/MutharasanE/german_investment_analysis`. The regulatory anchor running through every chapter is the EU AI Act: every methodological choice and every empirical finding is read back against what an Article 13 auditor would need to see.

## 1.6 Structure of the Thesis

The remainder of the thesis develops the argument in six steps. Chapter 2 reviews the literature on machine learning in asset pricing, on post-hoc explainability, on counterfactual and causal approaches to explanation, and on the trust-in-automation tradition that informs the survey design. Chapter 3 lays out the methodology in detail, from feature engineering and the CatBoost classifier through the suite of causal discovery algorithms to the LEWIS scoring procedure and the OLS specification used for the trust regression. Chapter 4 documents the data and the empirical setup, including the equity universe, stationarity treatment, and survey instrument. Chapter 5 reports the results — predictive performance, the discovered DAG, the LEWIS scores, the SHAP comparison, the survey descriptives, and the regression of trust on explanation features. Chapter 6 discusses what those findings mean for explanation design, for German-market practitioners, and for compliance with the EU AI Act. Chapter 7 concludes, sets out the limitations honestly, and points to the open questions that a follow-on study would need to address.

## Bibliography fragment

European Union (2024) *Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)*. Official Journal of the European Union, L 2024/1689.

Gu, S., Kelly, B. and Xiu, D. (2020) 'Empirical Asset Pricing via Machine Learning', *Review of Financial Studies*, 33(5), pp. 2223–2273.

Lopez de Prado, M. (2018) *Advances in Financial Machine Learning*. Hoboken: Wiley.

Lundberg, S. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', in *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Ribeiro, M. T., Singh, S. and Guestrin, C. (2016) '"Why Should I Trust You?": Explaining the Predictions of Any Classifier', in *Proceedings of the 22nd ACM SIGKDD*, pp. 1135–1144.

Rudin, C. (2019) 'Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead', *Nature Machine Intelligence*, 1(5), pp. 206–215.

Takahashi, R., Hara, S., Maeda, S. and Sasagawa, K. (2024) 'Counterfactual Explanations of Black-Box ML Models using Causal Discovery', arXiv:2402.02678.

Wachter, S., Mittelstadt, B. and Russell, C. (2017) 'Counterfactual Explanations Without Opening the Black Box', *Harvard Journal of Law & Technology*, 31(2), pp. 841–887.
