# Regulatory Compliance Report

Generated: 2026-04-16T22:09:14.061664

Pipeline: 2.0-dax40-daily

---

## EU_AI_Act_Article_13

**Requirement:** Transparency — High-risk AI systems must provide information enabling human oversight

**Implementation:** LEWIS counterfactual explanations provide feature-level causal necessity and sufficiency scores, enabling investment managers to understand the causal drivers of each recommendation

**Evidence:** `results/tables/lewis_scores_causal.csv`

**Gap:** Multi-class extension of LEWIS not yet peer-reviewed

## EU_AI_Act_Article_14

**Requirement:** Human oversight — meaningful ability to override

**Implementation:** Streamlit app presents causal explanation alongside prediction, allowing analyst to approve/reject

**Evidence:** `app.py survey functionality`

## BaFin_AI_Guidelines_2024

**Requirement:** Auditability of AI decision processes in banking

**Implementation:** All model artifacts, DAG, scores, and explanations are saved with timestamps and reproducible pipeline

**Evidence:** `models/ directory, pipeline.py`

## MiFID_II_Article_25

**Requirement:** Suitability — investment recommendations must be based on clients risk profile and justified

**Implementation:** Counterfactual feasibility constraints ensure recommendations are based on actionable, non-macro features only

**Evidence:** `results/tables/counterfactual_validity_report.csv`

## Model Quality Summary

- Test Accuracy: 0.4152
- Cohen's Kappa: 0.0421
- PC/DirectLiNGAM Agreement: 0.13043478260869565

## Known Limitations (document in thesis)

1. DirectLiNGAM assumes linear non-Gaussian relationships — financial data may violate this in nonlinear regimes
2. DAG may be unstable during structural breaks (COVID-19, rate hikes) — bootstrap stability analysis documents which periods show instability
3. Multi-class LEWIS extension is novel — no peer-reviewed validation exists yet (this IS the thesis contribution)
4. Short data window (4 months daily) limits regime diversity — document as constraint in thesis methodology chapter
5. Expert survey may have small sample size — document minimum required for statistical significance