# Regulatory Compliance Report

## EU_AI_Act_Article_13
- Requirement: Transparency for high-risk AI with understandable decision information
- Implementation: LEWIS causal necessity/sufficiency scores are generated per feature and saved as auditable tables
- Evidence: results/tables/lewis_scores_causal.csv
- Gap: Multi-class LEWIS extension remains a novel contribution and requires external peer validation

## EU_AI_Act_Article_14
- Requirement: Human oversight with meaningful intervention ability
- Implementation: Counterfactual feasibility constraints flag non-actionable macro features to support human override
- Evidence: results/tables/counterfactual_validity_report.csv
- Gap: Oversight quality depends on analyst training and policy integration

## BaFin_AI_Guidelines_2024
- Requirement: Auditability and traceability of model behavior in banking context
- Implementation: Pipeline saves reproducible artifacts for model, DAG, explanation tables, and logs
- Evidence: models/, results/tables/, results/pipeline.log
- Gap: Operational controls for production deployment are out of scope of thesis prototype

## MiFID_II_Article_25
- Requirement: Suitability and justification of investment recommendations
- Implementation: Actionability-aware counterfactual checks reduce infeasible recommendation drivers
- Evidence: results/tables/counterfactual_validity_report.csv
- Gap: Client-specific suitability dimensions require additional profile data
