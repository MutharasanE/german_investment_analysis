"""
Module: Governance Framework
Purpose: Generate regulatory compliance report mapping pipeline outputs to
         EU AI Act, BaFin guidelines, and MiFID II requirements.
Inputs:  All results from Steps 1-9
Outputs: results/reports/regulatory_compliance_report.json,
         results/reports/regulatory_compliance_report.md
Reference: EU AI Act (2024), BaFin AI Guidelines (2024), MiFID II Article 25
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def build_compliance_report(pipeline_results):
    """
    Generate structured regulatory compliance report.
    Maps each pipeline component to regulatory requirements with evidence.

    Args:
        pipeline_results: dict containing all results from pipeline steps.

    Returns:
        Compliance report dict.
    """
    metrics = pipeline_results.get("metrics", {})
    agreement = pipeline_results.get("agreement", {})

    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "2.0-dax40-daily",
            "data_source": "DAX 40 via Yahoo Finance (daily OHLCV)",
            "model": "CatBoost MultiClass (Buy/Hold/Sell)",
            "causal_method": "DirectLiNGAM with prior (b) + PC cross-check",
        },
        "EU_AI_Act_Article_13": {
            "requirement": "Transparency — High-risk AI systems must provide "
                           "information enabling human oversight",
            "implementation": "LEWIS counterfactual explanations provide feature-level "
                              "causal necessity and sufficiency scores, enabling investment "
                              "managers to understand the causal drivers of each recommendation",
            "evidence": "results/tables/lewis_scores_causal.csv",
            "gap": "Multi-class extension of LEWIS not yet peer-reviewed",
        },
        "EU_AI_Act_Article_14": {
            "requirement": "Human oversight — meaningful ability to override",
            "implementation": "Streamlit app presents causal explanation alongside "
                              "prediction, allowing analyst to approve/reject",
            "evidence": "app.py survey functionality",
        },
        "BaFin_AI_Guidelines_2024": {
            "requirement": "Auditability of AI decision processes in banking",
            "implementation": "All model artifacts, DAG, scores, and explanations "
                              "are saved with timestamps and reproducible pipeline",
            "evidence": "models/ directory, pipeline.py",
        },
        "MiFID_II_Article_25": {
            "requirement": "Suitability — investment recommendations must be based "
                           "on clients risk profile and justified",
            "implementation": "Counterfactual feasibility constraints ensure "
                              "recommendations are based on actionable, non-macro features only",
            "evidence": "results/tables/counterfactual_validity_report.csv",
        },
        "model_quality": {
            "test_accuracy": metrics.get("accuracy", "N/A"),
            "cohen_kappa": metrics.get("cohen_kappa", "N/A"),
            "causal_pc_agreement_rate": agreement.get("agreement_rate", "N/A"),
        },
        "known_limitations": [
            "DirectLiNGAM assumes linear non-Gaussian relationships — "
            "financial data may violate this in nonlinear regimes",
            "DAG may be unstable during structural breaks (COVID-19, rate hikes) — "
            "bootstrap stability analysis documents which periods show instability",
            "Multi-class LEWIS extension is novel — no peer-reviewed validation exists yet "
            "(this IS the thesis contribution)",
            "Short data window (4 months daily) limits regime diversity — "
            "document as constraint in thesis methodology chapter",
            "Expert survey may have small sample size — "
            "document minimum required for statistical significance",
        ],
    }

    return report


def save_report(report, results_dir="results/reports"):
    """
    Save compliance report as both JSON (structured) and Markdown (readable).

    Args:
        report: Compliance report dict.
        results_dir: Directory to save reports.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # JSON format
    with open(os.path.join(results_dir, "regulatory_compliance_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown format
    md_lines = [
        "# Regulatory Compliance Report",
        f"\nGenerated: {report['report_metadata']['generated_at']}",
        f"\nPipeline: {report['report_metadata']['pipeline_version']}",
        "",
        "---",
        "",
    ]

    for section_key, section in report.items():
        if section_key in ("report_metadata", "known_limitations", "model_quality"):
            continue

        md_lines.extend([
            f"## {section_key}",
            "",
            f"**Requirement:** {section.get('requirement', 'N/A')}",
            "",
            f"**Implementation:** {section.get('implementation', 'N/A')}",
            "",
            f"**Evidence:** `{section.get('evidence', 'N/A')}`",
            "",
        ])
        if "gap" in section:
            md_lines.append(f"**Gap:** {section['gap']}")
            md_lines.append("")

    # Model quality section
    mq = report.get("model_quality", {})
    md_lines.extend([
        "## Model Quality Summary",
        "",
        f"- Test Accuracy: {mq.get('test_accuracy', 'N/A')}",
        f"- Cohen's Kappa: {mq.get('cohen_kappa', 'N/A')}",
        f"- PC/DirectLiNGAM Agreement: {mq.get('causal_pc_agreement_rate', 'N/A')}",
        "",
    ])

    # Limitations
    md_lines.extend([
        "## Known Limitations (document in thesis)",
        "",
    ])
    for i, lim in enumerate(report.get("known_limitations", []), 1):
        md_lines.append(f"{i}. {lim}")

    with open(os.path.join(results_dir, "regulatory_compliance_report.md"), "w") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Compliance report saved to {results_dir}/ (JSON + Markdown)")


def run(pipeline_results, results_dir="results/reports"):
    """
    Execute Step 10: Generate regulatory compliance report.

    Args:
        pipeline_results: Aggregated results from all pipeline steps.
        results_dir: Directory to save reports.

    Returns:
        Compliance report dict.
    """
    logger.info("=" * 60)
    logger.info("STEP 10: GOVERNANCE FRAMEWORK")
    logger.info("=" * 60)

    report = build_compliance_report(pipeline_results)
    save_report(report, results_dir)

    return report
