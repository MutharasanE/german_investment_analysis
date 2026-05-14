"""
Module: 10_governance
Purpose: Generate regulatory compliance mapping and expert survey analysis artifacts.
Inputs:  results/tables/*, data/survey.db (optional), data/raw/survey.db (optional)
Outputs: results/reports/regulatory_compliance_report.json,
         results/reports/regulatory_compliance_report.md,
         results/tables/expert_survey_analysis.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def read_survey_votes(root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "survey.db",
        root.parent / "data" / "survey.db",
    ]
    for db_path in candidates:
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        try:
            votes = pd.read_sql_query("SELECT * FROM votes", conn)
            return votes
        finally:
            conn.close()

    return pd.DataFrame(columns=["preference", "comment", "expert_role", "timestamp"])


def build_compliance_report() -> dict[str, dict[str, str]]:
    return {
        "EU_AI_Act_Article_13": {
            "requirement": "Transparency for high-risk AI with understandable decision information",
            "implementation": "LEWIS causal necessity/sufficiency scores are generated per feature and saved as auditable tables",
            "evidence": "results/tables/lewis_scores_causal.csv",
            "gap": "Multi-class LEWIS extension remains a novel contribution and requires external peer validation",
        },
        "EU_AI_Act_Article_14": {
            "requirement": "Human oversight with meaningful intervention ability",
            "implementation": "Counterfactual feasibility constraints flag non-actionable macro features to support human override",
            "evidence": "results/tables/counterfactual_validity_report.csv",
            "gap": "Oversight quality depends on analyst training and policy integration",
        },
        "BaFin_AI_Guidelines_2024": {
            "requirement": "Auditability and traceability of model behavior in banking context",
            "implementation": "Pipeline saves reproducible artifacts for model, DAG, explanation tables, and logs",
            "evidence": "models/, results/tables/, results/pipeline.log",
            "gap": "Operational controls for production deployment are out of scope of thesis prototype",
        },
        "MiFID_II_Article_25": {
            "requirement": "Suitability and justification of investment recommendations",
            "implementation": "Actionability-aware counterfactual checks reduce infeasible recommendation drivers",
            "evidence": "results/tables/counterfactual_validity_report.csv",
            "gap": "Client-specific suitability dimensions require additional profile data",
        },
    }


def save_markdown_report(report: dict[str, dict[str, str]], out_path: Path) -> None:
    lines = ["# Regulatory Compliance Report", ""]
    for article, body in report.items():
        lines.append(f"## {article}")
        lines.append(f"- Requirement: {body['requirement']}")
        lines.append(f"- Implementation: {body['implementation']}")
        lines.append(f"- Evidence: {body['evidence']}")
        lines.append(f"- Gap: {body['gap']}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables_dir = root / "results" / "tables"
    reports_dir = root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    votes = read_survey_votes(root)

    if votes.empty:
        survey_summary = pd.DataFrame(
            [
                {
                    "n_votes": 0,
                    "pct_prefer_causal": np.nan,
                    "pct_prefer_shap": np.nan,
                    "p_value_binomtest_prefers_causal": np.nan,
                    "sample_size_warning": "No survey records found",
                }
            ]
        )
    else:
        pref = votes["preference"].astype(str)
        n = len(pref)
        n_causal = int((pref.str.lower() == "lewis").sum())
        n_shap = int((pref.str.lower() == "shap").sum())
        pval = binomtest(k=n_causal, n=max(1, n_causal + n_shap), p=0.5).pvalue if (n_causal + n_shap) > 0 else np.nan
        warning = "Sample below 30 responses; inferential significance is limited" if n < 30 else ""

        survey_summary = pd.DataFrame(
            [
                {
                    "n_votes": n,
                    "pct_prefer_causal": n_causal / max(1, n),
                    "pct_prefer_shap": n_shap / max(1, n),
                    "p_value_binomtest_prefers_causal": pval,
                    "sample_size_warning": warning,
                }
            ]
        )

    survey_summary.to_csv(tables_dir / "expert_survey_analysis.csv", index=False)

    report = build_compliance_report()

    with open(reports_dir / "regulatory_compliance_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    save_markdown_report(report, reports_dir / "regulatory_compliance_report.md")

    print(f"Saved compliance reports in: {reports_dir}")
    print(f"Saved survey analysis: {tables_dir / 'expert_survey_analysis.csv'}")


if __name__ == "__main__":
    main()
