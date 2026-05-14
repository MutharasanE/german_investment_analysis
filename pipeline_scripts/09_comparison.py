"""
Module: 09_comparison
Purpose: Produce SHAP vs LEWIS comparison tables and ranking disagreement analysis.
Inputs:  results/tables/shap_scores.csv, results/tables/lewis_scores_causal.csv,
         results/tables/lewis_scores_no_graph.csv
Outputs: results/tables/shap_vs_lewis_comparison.csv,
         results/tables/lewis_vs_shap_comparison.csv,
         results/plots/shap_vs_lewis_comparison.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tables_dir = root / "results" / "tables"
    plots_dir = root / "results" / "plots"

    shap_df = pd.read_csv(tables_dir / "shap_scores.csv")
    lewis_c = pd.read_csv(tables_dir / "lewis_scores_causal.csv")
    lewis_n = pd.read_csv(tables_dir / "lewis_scores_no_graph.csv")

    df = (
        shap_df[["feature", "shap_mean"]]
        .merge(lewis_c[["feature", "maxNesuf"]], on="feature", how="inner")
        .rename(columns={"maxNesuf": "lewis_causal"})
        .merge(lewis_n[["feature", "maxNesuf"]], on="feature", how="left")
        .rename(columns={"maxNesuf": "lewis_no_graph"})
    )

    df["rank_shap"] = df["shap_mean"].rank(ascending=False, method="average")
    df["rank_lewis"] = df["lewis_causal"].rank(ascending=False, method="average")
    df["rank_abs_diff"] = (df["rank_shap"] - df["rank_lewis"]).abs()

    corr, pval = spearmanr(df["rank_shap"], df["rank_lewis"])
    df["spearman_rank_corr"] = corr
    df["spearman_pvalue"] = pval

    df = df.sort_values("rank_abs_diff", ascending=False)
    df.to_csv(tables_dir / "shap_vs_lewis_comparison.csv", index=False)
    df.to_csv(tables_dir / "lewis_vs_shap_comparison.csv", index=False)

    top = df.sort_values("lewis_causal", ascending=True)
    x = range(len(top))

    plt.figure(figsize=(12, 7))
    plt.barh([i - 0.25 for i in x], top["shap_mean"], height=0.24, label="SHAP")
    plt.barh(x, top["lewis_causal"], height=0.24, label="LEWIS (causal)")
    plt.barh([i + 0.25 for i in x], top["lewis_no_graph"], height=0.24, label="LEWIS (no graph)")
    plt.yticks(list(x), top["feature"])
    plt.xlabel("Importance score")
    plt.title(f"SHAP vs LEWIS Comparison (Spearman={corr:.3f}, p={pval:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_vs_lewis_comparison.png", dpi=150)
    plt.close()

    disagreements = df[df["rank_abs_diff"] >= 3][["feature", "rank_shap", "rank_lewis", "rank_abs_diff"]]
    disagreements.to_csv(tables_dir / "comparison_key_disagreements.csv", index=False)

    print(f"Saved comparison table: {tables_dir / 'shap_vs_lewis_comparison.csv'}")


if __name__ == "__main__":
    main()
