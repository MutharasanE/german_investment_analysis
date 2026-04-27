"""
Module: SHAP vs LEWIS Comparison
Purpose: Side-by-side feature ranking comparison between SHAP (correlational) and
         LEWIS (causal). This is the core academic contribution of the thesis.
Inputs:  results/tables/shap_scores.csv, results/tables/lewis_scores_causal.csv,
         results/tables/lewis_scores_no_graph.csv
Outputs: results/tables/shap_vs_lewis_comparison.csv,
         results/plots/shap_vs_lewis_comparison.png,
         results/reports/key_disagreements.txt
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def build_comparison_table(shap_scores, lewis_causal, lewis_no_graph, results_dir="results"):
    """
    Build side-by-side comparison table of SHAP vs LEWIS rankings.

    Args:
        shap_scores: DataFrame with feature, shap_mean columns.
        lewis_causal: DataFrame with feature, maxNesuf_avg columns.
        lewis_no_graph: DataFrame with feature, maxNesuf_avg columns.
        results_dir: Base results directory.

    Returns:
        Comparison DataFrame, Spearman correlation between SHAP and LEWIS ranks.
    """
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)

    # Normalize scores to [0, 1] for comparison
    def normalize(series):
        rng = series.max() - series.min()
        if rng == 0:
            return series * 0
        return (series - series.min()) / rng

    comparison = pd.DataFrame({"feature": shap_scores["feature"].values})

    # SHAP importance
    shap_merged = shap_scores.set_index("feature")["shap_mean"]
    comparison["shap_importance"] = comparison["feature"].map(shap_merged).fillna(0)

    # LEWIS causal scores
    lewis_c = lewis_causal.set_index("feature")["maxNesuf_avg"]
    comparison["lewis_causal"] = comparison["feature"].map(lewis_c).fillna(0)

    # LEWIS no-graph scores
    lewis_ng = lewis_no_graph.set_index("feature")["maxNesuf_avg"]
    comparison["lewis_no_graph"] = comparison["feature"].map(lewis_ng).fillna(0)

    # Compute ranks (higher score = rank 1)
    comparison["rank_shap"] = rankdata(-comparison["shap_importance"]).astype(int)
    comparison["rank_lewis"] = rankdata(-comparison["lewis_causal"]).astype(int)
    comparison["rank_diff"] = abs(comparison["rank_shap"] - comparison["rank_lewis"])

    # Spearman rank correlation
    spr, p_value = spearmanr(comparison["rank_shap"], comparison["rank_lewis"])
    comparison.attrs["spearman_correlation"] = spr
    comparison.attrs["spearman_p_value"] = p_value

    comparison.to_csv(
        os.path.join(results_dir, "tables", "shap_vs_lewis_comparison.csv"), index=False
    )
    logger.info(f"SHAP vs LEWIS Spearman rank correlation: {spr:.4f} (p={p_value:.4f})")
    return comparison, spr, p_value


def identify_disagreements(comparison_df, results_dir="results"):
    """
    Identify features where SHAP and LEWIS strongly disagree in ranking.
    These represent potential spurious correlations that SHAP captures but LEWIS rejects.

    Args:
        comparison_df: DataFrame with rank_shap, rank_lewis columns.
        results_dir: Base results directory.

    Returns:
        List of disagreement dicts.
    """
    Path(os.path.join(results_dir, "reports")).mkdir(parents=True, exist_ok=True)

    disagreements = []
    for _, row in comparison_df.iterrows():
        if row["rank_diff"] >= 2:  # Significant rank difference
            reasoning = _explain_disagreement(row["feature"], row["rank_shap"], row["rank_lewis"])
            disagreements.append({
                "feature": row["feature"],
                "rank_shap": int(row["rank_shap"]),
                "rank_lewis": int(row["rank_lewis"]),
                "rank_diff": int(row["rank_diff"]),
                "reasoning": reasoning,
            })

    # Save disagreement report
    report_lines = [
        "KEY DISAGREEMENTS: SHAP vs LEWIS Feature Rankings",
        "=" * 60,
        "",
        "Features where SHAP (correlational) and LEWIS (causal) rankings",
        "differ by 2+ positions. These disagreements reveal where correlation",
        "does not imply causation in the investment domain.",
        "",
    ]

    if not disagreements:
        report_lines.append("No significant disagreements found (all rank diffs < 2).")
    else:
        for d in disagreements:
            report_lines.extend([
                f"Feature: {d['feature']}",
                f"  SHAP rank: {d['rank_shap']} | LEWIS rank: {d['rank_lewis']} (diff: {d['rank_diff']})",
                f"  Interpretation: {d['reasoning']}",
                "",
            ])

    with open(os.path.join(results_dir, "reports", "key_disagreements.txt"), "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Found {len(disagreements)} significant SHAP/LEWIS disagreements")
    return disagreements


def _explain_disagreement(feature, rank_shap, rank_lewis):
    """Generate domain-informed explanation for SHAP/LEWIS disagreement."""
    explanations = {
        "vix": ("VIX has high SHAP importance (correlated with crisis periods) but "
                "lower LEWIS Nesuf (cannot be intervened upon by investors — macro variable)"),
        "eur_usd": ("EUR/USD has high predictive power (correlated with export earnings) but "
                     "lower causal importance (currency movements are exogenous to individual stocks)"),
        "momentum": ("Momentum may rank high in SHAP (strong predictor) but differently in LEWIS "
                      "depending on whether causal graph identifies direct vs indirect effects"),
        "volatility": ("Volatility is both a strong predictor and a causal driver — "
                        "disagreement may reflect confounding through VIX or market regime"),
        "rsi_14": ("RSI captures mean-reversion patterns; SHAP may overweight it if correlated "
                    "with momentum, while LEWIS separates the causal contribution"),
        "volume_avg": ("Volume can be both a leading indicator and a consequence of price moves; "
                        "LEWIS identifies the causal direction while SHAP treats both equally"),
        "max_drawdown": ("Max drawdown captures tail risk; disagreement may indicate SHAP "
                          "conflates it with volatility while LEWIS distinguishes the causal path"),
    }
    if feature in explanations:
        return explanations[feature]
    if rank_shap < rank_lewis:
        return f"{feature} is more predictive (SHAP) than causally influential (LEWIS)"
    return f"{feature} is more causally influential (LEWIS) than predictive (SHAP)"


def plot_comparison(comparison_df, spr, p_value, results_dir="results"):
    """
    Plot side-by-side SHAP vs LEWIS feature importance bar chart.

    Args:
        comparison_df: Comparison DataFrame.
        spr: Spearman correlation.
        p_value: Spearman p-value.
        results_dir: Base results directory.
    """
    Path(os.path.join(results_dir, "plots")).mkdir(parents=True, exist_ok=True)

    features = comparison_df["feature"].values
    x = np.arange(len(features))
    width = 0.25

    # Normalize for visual comparison
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    shap_norm = norm(comparison_df["shap_importance"])
    lewis_c_norm = norm(comparison_df["lewis_causal"])
    lewis_ng_norm = norm(comparison_df["lewis_no_graph"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, shap_norm, width, label="SHAP (correlational)", color="#ff6b6b", alpha=0.85)
    ax.bar(x, lewis_c_norm, width, label="LEWIS causal (with DAG)", color="#4ecdc4", alpha=0.85)
    ax.bar(x + width, lewis_ng_norm, width, label="LEWIS no-graph (baseline)", color="#95e1d3", alpha=0.65)

    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Importance Score")
    ax.set_title(f"SHAP vs LEWIS Feature Importance\n(Spearman rank correlation: {spr:.3f}, p={p_value:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "shap_vs_lewis_comparison.png"), dpi=150)
    plt.close()
    logger.info("SHAP vs LEWIS comparison plot saved")


def run(shap_importance, lewis_causal, lewis_no_graph, results_dir="results"):
    """
    Execute Step 9: Full SHAP vs LEWIS comparison.

    Args:
        shap_importance: SHAP scores DataFrame.
        lewis_causal: Causal LEWIS scores DataFrame.
        lewis_no_graph: No-graph LEWIS scores DataFrame.
        results_dir: Base results directory.

    Returns:
        dict with comparison_df, spearman_corr, disagreements.
    """
    logger.info("=" * 60)
    logger.info("STEP 9: SHAP vs LEWIS COMPARISON")
    logger.info("=" * 60)

    # 9.1 Build comparison table
    comparison_df, spr, p_value = build_comparison_table(
        shap_importance, lewis_causal, lewis_no_graph, results_dir
    )

    # 9.2 Identify key disagreements
    disagreements = identify_disagreements(comparison_df, results_dir)

    # 9.3 Plot comparison
    plot_comparison(comparison_df, spr, p_value, results_dir)

    return {
        "comparison_df": comparison_df,
        "spearman_corr": spr,
        "spearman_p_value": p_value,
        "disagreements": disagreements,
    }
