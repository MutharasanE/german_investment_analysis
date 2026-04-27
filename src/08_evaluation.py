"""
Module: Evaluation Framework
Purpose: Full evaluation: ML metrics, counterfactual validity, DAG evaluation,
         AUC-ROC/PR curves. This is the most important section for thesis validity.
Inputs:  Model predictions, LEWIS scores, adjacency matrix
Outputs: results/tables/counterfactual_validity_report.csv,
         results/tables/dag_evaluation_metrics.csv,
         results/plots/auc_roc.png, results/plots/auc_pr.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Sell", "Hold", "Buy"]
MACRO_FEATURES = {"vix", "eur_usd"}  # Non-actionable features


def compute_ml_metrics(y_true, y_pred, y_prob=None, results_dir="results"):
    """
    Compute comprehensive ML metrics on test set.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (n_samples, n_classes).
        results_dir: Base results directory.

    Returns:
        dict with all metrics.
    """
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_dir, "plots")).mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
    }

    for i, name in enumerate(LABEL_NAMES):
        metrics[f"precision_{name}"] = round(precision[i], 4)
        metrics[f"recall_{name}"] = round(recall[i], 4)
        metrics[f"f1_{name}"] = round(f1[i], 4)

    # Weighted and macro F1
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["f1_weighted"] = round(f1_weighted, 4)
    metrics["f1_macro"] = round(f1_macro, 4)

    # AUC-ROC (one-vs-rest) if probabilities available
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            # Plot ROC curves per class
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#ff6b6b", "#ffd93d", "#4ecdc4"]
            for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
                if y_true_bin[:, i].sum() > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2,
                            label=f"{name} (AUC = {roc_auc:.3f})")
                    metrics[f"auc_roc_{name}"] = round(roc_auc, 4)

            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (One-vs-Rest)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "plots", "auc_roc.png"), dpi=150)
            plt.close()

            # AUC-PR curves
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
                if y_true_bin[:, i].sum() > 0:
                    prec_curve, rec_curve, _ = precision_recall_curve(
                        y_true_bin[:, i], y_prob[:, i]
                    )
                    pr_auc = auc(rec_curve, prec_curve)
                    ax.plot(rec_curve, prec_curve, color=color, lw=2,
                            label=f"{name} (AUC = {pr_auc:.3f})")
                    metrics[f"auc_pr_{name}"] = round(pr_auc, 4)

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curves (One-vs-Rest)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "plots", "auc_pr.png"), dpi=150)
            plt.close()

            logger.info("AUC-ROC and AUC-PR plots saved")
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, "tables", "ml_metrics.csv"), index=False
    )
    logger.info(f"ML metrics: accuracy={acc:.4f}, kappa={kappa:.4f}, F1_macro={f1_macro:.4f}")
    return metrics


def evaluate_counterfactual_validity(lewis_scores, feature_cols, results_dir="results"):
    """
    Evaluate counterfactual explanation quality: validity, actionability.

    Actionability constraints: macro features (VIX, EUR/USD, ECB rate) cannot be
    changed by an investor — flag counterfactuals involving these as INFEASIBLE.

    Args:
        lewis_scores: DataFrame with LEWIS scores per feature.
        feature_cols: Feature column names.
        results_dir: Base results directory.

    Returns:
        DataFrame with validity report.
    """
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)

    report = []
    for _, row in lewis_scores.iterrows():
        feat = row["feature"]
        is_actionable = feat not in MACRO_FEATURES
        nesuf = row.get("maxNesuf_avg", row.get("maxNesuf", 0))

        report.append({
            "feature": feat,
            "maxNesuf": round(nesuf, 4),
            "actionable": is_actionable,
            "feasible_counterfactual": is_actionable and nesuf > 0,
            "note": "macro (non-actionable)" if not is_actionable else "stock (actionable)",
        })

    report_df = pd.DataFrame(report)
    n_feasible = report_df["feasible_counterfactual"].sum()
    n_total = len(report_df)

    report_df.to_csv(
        os.path.join(results_dir, "tables", "counterfactual_validity_report.csv"), index=False
    )
    logger.info(f"Counterfactual validity: {n_feasible}/{n_total} features have feasible counterfactuals")
    return report_df


def evaluate_dag_quality(causal_scores, no_graph_scores, results_dir="results"):
    """
    Evaluate causal graph quality by comparing LEWIS scores with vs without graph.
    Uses Spearman Rank Correlation and MAE between the two score sets.

    Args:
        causal_scores: LEWIS scores with causal graph.
        no_graph_scores: LEWIS scores without causal graph.
        results_dir: Base results directory.

    Returns:
        dict with SPR and MAE metrics.
    """
    Path(os.path.join(results_dir, "tables")).mkdir(parents=True, exist_ok=True)

    # Merge on feature name
    merged = causal_scores[["feature", "maxNesuf_avg"]].merge(
        no_graph_scores[["feature", "maxNesuf_avg"]],
        on="feature", suffixes=("_causal", "_no_graph")
    )

    spr, p_value = spearmanr(merged["maxNesuf_avg_causal"], merged["maxNesuf_avg_no_graph"])
    mae = np.abs(merged["maxNesuf_avg_causal"] - merged["maxNesuf_avg_no_graph"]).mean()

    metrics = {
        "spearman_rank_correlation": round(spr, 4),
        "spearman_p_value": round(p_value, 4),
        "mae_causal_vs_no_graph": round(mae, 4),
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, "tables", "dag_evaluation_metrics.csv"), index=False
    )
    logger.info(f"DAG evaluation: SPR={spr:.4f} (p={p_value:.4f}), MAE={mae:.4f}")
    return metrics


def run(y_true, y_pred, y_prob, lewis_causal, lewis_no_graph, feature_cols,
        results_dir="results"):
    """
    Execute Step 8: Full evaluation framework.

    Args:
        y_true: True test labels.
        y_pred: Predicted test labels.
        y_prob: Predicted probabilities.
        lewis_causal: Causal LEWIS scores DataFrame.
        lewis_no_graph: No-graph LEWIS scores DataFrame.
        feature_cols: Feature column names.
        results_dir: Base results directory.

    Returns:
        dict with ml_metrics, counterfactual_report, dag_metrics.
    """
    logger.info("=" * 60)
    logger.info("STEP 8: EVALUATION FRAMEWORK")
    logger.info("=" * 60)

    # 8.1 Standard ML metrics
    ml_metrics = compute_ml_metrics(y_true, y_pred, y_prob, results_dir)

    # 8.2 Counterfactual validity
    cf_report = evaluate_counterfactual_validity(lewis_causal, feature_cols, results_dir)

    # 8.3 DAG evaluation
    dag_metrics = evaluate_dag_quality(lewis_causal, lewis_no_graph, results_dir)

    return {
        "ml_metrics": ml_metrics,
        "counterfactual_report": cf_report,
        "dag_metrics": dag_metrics,
    }
