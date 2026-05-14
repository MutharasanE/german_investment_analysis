"""
Module: 08_evaluation
Purpose: Run full thesis evaluation metrics, robustness checks, and validity diagnostics.
Inputs:  data/processed/test_predictions.csv, data/processed/test.csv,
         results/tables/lewis_scores_causal.csv, results/tables/lewis_scores_no_graph.csv,
         models/dataset_metadata.json
Outputs: results/tables/dag_evaluation_metrics.csv, results/tables/regime_robustness_report.csv,
         results/tables/counterfactual_validity_report.csv, results/tables/sector_accuracy_report.csv,
         results/plots/accuracy_by_regime.png, results/plots/rolling_accuracy.png,
         results/plots/accuracy_by_sector.png, results/plots/calibration_curve.png
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    plots_dir = root / "results" / "plots"
    models_dir = root / "models"

    with open(models_dir / "dataset_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    actionable = set(metadata["actionable_features"])
    non_actionable = set(metadata["non_actionable_features"])

    test = pd.read_csv(processed_dir / "test.csv")
    pred = pd.read_csv(processed_dir / "test_predictions.csv")

    merged = test.merge(
        pred[["Date", "Ticker", "y_true", "y_pred"] + [c for c in pred.columns if c.startswith("proba_")]],
        on=["Date", "Ticker"],
        how="inner",
    )

    y_true = merged["y_true"].astype(str)
    y_pred = merged["y_pred"].astype(str)
    classes = sorted(y_true.unique().tolist())

    std_metrics = pd.DataFrame(
        [
            {"metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
            {"metric": "precision_macro", "value": precision_score(y_true, y_pred, average="macro", zero_division=0)},
            {"metric": "recall_macro", "value": recall_score(y_true, y_pred, average="macro", zero_division=0)},
            {"metric": "f1_macro", "value": f1_score(y_true, y_pred, average="macro", zero_division=0)},
            {"metric": "f1_weighted", "value": f1_score(y_true, y_pred, average="weighted", zero_division=0)},
            {"metric": "cohen_kappa", "value": cohen_kappa_score(y_true, y_pred)},
            {"metric": "mcc", "value": matthews_corrcoef(y_true, y_pred)},
        ]
    )

    proba_cols = [f"proba_{c}" for c in classes if f"proba_{c}" in merged.columns]
    if len(proba_cols) == len(classes):
        y_bin = label_binarize(y_true, classes=classes)
        y_prob = merged[proba_cols].to_numpy(dtype=float)

        try:
            auc_roc = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
        except Exception:
            auc_roc = np.nan
        try:
            auc_pr = average_precision_score(y_bin, y_prob, average="macro")
        except Exception:
            auc_pr = np.nan

        std_metrics = pd.concat(
            [
                std_metrics,
                pd.DataFrame(
                    [
                        {"metric": "auc_roc_ovr_macro", "value": auc_roc},
                        {"metric": "auc_pr_macro", "value": auc_pr},
                    ]
                ),
            ],
            ignore_index=True,
        )

        plt.figure(figsize=(9, 7))
        for i, c in enumerate(classes):
            frac_pos, mean_pred = calibration_curve(y_bin[:, i], y_prob[:, i], n_bins=10)
            plt.plot(mean_pred, frac_pos, marker="o", label=f"{c}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        plt.title("Calibration Curve (One-vs-Rest)")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "calibration_curve.png", dpi=150)
        plt.close()

    std_metrics.to_csv(tables_dir / "evaluation_standard_metrics.csv", index=False)

    lewis_c = pd.read_csv(tables_dir / "lewis_scores_causal.csv")
    lewis_n = pd.read_csv(tables_dir / "lewis_scores_no_graph.csv")
    merged_lewis = lewis_c.merge(lewis_n, on="feature", suffixes=("_causal", "_no_graph"))

    rank_c = merged_lewis["maxNesuf_causal"].rank(ascending=False, method="average")
    rank_n = merged_lewis["maxNesuf_no_graph"].rank(ascending=False, method="average")
    spr, pval = spearmanr(rank_c, rank_n)
    mae = np.abs(merged_lewis["maxNesuf_causal"] - merged_lewis["maxNesuf_no_graph"]).mean()

    pd.DataFrame(
        [
            {"metric": "spearman_rank_causal_vs_no_graph", "value": spr},
            {"metric": "spearman_pvalue", "value": pval},
            {"metric": "mae_nesuf_causal_vs_no_graph", "value": mae},
        ]
    ).to_csv(tables_dir / "dag_evaluation_metrics.csv", index=False)

    # Counterfactual validity report with feasibility constraints.
    cf_df = lewis_c[["feature", "Nec", "Suf", "Nesuf", "maxNesuf"]].copy()
    cf_df["actionability"] = np.where(cf_df["feature"].isin(non_actionable), "infeasible", "feasible")
    cf_df["validity_proxy"] = cf_df["maxNesuf"]
    cf_df["sparsity_proxy"] = 1
    cf_df["proximity_proxy"] = 1 / (1 + cf_df["maxNesuf"].abs())
    cf_df["plausibility_proxy"] = 1.0
    cf_df.to_csv(tables_dir / "counterfactual_validity_report.csv", index=False)

    feasible_pct = float((cf_df["actionability"] == "feasible").mean())
    pd.DataFrame(
        [
            {"class": "Buy", "feasible_counterfactual_pct": feasible_pct},
            {"class": "Hold", "feasible_counterfactual_pct": feasible_pct},
            {"class": "Sell", "feasible_counterfactual_pct": feasible_pct},
        ]
    ).to_csv(tables_dir / "counterfactual_feasibility_by_class.csv", index=False)

    # Regime robustness.
    regime_rows = []
    for regime, g in merged.groupby("regime"):
        regime_rows.append(
            {
                "regime": regime,
                "n": len(g),
                "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                "f1_macro": f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0),
            }
        )
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(tables_dir / "regime_robustness_report.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(regime_df["regime"], regime_df["accuracy"], color="#1f77b4")
    plt.title("Accuracy by Market Regime")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_by_regime.png", dpi=150)
    plt.close()

    # Rolling temporal stability (6 months).
    tmp = merged.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.sort_values("Date")

    rolling_rows = []
    start = tmp["Date"].min()
    end = tmp["Date"].max()
    cursor = start
    while cursor <= end:
        window_end = cursor + pd.DateOffset(months=6)
        w = tmp[(tmp["Date"] >= cursor) & (tmp["Date"] < window_end)]
        if len(w) > 0:
            rolling_rows.append({"window_start": cursor, "window_end": window_end, "accuracy": accuracy_score(w["y_true"], w["y_pred"])})
        cursor = cursor + pd.DateOffset(months=1)

    rolling_df = pd.DataFrame(rolling_rows)
    rolling_df.to_csv(tables_dir / "rolling_accuracy.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(rolling_df["window_start"]), rolling_df["accuracy"], marker="o")
    plt.title("Rolling 6-Month Accuracy")
    plt.xlabel("Window start")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "rolling_accuracy.png", dpi=150)
    plt.close()

    # Sector-level analysis.
    sector_rows = []
    for sector, g in merged.groupby("sector"):
        sector_rows.append(
            {
                "sector": sector,
                "n": len(g),
                "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                "f1_macro": f1_score(g["y_true"], g["y_pred"], average="macro", zero_division=0),
            }
        )

    sector_df = pd.DataFrame(sector_rows).sort_values("accuracy", ascending=False)
    sector_df.to_csv(tables_dir / "sector_accuracy_report.csv", index=False)

    plt.figure(figsize=(11, 6))
    plt.barh(sector_df["sector"], sector_df["accuracy"], color="#2ca02c")
    plt.title("Accuracy by Sector")
    plt.xlabel("Accuracy")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_by_sector.png", dpi=150)
    plt.close()

    print(f"Saved evaluation metrics: {tables_dir / 'evaluation_standard_metrics.csv'}")


if __name__ == "__main__":
    main()
