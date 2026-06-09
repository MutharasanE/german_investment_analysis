"""
SHAP vs LEWIS comparison on the same CatBoost investment model.
Loads saved artifacts from run_investment.py — no retraining needed.

Usage: python run_shap_comparison.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier

from src.visualization import plot_lewis_vs_shap


def main():
    models_dir = Path("models")
    results_dir = Path("results/investment")
    data_path = Path("data/investment_dataset.csv")

    # Find available model keys
    meta_files = sorted(models_dir.glob("metadata_*.json"))
    if not meta_files:
        print("ERROR: No saved models found. Run `python run_investment.py` first.")
        return

    for meta_file in meta_files:
        key = meta_file.stem.replace("metadata_", "")
        print(f"\n{'='*60}")
        print(f"SHAP vs LEWIS comparison: {key}")
        print(f"{'='*60}")

        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)
        feature_cols = meta["feature_cols"]

        # Load model
        model = CatBoostClassifier()
        model.load_model(str(models_dir / f"catboost_{key}.cbm"))
        print(f"Loaded CatBoost model (accuracy: {meta['accuracy']:.4f})")

        # Load dataset
        df = pd.read_csv(data_path)
        X = df[feature_cols]
        print(f"Dataset: {len(df)} rows, {len(feature_cols)} features")

        # Compute SHAP values
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (APPROVE) explanations

        # Mean absolute SHAP value per feature (global importance)
        shap_importance = np.abs(shap_values).mean(axis=0)

        # Normalize to [0, 1]
        shap_normalized = shap_importance / shap_importance.max()

        # Load LEWIS scores
        lewis_df = pd.read_csv(results_dir / f"lewis_scores_{key}.csv")
        lewis_df = lewis_df.set_index("feature").reindex(feature_cols)
        lewis_scores = lewis_df["maxNesuf"].values

        # Normalize LEWIS to [0, 1]
        lewis_normalized = lewis_scores / lewis_scores.max()

        # Print comparison table
        print(f"\n{'Feature':<16} {'LEWIS (causal)':>15} {'SHAP (corr.)':>15} {'Rank LEWIS':>12} {'Rank SHAP':>12}")
        print("-" * 72)

        lewis_ranks = (-lewis_normalized).argsort().argsort() + 1
        shap_ranks = (-shap_normalized).argsort().argsort() + 1

        for i, feat in enumerate(feature_cols):
            print(f"{feat:<16} {lewis_normalized[i]:>15.4f} {shap_normalized[i]:>15.4f} {lewis_ranks[i]:>12d} {shap_ranks[i]:>12d}")

        # Spearman rank correlation between LEWIS and SHAP
        from scipy import stats
        corr, p_val = stats.spearmanr(lewis_normalized, shap_normalized)
        print(f"\nSpearman rank correlation (LEWIS vs SHAP): {corr:.4f} (p={p_val:.4f})")

        # Generate comparison plot
        plot_lewis_vs_shap(
            lewis_normalized, shap_normalized, feature_cols,
            title=f"LEWIS (Causal) vs SHAP (Correlational) — {key}",
            save_path=results_dir / f"lewis_vs_shap_{key}.png",
        )
        print(f"Saved plot to results/investment/lewis_vs_shap_{key}.png")

        # Save SHAP scores
        shap_df = pd.DataFrame({
            "feature": feature_cols,
            "shap_importance": shap_importance,
            "shap_normalized": shap_normalized,
        })
        shap_df.to_csv(results_dir / f"shap_scores_{key}.csv", index=False)
        print(f"Saved SHAP scores to results/investment/shap_scores_{key}.csv")

        # Save combined comparison
        combined = pd.DataFrame({
            "feature": feature_cols,
            "lewis_nesuf": lewis_scores,
            "lewis_normalized": lewis_normalized,
            "shap_importance": shap_importance,
            "shap_normalized": shap_normalized,
            "lewis_rank": lewis_ranks,
            "shap_rank": shap_ranks,
        })
        combined.to_csv(results_dir / f"lewis_vs_shap_{key}.csv", index=False)
        print(f"Saved comparison table to results/investment/lewis_vs_shap_{key}.csv")

        # SHAP summary plot (beeswarm — shows per-sample feature impact)
        fig_summary = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        plt.title(f"SHAP Summary — {key}", fontsize=14)
        plt.tight_layout()
        plt.savefig(results_dir / f"shap_summary_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot to results/investment/shap_summary_{key}.png")

        # Save raw SHAP values for Streamlit (per-sample explanations)
        np.save(models_dir / f"shap_values_{key}.npy", shap_values)
        print(f"Saved raw SHAP values to models/shap_values_{key}.npy")

    print("\nDone! SHAP comparison complete.")


if __name__ == "__main__":
    main()
