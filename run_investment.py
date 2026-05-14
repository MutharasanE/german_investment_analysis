"""
Entry point: Run causal XAI pipeline on DAX investment data.
Usage: python run_investment.py
"""

import json
from pathlib import Path

import numpy as np

from src.pipeline import run_investment_pipeline
from src.visualization import (
    plot_causal_graph, plot_nesuf_comparison, plot_reversal_probabilities,
)


def main():
    """Run pipeline on DAX investment data."""
    from src.data_loader import download_dax_data, build_investment_dataset

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    results_dir = Path("results/investment")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading DAX data...")
    all_data = download_dax_data()

    if not all_data:
        print("ERROR: Could not download data. Check internet/SSL.")
        return

    # Save raw stock data
    for ticker, price_df in all_data.items():
        price_df.to_csv(data_dir / f"{ticker.replace('.', '_')}.csv")
    print(f"  Saved raw stock data for {len(all_data)} tickers to data/")

    print(f"\nBuilding investment dataset from {len(all_data)} tickers...")
    df = build_investment_dataset(all_data)
    print(f"Dataset: {len(df)} rows, {df.columns.tolist()}")
    print(f"Decision split: {df['investment_decision'].value_counts().to_dict()}")

    # Save the full dataset
    df.to_csv(data_dir / "investment_dataset.csv", index=False)
    print("  Saved investment dataset to data/investment_dataset.csv")

    feature_cols = [
        "volatility", "momentum", "volume_avg", "return_1y", "max_drawdown",
        "ecb_rate", "eur_usd", "de_inflation", "vix",
    ]
    # Drop any macro features that failed to download
    feature_cols = [c for c in feature_cols if c in df.columns]

    results = run_investment_pipeline(
        df, feature_cols, target_col="investment_decision",
        n_bins=10, methods=["DirectLiNGAM"], priors=["b"],
    )

    # Save artifacts for Streamlit demo and SHAP comparison
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for key, res in results.items():
        # Save CatBoost model
        res["model"].save_model(str(models_dir / f"catboost_{key}.cbm"))
        print(f"  Saved CatBoost model to models/catboost_{key}.cbm")

        # Save adjacency matrix
        np.save(models_dir / f"adj_matrix_{key}.npy", res["adj_matrix"])

        # Save LEWIS scores as CSV
        res["scores_causal"].to_csv(results_dir / f"lewis_scores_{key}.csv", index=False)
        res["scores_no_graph"].to_csv(results_dir / f"lewis_scores_no_graph_{key}.csv", index=False)
        res["reversal"].to_csv(results_dir / f"reversal_scores_{key}.csv", index=False)

        # Save metadata (feature cols, accuracy, etc.)
        metadata = {
            "feature_cols": feature_cols,
            "target_col": "investment_decision",
            "accuracy": res["accuracy"],
            "method_key": key,
            "n_rows": len(df),
            "n_tickers": len(all_data),
        }
        with open(models_dir / f"metadata_{key}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print("  Saved model artifacts to models/")

    # Visualize
    for key, res in results.items():
        col_names = feature_cols + ["investment_decision"]

        plot_causal_graph(
            res["adj_matrix"], col_names,
            title=f"Causal Graph ({key})",
            save_path=results_dir / f"causal_graph_{key}.png",
        )

        scores_c = res["scores_causal"].set_index("feature").reindex(feature_cols)
        scores_n = res["scores_no_graph"].set_index("feature").reindex(feature_cols)

        plot_nesuf_comparison(
            scores_c["maxNesuf"].values, scores_n["maxNesuf"].values,
            feature_cols,
            title=f"Nesuf: Causal vs No Graph ({key})",
            save_path=results_dir / f"nesuf_comparison_{key}.png",
        )

        rev = res["reversal"].set_index("feature").reindex(feature_cols)
        plot_reversal_probabilities(
            rev["Nec"].values, rev["Suf"].values,
            feature_cols,
            title=f"Reversal Probabilities ({key})",
            save_path=results_dir / f"reversal_{key}.png",
        )

    print("\nDone! Check results/investment/, models/, and data/ for all outputs.")


if __name__ == "__main__":
    main()
