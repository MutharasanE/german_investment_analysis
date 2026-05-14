"""
Module: 03_labeling
Purpose: Generate Buy/Hold/Sell labels from real forward 3-month returns.
Inputs:  data/processed/features_engineered.csv
Outputs: data/processed/labeled_dataset.parquet, data/processed/labeled_dataset.csv,
         results/tables/label_distribution.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def assign_label(row: pd.Series) -> str:
    if row["forward_return_3m"] > row["buy_threshold"]:
        return "Buy"
    if row["forward_return_3m"] < row["sell_threshold"]:
        return "Sell"
    return "Hold"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    feature_parquet = processed_dir / "features_engineered.parquet"
    if feature_parquet.exists():
        df = pd.read_parquet(feature_parquet)
    else:
        df = pd.read_csv(processed_dir / "features_engineered.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Real forward return labeling from observed future prices only.
    df["forward_return_3m"] = df.groupby("Ticker", observed=True)["close"].shift(-63) / df["close"] - 1.0
    df["quarter"] = df["Date"].dt.to_period("Q").astype(str)

    grouped = df.groupby(["sector", "quarter"], observed=True)["forward_return_3m"]
    thresholds = grouped.agg(sector_median="median", sector_std="std").reset_index()
    thresholds["buy_threshold"] = thresholds["sector_median"] + 0.5 * thresholds["sector_std"].fillna(0)
    thresholds["sell_threshold"] = thresholds["sector_median"] - 0.5 * thresholds["sector_std"].fillna(0)

    df = df.merge(thresholds[["sector", "quarter", "buy_threshold", "sell_threshold"]], on=["sector", "quarter"], how="left")
    df = df.dropna(subset=["forward_return_3m", "buy_threshold", "sell_threshold"])

    # Sector-adjusted thresholds avoid persistent market-level bias across industries.
    # The +/- 0.5 * sector std rule targets practical class spread without synthetic rebalance.
    df["label"] = df.apply(assign_label, axis=1)
    df["year"] = df["Date"].dt.year

    distribution = (
        df.groupby(["year", "sector", "label"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    sector_year_total = distribution.groupby(["year", "sector"], observed=True)["count"].transform("sum")
    distribution["pct_within_sector_year"] = distribution["count"] / sector_year_total

    overall = df["label"].value_counts(normalize=True)
    imbalance_flag = (
        overall.get("Buy", 0) > 0.70 or overall.get("Hold", 0) > 0.70 or overall.get("Sell", 0) > 0.70
    )
    distribution["severe_imbalance_flag"] = imbalance_flag

    df.to_parquet(processed_dir / "labeled_dataset.parquet", index=False)
    # Keep a manageable CSV preview for quick inspection.
    df.head(50_000).to_csv(processed_dir / "labeled_dataset.csv", index=False)
    distribution.to_csv(tables_dir / "label_distribution.csv", index=False)

    print("Overall class distribution:")
    print(df["label"].value_counts(normalize=True).round(4).to_string())
    print(f"Severe imbalance (>70%) flag: {imbalance_flag}")
    print(f"Saved labeled dataset (full): {processed_dir / 'labeled_dataset.parquet'}")
    print(f"Saved labeled dataset (preview): {processed_dir / 'labeled_dataset.csv'}")
    print(f"Saved label distribution: {tables_dir / 'label_distribution.csv'}")


if __name__ == "__main__":
    main()
