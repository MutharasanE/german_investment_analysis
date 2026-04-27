"""
Module: Labeling
Purpose: Generate Buy/Hold/Sell labels from real forward 5-day returns.
Inputs:  Feature DataFrames per ticker (from Step 2)
Outputs: data/raw/labeled_dataset.csv, results/tables/label_distribution.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678 — extended to 3-class
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

FORWARD_DAYS = 5  # 1-week forward return for labeling


def compute_labels(features_df, price_df, forward_days=FORWARD_DAYS):
    """
    Compute Buy/Hold/Sell labels from forward returns.

    Uses distribution-based thresholds (0.5 std from mean) to create
    roughly balanced classes without artificial resampling.

    Args:
        features_df: DataFrame with computed features (date-indexed).
        price_df: DataFrame with Close prices (date-indexed).
        forward_days: Number of days for forward return computation.

    Returns:
        DataFrame with features + 'forward_return' + 'label' columns.
    """
    close = price_df["Close"].squeeze() if hasattr(price_df["Close"], "squeeze") else price_df["Close"]

    # Forward return: (price_t+n / price_t) - 1
    forward_return = close.shift(-forward_days) / close - 1
    forward_return.name = "forward_return"

    # Merge with features
    labeled = features_df.join(forward_return, how="inner")
    labeled = labeled.dropna(subset=["forward_return"])

    return labeled


def assign_labels(dataset_df):
    """
    Assign Buy/Hold/Sell labels using distribution-based thresholds.

    Thresholds: mean +/- 0.5 * std of forward returns.
    This gives roughly balanced classes (30/40/30 split).

    Args:
        dataset_df: DataFrame with 'forward_return' column.

    Returns:
        DataFrame with 'label' column added (Buy=2, Hold=1, Sell=0).
    """
    fwd = dataset_df["forward_return"]
    mean_ret = fwd.mean()
    std_ret = fwd.std()

    buy_threshold = mean_ret + 0.5 * std_ret
    sell_threshold = mean_ret - 0.5 * std_ret

    conditions = [
        fwd > buy_threshold,
        fwd < sell_threshold,
    ]
    choices = [2, 0]  # Buy=2, Sell=0
    dataset_df["label"] = np.select(conditions, choices, default=1)  # Hold=1

    logger.info(f"Label thresholds: sell < {sell_threshold:.4f} < hold < {buy_threshold:.4f} < buy")
    return dataset_df


def run(all_features, stock_data, output_dir="data/raw", results_dir="results/tables"):
    """
    Execute Step 3: Label all tickers and combine into single dataset.

    Args:
        all_features: dict {ticker: features_df} from Step 2.
        stock_data: dict {ticker: price_df} from Step 1.

    Returns:
        DataFrame: Combined labeled dataset with all tickers.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: LABELING")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_labeled = []

    for ticker, features_df in all_features.items():
        if ticker not in stock_data:
            continue
        price_df = stock_data[ticker]
        labeled = compute_labels(features_df, price_df)
        labeled["ticker"] = ticker
        all_labeled.append(labeled)

    if not all_labeled:
        raise RuntimeError("No tickers produced valid labeled data")

    dataset = pd.concat(all_labeled, axis=0)
    dataset = assign_labels(dataset)

    # Log class balance
    label_map = {0: "Sell", 1: "Hold", 2: "Buy"}
    dist = dataset["label"].value_counts().sort_index()
    total = len(dataset)
    logger.info("Class distribution:")
    for label_val, count in dist.items():
        logger.info(f"  {label_map.get(label_val, label_val)}: {count} ({100*count/total:.1f}%)")

    # Save label distribution
    dist_df = pd.DataFrame({
        "label": [label_map.get(k, k) for k in dist.index],
        "count": dist.values,
        "percentage": (100 * dist.values / total).round(1),
    })
    dist_df.to_csv(os.path.join(results_dir, "label_distribution.csv"), index=False)

    # Save full dataset
    dataset.to_csv(os.path.join(output_dir, "labeled_dataset.csv"))
    logger.info(f"Labeled dataset: {len(dataset)} rows, {dataset['ticker'].nunique()} tickers")

    return dataset
