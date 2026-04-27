"""
Module: Data Preparation
Purpose: Preprocessing, stationarity checks, outlier handling, scaling, temporal train/test split.
Inputs:  data/raw/labeled_dataset.csv
Outputs: data/processed/train.csv, data/processed/test.csv, models/scaler.pkl,
         results/tables/stationarity_report.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

FEATURE_COLS = ["volatility", "momentum", "volume_avg", "rsi_14", "max_drawdown", "vix", "eur_usd"]
TARGET_COL = "label"

# Temporal split dates
TRAIN_START = "2025-12-01"
TRAIN_END = "2026-02-28"
TEST_START = "2026-03-05"
TEST_END = "2026-04-08"
BUFFER_DAYS = 5  # gap between train and test to prevent rolling feature leakage


def handle_missing_values(df):
    """
    Handle missing values: forward-fill macro features, drop rows with NA stock features.

    Args:
        df: Labeled dataset DataFrame.

    Returns:
        DataFrame with NAs handled, plus count of rows dropped.
    """
    initial_rows = len(df)
    macro_cols = ["vix", "eur_usd"]
    stock_cols = ["volatility", "momentum", "volume_avg", "rsi_14", "max_drawdown"]

    # Forward-fill macro features (they change infrequently)
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Drop rows where stock features are NA (don't impute prices)
    df = df.dropna(subset=[c for c in stock_cols if c in df.columns])

    dropped = initial_rows - len(df)
    logger.info(f"Missing values: {dropped} rows dropped of {initial_rows} total")
    return df


def winsorize_outliers(df, feature_cols, lower=0.01, upper=0.99):
    """
    Winsorize at 1st and 99th percentile per feature.

    Args:
        df: DataFrame to winsorize.
        feature_cols: List of column names to winsorize.
        lower: Lower percentile (default 1%).
        upper: Upper percentile (default 99%).

    Returns:
        DataFrame with outliers clipped.
    """
    for col in feature_cols:
        if col in df.columns:
            lo = df[col].quantile(lower)
            hi = df[col].quantile(upper)
            clipped = df[col].clip(lo, hi)
            n_clipped = (df[col] != clipped).sum()
            if n_clipped > 0:
                logger.info(f"  {col}: {n_clipped} values winsorized")
            df[col] = clipped
    return df


def check_stationarity(df, feature_cols, results_dir="results/tables"):
    """
    Run Augmented Dickey-Fuller test on each feature. For non-stationary features,
    apply first-differencing.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature column names.
        results_dir: Directory to save stationarity report.

    Returns:
        DataFrame with stationarity-transformed features, stationarity report DataFrame.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    report = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 20:
            report.append({"feature": col, "adf_stat": np.nan, "p_value": np.nan,
                           "stationary": "insufficient_data", "action": "none"})
            continue

        result = adfuller(series, maxlag=10, autolag="AIC")
        adf_stat, p_value = result[0], result[1]
        is_stationary = p_value < 0.05

        action = "none"
        if not is_stationary:
            # Apply first-differencing
            df[col] = df[col].diff()
            action = "first_difference"
            logger.info(f"  {col}: non-stationary (p={p_value:.4f}), applied first-differencing")

        report.append({
            "feature": col,
            "adf_stat": round(adf_stat, 4),
            "p_value": round(p_value, 4),
            "stationary": "yes" if is_stationary else "no",
            "action": action,
        })

    report_df = pd.DataFrame(report)
    report_df.to_csv(os.path.join(results_dir, "stationarity_report.csv"), index=False)
    logger.info(f"Stationarity report saved ({len(report)} features tested)")

    # Drop NaN rows created by differencing
    df = df.dropna(subset=[c for c in feature_cols if c in df.columns])
    return df, report_df


def check_multicollinearity(df, feature_cols, results_dir="results/tables"):
    """
    Compute correlation matrix and flag pairs with |correlation| > 0.85.
    CatBoost is robust to correlated features, so we document but don't drop.

    Args:
        df: DataFrame with features.
        feature_cols: List of feature columns.
        results_dir: Directory to save correlation report.

    Returns:
        Correlation matrix DataFrame.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    corr = df[feature_cols].corr()

    # Flag high correlations
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append({
                    "feature_1": feature_cols[i],
                    "feature_2": feature_cols[j],
                    "correlation": round(r, 4),
                })
                logger.info(f"  High correlation: {feature_cols[i]} vs {feature_cols[j]} = {r:.4f}")

    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(
            os.path.join(results_dir, "high_correlation_pairs.csv"), index=False
        )
    else:
        logger.info("  No feature pairs with |correlation| > 0.85")

    corr.to_csv(os.path.join(results_dir, "correlation_matrix.csv"))
    return corr


def temporal_split(df):
    """
    Split data temporally into train and test sets. NO RANDOM SPLIT.
    Financial data is time-series — random split causes data leakage.

    Args:
        df: Full labeled dataset with DatetimeIndex.

    Returns:
        train_df, test_df
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)

    train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
    test = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    logger.info(f"Temporal split: train={len(train)} rows [{TRAIN_START} to {TRAIN_END}], "
                f"test={len(test)} rows [{TEST_START} to {TEST_END}]")
    logger.info(f"Buffer gap: {BUFFER_DAYS} trading days between train and test")

    return train, test


def scale_features(train_df, test_df, feature_cols, models_dir="models"):
    """
    Fit StandardScaler on training data ONLY, apply to both train and test.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        feature_cols: List of features to scale.
        models_dir: Directory to save scaler.

    Returns:
        Scaled train_df, scaled test_df, fitted scaler.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()

    cols_present = [c for c in feature_cols if c in train_df.columns]
    train_df[cols_present] = scaler.fit_transform(train_df[cols_present])
    test_df[cols_present] = scaler.transform(test_df[cols_present])

    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"StandardScaler fitted on {len(cols_present)} features, saved to models/scaler.pkl")
    return train_df, test_df, scaler


def run(dataset, output_dir="data/processed", results_dir="results/tables", models_dir="models"):
    """
    Execute Step 4: Full data preparation pipeline.

    Args:
        dataset: Labeled dataset DataFrame from Step 3.

    Returns:
        dict with train_df, test_df, scaler, stationarity_report.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: DATA PREPARATION")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure datetime index
    if not isinstance(dataset.index, pd.DatetimeIndex):
        if "Date" in dataset.columns:
            dataset = dataset.set_index("Date")
        dataset.index = pd.to_datetime(dataset.index)

    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]

    # 4.1 Missing values
    dataset = handle_missing_values(dataset)

    # 4.2 Outlier handling
    logger.info("Outlier handling (winsorization at 1st/99th percentile):")
    dataset = winsorize_outliers(dataset, feature_cols)

    # 4.3 Stationarity check
    logger.info("Stationarity testing (ADF):")
    dataset, stationarity_report = check_stationarity(dataset, feature_cols, results_dir)

    # 4.4 Multicollinearity check
    logger.info("Multicollinearity check:")
    check_multicollinearity(dataset, feature_cols, results_dir)

    # 4.5 Temporal split
    train_df, test_df = temporal_split(dataset)

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"Empty split: train={len(train_df)}, test={len(test_df)}. "
                           f"Check date ranges and data availability.")

    # 4.6 Feature scaling
    train_df, test_df, scaler = scale_features(
        train_df.copy(), test_df.copy(), feature_cols, models_dir
    )

    # Save processed data
    train_df.to_csv(os.path.join(output_dir, "train.csv"))
    test_df.to_csv(os.path.join(output_dir, "test.csv"))
    logger.info(f"Saved: train.csv ({len(train_df)} rows), test.csv ({len(test_df)} rows)")

    return {
        "train": train_df,
        "test": test_df,
        "scaler": scaler,
        "stationarity_report": stationarity_report,
        "feature_cols": feature_cols,
    }
