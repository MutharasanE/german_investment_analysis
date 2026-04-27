"""
Module: Feature Engineering
Purpose: Compute stock-level and macro features per ticker per date from real price data.
Inputs:  data/raw/prices_{ticker}.csv, data/raw/macro_data.csv
Outputs: data/raw/features_{ticker}.csv per ticker
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_stock_features(price_df):
    """
    Compute 5 stock-level features from daily OHLCV data.

    Args:
        price_df: DataFrame with columns Open, High, Low, Close, Volume (date-indexed).

    Returns:
        DataFrame with columns: volatility, momentum, volume_avg, rsi_14, max_drawdown.
    """
    close = price_df["Close"].squeeze() if hasattr(price_df["Close"], "squeeze") else price_df["Close"]
    volume = price_df["Volume"].squeeze() if hasattr(price_df["Volume"], "squeeze") else price_df["Volume"]

    features = pd.DataFrame(index=price_df.index)

    # 1. Volatility: 20-day rolling std of log returns (annualized)
    log_returns = np.log(close / close.shift(1))
    features["volatility"] = log_returns.rolling(20).std() * np.sqrt(252)

    # 2. Momentum: 21-day return (1-month)
    features["momentum"] = close.pct_change(21)

    # 3. Volume average: 20-day average volume (log-scaled)
    vol_clean = volume.replace(0, np.nan)
    features["volume_avg"] = np.log1p(vol_clean.rolling(20).mean())

    # 4. RSI 14: Relative Strength Index (14-day)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # 5. Max drawdown: 20-day trailing max peak-to-trough drop
    rolling_max = close.rolling(20).max()
    drawdown = (close - rolling_max) / rolling_max
    features["max_drawdown"] = drawdown.rolling(20).min()

    return features


def merge_macro_features(stock_features, macro_df):
    """
    Align macro features to stock dates using forward-fill.

    Args:
        stock_features: DataFrame with stock features (date-indexed).
        macro_df: DataFrame with vix, eur_usd columns (date-indexed).

    Returns:
        DataFrame with stock + macro features merged on date.
    """
    merged = stock_features.join(macro_df, how="left")
    merged = merged.ffill().bfill()
    return merged


def run(stock_data, macro_data, output_dir="data/raw"):
    """
    Execute Step 2: Compute features for all tickers.

    Args:
        stock_data: dict {ticker: price_df} from Step 1.
        macro_data: DataFrame with macro features from Step 1.

    Returns:
        dict {ticker: features_df} with all 7 features per ticker.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_features = {}

    for ticker, price_df in stock_data.items():
        stock_feats = compute_stock_features(price_df)
        features = merge_macro_features(stock_feats, macro_data)

        # Drop rows with NaN from rolling window warmup
        features = features.dropna()

        if len(features) < 20:
            logger.warning(f"{ticker}: only {len(features)} rows after feature computation, skipping")
            continue

        features.to_csv(os.path.join(output_dir, f"features_{ticker.replace('.', '_')}.csv"))
        all_features[ticker] = features
        logger.info(f"{ticker}: {len(features)} feature rows")

    logger.info(f"Feature engineering complete: {len(all_features)} tickers")
    return all_features
