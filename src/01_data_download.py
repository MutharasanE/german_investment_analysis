"""
Module: Data Download
Purpose: Download DAX 40 + S&P 500 daily price data and macro indicators from Yahoo Finance.
Inputs:  None (downloads from Yahoo Finance API)
Outputs: data/raw/prices_{ticker}.csv per ticker, data/raw/macro_data.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path

logger = logging.getLogger(__name__)

# DAX 40 tickers as of 2025
DAX40_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE",
    "BMW.DE", "BNR.DE", "CON.DE", "1COV.DE", "DB1.DE", "DBK.DE",
    "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HEI.DE", "HEN3.DE",
    "IFX.DE", "LIN.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE",
    "P911.DE", "PAH3.DE", "QGEN.DE", "RWE.DE", "SAP.DE", "SHL.DE",
    "SIE.DE", "SRT.DE", "SY1.DE", "VNA.DE", "VOW3.DE", "ZAL.DE",
    "RHM.DE", "ENR.DE", "DTG.DE", "HNR1.DE",
]

# S&P 500 — top 50 by market cap (representative subset to keep download time manageable)
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "PG", "MA", "JNJ",
    "COST", "HD", "MRK", "ABBV", "WMT", "NFLX", "CRM", "BAC",
    "CVX", "KO", "AMD", "PEP", "TMO", "LIN", "ORCL", "ACN", "MCD",
    "CSCO", "ADBE", "ABT", "WFC", "IBM", "GE", "PM", "NOW", "TXN",
    "QCOM", "MS", "CAT", "INTU", "GS", "DHR", "AMGN",
]

ALL_TICKERS = DAX40_TICKERS + SP500_TICKERS

# Download extra months for feature lookback (rolling windows need warmup)
DATA_START = "2025-07-01"
DATA_END = "2026-04-14"

# Effective analysis window (after feature warmup)
TRAIN_START = "2025-12-01"
TRAIN_END = "2026-02-28"
TEST_START = "2026-03-05"  # 5-day buffer to prevent leakage from rolling features
TEST_END = "2026-04-08"    # leave room for 5-day forward return labels


def download_stock_data(output_dir="data/raw"):
    """
    Download daily OHLCV data for all DAX 40 + S&P 500 tickers.

    Args:
        output_dir: Directory to save individual ticker CSVs.

    Returns:
        dict: {ticker: DataFrame} for successfully downloaded tickers.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stock_data = {}
    success, fail = 0, 0
    failed_tickers = []

    for i, ticker in enumerate(ALL_TICKERS):
        # Rate-limit: pause between requests to avoid Yahoo throttling
        if i > 0 and i % 10 == 0:
            logger.info(f"  [{i}/{len(ALL_TICKERS)}] pausing 3s to avoid rate limit...")
            time.sleep(3)
        elif i > 0:
            time.sleep(0.5)

        try:
            df = yf.download(ticker, start=DATA_START, end=DATA_END,
                             interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 20:
                logger.warning(f"{ticker}: insufficient data ({len(df)} rows), skipping")
                failed_tickers.append(ticker)
                fail += 1
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_csv(os.path.join(output_dir, f"prices_{ticker.replace('.', '_').replace('-', '_')}.csv"))
            stock_data[ticker] = df
            success += 1
            logger.info(f"{ticker}: {len(df)} rows downloaded")
        except Exception as e:
            logger.warning(f"{ticker}: download failed ({e})")
            failed_tickers.append(ticker)
            fail += 1

    # Retry failed tickers once (often just rate-limit timeouts)
    if failed_tickers:
        logger.info(f"Retrying {len(failed_tickers)} failed tickers after 10s cooldown...")
        time.sleep(10)
        for i, ticker in enumerate(failed_tickers):
            if i > 0:
                time.sleep(2)
            try:
                df = yf.download(ticker, start=DATA_START, end=DATA_END,
                                 interval="1d", progress=False, auto_adjust=False)
                if df.empty or len(df) < 20:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.to_csv(os.path.join(output_dir, f"prices_{ticker.replace('.', '_').replace('-', '_')}.csv"))
                stock_data[ticker] = df
                success += 1
                fail -= 1
                logger.info(f"{ticker}: {len(df)} rows downloaded (retry)")
            except Exception:
                pass

    logger.info(f"Stock download complete: {success}/{len(ALL_TICKERS)} succeeded, {fail} failed")
    logger.info(f"  DAX 40: {sum(1 for t in DAX40_TICKERS if t in stock_data)}/{len(DAX40_TICKERS)}")
    logger.info(f"  S&P 500: {sum(1 for t in SP500_TICKERS if t in stock_data)}/{len(SP500_TICKERS)}")
    return stock_data


def download_macro_data(output_dir="data/raw"):
    """
    Download macro indicators: VIX and EUR/USD from Yahoo Finance.

    Args:
        output_dir: Directory to save macro_data.csv.

    Returns:
        DataFrame with columns: vix, eur_usd (daily, date-indexed).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    macro = pd.DataFrame()

    # VIX - market fear gauge
    try:
        vix = yf.download("^VIX", start=DATA_START, end=DATA_END,
                           interval="1d", progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        macro["vix"] = vix["Close"]
        logger.info(f"VIX: {len(vix)} rows")
    except Exception as e:
        logger.warning(f"VIX download failed: {e}")

    # EUR/USD exchange rate
    try:
        eurusd = yf.download("EURUSD=X", start=DATA_START, end=DATA_END,
                              interval="1d", progress=False, auto_adjust=False)
        if isinstance(eurusd.columns, pd.MultiIndex):
            eurusd.columns = eurusd.columns.get_level_values(0)
        macro["eur_usd"] = eurusd["Close"]
        logger.info(f"EUR/USD: {len(eurusd)} rows")
    except Exception as e:
        logger.warning(f"EUR/USD download failed: {e}")

    macro = macro.ffill().bfill()
    macro.to_csv(os.path.join(output_dir, "macro_data.csv"))
    logger.info(f"Macro data saved: {len(macro)} rows, columns={list(macro.columns)}")
    return macro


def run(output_dir="data/raw"):
    """Execute Step 1: Download all data."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA DOWNLOAD (DAX 40 + S&P 500 Top 50)")
    logger.info("=" * 60)
    stock_data = download_stock_data(output_dir)
    macro_data = download_macro_data(output_dir)
    return {"stock_data": stock_data, "macro_data": macro_data}
