"""
Data loader for real German financial data.
Pulls DAX company data via yfinance and constructs investment decision dataset.
"""

import pandas as pd
import numpy as np


# Top DAX 40 companies
DAX_TICKERS = [
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BAS.DE",
    "MBG.DE", "BMW.DE", "MUV2.DE", "AIR.DE", "IFX.DE",
    "ADS.DE", "DB1.DE", "RWE.DE", "HEN3.DE", "VOW3.DE",
    "SY1.DE", "BEI.DE", "EOAN.DE", "FRE.DE", "MTX.DE",
]


def download_dax_data(tickers=None, period="5y", interval="1mo"):
    """
    Download DAX company stock data from Yahoo Finance.
    Returns dict of {ticker: DataFrame}.
    """
    import time
    import yfinance as yf

    tickers = tickers or DAX_TICKERS

    # Try batch download first (single request)
    try:
        print("  Attempting batch download...")
        batch = yf.download(tickers, period=period, interval=interval,
                            progress=False, group_by="ticker")
        all_data = {}
        for ticker in tickers:
            try:
                df = batch[ticker].dropna(how="all")
                if len(df) > 0:
                    all_data[ticker] = df
                    print(f"  {ticker}: {len(df)} rows")
            except (KeyError, Exception):
                print(f"  {ticker}: not in batch result")
        if all_data:
            return all_data
    except Exception as e:
        print(f"  Batch download failed: {e}")

    # Fallback: individual downloads with delays
    print("  Falling back to individual downloads...")
    all_data = {}
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(2)  # avoid rate limiting
        try:
            data = yf.download(ticker, period=period, interval=interval,
                               progress=False)
            if len(data) > 0:
                all_data[ticker] = data
                print(f"  {ticker}: {len(data)} rows")
            else:
                print(f"  {ticker}: FAILED (no data)")
        except Exception as e:
            print(f"  {ticker}: ERROR ({e})")
    return all_data


def compute_features(price_data, ticker_info=None):
    """
    Compute investment features from price data.

    Returns DataFrame with:
    - volatility: rolling 12-month std of returns
    - momentum: 6-month return
    - volume_avg: average trading volume
    - return_1y: 12-month return
    - max_drawdown: max drawdown over 12 months
    """
    df = price_data.copy()

    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(12).std()
    df["momentum"] = df["Close"].pct_change(6)
    df["volume_avg"] = df["Volume"].rolling(6).mean()
    df["return_1y"] = df["Close"].pct_change(12)

    # Max drawdown (12-month rolling)
    rolling_max = df["Close"].rolling(12).max()
    df["max_drawdown"] = (df["Close"] - rolling_max) / rolling_max

    return df.dropna()


def download_macro_data():
    """
    Download macroeconomic indicators from ECB Data API and yfinance.
    All sources are free, no API key required.

    Returns monthly DataFrame with:
    - ecb_rate: ECB main refinancing rate (%)
    - eur_usd: EUR/USD exchange rate
    - de_inflation: German HICP year-over-year (%)
    - vix: CBOE VIX index (VSTOXX proxy)
    """
    import yfinance as yf

    macro = pd.DataFrame()

    # ECB Main Refinancing Rate
    try:
        url = ("https://data-api.ecb.europa.eu/service/data/"
               "FM/B.U2.EUR.4F.KR.MRR_FR.LEV?format=csvdata&startPeriod=2015-01-01")
        df = pd.read_csv(url)
        ecb = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        ecb.columns = ["date", "ecb_rate"]
        ecb["date"] = pd.to_datetime(ecb["date"])
        # Forward-fill rate decisions to monthly frequency
        ecb = ecb.set_index("date").resample("MS").last().ffill()
        macro = ecb
        print("  ECB rate: OK")
    except Exception as e:
        print(f"  ECB rate: FAILED ({e})")

    # EUR/USD Exchange Rate (monthly)
    try:
        url = ("https://data-api.ecb.europa.eu/service/data/"
               "EXR/M.USD.EUR.SP00.A?format=csvdata&startPeriod=2015-01-01")
        df = pd.read_csv(url)
        fx = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        fx.columns = ["date", "eur_usd"]
        fx["date"] = pd.to_datetime(fx["date"])
        fx = fx.set_index("date")
        macro = macro.join(fx, how="outer") if len(macro) > 0 else fx
        print("  EUR/USD: OK")
    except Exception as e:
        print(f"  EUR/USD: FAILED ({e})")

    # German HICP Inflation (monthly, YoY %)
    try:
        url = ("https://data-api.ecb.europa.eu/service/data/"
               "ICP/M.DE.N.000000.4.ANR?format=csvdata&startPeriod=2015-01-01")
        df = pd.read_csv(url)
        cpi = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        cpi.columns = ["date", "de_inflation"]
        cpi["date"] = pd.to_datetime(cpi["date"])
        cpi = cpi.set_index("date")
        macro = macro.join(cpi, how="outer") if len(macro) > 0 else cpi
        print("  German inflation: OK")
    except Exception as e:
        print(f"  German inflation: FAILED ({e})")

    # VIX (VSTOXX proxy) — monthly from yfinance
    try:
        vix = yf.download("^VIX", period="10y", interval="1mo", progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix_monthly = vix[["Close"]].rename(columns={"Close": "vix"})
        macro = macro.join(vix_monthly, how="outer") if len(macro) > 0 else vix_monthly
        print("  VIX: OK")
    except Exception as e:
        print(f"  VIX: FAILED ({e})")

    macro = macro.ffill().bfill()
    return macro


def build_investment_dataset(all_data, sharpe_threshold=0.5):
    """
    Build a combined investment decision dataset from multiple tickers,
    enriched with macroeconomic indicators.

    Investment decision (binary):
    - 1 (APPROVE): risk-adjusted return > threshold
    - 0 (REJECT): otherwise

    Stock features: volatility, momentum, volume_avg, return_1y, max_drawdown
    Macro features: ecb_rate, eur_usd, de_inflation, vix
    """
    stock_cols = ["volatility", "momentum", "volume_avg", "return_1y", "max_drawdown"]
    rows = []

    # Download macro data
    print("  Downloading macro indicators...")
    macro = download_macro_data()
    has_macro = len(macro) > 0
    macro_cols = [c for c in macro.columns if c in ["ecb_rate", "eur_usd", "de_inflation", "vix"]]

    for ticker, price_data in all_data.items():
        features = compute_features(price_data)
        if len(features) < 5:
            continue

        for idx, row in features.iterrows():
            if row["volatility"] > 0:
                risk_adj_return = row["return_1y"] / row["volatility"]
            else:
                risk_adj_return = 0

            record = {col: row[col] for col in stock_cols if col in row.index}
            record["ticker"] = ticker
            record["date"] = idx
            record["investment_decision"] = 1 if risk_adj_return > sharpe_threshold else 0

            # Merge macro features by nearest month
            if has_macro:
                month_start = idx.replace(day=1)
                nearest_idx = macro.index.get_indexer([month_start], method="nearest")[0]
                if nearest_idx >= 0:
                    for col in macro_cols:
                        record[col] = macro.iloc[nearest_idx][col]

            rows.append(record)

    df = pd.DataFrame(rows)
    return df


def load_german_credit_uci():
    """
    Load the German Credit dataset from sklearn (simplified version).
    Useful for validating the pipeline before using real investment data.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame

    # Select numeric columns only for causal discovery
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = "class"

    # Binarize target: 'good' = 1, 'bad' = 0
    df["target"] = (df[target] == "good").astype(int)

    return df[numeric_cols + ["target"]]
