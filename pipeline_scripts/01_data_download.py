"""
Module: 01_data_download
Purpose: Download real market and macro data for DAX and S&P 500 constituents.
Inputs:  Yahoo Finance API (yfinance), FRED series via pandas_datareader.
Outputs: data/raw/prices_<ticker>.csv, data/raw/macro_data.csv, data/raw/ticker_metadata.csv,
         results/tables/download_log.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pandas_datareader import data as web


DAX_TICKERS = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE",
    "CON.DE", "1COV.DE", "DB1.DE", "DBK.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE",
    "HEI.DE", "HEN3.DE", "IFX.DE", "LIN.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE",
    "P911.DE", "PAH3.DE", "QGEN.DE", "RWE.DE", "SAP.DE", "SHL.DE", "SIE.DE", "SRT.DE",
    "SY1.DE", "VNA.DE", "VOW3.DE", "ZAL.DE",
]

YAHOO_MACRO_SYMBOLS = {
    "vix": "^VIX",
    "eur_usd": "EURUSD=X",
    "us_10y_yield": "^TNX",
    "de_10y_yield": "^IRDE10Y",
    "sp500_index": "^GSPC",
    "dax_index": "^GDAXI",
}

FRED_SERIES = {
    "ecb_rate": "ECBDFR",
    "de_cpi": "DEUCPIALLMINMEI",
    "us_cpi": "CPIAUCSL",
}


@dataclass
class DownloadResult:
    ticker: str
    market: str
    status: str
    rows: int
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: Download real data")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument(
        "--max-sp500",
        type=int,
        default=0,
        help="Optional cap for S&P 500 ticker count (0 = all constituents)",
    )
    return parser.parse_args()


def safe_ticker_name(ticker: str) -> str:
    return (
        ticker.replace("^", "")
        .replace("=", "")
        .replace("/", "_")
        .replace(".", "_")
    )


def get_sp500_tickers() -> list[str]:
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # Primary path: Wikipedia HTML fetched with browser-like headers.
    try:
        resp = requests.get(wiki_url, headers=headers, timeout=30)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        symbols = table["Symbol"].astype(str).tolist()
        return [s.replace(".", "-") for s in symbols]
    except Exception:
        pass

    # Fallback path: public CSV mirror of S&P 500 constituents.
    try:
        csv_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        table = pd.read_csv(csv_url)
        symbols = table["Symbol"].astype(str).tolist()
        return [s.replace(".", "-") for s in symbols]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load S&P 500 constituents from both Wikipedia and fallback source."
        ) from exc


def download_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("empty dataframe returned")
    df = df.reset_index()
    if "Date" not in df.columns:
        raise ValueError("Date column missing")
    df["Ticker"] = ticker
    return df


def fetch_ticker_metadata(ticker: str, market: str) -> dict[str, object]:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    sector = info.get("sector", "Unknown")
    pe_ratio = info.get("trailingPE")
    if pe_ratio is None:
        pe_ratio = np.nan
    return {"ticker": ticker, "market": market, "sector": sector, "pe_ratio": pe_ratio}


def _download_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    s = web.DataReader(series_id, "fred", start, end)[series_id]
    s.index = pd.to_datetime(s.index)
    return s


def build_macro_data(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D")
    macro = pd.DataFrame(index=idx)

    for col, symbol in YAHOO_MACRO_SYMBOLS.items():
        try:
            series = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
            if series.empty:
                macro[col] = np.nan
                continue
            macro[col] = series["Close"].reindex(idx)
        except Exception:
            macro[col] = np.nan

    for col, series_id in FRED_SERIES.items():
        try:
            macro[col] = _download_fred_series(series_id, start, end).reindex(idx)
        except Exception:
            macro[col] = np.nan

    # Daily aligned, forward-fill infrequent macro series.
    macro = macro.sort_index().ffill().bfill()
    macro.index.name = "Date"
    return macro.reset_index()


def save_prices(
    tickers: Iterable[str],
    market: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> tuple[list[DownloadResult], list[dict[str, object]]]:
    logs: list[DownloadResult] = []
    metadata_rows: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            df = download_price_history(ticker, start, end)
            out = raw_dir / f"prices_{safe_ticker_name(ticker)}.csv"
            df.to_csv(out, index=False)
            logs.append(DownloadResult(ticker=ticker, market=market, status="ok", rows=len(df), error=""))
        except Exception as exc:
            logs.append(
                DownloadResult(
                    ticker=ticker,
                    market=market,
                    status="failed",
                    rows=0,
                    error=str(exc),
                )
            )

        metadata_rows.append(fetch_ticker_metadata(ticker, market))

    return logs, metadata_rows


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    results_tables = root / "results" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_tables.mkdir(parents=True, exist_ok=True)

    sp500_tickers = get_sp500_tickers()
    if args.max_sp500 > 0:
        sp500_tickers = sp500_tickers[: args.max_sp500]

    dax_logs, dax_meta = save_prices(DAX_TICKERS, "DAX", args.start, args.end, raw_dir)
    sp_logs, sp_meta = save_prices(sp500_tickers, "SP500", args.start, args.end, raw_dir)

    macro = build_macro_data(args.start, args.end)
    macro.to_csv(raw_dir / "macro_data.csv", index=False)

    logs_df = pd.DataFrame([r.__dict__ for r in dax_logs + sp_logs])
    logs_df.to_csv(results_tables / "download_log.csv", index=False)

    metadata_df = pd.DataFrame(dax_meta + sp_meta).drop_duplicates(subset=["ticker"], keep="last")
    metadata_df.to_csv(raw_dir / "ticker_metadata.csv", index=False)

    success = int((logs_df["status"] == "ok").sum())
    failed = int((logs_df["status"] == "failed").sum())
    print(f"Download completed. Success={success}, Failed={failed}")
    print(f"Saved macro data to: {raw_dir / 'macro_data.csv'}")
    print(f"Saved logs to: {results_tables / 'download_log.csv'}")


if __name__ == "__main__":
    main()
