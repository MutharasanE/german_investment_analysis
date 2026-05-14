"""
Module: 02_feature_engineering
Purpose: Compute stock-level and macro-aligned features from real downloaded data.
Inputs:  data/raw/prices_<ticker>.csv, data/raw/macro_data.csv, data/raw/ticker_metadata.csv
Outputs: data/processed/features_engineered.parquet, data/processed/features_engineered.csv,
         results/tables/stationarity_report.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


FEATURE_COLS = [
    "volatility",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "return_1y",
    "max_drawdown",
    "volume_avg",
    "rsi_14",
    "macd_signal",
    "beta_market",
    "pe_ratio",
    "ecb_rate",
    "us_10y_yield",
    "vix",
    "eur_usd",
    "de_inflation",
    "us_inflation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: Feature engineering")
    parser.add_argument("--adf-p-threshold", type=float, default=0.05)
    return parser.parse_args()


def infer_ticker(price_df: pd.DataFrame, file_path: Path) -> str:
    """Infer ticker from data first, then fallback to filename."""
    if "Ticker" in price_df.columns:
        vals = (
            price_df["Ticker"]
            .astype(str)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "None": np.nan, "Ticker": np.nan})
            .dropna()
        )
        if not vals.empty:
            return str(vals.iloc[0])

    stem = file_path.stem
    if stem.startswith("prices_"):
        tok = stem[len("prices_") :]
        # files for German tickers are saved as XXX_DE -> XXX.DE
        if tok.endswith("_DE"):
            tok = tok[:-3] + ".DE"
        return tok
    return stem


def clean_price_dataframe(price_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Yahoo CSV structure and coerce numeric columns safely."""
    out = price_df.copy()
    if "Date" not in out.columns:
        return out.iloc[0:0]

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).copy()

    numeric_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = [c for c in ["Close", "Volume"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)

    return out.sort_values("Date")


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def compute_macd_signal(close: pd.Series) -> pd.Series:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    return macd.ewm(span=9, adjust=False).mean()


def compute_stock_features(price_df: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
    df = price_df.copy().sort_values("Date")
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    close = pd.to_numeric(df[close_col], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    valid_mask = close.notna() & volume.notna()
    df = df.loc[valid_mask].copy()
    close = close.loc[valid_mask]
    volume = volume.loc[valid_mask]

    if df.empty:
        return df

    # Align all rolling calculations to Date index for robust joins.
    df = df.set_index("Date", drop=False)
    close = pd.Series(close.to_numpy(), index=df.index)
    volume = pd.Series(volume.to_numpy(), index=df.index)

    log_ret = np.log(close.replace(0, np.nan)).diff()
    simple_ret = close.pct_change()

    df["close"] = close
    df["volatility"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["momentum_1m"] = close.pct_change(21)
    df["momentum_3m"] = close.pct_change(63)
    df["momentum_6m"] = close.pct_change(126)
    df["return_1y"] = close.pct_change(252)

    rolling_max = close.rolling(252).max()
    drawdown = close / rolling_max - 1
    df["max_drawdown"] = drawdown.rolling(252).min()

    df["volume_avg"] = np.log1p(volume.rolling(20).mean())
    df["rsi_14"] = compute_rsi(close, window=14)
    df["macd_signal"] = compute_macd_signal(close)

    market_ret_aligned = market_returns.reindex(df.index)
    merged_returns = pd.concat([simple_ret.rename("asset"), market_ret_aligned.rename("market")], axis=1)
    cov = merged_returns["asset"].rolling(60).cov(merged_returns["market"])
    var = merged_returns["market"].rolling(60).var()
    df["beta_market"] = cov / (var + 1e-12)

    return df.reset_index(drop=True)


def adf_stationarity_report(data: pd.DataFrame, feature_cols: list[str], p_threshold: float) -> pd.DataFrame:
    rows = []
    max_adf_points = 10_000

    for col in feature_cols:
        series = data[col].dropna()
        if len(series) < 100:
            rows.append(
                {
                    "feature": col,
                    "adf_pvalue": np.nan,
                    "stationary": False,
                    "transformation_applied": "insufficient_data",
                }
            )
            continue

        # Keep ADF runtime bounded on very large real-world datasets.
        if len(series) > max_adf_points:
            step = max(1, len(series) // max_adf_points)
            series = series.iloc[::step]

        try:
            # Keep lag search bounded for stability on very large series.
            pval = adfuller(series, maxlag=5, autolag=None)[1]
        except Exception:
            pval = np.nan

        stationary = bool(np.isfinite(pval) and pval <= p_threshold)
        rows.append(
            {
                "feature": col,
                "adf_pvalue": pval,
                "stationary": stationary,
                "transformation_applied": "none" if stationary else "first_difference",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    macro = pd.read_csv(raw_dir / "macro_data.csv")
    macro["Date"] = pd.to_datetime(macro["Date"])
    macro = macro.sort_values("Date")

    if "de_cpi" in macro.columns:
        macro["de_inflation"] = macro["de_cpi"].pct_change(252) * 100
    else:
        macro["de_inflation"] = np.nan

    if "us_cpi" in macro.columns:
        macro["us_inflation"] = macro["us_cpi"].pct_change(252) * 100
    else:
        macro["us_inflation"] = np.nan

    metadata = pd.read_csv(raw_dir / "ticker_metadata.csv")
    price_files = sorted(raw_dir.glob("prices_*.csv"))

    all_frames: list[pd.DataFrame] = []

    total_files = len(price_files)
    for i, file_path in enumerate(price_files, start=1):
        price_df = pd.read_csv(file_path)
        if "Date" not in price_df.columns:
            continue

        ticker = infer_ticker(price_df, file_path)
        price_df = clean_price_dataframe(price_df)
        if price_df.empty:
            continue

        if ticker.endswith(".DE"):
            market_returns = macro.set_index("Date")["dax_index"].pct_change()
            market = "DAX"
        else:
            market_returns = macro.set_index("Date")["sp500_index"].pct_change()
            market = "SP500"

        feat = compute_stock_features(price_df, market_returns)
        if feat.empty:
            continue

        feat = feat.merge(macro, on="Date", how="left", suffixes=("", "_macro"))
        feat["market"] = market

        row_meta = metadata[metadata["ticker"] == ticker]
        feat["sector"] = row_meta["sector"].iloc[0] if not row_meta.empty else "Unknown"
        feat["pe_ratio"] = row_meta["pe_ratio"].iloc[0] if not row_meta.empty else np.nan

        feat["market_close"] = feat["dax_index"] if market == "DAX" else feat["sp500_index"]

        keep_cols = [
            "Date",
            "Ticker",
            "close",
            "sector",
            "market",
            "market_close",
            "volatility",
            "momentum_1m",
            "momentum_3m",
            "momentum_6m",
            "return_1y",
            "max_drawdown",
            "volume_avg",
            "rsi_14",
            "macd_signal",
            "beta_market",
            "pe_ratio",
            "ecb_rate",
            "us_10y_yield",
            "vix",
            "eur_usd",
            "de_inflation",
            "us_inflation",
        ]
        keep_cols = [c for c in keep_cols if c in feat.columns]
        feat = feat[keep_cols].copy()

        all_frames.append(feat)

        if i % 50 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} price files...")

    if not all_frames:
        raise ValueError("No price files found. Run 01_data_download.py first.")

    data = pd.concat(all_frames, ignore_index=True)
    data = data.sort_values(["Date"], kind="mergesort")

    report = adf_stationarity_report(data, FEATURE_COLS, args.adf_p_threshold)

    non_stationary = report.loc[report["transformation_applied"] == "first_difference", "feature"].tolist()
    for col in non_stationary:
        data[col] = data.groupby("Ticker", observed=True)[col].diff()

    report.to_csv(tables_dir / "stationarity_report.csv", index=False)
    data.to_parquet(processed_dir / "features_engineered.parquet", index=False)

    # Write a smaller CSV preview for manual inspection without huge IO cost.
    data.head(50_000).to_csv(processed_dir / "features_engineered.csv", index=False)

    print(f"Saved engineered features (full): {processed_dir / 'features_engineered.parquet'}")
    print(f"Saved engineered features (preview): {processed_dir / 'features_engineered.csv'}")
    print(f"Saved stationarity report: {tables_dir / 'stationarity_report.csv'}")


if __name__ == "__main__":
    main()
