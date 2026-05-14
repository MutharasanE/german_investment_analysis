"""
Module: 04_data_preparation
Purpose: Prepare labeled data with temporal split, scaling, and robustness metadata.
Inputs:  data/processed/labeled_dataset.csv
Outputs: data/processed/train.csv, validation.csv, test.csv, models/scaler.pkl,
         results/tables/collinearity_report.csv
Reference: Takahashi et al. (2024) arXiv:2402.02678
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

NON_ACTIONABLE = {"vix", "ecb_rate", "eur_usd", "us_10y_yield", "de_inflation", "us_inflation"}


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["Ticker", "Date"])
    out["market_ma_200"] = out.groupby("market", observed=True)["market_close"].transform(
        lambda s: s.rolling(200, min_periods=200).mean()
    )

    bull = (out["vix"] < 20) & (out["market_close"] > out["market_ma_200"])
    crisis = out["vix"] > 30
    out["regime"] = np.select([bull, crisis], ["Bull", "Crisis"], default="Neutral")
    return out


def winsorize(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    out = df.copy()
    summary: list[dict[str, float]] = []

    for col in cols:
        q1 = out[col].quantile(0.01)
        q99 = out[col].quantile(0.99)
        before = out[col].copy()
        out[col] = out[col].clip(lower=q1, upper=q99)
        changed = float((before != out[col]).mean())
        summary.append({"feature": col, "p01": q1, "p99": q99, "changed_fraction": changed})

    return out, summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed_dir = root / "data" / "processed"
    tables_dir = root / "results" / "tables"
    models_dir = root / "models"

    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labeled_parquet = processed_dir / "labeled_dataset.parquet"
    if labeled_parquet.exists():
        df = pd.read_parquet(labeled_parquet)
    else:
        df = pd.read_csv(processed_dir / "labeled_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    before_rows = len(df)

    # Missing value handling policy: macro forward fill; stock features strict drop.
    macro_cols = ["ecb_rate", "us_10y_yield", "vix", "eur_usd", "de_inflation", "us_inflation", "market_close"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    required_cols = FEATURE_COLS + ["label", "Date", "Ticker", "sector", "market", "market_close"]
    required_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=required_cols)

    dropped_rows = before_rows - len(df)

    df, winsor_summary = winsorize(df, FEATURE_COLS)
    pd.DataFrame(winsor_summary).to_csv(tables_dir / "winsorization_report.csv", index=False)

    df = add_market_regime(df)

    # Temporal split with 21-day leakage buffer.
    train_end = pd.Timestamp("2021-12-31")
    val_start = pd.Timestamp("2022-01-01") + pd.Timedelta(days=21)
    val_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=21)
    test_end = pd.Timestamp("2024-12-31")

    train = df[df["Date"] <= train_end].copy()
    validation = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].copy()
    test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

    scaler = StandardScaler()
    train[FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
    validation[FEATURE_COLS] = scaler.transform(validation[FEATURE_COLS])
    test[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

    # Multicollinearity diagnostics are documented, not auto-removed.
    corr = train[FEATURE_COLS].corr().abs()
    high_corr_rows = []
    for i, col_i in enumerate(FEATURE_COLS):
        for col_j in FEATURE_COLS[i + 1 :]:
            c = corr.loc[col_i, col_j]
            if c > 0.85:
                high_corr_rows.append({"feature_a": col_i, "feature_b": col_j, "abs_corr": c})

    pd.DataFrame(high_corr_rows).to_csv(tables_dir / "collinearity_report.csv", index=False)

    train.to_csv(processed_dir / "train.csv", index=False)
    validation.to_csv(processed_dir / "validation.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    joblib.dump(scaler, models_dir / "scaler.pkl")

    metadata = {
        "feature_cols": FEATURE_COLS,
        "target_col": "label",
        "non_actionable_features": sorted(NON_ACTIONABLE),
        "actionable_features": sorted(set(FEATURE_COLS) - NON_ACTIONABLE),
        "rows_before_missing_handling": before_rows,
        "rows_after_missing_handling": len(df),
        "rows_dropped_missing": dropped_rows,
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }

    with open(models_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved data preparation outputs:")
    print(f"  train: {processed_dir / 'train.csv'} ({len(train)} rows)")
    print(f"  validation: {processed_dir / 'validation.csv'} ({len(validation)} rows)")
    print(f"  test: {processed_dir / 'test.csv'} ({len(test)} rows)")
    print(f"  metadata: {models_dir / 'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
