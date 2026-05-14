"""
Standalone holdout evaluation for the investment model.

Creates a proper unseen test split (time-based by default), trains CatBoost on
train data only, and reports confusion matrix + precision/recall/F1 on test data.

Usage examples:
  python run_investment_split_eval.py
  python run_investment_split_eval.py --split random --test-size 0.2 --random-state 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def _fit_equal_freq_bins(train_series: pd.Series, n_bins: int) -> np.ndarray:
    """Fit quantile-based bin edges on train data only."""
    # np.quantile can return duplicate edges for low-cardinality columns.
    edges = np.quantile(train_series.to_numpy(), q=np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)

    # Guarantee at least two edges so pd.cut works.
    if len(edges) < 2:
        val = float(edges[0]) if len(edges) == 1 else float(train_series.iloc[0])
        edges = np.array([val - 1e-10, val + 1e-10])

    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return edges


def _apply_bins(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Apply pre-fitted edges and return integer bin ids."""
    binned = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    # Values outside learned range should not happen due to edge padding, but guard anyway.
    return binned.fillna(0).astype(int)


def _discretize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discretize features with train-only fitted bins; keep binary target as-is."""
    train_disc = pd.DataFrame(index=train_df.index)
    test_disc = pd.DataFrame(index=test_df.index)

    for col in feature_cols:
        edges = _fit_equal_freq_bins(train_df[col], n_bins=n_bins)
        train_disc[col] = _apply_bins(train_df[col], edges)
        test_disc[col] = _apply_bins(test_df[col], edges)

    train_disc[target_col] = train_df[target_col].astype(int)
    test_disc[target_col] = test_df[target_col].astype(int)
    return train_disc, test_disc


def _time_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: earlier rows for train, most recent rows for test."""
    if "date" not in df.columns:
        raise ValueError("Time split requested, but 'date' column is missing in dataset.")

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = temp.dropna(subset=["date"])

    if temp.empty:
        raise ValueError("No valid rows left after parsing the 'date' column.")

    # Sort deterministically, even when ticker is unavailable.
    sort_cols = ["date"]
    if "ticker" in temp.columns:
        sort_cols.append("ticker")
    temp = temp.sort_values(sort_cols)

    n_test = max(1, int(len(temp) * test_size))
    n_train = len(temp) - n_test

    if n_train <= 0:
        raise ValueError("Test size too large: train set would be empty.")

    return temp.iloc[:n_train].copy(), temp.iloc[n_train:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout evaluation for investment CatBoost model")
    parser.add_argument("--data", default="data/investment_dataset.csv", help="Path to investment dataset")
    parser.add_argument("--split", choices=["time", "random"], default="time",
                        help="Split strategy. Default is time-based" \
                        " split.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction in (0,1). Default: 0.2")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for random split")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of equal-frequency bins for features")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run 'python run_investment.py' first."
        )

    results_dir = Path("results/investment")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    feature_cols = [
        "volatility", "momentum", "volume_avg", "return_1y", "max_drawdown",
        "ecb_rate", "eur_usd", "de_inflation", "vix",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = "investment_decision"

    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df.dropna(subset=required).copy()

    if args.split == "time":
        train_df, test_df = _time_split(df, args.test_size)
    else:
        # Prefer stratified split, but fall back when class counts are too small.
        stratify = df[target_col]
        min_class_count = int(stratify.value_counts().min())
        if min_class_count < 2:
            stratify = None

        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )

    train_disc, test_disc = _discretize_train_test(
        train_df,
        test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        n_bins=args.n_bins,
    )

    X_train = train_disc[feature_cols]
    y_train = train_disc[target_col]
    X_test = test_disc[feature_cols]
    y_test = test_disc[target_col]

    model = CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "split_strategy": args.split,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_bins": args.n_bins,
        "n_total": int(len(df)),
        "n_train": int(len(train_disc)),
        "n_test": int(len(test_disc)),
        "feature_cols": feature_cols,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    # Save artifacts for comparison with existing in-sample run.
    stem = f"holdout_eval_{args.split}"
    with open(results_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).drop(columns=["feature_cols", "confusion_matrix"]).to_csv(
        results_dir / f"{stem}.csv", index=False
    )

    pred_df = pd.DataFrame({
        "y_true": y_test.to_numpy().astype(int),
        "y_pred": y_pred,
    })
    pred_df.to_csv(results_dir / f"{stem}_predictions.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Holdout evaluation ({args.split} split)")
    print("=" * 60)
    print(f"Train size: {metrics['n_train']} | Test size: {metrics['n_test']}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print(f"Saved: {results_dir / (stem + '.json')}")
    print(f"Saved: {results_dir / (stem + '.csv')}")
    print(f"Saved: {results_dir / (stem + '_predictions.csv')}")


if __name__ == "__main__":
    main()
