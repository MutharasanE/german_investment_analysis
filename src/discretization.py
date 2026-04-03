"""Equal-width and equal-frequency discretization utilities."""

import numpy as np
import pandas as pd


def equal_width_discretize(series, n_bins=10):
    """Discretize a series into equal-width bins. Returns integer bin labels 0..n_bins-1."""
    bins = np.linspace(series.min(), series.max(), n_bins + 1)
    bins[0] -= 1e-10  # include minimum
    bins[-1] += 1e-10  # include maximum
    return pd.cut(series, bins=bins, labels=False, include_lowest=True).astype(int)


def equal_freq_discretize(series, n_bins=10):
    """Discretize a series into equal-frequency bins. Returns integer bin labels 0..n_bins-1."""
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop").astype(int)


def binarize_target(series, method="equal_width"):
    """Binarize target variable into 0/1 using median or equal-width split."""
    if method == "equal_width":
        mid = (series.min() + series.max()) / 2
        return (series >= mid).astype(int)
    elif method == "median":
        return (series >= series.median()).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")


def discretize_dataframe(df, target_col="Y", n_bins=10, method="equal_width"):
    """
    Discretize all columns in a DataFrame.
    Features get n_bins bins, target gets binarized.
    """
    result = pd.DataFrame()
    for col in df.columns:
        if col == target_col:
            result[col] = binarize_target(df[col], method="equal_width")
        else:
            if method == "equal_width":
                result[col] = equal_width_discretize(df[col], n_bins)
            else:
                result[col] = equal_freq_discretize(df[col], n_bins)
    return result
