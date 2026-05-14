import numpy as np
import pandas as pd
import pytest

from run_investment_split_eval import (
    _discretize_train_test,
    _fit_equal_freq_bins,
    _time_split,
)


def test_fit_equal_freq_bins_handles_constant_series() -> None:
    s = pd.Series([5.0] * 10)
    edges = _fit_equal_freq_bins(s, n_bins=5)

    assert len(edges) >= 2
    assert edges[0] < edges[-1]


def test_time_split_raises_when_date_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="date"):
        _time_split(df, test_size=0.2)


def test_time_split_works_without_ticker_column() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04"],
            "investment_decision": [1, 0, 1, 0],
            "volatility": [0.1, 0.2, 0.3, 0.4],
        }
    )

    train_df, test_df = _time_split(df, test_size=0.25)

    assert len(train_df) == 3
    assert len(test_df) == 1
    assert pd.to_datetime(train_df["date"]).max() < pd.to_datetime(test_df["date"]).min()


def test_discretize_train_test_creates_integer_bins() -> None:
    train_df = pd.DataFrame(
        {
            "volatility": [0.1, 0.2, 0.3, 0.4],
            "momentum": [1.0, 0.5, -0.1, 0.0],
            "investment_decision": [0, 0, 1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "volatility": [0.15, 0.35],
            "momentum": [0.9, -0.2],
            "investment_decision": [0, 1],
        }
    )

    train_disc, test_disc = _discretize_train_test(
        train_df,
        test_df,
        feature_cols=["volatility", "momentum"],
        target_col="investment_decision",
        n_bins=3,
    )

    for col in ["volatility", "momentum"]:
        assert np.issubdtype(train_disc[col].dtype, np.integer)
        assert np.issubdtype(test_disc[col].dtype, np.integer)
        assert train_disc[col].isna().sum() == 0
        assert test_disc[col].isna().sum() == 0

    assert train_disc["investment_decision"].tolist() == [0, 0, 1, 1]
    assert test_disc["investment_decision"].tolist() == [0, 1]
