"""Generate trust_drivers_plot.png from the saved regression coefficients.

This is a build helper (no MongoDB required) used to produce the figure that the
OLS pipeline normally creates. It rebuilds the regression matrix from the local
survey CSV and re-fits the model, then writes a 2x2 diagnostic panel.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent
SURVEY = ROOT / "survey_detailed.csv"
OUT = ROOT / "results" / "plots" / "trust_drivers_plot.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def parse_decision(decision_str: str):
    """Parse the pipe-delimited decision blob into structured fields."""
    out = {"num_features": np.nan, "attention_flag": np.nan}
    if not isinstance(decision_str, str):
        return out
    parts = decision_str.split("|")
    for p in parts:
        if p.startswith("features:"):
            feats = p.split(":", 1)[1]
            out["num_features"] = len([f for f in feats.split(",") if f.strip()])
        elif p.startswith("attention:"):
            out["attention_flag"] = 1 if p.split(":", 1)[1].strip() == "pass" else 0
    return out


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["decision"].apply(parse_decision).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df["is_counterfactual_b"] = (df["method_for_b"] == "COUNTERFACTUAL").astype(int)
    accuracy_map = {
        "Yes, highly accurate": 2,
        "Somewhat accurate": 1,
        "No, they seem theoretical": 0,
    }
    df["accuracy_score"] = df["mechanics_feedback"].map(accuracy_map)
    keep = [
        "trust_score",
        "is_counterfactual_b",
        "accuracy_score",
        "num_features",
        "confidence_score",
        "attention_flag",
    ]
    return df[keep].dropna()


def fit_ols(matrix: pd.DataFrame):
    y = matrix["trust_score"].astype(float)
    X = matrix.drop(columns=["trust_score"]).astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X, y


def main():
    df = pd.read_csv(SURVEY)
    matrix = build_matrix(df)
    print(f"Regression matrix: {matrix.shape[0]} rows, {matrix.shape[1] - 1} predictors")

    model, X, y = fit_ols(matrix)
    print(model.summary())

    yhat = model.predict(X)
    resid = y - yhat

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Actual vs predicted
    ax = axes[0, 0]
    ax.scatter(yhat, y, alpha=0.7, color="black")
    lims = [min(y.min(), yhat.min()) - 0.5, max(y.max(), yhat.max()) + 0.5]
    ax.plot(lims, lims, color="grey", linestyle="--")
    ax.set_xlabel("Predicted trust")
    ax.set_ylabel("Observed trust")
    ax.set_title("Actual vs predicted trust")

    # Residuals vs predicted
    ax = axes[0, 1]
    ax.scatter(yhat, resid, alpha=0.7, color="black")
    ax.axhline(0, color="grey", linestyle="--")
    ax.set_xlabel("Predicted trust")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted")

    # Q-Q plot
    ax = axes[1, 0]
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Q-Q plot of residuals")
    for line in ax.get_lines():
        line.set_color("black")

    # Coefficient bars (excluding intercept)
    ax = axes[1, 1]
    coefs = model.params.drop("const")
    cis = model.conf_int().drop("const")
    err_low = coefs - cis[0]
    err_high = cis[1] - coefs
    y_pos = np.arange(len(coefs))
    ax.barh(y_pos, coefs, xerr=[err_low, err_high], color="lightgrey", edgecolor="black")
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index)
    ax.set_xlabel("Coefficient (95 % CI)")
    ax.set_title("OLS coefficients on trust score")

    fig.suptitle(
        f"Drivers of expert trust (n = {len(matrix)}, R^2 = {model.rsquared:.3f})",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")

    # Also persist a JSON summary so the writeup can reference live values
    summary = {
        "n": int(len(matrix)),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": {
            name: {
                "coef": float(model.params[name]),
                "se": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p": float(model.pvalues[name]),
                "ci_low": float(model.conf_int().loc[name, 0]),
                "ci_high": float(model.conf_int().loc[name, 1]),
            }
            for name in model.params.index
        },
    }
    out_json = ROOT / "results" / "plots" / "trust_drivers_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
