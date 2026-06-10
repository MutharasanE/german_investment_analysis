"""Build survey descriptive plots used in the thesis."""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SURVEY = ROOT / "survey_detailed.csv"
OUT_DIR = ROOT / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SURVEY)

# 1. Preference distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
order = ["FEATURE_IMPORTANCE", "COUNTERFACTUAL", "No"]
labels = ["SHAP\n(feature importance)", "LEWIS\n(counterfactual)", "No\npreference"]
counts = [int((df["chosen_method"] == m).sum()) for m in order]
bars = ax.bar(labels, counts, color=["#bdbdbd", "#525252", "#f0f0f0"], edgecolor="black")
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(c), ha="center", fontsize=11, weight="bold")
ax.set_ylabel("Number of evaluations")
ax.set_title(f"Expert preference for explanation type (n = {len(df)})")
ax.set_ylim(0, max(counts) + 3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "expert_preference.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Trust score distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["trust_score"], bins=range(1, 12), color="lightgrey", edgecolor="black")
ax.set_xlabel("Trust score (1 = none, 10 = full)")
ax.set_ylabel("Frequency")
ax.set_title(f"Trust score distribution (mean = {df['trust_score'].mean():.2f}, "
             f"sd = {df['trust_score'].std():.2f})")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "trust_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Confidence vs trust scatter (the regression headline)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["confidence_score"], df["trust_score"], alpha=0.7, color="black")
m, b = pd.np.polyfit(df["confidence_score"], df["trust_score"], 1) if False else (0, 0)
import numpy as np
m, b = np.polyfit(df["confidence_score"], df["trust_score"], 1)
xs = np.linspace(df["confidence_score"].min(), df["confidence_score"].max(), 50)
ax.plot(xs, m * xs + b, color="grey", linestyle="--")
ax.set_xlabel("Confidence (1-10)")
ax.set_ylabel("Trust (1-10)")
ax.set_title("Trust vs confidence (the only significant OLS predictor)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "trust_vs_confidence.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4. Mechanics-feedback distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
mech_order = ["Yes, highly accurate", "Somewhat accurate", "No, they seem theoretical"]
mech_labels = ["Highly\naccurate", "Somewhat\naccurate", "Theoretical"]
mech_counts = [int((df["mechanics_feedback"] == m).sum()) for m in mech_order]
bars = ax.bar(mech_labels, mech_counts, color=["#525252", "#bdbdbd", "#f0f0f0"], edgecolor="black")
for bar, c in zip(bars, mech_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(c), ha="center", fontsize=11, weight="bold")
ax.set_ylabel("Number of evaluations")
ax.set_title("Perceived mechanical accuracy of the explanations")
ax.set_ylim(0, max(mech_counts) + 3)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "mechanics_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 5. LEWIS vs SHAP rankings (custom black-and-white version)
import numpy as np
lvs = pd.read_csv(ROOT / "results" / "investment" / "lewis_vs_shap_DirectLiNGAM(b).csv")
lvs = lvs.sort_values("lewis_rank")
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(lvs))
width = 0.4
ax.bar(x - width / 2, lvs["lewis_normalized"], width, label="LEWIS (causal)",
       color="#525252", edgecolor="black")
ax.bar(x + width / 2, lvs["shap_normalized"], width, label="SHAP (correlational)",
       color="#bdbdbd", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(lvs["feature"], rotation=30, ha="right")
ax.set_ylabel("Normalised importance [0, 1]")
ax.set_title("LEWIS (causal) vs SHAP (correlational) rankings on the DAX panel")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "lewis_vs_shap_bw.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(" ", p)
