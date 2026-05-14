import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Data
labels = ['Buy', 'Hold', 'Sell']

# Old CM
cm_old = np.array([
    [1347, 13661,  1605],
    [ 992, 24713,  1549],
    [ 600, 13655,  1878]
])

# New CM
cm_new = np.array([
    [ 6690, 12324,  9045],
    [ 7724, 24334, 12957],
    [ 4988, 12628,  9310]
])

# Normalize by row (True labels) to get recall percentages
cm_old_norm = cm_old.astype('float') / cm_old.sum(axis=1)[:, np.newaxis]
cm_new_norm = cm_new.astype('float') / cm_new.sum(axis=1)[:, np.newaxis]

os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# 1. Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm_old_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Previous Model (Accuracy-optimized)\nNormalized Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_new_norm, annot=True, fmt=".1%", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("Updated Model (Class-Weighted + F1)\nNormalized Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("results/plots/cm_comparison.png", dpi=150)
print("Saved results/plots/cm_comparison.png")

def get_metrics(cm):
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    precisions = np.diag(cm) / np.sum(cm, axis=0)
    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    return recalls, precisions, f1s, acc

r_old, p_old, f1_old, acc_old = get_metrics(cm_old)
r_new, p_new, f1_new, acc_new = get_metrics(cm_new)

df_comp = pd.DataFrame({
    "Metric": [
        "Accuracy", 
        "Macro F1",
        "Buy Recall (Accuracy of actual Buys)",
        "Buy Precision",
        "Buy F1",
        "Hold Recall",
        "Hold Precision",
        "Hold F1",
        "Sell Recall (Accuracy of actual Sells)",
        "Sell Precision",
        "Sell F1"
    ],
    "Previous Model (Unweighted)": [
        acc_old, np.mean(f1_old),
        r_old[0], p_old[0], f1_old[0],
        r_old[1], p_old[1], f1_old[1],
        r_old[2], p_old[2], f1_old[2]
    ],
    "Updated Model (Weighted)": [
        acc_new, np.mean(f1_new),
        r_new[0], p_new[0], f1_new[0],
        r_new[1], p_new[1], f1_new[1],
        r_new[2], p_new[2], f1_new[2]
    ]
})

df_comp = df_comp.round(4)
df_comp.to_csv("results/tables/model_comparison_metrics.csv", index=False)
print("Saved results/tables/model_comparison_metrics.csv")
