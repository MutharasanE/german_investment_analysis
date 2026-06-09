"""
MongoDB Survey Analytics Dashboard
Usage: python survey_analytics.py
"""

import io
import os
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import json

MONGO_URI = os.environ["MONGO_URI"]

client = MongoClient(MONGO_URI)
db = client["thesis_survey"]
votes = db["votes"]

# Fetch data
df = pd.DataFrame(list(votes.find()))
df = df.drop("_id", axis=1, errors="ignore")

# === PREFERENCE ANALYSIS ===
print("\n" + "="*80)
print("PREFERENCE ANALYSIS")
print("="*80)

pref_counts = df["preference"].value_counts()
print(f"\nPreference Votes:")
for pref, count in pref_counts.items():
    pct = 100 * count / len(df)
    print(f"  {pref}: {count} ({pct:.1f}%)")

# === METHOD COMPARISON ===
print("\n" + "="*80)
print("METHOD TYPE COMPARISON")
print("="*80)

method_counts = {
    "Method A (Feature Importance)": df[df['method_for_a'] == 'FEATURE_IMPORTANCE'].shape[0],
    "Method B (Counterfactual)": df[df['method_for_b'] == 'COUNTERFACTUAL'].shape[0],
    "No preference": df[df['preference'] == 'No'].shape[0],
}

for method, count in method_counts.items():
    print(f"  {method}: {count}")

# === TRUST & CONFIDENCE ===
print("\n" + "="*80)
print("TRUST & CONFIDENCE SCORES")
print("="*80)

print(f"Trust Score (mean): {df['trust_score'].mean():.2f}/10 (std: {df['trust_score'].std():.2f})")
print(f"Confidence Score (mean): {df['confidence_score'].mean():.2f}/10 (std: {df['confidence_score'].std():.2f})")

# Score distribution
print("\nTrust Score Distribution:")
for score in sorted(df['trust_score'].unique()):
    count = (df['trust_score'] == score).sum()
    pct = 100 * count / len(df)
    bar = "[" + "#" * int(pct/5) + " " * (20 - int(pct/5)) + "]"
    print(f"  {score}/10: {count:2d} {bar} {pct:.1f}%")

# === ACCURACY PERCEPTION ===
print("\n" + "="*80)
print("PERCEIVED ACCURACY")
print("="*80)

if 'mechanics_feedback' in df.columns:
    accuracy = df['mechanics_feedback'].value_counts()
    for rating, count in accuracy.items():
        pct = 100 * count / len(df)
        print(f"  {rating}: {count} ({pct:.1f}%)")

# === BY TICKER ===
print("\n" + "="*80)
print("RESPONSES BY COMPANY (Ticker)")
print("="*80)

ticker_counts = df['ticker'].value_counts()
for ticker, count in ticker_counts.head(10).items():
    print(f"  {ticker}: {count}")

# === COMMENTS ANALYSIS ===
print("\n" + "="*80)
print("SAMPLE EXPERT COMMENTS")
print("="*80)

comments = df[df['comment'].notna() & (df['comment'] != '') & (df['comment'] != '.')]
if len(comments) > 0:
    print(f"\nFound {len(comments)} detailed comments:\n")
    for idx, row in comments.head(3).iterrows():
        print(f"[{row['ticker']}] {row['expert_name'] or 'Anonymous'}:")
        comment_text = str(row['comment'])[:300] + "..."
        print(f"  {comment_text}\n")

# === EXPORT SUMMARY ===
print("\n" + "="*80)
print("EXPORTING DETAILED CSV...")
print("="*80)

summary = {
    "Total Responses": len(df),
    "Avg Trust Score": f"{df['trust_score'].mean():.2f}",
    "Avg Confidence": f"{df['confidence_score'].mean():.2f}",
    "Preference A %": f"{100 * (df['preference']=='A').sum() / len(df):.1f}%",
    "Preference B %": f"{100 * (df['preference']=='B').sum() / len(df):.1f}%",
    "No Preference %": f"{100 * (df['preference']=='No').sum() / len(df):.1f}%",
}

print("\nSUMMARY REPORT:")
for key, val in summary.items():
    print(f"  {key}: {val}")

# Save detailed CSV
df.to_csv("survey_detailed.csv", index=False)
print("\n[OK] Detailed export: survey_detailed.csv")

client.close()
