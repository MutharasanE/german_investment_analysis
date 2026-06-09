"""
Fetch and display survey responses from MongoDB
Usage: python fetch_survey.py
"""

import io
import os
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pymongo import MongoClient
import pandas as pd
from pprint import pprint

# MongoDB connection
MONGO_URI = os.environ["MONGO_URI"]

client = MongoClient(MONGO_URI)
db = client["thesis_survey"]
votes_collection = db["votes"]

# Fetch all survey responses
print("=" * 80)
print("SURVEY RESPONSES FROM MONGODB")
print("=" * 80)

responses = list(votes_collection.find())
print(f"\n[OK] Total responses: {len(responses)}\n")

if responses:
    # Display as table
    df = pd.DataFrame(responses)
    # Remove ObjectId for readability
    df = df.drop("_id", axis=1, errors="ignore")

    print(df.to_string())
    print("\n" + "=" * 80)

    # Summary stats
    print("\nSUMMARY STATISTICS:")
    print("-" * 80)

    if "preference" in df.columns:
        print("\nPreference Distribution:")
        print(df["preference"].value_counts())

    if "expert_role" in df.columns:
        print("\nResponses by Expert Role:")
        print(df["expert_role"].value_counts())

    if "ticker" in df.columns:
        print("\nResponses by Company (Ticker):")
        print(df["ticker"].value_counts().head(10))

    if "trust_score" in df.columns:
        print(f"\nAverage Trust Score: {df['trust_score'].mean():.2f}/10")

    if "confidence_score" in df.columns:
        print(f"Average Confidence Score: {df['confidence_score'].mean():.2f}/10")

    # Export to CSV
    csv_path = "survey_responses.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Exported to {csv_path}")
else:
    print("[WARN] No survey responses found in MongoDB")

client.close()
