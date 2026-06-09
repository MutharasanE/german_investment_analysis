"""
Upload synthetic survey responses to MongoDB.
Run in VSCode: python upload_synthetic.py
"""
import io
import os
import sys
import json
from datetime import datetime
from pymongo import MongoClient

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = "thesis_survey"
COLLECTION = "votes"
JSON_FILE = "synthetic_survey_responses.json"


def main():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    for r in records:
        if isinstance(r.get("timestamp"), str):
            r["timestamp"] = datetime.fromisoformat(r["timestamp"])

    client = MongoClient(MONGO_URI)
    coll = client[DB_NAME][COLLECTION]

    before = coll.count_documents({})
    result = coll.insert_many(records)
    after = coll.count_documents({})

    print(f"[OK] Inserted {len(result.inserted_ids)} records")
    print(f"     Collection size: {before} -> {after}")
    client.close()


if __name__ == "__main__":
    main()
