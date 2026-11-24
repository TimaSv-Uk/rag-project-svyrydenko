import json
with open("data/clean_american_novelists.jsonl", encoding="utf-8") as f:
    doc = json.loads(f.readline())
print(repr(doc["text"][:1500]))
