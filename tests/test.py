import json
import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Student-Project/1.0 (timofii.educational.project@gmail.com)"}
missing = [t.strip() for t in open("missing_titles.txt", encoding="utf-8") if t.strip()]
sample = missing[:3]

for title in sample:
    r = requests.get(API, params={
        "action": "query",
        "prop": "extracts|info|revisions",
        "explaintext": 1,
        "inprop": "url",
        "rvprop": "size",
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }, headers=HEADERS, timeout=30)
    print(title)
    print(json.dumps(r.json().get("query", {}), ensure_ascii=False, indent=2)[:800])
    print("-" * 40)
