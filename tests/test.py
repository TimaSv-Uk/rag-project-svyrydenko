import json

# 1. Збираємо множину цільових заголовків (title або source_title) з eval-set.jsonl
target_titles = set()
with open("tests/eval-set.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        title = item.get("source_title") or item.get("title")
        if title:
            target_titles.add(title)

# 2. Шукаємо всі чанки в корпусі, заголовок яких є серед target_titles
matched_chunks = []
with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_title = doc.get("source_title") or doc.get("title")
        if doc_title in target_titles:
            matched_chunks.append(doc)

print(f"Знайдено чанків: {len(matched_chunks)}")
