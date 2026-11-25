import json

target_titles = set()
with open("tests/eval-set.jsonl", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line, strict=False)
        title = item.get("source_title") or item.get("title")
        if title:
            target_titles.add(title)
print(target_titles)

output_path = "tests/matched_chunks.jsonl"
count = 0

with (
    open("data/chunks.jsonl", encoding="utf-8") as infile,
    open(output_path, "w", encoding="utf-8") as outfile,
):
    for line in infile:
        if not line.strip():
            continue
        doc = json.loads(line, strict=False)
        doc_title = doc.get("source_title") or doc.get("title")
        if doc_title in target_titles:
            outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

print(f"Saved {count} chunks to {output_path}")
