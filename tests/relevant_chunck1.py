import json
from pathlib import Path

CORPUS_PATH = Path("chunks.jsonl")
OUT_PATH = Path("tests/eval-set.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            # Припускаємо поля на кшталт id / chunk_id та text / content / chunk_text
            chunk_id = item.get("chunk_id") or item.get("id") or f"chunk_{idx}"
            text = item.get("text") or item.get("content") or item.get("chunk_text") or ""
            chunks.append({
                "chunk_id": str(chunk_id),
                "text": text,
                "title": item.get("title", ""),
            })
    return chunks


# 2. 20 вручну сформульованих запитів під сценарій (американські романісти, XX ст., American Dream тощо)
# Для кожного запиту вказуємо ключові слова для швидкого пошуку відповідного чанка в корпусі
SEED_QUERIES = [
    ("q01", "Which American novelists of the early 20th century explored the American Dream and its disillusionment?", ["American Dream", "Fitzgerald"]),
    ("q02", "What themes and social critiques appear in F. Scott Fitzgerald's novels?", ["Fitzgerald", "Gatsby"]),
    ("q03", "How does Ernest Hemingway's writing style and major novels reflect post-war disillusionment?", ["Hemingway", "war"]),
    ("q04", "What are John Steinbeck's most famous novels depicting working-class struggles and the Great Depression?", ["Steinbeck", "Grapes of Wrath"]),
    ("q05", "How is the American Dream portrayed as an illusion or unfulfilled promise in 20th-century American fiction?", ["American Dream", "illusion"]),
    ("q06", "Which novels by William Faulkner focus on the American South and family decline?", ["Faulkner", "Mississippi"]),
    ("q07", "What role does social class and wealth play in early 20th-century American novels?", ["social class", "wealth", "novel"]),
    ("q08", "Which American novelists won major literary prizes such as the Pulitzer Prize or Nobel Prize in Literature?", ["Pulitzer", "Nobel", "novelist"]),
    ("q09", "How did modernism influence 20th-century American novelists and narrative structure?", ["modernism", "novel"]),
    ("q10", "Which novels address racial issues and identity in 20th-century American literature?", ["race", "identity", "novel"]),
    ("q11", "What are key themes in Ralph Ellison's Invisible Man?", ["Ellison", "Invisible Man"]),
    ("q12", "How do Sinclair Lewis's novels satirize middle-class American life and conformity?", ["Sinclair Lewis", "satire"]),
    ("q13", "Which American novelists wrote about alienation and individualism in modern society?", ["alienation", "novel"]),
    ("q14", "What are notable novels by Edith Wharton examining high society and social conventions?", ["Wharton", "society"]),
    ("q15", "How does the setting of New York appear in American novels of the 20th century?", ["New York", "novel"]),
    ("q16", "Which female American novelists of the 20th century explored women's roles and independence?", ["women", "novelist", "novel"]),
    ("q17", "What historical events shaped American fiction in the first half of the 20th century?", ["century", "war", "novel"]),
    ("q18", "How do coming-of-age themes appear in classic American novels?", ["coming-of-age", "novel"]),
    ("q19", "Which American novelists examined materialism and consumer culture?", ["materialism", "American Dream"]),
    ("q20", "What distinguishes realistic and naturalistic American novels from earlier romantic works?", ["realism", "naturalism", "novel"]),
]


def find_best_chunk(chunks: list[dict], keywords: list[str]) -> dict | None:
    best = None
    best_score = -1
    kw_lower = [k.lower() for k in keywords]
    for ch in chunks:
        txt_lower = ch["text"].lower()
        score = sum(txt_lower.count(k) for k in kw_lower)
        # Невеликий бонус за збіг усіх ключових слів
        if all(k in txt_lower for k in kw_lower):
            score += 5
        if score > best_score and score > 0:
            best_score = score
            best = ch
    return best


def main():
    chunks = load_chunks(CORPUS_PATH)
    records = []

    for qid, query, keywords in SEED_QUERIES:
        match = find_best_chunk(chunks, keywords)
        if match is None:
            print(f"[WARNING] Не знайдено збігу для {qid}: {query}")
            continue
        records.append({
            "id": qid,
            "query": query,
            "relevant_chunk_id": match["chunk_id"],
            "relevant_chunk_text": match["text"],
            "source_title": match["title"],
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Збережено {len(records)} пар у {OUT_PATH}")


if __name__ == "__main__":
    main()
