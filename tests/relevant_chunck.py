import json
import re
from pathlib import Path

CORPUS_PATH = Path("data/chunks.jsonl")
REVIEW_PATH = Path("tests/candidates_review.json")
OUT_PATH = Path("tests/eval-set.jsonl")
REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

MIN_WORDS = 55
MAX_WORDS = 400


def is_clean_prose(text: str) -> bool:
    """Відсіює списки нагород, бібліографії, занадто короткі та занадто довгі чанки."""
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if text.count("\n") > 5:
        return False
    if len(re.findall(r"\(\d{4}\)", text)) >= 4:
        return False
    return True


def ends_mid_sentence(text: str) -> bool:
    """Чанк, що обривається посеред речення, погано підходить для оцінювання."""
    return not re.search(r"[.!?\"»)]\s*$", text.strip())


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text") or item.get("content") or item.get("chunk_text") or ""
            title = item.get("source_title") or item.get("title") or ""
            if not text or not title:
                continue
            if not is_clean_prose(text) or ends_mid_sentence(text):
                continue
            chunks.append({
                "chunk_id": str(item.get("chunk_id") or item.get("id") or f"chunk_{idx}"),
                "text": text,
                "source_title": title,
            })
    return chunks


def candidates(chunks, title_substr: str, must_terms: list[str], k: int = 5):
    """Кандидати з потрібної статті, що містять принаймні один ключовий термін.
    Вступ статті (_c0000) отримує бонус, бо найчастіше описує творчість автора."""
    scored = []
    for ch in chunks:
        if title_substr.lower() not in ch["source_title"].lower():
            continue
        hits = sum(len(re.findall(p, ch["text"], re.IGNORECASE)) for p in must_terms)
        if hits == 0:
            continue
        bonus = 3 if ch["chunk_id"].endswith("_c0000") else 0
        scored.append((hits + bonus, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:k]]


# Запити та пошукові терміни. Заповніть "title" і "terms" для кожного запиту.
TOPICS = [
    {"id": "q01", "query": "Which American novelists of the early 20th century explored the American Dream and its disillusionment?",
     "title": "Fitzgerald", "terms": [r"\bAmerican Dream\b", r"\bdisillusion"]},
    {"id": "q02", "query": "How did F. Scott Fitzgerald portray the Jazz Age and wealth in his novels?",
     "title": "F. Scott Fitzgerald", "terms": [r"\bJazz Age\b", r"\bwealth\b"]},
    {"id": "q03", "query": "How does Ernest Hemingway's writing style and major novels reflect war and disillusionment?",
     "title": "Ernest Hemingway", "terms": [r"\bwar\b", r"\bdisillusion"]},
    {"id": "q04", "query": "How did John Steinbeck depict working-class struggles and the Great Depression in his novels?",
     "title": "Steinbeck", "terms": [r"\bDepression\b", r"\bmigrant\b", r"\bworking class\b"]},
    {"id": "q06", "query": "Which themes define William Faulkner's novels about the American South?",
     "title": "William Faulkner", "terms": [r"\bSouth\b", r"\bYoknapatawpha\b"]},
    {"id": "q07", "query": "How did Sinclair Lewis satirize middle-class American society and conformity?",
     "title": "Sinclair Lewis", "terms": [r"\bBabbitt\b", r"\bMain Street\b", r"\bsatir"]},
    {"id": "q08", "query": "What social conventions did Edith Wharton examine in her novels about New York society?",
     "title": "Edith Wharton", "terms": [r"\bNew York\b", r"\bsociety\b", r"\bconvention"]},
    {"id": "q09", "query": "What are the central themes of Ralph Ellison's novel Invisible Man?",
     "title": "Ralph Ellison", "terms": [r"\bInvisible Man\b", r"\bidentity\b"]},
    {"id": "q10", "query": "How did Richard Wright address racial injustice in 20th-century American fiction?",
     "title": "Richard Wright", "terms": [r"\bracis", r"\bracial\b", r"\bNative Son\b"]},
    {"id": "q11", "query": "How did Toni Morrison explore African American history and identity in her novels?",
     "title": "Toni Morrison", "terms": [r"\bBeloved\b", r"\bhistory\b", r"\bidentity\b"]},
    {"id": "q12", "query": "What themes of alienation and youth appear in J. D. Salinger's The Catcher in the Rye?",
     "title": "Catcher in the Rye", "terms": [r"\balienat", r"\byouth\b", r"\bHoldén\b|\bHolden\b"]},
    {"id": "q13", "query": "How did Theodore Dreiser contribute to American literary naturalism?",
     "title": "Theodore Dreiser", "terms": [r"\bnaturalis"]},
    {"id": "q14", "query": "What role did Upton Sinclair's novels play in exposing industrial and social conditions?",
     "title": "Upton Sinclair", "terms": [r"\bThe Jungle\b", r"\bmeatpacking\b|\bindustr"]},
    {"id": "q15", "query": "How did Jack London depict survival, nature, and social class in his fiction?",
     "title": "Jack London", "terms": [r"\bCall of the Wild\b", r"\bMartin Eden\b", r"\bsurviv"]},
    {"id": "q16", "query": "How did Zora Neale Hurston portray Black culture and women's independence in her novels?",
     "title": "Zora Neale Hurston", "terms": [r"\bTheir Eyes Were Watching God\b", r"\bJonah\b"]},
    {"id": "q17", "query": "How did John Dos Passos structure his modernist trilogy about early 20th-century America?",
     "title": "John Dos Passos", "terms": [r"\bU\.S\.A\.", r"\btrilogy\b"]},
    {"id": "q18", "query": "What themes of war and absurdity define Joseph Heller's Catch-22?",
     "title": "Joseph Heller", "terms": [r"\bCatch-22\b", r"\babsurd"]},
    {"id": "q19", "query": "How did Kurt Vonnegut blend science fiction and anti-war satire in his novels?",
     "title": "Kurt Vonnegut", "terms": [r"\bSlaughterhouse-Five\b", r"\bscience fiction\b"]},
    {"id": "q20", "query": "How did Willa Cather depict frontier life and immigrants on the American Great Plains?",
     "title": "Willa Cather", "terms": [r"\bPlains\b", r"\bfrontier\b", r"\bimmigra"]},
]

# Ручний вибір: заповнюється після перегляду candidates_review.json.
# Ключ — id запиту, значення — chunk_id обраного чанку.
SELECTED: dict[str, str] = {}


def build_review(chunks):
    review = []
    for t in TOPICS:
        cands = candidates(chunks, t["title"], t["terms"])
        review.append({
            "id": t["id"],
            "query": t["query"],
            "candidates": [
                {"chunk_id": c["chunk_id"], "source_title": c["source_title"],
                 "preview": c["text"][:300]}
                for c in cands
            ],
        })
    REVIEW_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Кандидати для перегляду: {REVIEW_PATH}")


def build_eval_set(chunks):
    by_id = {c["chunk_id"]: c for c in chunks}
    used = set()
    items = []
    for t in TOPICS:
        chunk_id = SELECTED.get(t["id"])
        if not chunk_id:
            print(f"[SKIP] {t['id']}: не вибрано чанк")
            continue
        if chunk_id in used:
            print(f"[DUP] {t['id']}: {chunk_id} уже використано")
            continue
        ch = by_id.get(chunk_id)
        if ch is None:
            print(f"[MISS] {t['id']}: {chunk_id} не знайдено серед чистих чанків")
            continue
        used.add(chunk_id)
        items.append({
            "id": t["id"],
            "query": t["query"],
            "relevant_chunk_id": ch["chunk_id"],
            "relevant_chunk_text": ch["text"],
            "source_title": ch["source_title"],
        })

    if not (15 <= len(items) <= 25):
        raise SystemExit(f"Пар {len(items)}, потрібно 15–25. Додайте вибір у SELECTED.")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Збережено {len(items)} пар у {OUT_PATH}")


def main():
    chunks = load_chunks(CORPUS_PATH)

    titles = sorted({c["source_title"] for c in chunks})
    print([t for t in titles if any(k in t for k in ["Hemingway", "Steinbeck", "Faulkner", "Salinger", "Hurston", "Vonnegut", "Fitzgerald"])])
    print(f"Чистих прозових чанків: {len(chunks)}")
    build_review(chunks)
    if SELECTED:
        build_eval_set(chunks)
    else:
        print("Перегляньте candidates_review.json, заповніть SELECTED і запустіть знову.")


if __name__ == "__main__":
    main()
