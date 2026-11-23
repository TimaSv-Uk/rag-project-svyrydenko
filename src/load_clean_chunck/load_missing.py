import json
import time
from collections import Counter
from pathlib import Path

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Student-Project/1.0 (timofii.educational.project@gmail.com)"}
MISSING = Path("missing_titles.txt")
SRC = Path("raw_american_novelists.jsonl")          # дамп, лише для дедуплікації
OUT = Path("raw_american_novelists_api.jsonl")      # нові статті з API
SKIPPED_LOG = Path("skipped_api.jsonl")
DELAY = 0.5
RETRIES = 2


def fetch_one(title: str) -> dict | None:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", [])
    return pages[0] if pages else None


def fetch_with_retry(title: str) -> dict | None:
    for attempt in range(RETRIES + 1):
        p = fetch_one(title)
        if p and (p.get("missing") or p.get("extract")):
            return p
        time.sleep(DELAY * (attempt + 1))
    return p  # останнє значення, навіть порожнє


def load_existing_titles(path: Path) -> set[str]:
    titles = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                titles.add(json.loads(line)["title"])
    return titles

def append_api_to_source(src_path: Path = SRC, api_path: Path = OUT) -> int:
    """Appends newly fetched API JSONL records into the existing JSONL file."""
    if not api_path.exists():
        print(f"Файл {api_path} не знайдено, нічого додавати.")
        return 0

    existing_titles = load_existing_titles(src_path) if src_path.exists() else set()
    added_count = 0

    # Ensure the target file ends with '\n' before appending
    if src_path.exists() and src_path.stat().st_size > 0:
        with src_path.open("rb") as f:
            f.seek(-1, 2)
            last_byte = f.read(1)
        if last_byte != b"\n":
            with src_path.open("a", encoding="utf-8") as f:
                f.write("\n")

    with (
        api_path.open(encoding="utf-8") as fin,
        src_path.open("a", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title")
            if title and title not in existing_titles:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_titles.add(title)
                added_count += 1

    print(f"Додано {added_count} статей з {api_path} у {src_path}.")
    return added_count

def load_missing():
    existing = load_existing_titles(SRC)
    titles = [t.strip() for t in MISSING.read_text(encoding="utf-8").splitlines() if t.strip()]
    print(f"Запитів: {len(titles)}, у дампі вже є: {len(existing)}", flush=True)

    reasons = Counter()
    skipped_rows = []
    seen_in_run = set()

    with OUT.open("w", encoding="utf-8") as out:
        for i, title in enumerate(titles, 1):
            p = fetch_with_retry(title)
            if p is None:
                reason, canon = "no_response", title
            else:
                canon = p.get("title", title)
                if p.get("invalid"):
                    reason = "invalid"
                elif p.get("missing"):
                    reason = "missing_page"
                elif not p.get("extract"):
                    reason = "empty_extract"
                elif canon in existing:
                    reason = "duplicate_in_dump"
                elif canon in seen_in_run:
                    reason = "duplicate_in_api"
                else:
                    reason = "added"

            reasons[reason] += 1
            if reason == "added" and p is not None:
                seen_in_run.add(canon)
                out.write(json.dumps({
                    "id": str(p["pageid"]),
                    "url": p["fullurl"] ,
                    "title": canon,
                    "text": p["extract"],
                    "source": "wikipedia-api",
                }, ensure_ascii=False) + "\n")
            else:
                skipped_rows.append({"requested": title, "title": canon, "reason": reason})

            if i % 50 == 0 or i == len(titles):
                print(f"[{i}/{len(titles)}] {dict(reasons)}", flush=True)
            time.sleep(DELAY)

    with SKIPPED_LOG.open("w", encoding="utf-8") as f:
        for row in skipped_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nПідсумок:")
    for reason, n in reasons.most_common():
        print(f"  {reason}: {n}")
    print(f"Статті: {OUT}, пропуски: {SKIPPED_LOG}")

    append_api_to_source(SRC, OUT)

if __name__ == "__main__":
    load_missing()
