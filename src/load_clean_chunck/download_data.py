from pathlib import Path
import duckdb
import json
import requests
import pandas as pd

def get_category_members(category_title: str) -> list:
    """Fetch all article titles belonging to a specific Wikipedia category."""
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "RAG-Student-Project/1.0 (timofii.educational.project@gmail.com)"}
    titles = []
    cmcontinue = ""

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_title}",
            "cmlimit": "max",
            "cmtype": "page",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend([m["title"] for m in members])

        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break

    return titles


def download_data():
    target_titles_save = Path("target_titles.json")

    if target_titles_save.exists():
        print(f"Loading cached titles from {target_titles_save}...")
        target_titles = json.loads(target_titles_save.read_text(encoding="utf-8"))
    else:
        print("Fetching category titles from Wikipedia API...")
        cat_19th = get_category_members("19th-century_American_novelists")
        cat_20th = get_category_members("20th-century_American_novelists")
        target_titles = sorted(set(cat_19th + cat_20th))
        target_titles_save.write_text(
            json.dumps(target_titles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Target titles: {len(target_titles)}")

    con = duckdb.connect()
    con.execute("SET memory_limit='2GB';")
    con.execute("SET threads=2;")
    con.execute("SET preserve_insertion_order=false;")

    con.execute("CREATE TABLE target_titles (title VARCHAR PRIMARY KEY);")
    con.executemany("INSERT INTO target_titles VALUES (?)", [(t,) for t in target_titles])

    shards = [r[0] for r in con.execute(
        "SELECT file FROM glob('hf://datasets/wikimedia/wikipedia/20231101.en/*.parquet') ORDER BY file"
    ).fetchall()]
    print(f"Found {len(shards)} shards", flush=True)

    found_titles = set()
    found = 0
    with open("raw_american_novelists.jsonl", "w", encoding="utf-8") as out:
        for i, shard in enumerate(shards, 1):
            print(f"[{i}/{len(shards)}] reading {shard.split('/')[-1]}...", flush=True)
            rows = con.execute(
                """
                SELECT id, url, title, text
                FROM read_parquet(?)
                WHERE title IN (SELECT title FROM target_titles)
                """,
                [shard],
            ).fetchall()
            for id_, url, title, text in rows:
                out.write(json.dumps(
                    {"id": id_, "url": url, "title": title, "text": text},
                    ensure_ascii=False) + "\n")
                found_titles.add(title)
            found += len(rows)
            print(f"    {len(rows)} matches (total {found})", flush=True)

    missing = sorted(set(target_titles) - found_titles)
    print(f"Saved {found} articles. Target: {len(target_titles)}, missing: {len(missing)}")
    Path("missing_titles.txt").write_text("\n".join(missing), encoding="utf-8")
    print("Список відсутніх записано в missing_titles.txt")

if __name__ == "__main__":
    download_data()

