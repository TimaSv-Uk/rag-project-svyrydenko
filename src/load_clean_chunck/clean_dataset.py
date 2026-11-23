import json
from pathlib import Path


def clean_dataset(data_filename: str, save_filename: str) -> int:
    kept, dropped_short_count = [], 0
    with open(data_filename, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text") or ""
            if len(text) < 100:
                dropped_short_count += 1
                continue
            kept.append(doc)

    out_path = Path(save_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for d in kept:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    return dropped_short_count


if __name__ == "__main__":
    removed = clean_dataset(
        "data/raw_american_novelists.jsonl",
        "data/clean_american_novelists.jsonl",
    )
    print(f"Number of removed short articles: {removed}")
