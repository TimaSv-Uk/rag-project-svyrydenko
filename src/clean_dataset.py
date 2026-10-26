import json


def clean_dataset(data_filename, save_filename):
    kept, dropped_short_count = [], 0
    with open(data_filename, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if len(doc["text"]) < 100:
                dropped_short_count += 1
                continue
            kept.append(doc)

    with open(save_filename, "w", encoding="utf-8") as out:
        for d in kept:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    return dropped_short_count


if __name__ == "__main__":
    print(f"Number of removed short articles: {clean_dataset("raw_american_novelists.jsonl", "clean_american_novelists.jsonl")}")
