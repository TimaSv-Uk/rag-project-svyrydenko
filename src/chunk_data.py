import json

INPUT_FILE = "clean_american_novelists.jsonl"
OUTPUT_FILE = "chunks.jsonl"

CHUNK_SIZE = 250
OVERLAP = 40
STEP = CHUNK_SIZE - OVERLAP


def chunk_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    assert 0 <= overlap < chunk_size
    step = chunk_size - overlap
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


if __name__ == "__main__":
    n_docs, n_chunks = 0, 0
    sizes = []

    with (
        open(INPUT_FILE, encoding="utf-8") as fin,
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            doc = json.loads(line)
            n_docs += 1

            for idx, chunk_text in enumerate(chunk_words(doc["text"])):
                record = {
                    "chunk_id": f"{doc['id']}_c{idx:04d}",  # стабільний глобальний ID
                    "doc_id": doc["id"],  # документ-джерело
                    "chunk_index": idx,  # порядковий номер у документі
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": chunk_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                sizes.append(len(chunk_text.split()))
                n_chunks += 1

    print(f"Документів: {n_docs}, чанків: {n_chunks}")
    print(
        f"Слів у чанку: min={min(sizes)}, max={max(sizes)}, "
        f"avg={sum(sizes) / len(sizes):.0f}"
    )
