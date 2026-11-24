import json
import re

INPUT_FILE = "data/clean_american_novelists.jsonl"
OUTPUT_FILE = "data/chunks.jsonl"

CHUNK_SIZE = 250
OVERLAP = 40
MIN_CHUNK_WORDS = 30

HEADING_MARK = re.compile(r"^=+\s*(.+?)\s*=+$")


def is_heading_line(line: str, lines: list[str], i: int) -> bool:
    """Заголовок без розмітки: короткий рядок без розділових знаків у кінці,
    за яким іде абзац тексту."""
    if len(line) > 60 or len(line.split()) > 8:
        return False
    if line[-1] in ".!?:;,\"'”’)-–—":
        return False
    if " – " in line or " - " in line:
        return False
    if not line[0].isupper() and not line[0].isdigit():
        return False
    nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
    return len(nxt) > 60


def split_sections(doc_text: str) -> list[tuple[str, str]]:
    """Розбиває текст документа на (заголовок, тіло) по рядках.
    Підтримує і розмітку == Заголовок ==, і простий рядок-заголовок."""
    sections = []
    heading = "Introduction"
    buf: list[str] = []

    lines = doc_text.split("\n")
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        m = HEADING_MARK.match(line)
        if m:
            name = m.group(1)
        elif is_heading_line(line, lines, i):
            name = line
        else:
            buf.append(line)
            continue

        if buf:
            sections.append((heading, " ".join(buf)))
        heading = name
        buf = []

    if buf:
        sections.append((heading, " ".join(buf)))
    return sections


def chunk_section(heading: str, body: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Ділить тіло секції на чанки, додаючи заголовок на початок кожного."""
    if not body or not body.strip():
        return []

    words = body.split()
    heading_words = heading.split() if heading != "Introduction" else []
    effective_chunk_size = max(50, chunk_size - len(heading_words))

    if len(words) <= effective_chunk_size:
        full_text = f"{heading}\n{body}" if heading != "Introduction" else body
        return [full_text]

    chunks = []
    step = effective_chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + effective_chunk_size, len(words))
        chunk_body = " ".join(words[start:end])

        if heading != "Introduction":
            chunk_text = f"{heading}\n{chunk_body}"
        else:
            chunk_text = chunk_body

        chunks.append(chunk_text)

        if end == len(words):
            break
        start += step

    return chunks


def chunk_data():
    n_docs, n_chunks, n_skipped = 0, 0, 0
    sizes = []

    with (
        open(INPUT_FILE, encoding="utf-8") as fin,
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            n_docs += 1

            sections = split_sections(doc["text"])
            doc_chunk_idx = 0

            for heading, body in sections:
                for chunk_text in chunk_section(heading, body):
                    if len(chunk_text.split()) < MIN_CHUNK_WORDS:
                        n_skipped += 1
                        continue

                    record = {
                        "chunk_id": f"{doc['id']}_c{doc_chunk_idx:04d}",
                        "doc_id": doc["id"],
                        "chunk_index": doc_chunk_idx,
                        "title": doc["title"],
                        "source": doc.get("source") or "hugging face",
                        "heading": heading,
                        "url": doc["url"],
                        "text": chunk_text,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sizes.append(len(chunk_text.split()))
                    n_chunks += 1
                    doc_chunk_idx += 1

    print(f"Документів: {n_docs}, чанків: {n_chunks}, відкинуто коротких: {n_skipped}")
    if sizes:
        print(
            f"Слів у чанку: min={min(sizes)}, max={max(sizes)}, "
            f"avg={sum(sizes) / len(sizes):.0f}"
        )


if __name__ == "__main__":
    chunk_data()
