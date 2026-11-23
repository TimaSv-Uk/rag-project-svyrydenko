import json
import re

INPUT_FILE = "data/clean_american_novelists.jsonl"
OUTPUT_FILE = "data/chunks.jsonl"

CHUNK_SIZE = 250
OVERLAP = 40


def is_heading(line: str) -> bool:
    """Heuristically determines if a single line is a Wikipedia section heading."""
    line = line.strip()
    if not line:
        return False
    # Headings are short (usually 1-6 words, under 60 chars)
    if len(line) > 60 or len(line.split()) > 8:
        return False
    # Headings don't end with sentence punctuation or list separators
    if line[-1] in ".!?:;,\"'”’)-–—":
        return False
    # Exclude bullet/list items that use dashes (e.g., "West Side Story – 1957")
    if " – " in line or " - " in line:
        return False
    # Must start with an uppercase letter or digit
    if not line[0].isupper() and not line[0].isdigit():
        return False
    return True


def is_categories_block(block: str) -> bool:
    """Detects the trailing Wikipedia categories list (many short lines, no punctuation)."""
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if len(lines) < 5:
        return False
    # Category lines are short, have no ending periods, and rarely contain long prose
    avg_len = sum(len(l) for l in lines) / len(lines)
    no_periods = sum(1 for l in lines if not l.endswith(".")) / len(lines)
    return avg_len < 45 and no_periods > 0.9


def parse_wiki_sections(doc_text: str):
    """Groups paragraphs under their true Wikipedia headings."""
    blocks = doc_text.split("\n\n")
    sections = []
    current_heading = "Introduction"
    current_paragraphs = []

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        # Check if the final block is the Wikipedia categories dump
        if i == len(blocks) - 1 and is_categories_block(block):
            if current_paragraphs or current_heading != "Introduction":
                sections.append((current_heading, "\n\n".join(current_paragraphs)))
            sections.append(("Categories", block))
            return sections

        lines = block.split("\n", 1)
        first_line = lines[0].strip()

        # Case 1: Block starts with a heading (either standalone like "Career" or attached like "Early life\nBorn...")
        if is_heading(first_line):
            # Save the accumulated section before starting the new one
            if current_paragraphs or current_heading != "Introduction":
                sections.append((current_heading, "\n\n".join(current_paragraphs)))

            current_heading = first_line
            current_paragraphs = []

            # If there is body text attached right after the heading via single \n
            if len(lines) > 1 and lines[1].strip():
                current_paragraphs.append(lines[1].strip())
        else:
            # Case 2: Regular paragraph belonging to the current_heading
            current_paragraphs.append(block)

    # Flush the last remaining section
    if current_paragraphs or current_heading != "Introduction":
        sections.append((current_heading, "\n\n".join(current_paragraphs)))

    return sections


def chunk_section(heading: str, body: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Chunks a section's text, preserving the heading context in each chunk."""
    if not body:
        return [heading]

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
    n_docs, n_chunks = 0, 0
    sizes = []

    with (
        open(INPUT_FILE, encoding="utf-8") as fin,
        open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            doc = json.loads(line)
            n_docs += 1

            sections = parse_wiki_sections(doc["text"])
            doc_chunk_idx = 0

            for heading, body in sections:
                # Optional: skip empty sections (like "Directing", "References") or "Categories" if unwanted
                # if not body or heading == "Categories":
                #     continue

                section_chunks = chunk_section(heading, body)

                for chunk_text in section_chunks:
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

    print(f"Документів: {n_docs}, чанків: {n_chunks}")
    print(
        f"Слів у чанку: min={min(sizes)}, max={max(sizes)}, "
        f"avg={sum(sizes) / len(sizes):.0f}"
    )

if __name__ == "__main__":
    chunk_data()
