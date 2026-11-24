import json
import re
from collections import Counter


STOPWORDS = {"the", "and", "was", "his", "her", "that", "with", "for", "from", "were",
             "had", "has", "she", "him", "their", "this", "which", "also", "after",
             "into", "who", "not", "but", "are", "its", "one", "other", "than", "they"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_headings(path, min_count=300):
    """Count heading-like lines (first line after a blank line)."""
    counts = Counter()
    for doc in read_jsonl(path):
        for part in doc["text"].split("\n\n")[1:]:
            part = part.strip()
            if part:
                counts[part.split("\n")[0].strip()] += 1
    return {h: c for h, c in counts.most_common() if c > min_count}


def chunk_stats(path):
    """Number of chunks and average chunk size in words."""
    sizes = [len(c["text"].split()) for c in read_jsonl(path)]
    return len(sizes), sum(sizes) / len(sizes)


def top_words(path, k=20):
    """Top-k most frequent words, stopwords removed."""
    counter = Counter()
    for doc in read_jsonl(path):
        for w in re.findall(r"[a-z']+", doc["text"].lower()):
            if len(w) >= 3 and w not in STOPWORDS:
                counter[w] += 1
    return counter.most_common(k)


def print_dataset_stats():
    print(count_headings("data/raw_american_novelists.jsonl"))

    n, avg = chunk_stats("data/chunks.jsonl")
    print(f"Chunks: {n}, average size: {avg:.1f} words (~{avg * 1.3:.0f} tokens)")

    for word, count in top_words("data/clean_american_novelists.jsonl"):
        print(word, count)
if __name__ == "__main__":
    print_dataset_stats()
