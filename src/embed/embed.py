import json

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, DATA_DIR, EMB_PATH, IDS_PATH, MODEL_NAME, EMBED_BATCH


def load_chunks(path=CHUNKS_PATH) -> list[dict]:
    """Повертає всі чанки з chunks.jsonl у незмінному порядку (без фільтрації)."""
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_text(chunk: dict) -> str:
    return chunk.get("text") or chunk.get("content") or chunk.get("chunk_text") or ""


def get_id(chunk: dict) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id"))


def embed_all() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    chunks = load_chunks()
    ids = [get_id(c) for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("chunk_id не унікальні, перевірте chunks.jsonl")

    print(f"Завантаження моделі {MODEL_NAME}...", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    print(f"Максимальна довжина входу: {model.max_seq_length} токенів")

    texts = [get_text(c) for c in chunks]
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,   # для косинусної подібності
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(EMB_PATH, emb)
    IDS_PATH.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    print(f"Збережено embeddings {emb.shape} у {EMB_PATH}")
