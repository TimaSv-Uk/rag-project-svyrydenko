import json

import chromadb
import numpy as np

from src.config import CHROMA_DIR, COLLECTION, EMB_PATH, IDS_PATH, ADD_BATCH
from src.embed.embed import load_chunks, get_text, get_id


def build_metadata(chunk: dict) -> dict:
    """Метадані чанка з Лабораторної 2. Chroma приймає лише str, int, float, bool."""
    meta = {}
    for key, value in chunk.items():
        if key in ("text", "content", "chunk_text"):
            continue
        if value is None:
            meta[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = json.dumps(value, ensure_ascii=False)
    return meta


def build_index() -> chromadb.Collection:
    chunks = load_chunks()
    emb = np.load(EMB_PATH)
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))

    if len(chunks) != len(emb) or ids != [get_id(c) for c in chunks]:
        raise ValueError("embeddings не відповідають chunks.jsonl, запустіть етап embed заново")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)   # повний перебудова індексу
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), ADD_BATCH):
        end = min(start + ADD_BATCH, len(chunks))
        batch = chunks[start:end]
        collection.add(
            ids=ids[start:end],
            embeddings=emb[start:end].tolist(),
            documents=[get_text(c) for c in batch],
            metadatas=[build_metadata(c) for c in batch],
        )
        print(f"Додано {end}/{len(chunks)}", flush=True)

    print(f"Індекс '{COLLECTION}' містить {collection.count()} векторів у {CHROMA_DIR}")
    return collection


def sanity_check(n: int = 20) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION)
    emb = np.load(EMB_PATH)
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    texts = {}
    for c in load_chunks():
        texts[get_id(c)] = get_text(c)

    ok = 0
    for i in range(n):
        res = col.query(query_embeddings=[emb[i].tolist()], n_results=5)
        top_ids = res["ids"][0]
        if ids[i] in top_ids or any(texts[t] == texts[ids[i]] for t in top_ids[:1]):
            ok += 1
    print(f"Перевірка: {ok}/{n}")
