import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION, MODEL_NAME

TOP_K = 5

_model = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


def search(query: str, k: int = TOP_K) -> list[dict]:
    """Повертає k найближчих чанків. score = косинусна подібність (1 - відстань Chroma)."""
    q_emb = get_model().encode(
        [query],
        normalize_embeddings=True,   # та сама нормалізація, що й при індексації
        convert_to_numpy=True,
    )[0].tolist()

    res = get_collection().query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "chunk_id": cid,
            "score": round(1.0 - dist, 4),
            "title": meta.get("title", ""),
            "heading": meta.get("heading", ""),
            "url": meta.get("url", ""),
            "text": doc,
        })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошук top-k чанків за запитом")
    parser.add_argument("query", nargs="+", help="Текст запиту (можна без лапок)")
    parser.add_argument("-k", type=int, default=TOP_K, help="Кількість результатів")
    parser.add_argument("--snippet", type=int, default=200, help="Довжина фрагмента тексту")
    parser.add_argument("--json", action="store_true", help="Вивід у форматі JSON")
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        raise SystemExit("Порожній запит")

    hits = search(query, args.k)

    if args.json:
        print(json.dumps({"query": query, "results": hits}, ensure_ascii=False, indent=2))
        return

    print(f"Запит: {query}\n")
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:args.snippet].replace("\n", " ")
        print(f"{i}. {h['chunk_id']}  score={h['score']:.4f}  [{h['title']} / {h['heading']}]")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
