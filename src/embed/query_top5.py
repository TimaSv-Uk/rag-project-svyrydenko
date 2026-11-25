import argparse
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DIR, COLLECTION, MODEL_NAME


def distance_to_similarity(distance: float, space: str = "l2") -> float:
    """Переводить відстань Chroma у оцінку подібності (вище = краще)."""
    if space == "cosine":
        return 1.0 - distance
    if space == "ip":
        return -distance
    return 1.0 / (1.0 + distance)


def search_top_k(query: str, top_k: int = 5, space: str = "l2") -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        [query],
        normalize_embeddings=(space == "cosine"),
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids_list = results.get("ids") or [[]]
    distances_list = results.get("distances") or [[]]
    documents_list = results.get("documents") or [[]]
    metadatas_list = results.get("metadatas") or [[]]

    ids = ids_list[0] if ids_list else []
    distances = distances_list[0] if distances_list else []
    documents = documents_list[0] if documents_list else [""] * len(ids)
    metadatas = metadatas_list[0] if metadatas_list else [{}] * len(ids)

    items = []
    for chunk_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
        similarity = distance_to_similarity(float(dist), space=space)
        snippet = " ".join((doc or "").split())[:220]
        items.append({
            "chunk_id": chunk_id,
            "similarity": similarity,
            "distance": float(dist),
            "title": (meta or {}).get("title") or (meta or {}).get("source_title") or "",
            "snippet": snippet,
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошук top-5 найближчих чанків у Chroma.")
    parser.add_argument("query", type=str, help="Пошуковий запит у лапках")
    parser.add_argument("--top-k", type=int, default=5, help="Кількість результатів (за замовчуванням 5)")
    parser.add_argument(
        "--space",
        type=str,
        default="l2",
        choices=["l2", "cosine", "ip"],
        help="Метрика простору індексу Chroma",
    )
    args = parser.parse_args()

    hits = search_top_k(args.query, top_k=args.top_k, space=args.space)
    print(f'\nЗапит: "{args.query}" (top-{args.top_k})\n' + "-" * 70)
    for rank, h in enumerate(hits, start=1):
        title_part = f" | {h['title']}" if h["title"] else ""
        print(
            f"{rank}. id={h['chunk_id']} | similarity={h['similarity']:.4f} "
            f"(dist={h['distance']:.4f}){title_part}"
        )
        # if h["snippet"]:
        #     print(f"   {h['snippet']}...")


if __name__ == "__main__":
    main()
