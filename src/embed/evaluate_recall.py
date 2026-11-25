import argparse
import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DIR, COLLECTION, MODEL_NAME, ROOT

EVAL_PATH = ROOT / "tests" / "eval-set.jsonl"


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line, strict=False)
            rel = row.get("relevant_chunk_id", [])
            if isinstance(rel, str):
                rel = [rel]
            row["relevant_set"] = {str(x) for x in rel if x}
            items.append(row)
    return items


def evaluate(
    eval_items: list[dict],
    collection,
    model: SentenceTransformer,
    ks: tuple[int, ...] = (5, 10),
    normalize: bool = True,
    where_filter: dict | None = None,
    per_item_title: bool = True,
) -> dict:
    max_k = max(ks)
    queries = [item["query"] for item in eval_items]
    embeddings = model.encode(
        queries,
        batch_size=32,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).tolist()

    retrieved_lists: list[list[str]] = []
    if per_item_title:
        for item, emb in zip(eval_items, embeddings):
            query_kwargs: dict = {
                "query_embeddings": [emb],
                "n_results": max_k,
            }
            if item.get("title"):
                query_kwargs["where"] = {"title": item["title"]}
            res = collection.query(**query_kwargs)
            ids_nested = res.get("ids") or [[]]
            retrieved_lists.append(ids_nested[0] if ids_nested else [])
    else:
        query_kwargs = {
            "query_embeddings": embeddings,
            "n_results": max_k,
        }
        if where_filter:
            query_kwargs["where"] = where_filter
        res = collection.query(**query_kwargs)
        retrieved_lists = res.get("ids") or []

    fractional_recalls = {k: [] for k in ks}
    hit_recalls = {k: [] for k in ks}
    misses_at_10 = []

    for item, ret_ids in zip(eval_items, retrieved_lists):
        rel_set = item["relevant_set"]
        if not rel_set:
            continue
        for k in ks:
            top_k_set = set(ret_ids[:k])
            overlap = len(top_k_set & rel_set)
            fractional_recalls[k].append(overlap / len(rel_set))
            hit_recalls[k].append(1.0 if overlap > 0 else 0.0)

        if not (set(ret_ids[:10]) & rel_set):
            misses_at_10.append({
                "id": item.get("id"),
                "query": item["query"],
                "expected": sorted(rel_set),
                "got_top5": ret_ids[:5],
            })

    summary = {}
    for k in ks:
        summary[f"recall@{k}_fractional"] = sum(fractional_recalls[k]) / len(fractional_recalls[k])
        summary[f"recall@{k}_hit"] = sum(hit_recalls[k]) / len(hit_recalls[k])
    summary["misses_at_10"] = misses_at_10
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Оцінка Recall@5 та Recall@10 для індексу Chroma.")
    parser.add_argument("--eval-file", type=Path, default=EVAL_PATH)
    parser.add_argument("--no-normalize", action="store_true", help="Вимкнути L2-нормалізацію запитів")
    parser.add_argument("--ef-search", type=int, default=200, help="Параметр HNSW ef_search (0 — не змінювати)")
    parser.add_argument("--filter-era", type=str, default="", help="Опційний фільтр метаданих era")
    parser.add_argument("--filter-title", type=str, default="", help="Фіксований фільтр метаданих title для всіх запитів")
    parser.add_argument("--no-title-filter", action="store_true", help="Шукати по всій базі без фільтра за title запиту")
    args = parser.parse_args()

    eval_items = load_eval_set(args.eval_file)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION)

    if args.ef_search > 0:
        updated_meta = {**(collection.metadata or {}), "hnsw:ef_search": args.ef_search}
        collection.modify(metadata=updated_meta)

    print(f"Колекція: {COLLECTION} | Всього чанків у базі: {collection.count()}")
    print(f"Метадані індексу: {collection.metadata}")

    conditions = []
    if args.filter_era:
        conditions.append({"era": args.filter_era})
    if args.filter_title:
        conditions.append({"title": args.filter_title})

    where_clause = None
    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}

    per_item_title = not args.no_title_filter and not args.filter_title

    metrics = evaluate(
        eval_items,
        collection,
        model,
        ks=(5, 10),
        normalize=not args.no_normalize,
        where_filter=where_clause,
        per_item_title=per_item_title,
    )

    print(f"Кількість запитів: {len(eval_items)}")
    print(f"Recall@5  (частковий / хіт): {metrics['recall@5_fractional']:.4f} / {metrics['recall@5_hit']:.4f}")
    print(f"Recall@10 (частковий / хіт): {metrics['recall@10_fractional']:.4f} / {metrics['recall@10_hit']:.4f}")
    print(f"Пропущено навіть у top-10: {len(metrics['misses_at_10'])}")
    for m in metrics["misses_at_10"]:
        print(f"  - [{m['id']}] expected={m['expected']} | got_top5={m['got_top5']}")


if __name__ == "__main__":
    main()
