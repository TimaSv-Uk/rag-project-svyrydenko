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


def run_config(
    eval_items: list[dict],
    collection,
    model: SentenceTransformer,
    normalize: bool = False,
    ef_search: int | None = None,
    use_title_filter: bool = False,
    use_era_filter: bool = False,
    fetch_k: int = 10,
) -> dict:
    if ef_search is not None:
        collection.modify(metadata={"hnsw:ef_search": ef_search})

    queries = [item["query"] for item in eval_items]
    embeddings = model.encode(
        queries,
        batch_size=32,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    ).tolist()

    r5_frac, r5_hit, r10_frac, r10_hit = [], [], [], []

    for item, emb in zip(eval_items, embeddings):
        rel_set = item["relevant_set"]
        if not rel_set:
            continue

        conditions = []
        if use_title_filter and item.get("title"):
            conditions.append({"title": item["title"]})
        if use_era_filter and item.get("era"):
            conditions.append({"era": item["era"]})

        kwargs = {
            "query_embeddings": [emb],
            "n_results": max(10, fetch_k),
        }
        if len(conditions) == 1:
            kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            kwargs["where"] = {"$and": conditions}

        res = collection.query(**kwargs)
        ids_nested = res.get("ids") or [[]]
        ret_ids = ids_nested[0] if ids_nested else []

        top5 = set(ret_ids[:5])
        top10 = set(ret_ids[:10])

        ov5 = len(top5 & rel_set)
        ov10 = len(top10 & rel_set)

        r5_frac.append(ov5 / len(rel_set))
        r5_hit.append(1.0 if ov5 > 0 else 0.0)
        r10_frac.append(ov10 / len(rel_set))
        r10_hit.append(1.0 if ov10 > 0 else 0.0)

    n = max(1, len(r5_frac))
    return {
        "r5_frac": sum(r5_frac) / n,
        "r5_hit": sum(r5_hit) / n,
        "r10_frac": sum(r10_frac) / n,
        "r10_hit": sum(r10_hit) / n,
    }


def main() -> None:
    eval_items = load_eval_set(EVAL_PATH)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION)

    experiments = [
        ("1. Baseline (no norm, default ef)", dict(normalize=False, ef_search=None, use_title_filter=False, use_era_filter=False, fetch_k=10)),
        ("2. Query L2 normalize", dict(normalize=True, ef_search=None, use_title_filter=False, use_era_filter=False, fetch_k=10)),
        ("3. Norm + ef_search=100", dict(normalize=True, ef_search=100, use_title_filter=False, use_era_filter=False, fetch_k=10)),
        ("4. Norm + ef_search=200", dict(normalize=True, ef_search=200, use_title_filter=False, use_era_filter=False, fetch_k=10)),
        ("5. Norm + ef_200 + title filter", dict(normalize=True, ef_search=200, use_title_filter=True, use_era_filter=False, fetch_k=10)),
    ]

    print(f"{'Конфігурація':<32} | {'R@5 frac':>8} | {'R@5 hit':>7} | {'R@10 frac':>9} | {'R@10 hit':>8}")
    print("-" * 76)
    for name, cfg in experiments:
        m = run_config(eval_items, collection, model, **cfg)
        print(f"{name:<32} | {m['r5_frac']:>8.4f} | {m['r5_hit']:>7.4f} | {m['r10_frac']:>9.4f} | {m['r10_hit']:>8.4f}")


if __name__ == "__main__":
    main()
