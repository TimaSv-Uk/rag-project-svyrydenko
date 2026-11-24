import json
import numpy as np
import chromadb
from src.config import CHROMA_DIR, COLLECTION, EMB_PATH, IDS_PATH

emb = np.load(EMB_PATH)
ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION)

print("форма emb:", emb.shape, "норми:", np.linalg.norm(emb[:3], axis=1))

for i in [3, 12, 16, 17]:
    got = col.get(ids=[ids[i]], include=["embeddings"])
    stored = np.array(got["embeddings"][0], dtype=np.float32)
    cos_stored = float(np.dot(stored, emb[i]) / (np.linalg.norm(stored) * np.linalg.norm(emb[i])))

    sims = emb @ emb[i]                      # точний пошук по всьому корпусу
    exact_top = np.argsort(-sims)[:3]
    print(f"{ids[i]}: cos(stored, emb)={cos_stored:.4f}, "
          f"точний top-3={[ids[j] for j in exact_top]}, sims={[round(float(sims[j]),4) for j in exact_top]}")
