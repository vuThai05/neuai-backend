"""Script to generate embeddings for knowledge base using BGE-M3 model."""

from typing import List

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from src.utils.config import MONGO_DB_NAME, get_mongo_client


def embed_knowledge_base(
    batch_size: int = 16,
    collection_name: str = "knowledge_base",
    embedding_field: str = "embedding",
    sparse_field: str = "sparse_embedding",
    use_sparse: bool = True,
) -> int:
    """
    Generate embeddings for knowledge base using BGE-M3 model.
    
    Hybrid search (dense + sparse) typically provides better results than dense-only.
    """
    client = get_mongo_client()
    db = client[MONGO_DB_NAME]
    kb_col = db[collection_name]

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    cursor = kb_col.find({})
    docs = list(cursor)
    if not docs:
        print("No documents found in collection 'knowledge_base'. Please run scripts/index_mongo.py first.")
        return 0

    total = len(docs)
    print(f"Starting embedding generation for {total} documents using BGE-M3...")
    if use_sparse:
        print("  - Dense embeddings: ON")
        print("  - Sparse embeddings: ON (hybrid search)")

    updated = 0
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        texts: List[str] = [str(d.get("text", "")) for d in batch]

        outputs = model.encode(
            texts,
            return_dense=True,
            return_sparse=use_sparse,
            return_colbert_vecs=False,
        )
        dense_vectors = outputs["dense_vecs"]
        sparse_vectors = outputs.get("sparse_vecs", []) if use_sparse else []

        for idx, (doc, vec) in enumerate(zip(batch, dense_vectors)):
            # Convert to float32 to reduce storage size
            vec32 = np.asarray(vec, dtype=np.float32).tolist()
            
            update_data = {
                embedding_field: vec32,
                "embedding_dim": len(vec32),
                "embedding_model": "BAAI/bge-m3",
            }
            
            if use_sparse and sparse_vectors:
                sparse_dict = sparse_vectors[idx]
                sparse_dict_clean = {int(k): float(v) for k, v in sparse_dict.items()}
                update_data[sparse_field] = sparse_dict_clean
            
            kb_col.update_one(
                {"_id": doc["_id"]},
                {"$set": update_data},
            )
            updated += 1

        print(f"Processed {min(i + batch_size, total)}/{total} documents")

    print(f"Hoan thanh. Da cap nhat embedding cho {updated} documents.")
    return updated


if __name__ == "__main__":
    embed_knowledge_base()

