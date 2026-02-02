"""RAG retriever implementation using BGE-M3 with hybrid search."""

from typing import Any, Dict, List, Optional

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from src.utils.config import MONGO_DB_NAME, get_mongo_client

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between vectors."""
    if a.ndim == 1:
        a = a[None, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


def _sparse_sim(query_sparse: Dict[int, float], doc_sparse: Dict[int, float]) -> float:
    """Calculate sparse similarity (BM25-like) between query and document."""
    if not query_sparse or not doc_sparse:
        return 0.0
    
    score = 0.0
    for token_id, weight in query_sparse.items():
        if token_id in doc_sparse:
            score += weight * doc_sparse[token_id]
    return float(score)


class RAGRetriever:
    """
    RAG retriever using BGE-M3 with hybrid search (dense + sparse embeddings).
    
    Score ranges:
    - Dense score: [-1, 1] (cosine similarity)
    - Final hybrid score: [0, 1] (normalized combination)
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        embedding_field: str = "embedding",
        sparse_field: str = "sparse_embedding",
        top_k: int = 5,
        min_score: Optional[float] = None,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        """Initialize RAG retriever."""
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.sparse_field = sparse_field
        self.top_k = top_k
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.client = get_mongo_client()
        self.db = self.client[MONGO_DB_NAME]
        self.col = self.db[self.collection_name]

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self._load_embeddings_cache()

    def _load_embeddings_cache(self) -> None:
        """Load all embeddings from MongoDB into memory cache."""
        docs = list(self.col.find({self.embedding_field: {"$exists": True}}))
        if not docs:
            raise RuntimeError(
                "No embeddings found in MongoDB. Please run scripts/embed_bge_m3.py first."
            )

        self.doc_ids: List[Any] = [d["_id"] for d in docs]
        self.doc_texts: List[str] = [d.get("text", "") for d in docs]
        self.doc_sources: List[Dict[str, Any]] = [d.get("source", {}) for d in docs]

        emb_list = [np.array(d[self.embedding_field], dtype=np.float32) for d in docs]
        self.embeddings = np.stack(emb_list, axis=0)

        self.sparse_embeddings: List[Dict[int, float]] = []
        if self.use_hybrid:
            for d in docs:
                sparse = d.get(self.sparse_field)
                if sparse and isinstance(sparse, dict):
                    self.sparse_embeddings.append({int(k): float(v) for k, v in sparse.items()})
                else:
                    self.sparse_embeddings.append({})
        
        print(f"Loaded {len(self.doc_ids)} embeddings into RAM for RAG.")
        if self.use_hybrid:
            sparse_count = sum(1 for s in self.sparse_embeddings if s)
            print(f"  - Dense embeddings: {len(self.embeddings)}")
            print(f"  - Sparse embeddings: {sparse_count}/{len(self.sparse_embeddings)}")

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using hybrid search (dense + sparse) or dense-only."""
        if top_k is None:
            top_k = self.top_k

        outputs = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=self.use_hybrid,
            return_colbert_vecs=False,
        )
        q_vec = np.asarray(outputs["dense_vecs"][0], dtype=np.float32)
        dense_sims = _cosine_sim(q_vec, self.embeddings)[0]
        
        if self.use_hybrid and outputs.get("sparse_vecs") and self.sparse_embeddings:
            q_sparse = outputs["sparse_vecs"][0]
            
            sparse_sims = np.array([
                _sparse_sim(q_sparse, doc_sparse) 
                for doc_sparse in self.sparse_embeddings
            ])
            
            # Normalize sparse scores to [0, 1] for combination
            if sparse_sims.max() > sparse_sims.min():
                sparse_sims = (sparse_sims - sparse_sims.min()) / (sparse_sims.max() - sparse_sims.min())
            else:
                sparse_sims = np.full_like(sparse_sims, 0.5)
            
            # Combine dense and sparse scores
            dense_normalized = (dense_sims + 1) / 2
            final_scores = self.dense_weight * dense_normalized + self.sparse_weight * sparse_sims
        else:
            final_scores = (dense_sims + 1) / 2
        
        if self.min_score is not None:
            valid_mask = final_scores >= self.min_score
            if not valid_mask.any():
                top_idx = np.argsort(-final_scores)[:top_k]
            else:
                valid_indices = np.where(valid_mask)[0]
                valid_scores = final_scores[valid_indices]
                top_idx = valid_indices[np.argsort(-valid_scores)[:top_k]]
        else:
            top_idx = np.argsort(-final_scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            results.append(
                {
                    "_id": self.doc_ids[idx],
                    "score": float(final_scores[idx]),
                    "dense_score": float(dense_sims[idx]),
                    "text": self.doc_texts[idx],
                    "source": self.doc_sources[idx],
                }
            )
        return results


def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into a context string for LLM."""
    parts = []
    for i, d in enumerate(retrieved_docs, start=1):
        meta = d.get("source", {})
        link = meta.get("permalink_url") or ""
        dense_score = d.get("dense_score")
        if dense_score is not None:
            parts.append(
                f"[DOC {i}] final_score={d['score']:.3f} (dense={dense_score:.3f})\n"
                f"text: {d['text']}\n"
                f"source: {link}\n"
            )
        else:
            parts.append(
                f"[DOC {i}] score={d['score']:.3f}\n"
                f"text: {d['text']}\n"
                f"source: {link}\n"
            )
    return "\n\n".join(parts)


def build_prompt(user_question: str, context: str) -> str:
    """Build prompt for LLM (compatible with Gemini, OpenAI, Groq, etc.)."""
    return (
        "Ban la tro ly tra loi cau hoi dua tren thong tin duoc cung cap.\n"
        "Neu khong tim duoc cau tra loi trong context thi noi ro la khong ro.\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== CAU HOI NGUOI DUNG ===\n{user_question}\n\n"
        "=== TRA LOI ===\n"
    )

