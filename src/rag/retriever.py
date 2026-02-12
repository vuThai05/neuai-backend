"""RAG retriever implementation using BGE-M3 with hybrid search."""

from typing import Any, Dict, List, Optional

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient

from src.utils.config import MONGO_DB_NAME, get_mongo_client, get_qdrant_client, QDRANT_COLLECTION_NAME


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
    
    Data flow:
    - Posts/comments source: MongoDB (Postandcmt DB)
    - Vector database: Qdrant (knowledge_base collection)
    
    Score ranges:
    - Dense score: [-1, 1] (cosine similarity)
    - Final hybrid score: [0, 1] (normalized combination)
    """

    def __init__(
        self,
        collection_name: str = None,
        top_k: int = 5,
        min_score: Optional[float] = None,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        """Initialize RAG retriever."""
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.top_k = top_k
        self.min_score = min_score
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Qdrant client for vector database
        self.qdrant_client = get_qdrant_client()
        
        # Load model
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        
        # Load embeddings and comments from Qdrant
        self._load_embeddings_cache()
        self._load_comments_cache()

    def _load_embeddings_cache(self) -> None:
        """Load all embeddings from Qdrant into memory cache."""
        # Scroll through all points in Qdrant collection
        points, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Lấy tối đa 10k points
            with_payload=True,
            with_vectors=True,
        )
        
        if not points:
            raise RuntimeError(
                f"No embeddings found in Qdrant collection '{self.collection_name}'. "
                "Please run scripts/index_mongo.py and scripts/embed_bge_m3.py first."
            )
        
        # Filter out points with zero vectors (chưa được embed)
        valid_points = []
        for p in points:
            if p.vector and isinstance(p.vector, list) and sum(p.vector) != 0:
                valid_points.append(p)
        
        if not valid_points:
            raise RuntimeError(
                f"Found {len(points)} points in Qdrant but none have embeddings. "
                "Please run scripts/embed_bge_m3.py to generate embeddings."
            )
        
        # Extract data from Qdrant points
        self.point_ids: List[int] = [p.id for p in valid_points]
        self.doc_ids: List[str] = [p.payload.get("doc_id", "") for p in valid_points]
        self.doc_texts: List[str] = [p.payload.get("text", "") for p in valid_points]
        self.doc_sources: List[Dict[str, Any]] = [p.payload.get("source", {}) for p in valid_points]
        
        # Stack vectors into numpy array
        emb_list = [np.array(p.vector, dtype=np.float32) for p in valid_points]
        self.embeddings = np.stack(emb_list, axis=0)
        
        # Load sparse embeddings if hybrid mode
        self.sparse_embeddings: List[Dict[int, float]] = []
        if self.use_hybrid:
            for p in valid_points:
                sparse = p.payload.get("sparse_embedding")
                if sparse and isinstance(sparse, dict):
                    self.sparse_embeddings.append({int(k): float(v) for k, v in sparse.items()})
                else:
                    self.sparse_embeddings.append({})
        
        print(f"Loaded {len(self.doc_ids)} embeddings from Qdrant into RAM for RAG.")
        if self.use_hybrid:
            sparse_count = sum(1 for s in self.sparse_embeddings if s)
            print(f"  - Dense embeddings: {len(self.embeddings)}")
            print(f"  - Sparse embeddings: {sparse_count}/{len(self.sparse_embeddings)}")

    def _load_comments_cache(self) -> None:
        """Load comments mapping from Qdrant: post_id -> list of comments."""
        # Load tất cả points từ Qdrant (không filter để tránh cần index)
        points, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        
        self.comments_by_post: Dict[str, List[Dict[str, Any]]] = {}
        
        # Filter comments trong Python
        for point in points:
            payload = point.payload
            # Chỉ lấy points có type="comment"
            if payload.get("type") == "comment":
                post_id = payload.get("source", {}).get("post_id")
                if post_id:
                    if post_id not in self.comments_by_post:
                        self.comments_by_post[post_id] = []
                    self.comments_by_post[post_id].append({
                        "text": payload.get("text", ""),
                        "comment_id": payload.get("source", {}).get("comment_id"),
                    })
        
        total_comments = sum(len(comments) for comments in self.comments_by_post.values())
        print(f"Loaded {total_comments} comments from Qdrant for {len(self.comments_by_post)} posts.")

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

    def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Lấy post từ post_id trong Qdrant knowledge_base."""
        doc_id = f"post::{post_id}"
        
        # Tìm trong cache trước
        try:
            idx = self.doc_ids.index(doc_id)
            return {
                "_id": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "source": self.doc_sources[idx],
                "score": 1.0,  # Default score khi query trực tiếp
            }
        except ValueError:
            # Nếu không có trong cache, query từ Qdrant
            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
                scroll_filter={
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}}
                    ]
                }
            )
            
            if points:
                payload = points[0].payload
                return {
                    "_id": payload.get("doc_id"),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", {}),
                    "score": 1.0,
                }
        
        return None


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


def build_single_context(doc: Dict[str, Any], retriever: Optional["RAGRetriever"] = None) -> str:
    """
    Format a single document (post) and its comments into context string for LLM.
    
    Args:
        doc: Document dictionary (should be a post)
        retriever: RAGRetriever instance to access comments cache
    """
    meta = doc.get("source", {})
    link = meta.get("permalink_url") or ""
    post_id = meta.get("post_id")
    
    context_parts = [
        "=== BAI VIET ===",
        f"text: {doc['text']}",
        f"source: {link}",
    ]
    
    # Lấy comments của bài viết này nếu có retriever và post_id
    comments_text = []
    if retriever and post_id and hasattr(retriever, 'comments_by_post'):
        comments = retriever.comments_by_post.get(post_id, [])
        if comments:
            comments_text.append("\n=== COMMENTS ===")
            for i, cmt in enumerate(comments, start=1):
                comment_text = cmt.get("text", "").strip()
                if comment_text and comment_text != "[NO_MESSAGE]":
                    comments_text.append(f"Comment {i}: {comment_text}")
    
    if comments_text:
        context_parts.extend(comments_text)
    
    return "\n".join(context_parts)


def build_prompt(user_question: str, context: str) -> str:
    """Build prompt for LLM (compatible with Gemini, OpenAI, Groq, etc.)."""
    return (
        "Ban la tro ly tra loi cau hoi CHI DUA TREN noi dung trong context duoc cung cap.\n"
        "YEU CAU:\n"
        "- Chi su dung thong tin co trong context (BAI VIET va COMMENTS), TUYET DOI khong duoc suy doan hoac bia noi dung.\n"
        "- Neu BAI VIET va COMMENTS co nhieu quan diem khac nhau, hay tong hop va phan anh ca hai phia mot cach ngan gon.\n"
        "- Noi dung tu COMMENTS chi duoc su dung khi thuoc dung bai viet dang duoc tra loi.\n"
        "- Tra loi ngan gon, day du y chinh, toi uu token, khong lan man.\n"
        "- Neu khong tim duoc cau tra loi trong context (bao gom ca BAI VIET va COMMENTS), hay tra loi: 'Hiện chưa có dữ liệu để trả lời câu hỏi này.'\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== CAU HOI NGUOI DUNG ===\n{user_question}\n\n"
        "=== TRA LOI ===\n"
    )

