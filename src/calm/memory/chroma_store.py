"""
Mô-đun bộ nhớ vector ChromaDB — Reflexion framework (RULE 6).

Lưu trữ văn bản theo collection; mỗi agent có thể dùng collection riêng.
Bắt buộc persist_directory. Embedding mặc định HuggingFace; tùy chọn OpenAI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """
    Bộ nhớ vector dựa trên ChromaDB; chỉ lưu văn bản; bắt buộc persist_directory.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path,
        embedding_model: str = "text-embedding-3-small",
        k: int = 3,
        similarity_threshold: float = 0.65,
        use_openai_embeddings: bool = False,
        use_reranker: bool = False,
        rerank_top_k: int | None = None,
        score_fusion_embedding_weight: float = 0.7,
    ) -> None:
        """
        Khởi tạo với tên collection, thư mục lưu trữ và tham số truy xuất.

        Mặc định use_openai_embeddings=False (dùng HuggingFace, không cần API key).
        use_reranker: Bật BGE cross-encoder reranker.
        rerank_top_k: Số doc đưa vào rerank (mặc định k*5).
        score_fusion_embedding_weight: Trọng số embedding trong fusion (0.7 = 70% embedding + 30% rerank).
        """
        self.collection_name = collection_name
        self.persist_directory = str(
            Path(persist_directory).expanduser().resolve()
        )
        self.embedding_model = embedding_model
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.use_openai_embeddings = use_openai_embeddings
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k or max(k * 5, 15)
        self.score_fusion_embedding_weight = score_fusion_embedding_weight
        self._client = None
        self._collection = None

    def _get_collection(self):
        """Khởi tạo lazy collection; mặc định HuggingFace; set use_openai_embeddings=True và OPENAI_API_KEY để dùng OpenAI."""
        if self._collection is not None:
            return self._collection
        import os

        from langchain_chroma import Chroma

        use_openai = (
            self.use_openai_embeddings and bool(os.environ.get("OPENAI_API_KEY"))
        )
        if use_openai:
            try:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(model=self.embedding_model)
            except Exception as e:
                logger.warning("OpenAIEmbeddings failed (%s), using HuggingFace", e)
                use_openai = False
        if not use_openai:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import (  # fallback if langchain-huggingface not installed
                    HuggingFaceEmbeddings,
                )

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        self._collection = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )
        return self._collection

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Lưu danh sách văn bản vào collection (chỉ text, không mảng)."""
        if not texts:
            return
        store = self._get_collection()
        metadatas = metadatas or [{}] * len(texts)
        if len(metadatas) != len(texts):
            metadatas = [{}] * len(texts)
        store.add_texts(texts=texts, metadatas=metadatas)

    def _distance_to_relevance(self, score: float) -> float:
        """
        Chuyển score từ Chroma về relevance [0, 1].
        Chroma có thể trả về:
        - Cosine similarity raw [-1, 1] (từ relevance_scores)
        - Cosine distance [0, 2] hoặc L2 (từ similarity_search_with_score)
        Hàm xử lý cả hai để luôn có relevance trong [0, 1].
        """
        if -1.0 <= score <= 1.0:
            # Cosine similarity [-1, 1] -> [0, 1]
            return (score + 1.0) / 2.0
        if 0 <= score <= 2.0:
            # Cosine distance [0, 2] -> [0, 1] (0=giống, 2=khác)
            return max(0.0, 1.0 - score / 2.0)
        # L2 hoặc metric khác: clip về [0, 1]
        return max(0.0, min(1.0, 1.0 / (1.0 + score)))

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[Any]:
        """
        Truy vấn tương đồng top-k; lọc theo ngưỡng similarity.
        Hỗ trợ BGE reranker + score fusion khi use_reranker=True.

        Tham số:
            query: Câu truy vấn.
            k: Số kết quả (mặc định self.k).
            threshold: Ngưỡng điểm (mặc định self.similarity_threshold).

        Trả về:
            Danh sách tài liệu thỏa ngưỡng; rỗng nếu không có kết quả.
        """
        store = self._get_collection()
        k = k or self.k
        threshold = (
            threshold if threshold is not None else self.similarity_threshold
        )
        fetch_k = self.rerank_top_k if self.use_reranker else k
        try:
            results = store.similarity_search_with_score(query, k=fetch_k)
            normalized = [
                (doc, self._distance_to_relevance(score))
                for doc, score in results
            ]
            if self.use_reranker and normalized:
                from calm.memory.reranker import rerank

                docs_for_rerank = [d for d, _ in normalized]
                emb_scores = {id(d): s for d, s in normalized}
                reranked = rerank(
                    query=query,
                    documents=docs_for_rerank,
                    top_k=min(k * 2, len(normalized)),
                )
                w_emb = self.score_fusion_embedding_weight
                w_rerank = 1.0 - w_emb
                fused = [
                    (
                        doc,
                        w_emb * emb_scores.get(id(doc), 0.5) + w_rerank * r_score,
                    )
                    for doc, r_score in reranked
                ]
                fused.sort(key=lambda x: x[1], reverse=True)
                normalized = fused[:k]
            else:
                normalized = normalized[:k]
            filtered = [doc for doc, rel in normalized if rel >= threshold]
            return filtered
        except Exception:
            return store.similarity_search(query, k=k)
