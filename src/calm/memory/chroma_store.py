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
    ) -> None:
        """
        Khởi tạo với tên collection, thư mục lưu trữ và tham số truy xuất.

        Mặc định use_openai_embeddings=False (dùng HuggingFace, không cần API key).
        """
        self.collection_name = collection_name
        self.persist_directory = str(
            Path(persist_directory).expanduser().resolve()
        )
        self.embedding_model = embedding_model
        self.k = k
        self.similarity_threshold = similarity_threshold
        self.use_openai_embeddings = use_openai_embeddings
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
            from langchain_community.embeddings import HuggingFaceEmbeddings

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

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[Any]:
        """
        Truy vấn tương đồng top-k; lọc theo ngưỡng similarity.

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
        try:
            results = store.similarity_search_with_relevance_scores(
                query, k=k
            )
            filtered = [doc for doc, score in results if score >= threshold]
            return filtered
        except Exception:
            return store.similarity_search(query, k=k)
