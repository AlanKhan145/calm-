"""
File: chroma_store.py
Description: ChromaDB memory store — Reflexion framework (RULE 6).
             Separate collections, text only, persist_directory required.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """ChromaDB-backed memory. Text only. Persist directory required (RULE 6)."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path,
        embedding_model: str = "text-embedding-3-small",
        k: int = 3,
        similarity_threshold: float = 0.65,
        use_openai_embeddings: bool = False,
    ) -> None:
        """Initialize with collection name, persist path, and retrieval params.
        Mặc định use_openai_embeddings=False — dùng HuggingFace (không cần API key).
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
        """Lazy init. Mặc định HuggingFace (không API key). Set use_openai_embeddings=True + OPENAI_API_KEY để dùng OpenAI."""
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
        """Store verbal reinforcement text only (no arrays)."""
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
        """Top-k retrieval. Returns empty list when no match (NP-7.6)."""
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
