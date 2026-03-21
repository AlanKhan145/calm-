"""
BGE Cross-Encoder Reranker — cải thiện chất lượng retrieval.

Dùng BAAI/bge-reranker-base (hoặc -large) để rerank kết quả từ vector search.
Chạy local, không cần API key.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RERANKER: Any = None


def get_reranker(model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = True) -> Any | None:
    """
    Lazy load BGE reranker. Trả về None nếu FlagEmbedding chưa cài.
    """
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        from FlagEmbedding import FlagReranker

        _RERANKER = FlagReranker(model_name, use_fp16=use_fp16)
        logger.info("[Reranker] Loaded %s", model_name)
        return _RERANKER
    except ImportError:
        logger.debug(
            "[Reranker] FlagEmbedding not installed. pip install FlagEmbedding"
        )
        return None


def rerank(
    query: str,
    documents: list[Any],
    top_k: int | None = None,
    model_name: str = "BAAI/bge-reranker-base",
) -> list[tuple[Any, float]]:
    """
    Rerank documents theo relevance với query.

    Args:
        query: Câu truy vấn.
        documents: Danh sách Document (LangChain) hoặc chuỗi.
        top_k: Số kết quả giữ lại (mặc định giữ tất cả).
        model_name: Tên model BGE reranker.

    Returns:
        List[(doc, score)] đã sắp xếp theo score giảm dần.
        Score trong [0, 1] (sigmoid từ logits).
    """
    reranker_obj = get_reranker(model_name=model_name)
    if reranker_obj is None:
        return [(d, 1.0) for d in documents]

    if not documents:
        return []

    # Lấy page_content từ Document
    texts = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in documents
    ]

    try:
        pairs = [[query, t] for t in texts]
        scores = reranker_obj.compute_score(pairs, normalize=True)
        if hasattr(scores, "tolist"):
            norm_scores = scores.tolist()
        elif isinstance(scores, (int, float)):
            norm_scores = [float(scores)]
        else:
            norm_scores = list(scores)
    except Exception as e:
        logger.warning("[Reranker] Failed: %s", e)
        return [(d, 1.0) for d in documents]

    doc_scores = list(zip(documents, norm_scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        doc_scores = doc_scores[:top_k]
    return doc_scores
