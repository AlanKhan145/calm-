"""
Memory Agent — quản lý bộ nhớ đa tầng (short-term, long-term, episodic).

- Short-term: Buffer N turns gần nhất (query, response) — context cho reasoning
- Long-term: ChromaMemoryStore (vector DB) — tri thức lâu dài
- Episodic: Log session (query, result, metadata) — phục vụ reflexion, analytics
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Một sự kiện trong session."""

    query: str
    result: dict[str, Any]
    task_type: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
            "result_summary": {
                k: v
                for k, v in self.result.items()
                if k
                in {
                    "task_type",
                    "answer",
                    "risk_level",
                    "confidence",
                    "decision",
                    "approved",
                }
            },
        }


class MemoryAgent:
    """
    Quản lý memory đa tầng cho CALM.

    - Short-term: deque của (query, response) — dùng khi build context
    - Long-term: ChromaMemoryStore — add_texts, similarity_search
    - Episodic: list Episode — session log
    """

    def __init__(
        self,
        long_term_store,
        short_term_size: int = 5,
        episodic_max: int = 100,
    ) -> None:
        """
        Args:
            long_term_store: ChromaMemoryStore (hoặc tương thích).
            short_term_size: Số (query, response) giữ trong short-term.
            episodic_max: Số episode tối đa trong session.
        """
        self.long_term = long_term_store
        self.short_term_size = short_term_size
        self.episodic_max = episodic_max
        self._short_term: deque[tuple[str, Any]] = deque(maxlen=short_term_size)
        self._episodic: list[Episode] = []

    def add_episode(self, query: str, result: dict[str, Any], task_type: str = "") -> None:
        """Ghi một episode vào episodic memory."""
        ep = Episode(query=query, result=result, task_type=task_type)
        self._episodic.append(ep)
        if len(self._episodic) > self.episodic_max:
            self._episodic = self._episodic[-self.episodic_max :]
        logger.debug("[MemoryAgent] Episode added: %s", query[:50])

    def add_short_term(self, query: str, response: Any) -> None:
        """Thêm (query, response) vào short-term buffer."""
        self._short_term.append((query, response))

    def get_short_term_context(self, max_chars: int = 2000) -> str:
        """Lấy context từ short-term (N turns gần nhất)."""
        if not self._short_term:
            return ""
        parts = []
        total = 0
        for q, r in reversed(list(self._short_term)):
            s = f"Q: {q}\nA: {json.dumps(r, default=str)[:500]}"
            if total + len(s) > max_chars:
                break
            parts.append(s)
            total += len(s)
        return "\n\n".join(reversed(parts))

    def get_episodic_summary(self, last_n: int = 5) -> list[dict[str, Any]]:
        """Lấy N episode gần nhất dạng dict."""
        return [e.to_dict() for e in self._episodic[-last_n:]]

    def get_relevant_context(
        self,
        query: str,
        k: int | None = None,
        include_short_term: bool = True,
        short_term_max_chars: int = 1500,
    ) -> dict[str, Any]:
        """
        Hợp nhất context từ long-term + short-term.

        Returns:
            {
                "long_term_docs": [...],  # từ Chroma
                "short_term_text": "...", # từ buffer
                "episodic_summary": [...],  # N episode gần nhất
            }
        """
        long_docs = self.long_term.similarity_search(query, k=k or self.long_term.k)
        out: dict[str, Any] = {
            "long_term_docs": long_docs,
            "short_term_text": "",
            "episodic_summary": [],
        }
        if include_short_term:
            out["short_term_text"] = self.get_short_term_context(
                max_chars=short_term_max_chars
            )
        out["episodic_summary"] = self.get_episodic_summary()
        return out

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Proxy sang long-term store."""
        self.long_term.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> list[Any]:
        """Proxy sang long-term store."""
        return self.long_term.similarity_search(
            query=query, k=k, threshold=threshold
        )
