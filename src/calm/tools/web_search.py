"""
Mô-đun tìm kiếm web DuckDuckGo (FR-D03).

Mỗi lần gọi search() đều qua SafetyChecker; nếu không an toàn thì ném PermissionError.
Trả về danh sách dict (title, snippet, url).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebSearchTool:
    """DuckDuckGo search for news and reports."""

    def __init__(
        self,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        """Initialize with safety checker and config."""
        self.safety_checker = safety_checker
        self.config = config or {}
        self.max_results = self.config.get("max_news_results", 10)

    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for wildfire-related content."""
        action = f"DuckDuckGo search: {query}"
        self.safety_checker.check_or_raise(action)
        max_results = max_results or self.max_results
        try:
            from duckduckgo_search import DDGS

            results = list(DDGS().text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "url": r.get("href"),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return []
