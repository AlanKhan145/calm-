"""
File: arxiv_tool.py
Description: ArXiv API (FR-D04). Academic papers. Safety check before call.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ArXivTool:
    """ArXiv search for wildfire/academic papers."""

    def __init__(
        self,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        """Initialize with safety checker and config."""
        self.safety_checker = safety_checker
        self.config = config or {}
        self.max_papers = self.config.get("max_arxiv_papers", 3)

    def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search ArXiv for papers."""
        action = f"ArXiv search: {query}"
        self.safety_checker.check_or_raise(action)
        max_results = max_results or self.max_papers
        try:
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=max_results)
            results = list(client.results(search))
            return [
                {
                    "title": r.title,
                    "summary": r.summary,
                    "url": r.entry_id,
                    "authors": [a.name for a in r.authors],
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("ArXiv search failed: %s", e)
            return []
