"""
Heuristic intent từ text query (khi router/planner không đủ tin cậy).

Ưu tiên tín hiệu dự báo (predict / next N days / wildfire risk) trước tín hiệu QA
để tránh nhầm câu hỏi dạng “what are the risks of …” sang prediction.
"""

from __future__ import annotations

import re
from typing import List

_PREDICTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(?:predict|forecast|forecasting)\b", re.I),
    re.compile(r"\bnext\s+\d+\s+days?\b", re.I),
    re.compile(r"\bnext\s+\d+\s+weeks?\b", re.I),
    re.compile(r"\bnext\s+week\b", re.I),
    re.compile(r"\b(?:likelihood|probability)\b", re.I),
    re.compile(r"\bwildfire\s+risk\b", re.I),
    re.compile(r"\bfire\s+danger\b", re.I),
    re.compile(r"\bfire\s+risk\b", re.I),
    re.compile(r"\brisk\s+(?:for|in|over|around)\b", re.I),
    re.compile(r"\b(?:assess|assessment|estimate)\s+(?:the\s+)?(?:wildfire\s+)?risk\b", re.I),
]

_QA_HINT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+(?:are|is|were|was|do|does|did)\b", re.I),
    re.compile(r"\bwhy\b", re.I),
    re.compile(r"\bhow\s+(?:does|do|can|could|would|is|are|to)\b", re.I),
    re.compile(r"\bexplain\b", re.I),
    re.compile(r"\bdescribe\b", re.I),
    re.compile(r"\bcauses?\s+of\b", re.I),
    re.compile(r"\binformation\s+about\b", re.I),
]


def query_suggests_prediction(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return any(p.search(q) for p in _PREDICTION_PATTERNS)


def query_suggests_qa(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    return any(p.search(q) for p in _QA_HINT_PATTERNS)


def infer_task_from_keywords(query: str) -> str:
    """Trả về 'prediction' hoặc 'qa' dựa trên pattern; mặc định qa."""
    if query_suggests_prediction(query):
        return "prediction"
    if query_suggests_qa(query):
        return "qa"
    return "qa"
