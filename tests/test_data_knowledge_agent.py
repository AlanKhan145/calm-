"""
File: test_data_knowledge_agent.py
Description: Tests for Data & Knowledge Agent — dedup (FR-D05), dict output.
Author: CALM Team
Created: 2026-03-13
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calm.agents.data_knowledge_agent import DataKnowledgeAgent


def test_data_agent_returns_dict():
    """NP-3.4: Never return DataFrame; must be dict/list."""
    memory = MagicMock()
    memory.similarity_search.return_value = []
    memory.add_texts.return_value = None

    agent = DataKnowledgeAgent(
        llm=MagicMock(),
        tools={},
        memory_store=memory,
        config={"dedup_check": False},
    )
    result = agent.collect("wildfire risk California")
    assert isinstance(result, dict)
    assert "retrieved_data" in result or "dedup" in result
