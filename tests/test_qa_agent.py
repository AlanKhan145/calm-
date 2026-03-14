"""
File: test_qa_agent.py
Description: Tests for QA Agent.
Author: CALM Team
Created: 2026-03-13
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calm.agents.qa_agent import WildfireQAAgent


@pytest.fixture
def mock_qa_flow(mock_llm):
    """Evidence Pass -> [APPROVED] -> formalizer produces answer."""
    mock_llm.invoke.side_effect = [
        MagicMock(
            content='{"evaluation_verdict":{"overall_sufficiency":"Pass"}} '
            '[APPROVED]'
        ),
        MagicMock(
            content='{"answer":"Drought and heat.","reasoning_chain":[],'
            '"citations":[],"confidence":0.8}'
        ),
    ]
    return mock_llm


def test_qa_produces_final_output(mock_qa_flow):
    """Test QA agent produces final_output with answer or error."""
    data_agent = MagicMock()
    data_agent.retrieve.return_value = {
        "retrieved_data": [{"data_content": "text", "source": "web"}],
    }
    web = MagicMock()
    web.search.return_value = []
    memory = MagicMock()
    memory.similarity_search.return_value = []

    qa = WildfireQAAgent(
        llm=mock_qa_flow,
        data_agent=data_agent,
        web_search_tool=web,
        memory_store=memory,
        config={},
    )
    result = qa.invoke("What caused Canadian wildfires?")
    assert "final_output" in result
    out = result.get("final_output") or {}
    assert "answer" in out or "error" in result
