"""
File: test_planning_agent.py
Description: Tests for Planning Agent — URSA 3-node, [APPROVED], f_max retry.
Author: CALM Team
Created: 2026-03-13
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calm.agents.planning_agent import PlanningAgent


def test_approved_on_first_reflection(mock_llm_plan_approved):
    """Test plan approved on first reflection returns valid JSON."""
    agent = PlanningAgent(llm=mock_llm_plan_approved, config={})
    result = agent.invoke("Wildfire risk assessment Amazon 7 days")
    assert result["approved"] is True
    assert isinstance(result["final_output"], list)
    step = result["final_output"][0]
    required_keys = [
        "step_id",
        "action",
        "agent",
        "prompt",
        "expected_output",
        "success_criteria",
    ]
    assert all(k in step for k in required_keys)


def test_force_finalize_at_n_max(mock_llm):
    """URSA Code Block 1: force finalize at n_max, never crash."""
    valid_json = (
        '[{"step_id":"s1","action":"test","agent":"qa","prompt":"Test step",'
        '"parameters":{},"expected_output":[],"success_criteria":[]}]'
    )
    mock_llm.invoke.side_effect = [
        MagicMock(content="Plan v1"),
        MagicMock(content="Needs revision: add data step"),
        MagicMock(content="Plan v2"),
        MagicMock(content="Needs revision: add validation"),
        MagicMock(content="Plan v3"),
        MagicMock(content="Needs revision still"),
        MagicMock(content=valid_json),
    ]
    agent = PlanningAgent(llm=mock_llm, config={}, n_max=3)
    result = agent.invoke("test query")
    assert result["final_output"] is not None


def test_json_retry_f_max(mock_llm):
    """URSA Code Block 1 lines 12-20: retry on bad JSON."""
    mock_llm.invoke.side_effect = [
        MagicMock(content="Initial plan"),
        MagicMock(content="Looks good. [APPROVED]"),
        MagicMock(content="not json {{bad}}"),
        MagicMock(
            content='[{"step_id":"s1","action":"test","agent":"qa","prompt":"Test step",'
            '"parameters":{},"expected_output":[],"success_criteria":[]}]'
        ),
    ]
    agent = PlanningAgent(llm=mock_llm, config={}, f_max=3)
    result = agent.invoke("test")
    assert result["approved"] is True
