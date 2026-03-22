"""
File: conftest.py
Description: Pytest fixtures for CALM tests. All API calls mocked (NP-7.2).
Author: CALM Team
Created: 2026-03-13
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

src = Path(__file__).resolve().parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


@pytest.fixture
def mock_llm():
    """Return mock LLM for testing."""
    return MagicMock()


@pytest.fixture
def mock_llm_plan_approved(mock_llm):
    """Simulates approved planning flow (URSA Code Block 1 path)."""
    mock_llm.invoke.side_effect = [
        MagicMock(content="Step 1: collect data. Step 2: predict..."),
        MagicMock(content="Plan is complete and feasible. [APPROVED]"),
        MagicMock(
            content='[{"step_id":"step-1","action":"call_agent",'
            '"agent":"data_knowledge","prompt":"Collect data for Amazon region",'
            '"parameters":{},"expected_output":[],"success_criteria":[]}]'
        ),
    ]
    return mock_llm


@pytest.fixture
def mock_rsen_plausible(mock_llm):
    """Simulates RSEN parallel validation → Plausible."""
    weather_json = json.dumps({
        "weather_report": {
            "key_findings": ["High temp", "Low humidity"],
            "fire_weather_impact_score": "High",
            "confidence_score": 0.9,
            "impact_assessment": "...",
            "rationale": "...",
        }
    })
    geo_json = json.dumps({
        "geospatial_report": {
            "key_findings": ["Dry chaparral", "Steep slope"],
            "fire_geospatial_impact_score": "High",
            "confidence_score": 0.85,
            "impact_assessment": "...",
            "rationale": "...",
        }
    })
    coord_json = json.dumps({
        "final_prediction": {"risk_level": "High", "confidence": 0.88},
        "validation_decision": "Plausible",
        "reasoning_summary": {
            "weather_factors": "...",
            "geospatial_factors": "...",
            "synthesis": "...",
        },
        "final_rationale": "Both weather and terrain support fire risk.",
    })
    mock_llm.invoke.side_effect = [
        MagicMock(content=weather_json),
        MagicMock(content=geo_json),
        MagicMock(content=coord_json),
    ]
    return mock_llm
