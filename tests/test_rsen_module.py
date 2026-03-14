"""
File: test_rsen_module.py
Description: Tests for RSEN — parallel independence (NP-1.4, NP-1.5).
Author: CALM Team
Created: 2026-03-13
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calm.agents.rsen_module import RSENModule


def test_rsen_parallel_independence(mock_rsen_plausible):
    """NP-1.5: Weather analyst must NOT see spatial_data; Geo not met_data."""
    memory = MagicMock()
    memory.similarity_search.return_value = []
    memory.add_texts.return_value = None

    rsen = RSENModule(llm=mock_rsen_plausible, memory_store=memory, k=3)
    result = rsen.validate(
        prediction={"risk_level": "High", "confidence": 0.8},
        met_data={"temperature": 42.15, "humidity": 15.50, "wind": 5.50},
        spatial_data={"fuel_type": "Shrubland", "slope": 30},
    )
    assert result["validation_decision"] in ["Plausible", "Implausible"]

    weather_call_args = str(mock_rsen_plausible.invoke.call_args_list[0])
    assert "42.15" in weather_call_args
    geo_call_args = str(mock_rsen_plausible.invoke.call_args_list[1])
    assert "Shrubland" in geo_call_args
    assert "Shrubland" not in weather_call_args
    assert "42.15" not in geo_call_args
