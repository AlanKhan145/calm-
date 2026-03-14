"""
File: test_safety_check.py
Description: Tests for Safety Check — NP-5, URSA Code Block 2.
Author: CALM Team
Created: 2026-03-13
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calm.tools.safety_check import SafetyChecker


def test_safety_check_blocks_unsafe():
    """Test [NO] in response raises PermissionError."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="[NO] Unsafe command")
    checker = SafetyChecker(llm=llm)
    with pytest.raises(PermissionError):
        checker.check_or_raise("rm -rf /mnt/satellite_data/")


def test_safety_check_allows_safe():
    """Test [YES] in response returns True."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="[YES] Safe")
    checker = SafetyChecker(llm=llm)
    assert checker.is_safe("search wildfire news") is True
