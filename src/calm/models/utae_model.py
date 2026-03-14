"""
File: utae_model.py
Description: UTAE wildfire model — CALM §5.2.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_utae(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load UTAE checkpoint."""
    logger.info("UTAE load stub: %s", checkpoint_path)
    return None


def predict_utae(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run UTAE prediction."""
    return {"risk_level": "Unknown", "confidence": 0.0, "error": "model stub"}
