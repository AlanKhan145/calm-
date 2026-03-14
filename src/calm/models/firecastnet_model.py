"""
File: firecastnet_model.py
Description: FireCastNet wildfire model — CALM §5.2. Best performer in paper.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_firecastnet(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load FireCastNet checkpoint."""
    logger.info("FireCastNet load stub: %s", checkpoint_path)
    return None


def predict_firecastnet(
    model: Any,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Run FireCastNet prediction."""
    return {"risk_level": "Unknown", "confidence": 0.0, "error": "model stub"}
