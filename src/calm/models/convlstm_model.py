"""
File: convlstm_model.py
Description: ConvLSTM wildfire model — CALM §5.2.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_convlstm(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load ConvLSTM checkpoint."""
    logger.info("ConvLSTM load stub: %s", checkpoint_path)
    return None


def predict_convlstm(
    model: Any,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Run ConvLSTM prediction."""
    return {"risk_level": "Unknown", "confidence": 0.0, "error": "model stub"}
