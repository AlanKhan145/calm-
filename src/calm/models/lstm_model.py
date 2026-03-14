"""
File: lstm_model.py
Description: LSTM wildfire model — CALM §5.2. SeasFire: 0.25° × 0.25°.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_lstm(checkpoint_path: str, device: str = "cpu") -> Any:
    """Load LSTM checkpoint. Stub for model interface."""
    logger.info("LSTM load stub: %s", checkpoint_path)
    return None


def predict_lstm(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run LSTM prediction. Returns risk_level, confidence."""
    return {"risk_level": "Unknown", "confidence": 0.0, "error": "model stub"}
