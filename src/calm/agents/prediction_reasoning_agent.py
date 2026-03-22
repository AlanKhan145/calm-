"""
File: prediction_reasoning_agent.py
Description: Prediction & Reasoning Agent — runs wildfire models
             (SeasFire GRU, LSTM, FireCastNet), feeds into RSEN.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PredictionReasoningAgent:
    """
    Chạy model dự đoán cháy rừng. model_runner có thể là:
    - SeasFireModelRunner: load seasfire-ml checkpoint, heuristic fallback khi không có features
    - Bất kỳ object có .predict(parameters) -> {risk_level, confidence, result}
    Trả về dict cho RSEN. Không bịa dữ liệu (NP-5.1).
    """

    def __init__(
        self,
        model_runner=None,
        config: dict | None = None,
    ) -> None:
        """Initialize with optional model runner and config."""
        self.model_runner = model_runner
        self.config = config or {}

    def predict(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Run model. On failure: return {"error": "...", "result": null}.
        NEVER generate placeholder/synthetic prediction.
        """
        if not self.model_runner:
            return {
                "error": "model unavailable",
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }
        try:
            result = self.model_runner.predict(parameters)
            return result
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            return {
                "error": str(e),
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }
