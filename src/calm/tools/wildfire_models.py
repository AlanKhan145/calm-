"""
File: wildfire_models.py
Description: Wildfire model runner — LSTM, ConvLSTM, UTAE, FireCastNet.
             SeasFire: 0.25° × 0.25°, 59 variables.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WildfireModelRunner:
    """Unified interface for wildfire prediction models."""

    def __init__(self, config: dict | None = None) -> None:
        """Initialize with config (model, checkpoint, device)."""
        self.config = config or {}
        self.model_name = self.config.get("model", "firecastnet")
        self.checkpoint = self.config.get("checkpoint", "")
        self.device = self.config.get("device", "cpu")
        self._model = None

    def load(self) -> None:
        """Load model checkpoint."""
        if not self.checkpoint or not Path(self.checkpoint).exists():
            logger.warning("Checkpoint not found: %s", self.checkpoint)
            return
        try:
            if self.model_name == "lstm":
                from calm.models.lstm_model import load_lstm

                self._model = load_lstm(self.checkpoint, self.device)
            elif self.model_name == "convlstm":
                from calm.models.convlstm_model import load_convlstm

                self._model = load_convlstm(self.checkpoint, self.device)
            elif self.model_name == "utae":
                from calm.models.utae_model import load_utae

                self._model = load_utae(self.checkpoint, self.device)
            else:
                from calm.models.firecastnet_model import load_firecastnet

                self._model = load_firecastnet(self.checkpoint, self.device)
        except ImportError:
            logger.warning("Model module not available: %s", self.model_name)

    def predict(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Run prediction. Never fabricate (NP-5.1)."""
        if self._model is None:
            return {
                "error": "model unavailable",
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }
        try:
            if self.model_name == "lstm":
                from calm.models.lstm_model import predict_lstm

                return predict_lstm(self._model, parameters)
            elif self.model_name == "convlstm":
                from calm.models.convlstm_model import predict_convlstm

                return predict_convlstm(self._model, parameters)
            elif self.model_name == "utae":
                from calm.models.utae_model import predict_utae

                return predict_utae(self._model, parameters)
            else:
                from calm.models.firecastnet_model import predict_firecastnet

                return predict_firecastnet(self._model, parameters)
        except Exception as e:
            logger.exception("Prediction failed: %s", e)
            return {
                "error": str(e),
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }
