"""
SeasFireRunner — adapter tới seasfire-ml. Batch inference, lưu output.

Stub: chưa nối thật seasfire-ml. Khi có repo seasfire-ml:
- SeasFireFeatureBuilder: đọc zarr, tạo features
- predict_batch(): load checkpoint, chạy inference
- lưu predictions.parquet → PredictionArtifactStore
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calm.artifact.prediction_store import PredictionArtifactStore

logger = logging.getLogger(__name__)


class SeasFireRunner:
    """
    Adapter cho seasfire-ml. Chạy batch inference một lần, lưu artifact.
    Agent chỉ đọc lát cắt từ artifact store.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "",
        model_name: str = "firecastnet",
        artifact_store: "PredictionArtifactStore | None" = None,
        config: dict | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint_path) if checkpoint_path else None
        self.model_name = model_name
        self.artifact_store = artifact_store
        self.config = config or {}
        self._model = None

    def load(self) -> bool:
        """Load checkpoint từ seasfire-ml. Stub: chưa implement."""
        if not self.checkpoint or not self.checkpoint.exists():
            logger.warning("SeasFireRunner: checkpoint not found %s", self.checkpoint)
            return False
        try:
            # TODO: from seasfire_ml import load_model; self._model = load_model(...)
            return False
        except ImportError:
            logger.warning("SeasFireRunner: seasfire-ml not installed")
            return False

    def predict_batch(
        self,
        features_path: str | Path,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Chạy batch inference, lưu predictions.parquet.
        Stub: trả về placeholder nếu chưa có model.
        """
        if self.artifact_store:
            cached = self.artifact_store.get(params)
            if cached:
                return cached

        if self._model is None and not self.load():
            return {
                "error": "model unavailable",
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }

        # TODO: actual inference from features_path
        out = {
            "risk_level": "Unknown",
            "confidence": 0.0,
            "result": None,
        }
        if self.artifact_store:
            self.artifact_store.put(params, out)
        return out
