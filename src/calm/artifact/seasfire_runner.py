"""
SeasFireRunner — adapter tới seasfire-ml. Load checkpoint, batch inference.

Khi có seasfire-ml: load GRU checkpoint, chạy inference.
Khi không: fallback heuristic từ met_data (temp, humidity).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calm.artifact.prediction_store import PredictionArtifactStore

logger = logging.getLogger(__name__)

# Cố gắng import seasfire-ml (optional)
_SEASFIRE_AVAILABLE = False
_SEASFIRE_GRU = None
try:
    seasfire_path = os.environ.get("SEASFIRE_ML_PATH")
    if seasfire_path and Path(seasfire_path).exists():
        if seasfire_path not in sys.path:
            sys.path.insert(0, seasfire_path)
        from models import GRUModel as _GRU
        _SEASFIRE_GRU = _GRU
        _SEASFIRE_AVAILABLE = True
except ImportError:
    pass


class SeasFireRunner:
    """
    Adapter cho seasfire-ml. Load checkpoint, chạy inference.
    Khi không có seasfire-ml hoặc không có features: dùng heuristic từ met_data.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "",
        model_name: str = "gru",
        artifact_store: "PredictionArtifactStore | None" = None,
        config: dict | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint_path) if checkpoint_path else None
        self.model_name = model_name
        self.artifact_store = artifact_store
        self.config = config or {}
        self._model = None
        # GRU architecture (theo train_gru.py)
        self._input_size = self.config.get("input_size", 59)
        self._hidden_size = self.config.get("hidden_size", 64)
        self._num_layers = self.config.get("num_layers", 2)
        self._output_size = 1
        self._dropout = 0.1

    def load(self) -> bool:
        """Load checkpoint từ seasfire-ml. Trả về True nếu thành công."""
        if not self.checkpoint or not self.checkpoint.exists():
            logger.warning("SeasFireRunner: checkpoint not found %s", self.checkpoint)
            return False
        if not _SEASFIRE_AVAILABLE or _SEASFIRE_GRU is None:
            logger.warning(
                "SeasFireRunner: seasfire-ml not installed. "
                "Set SEASFIRE_ML_PATH to seasfire-ml repo path."
            )
            return False
        try:
            import torch
            self._model = _SEASFIRE_GRU(
                input_size=self._input_size,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                output_size=self._output_size,
                dropout=self._dropout,
            )
            state_dict = torch.load(str(self.checkpoint), map_location="cpu")
            self._model.load_state_dict(state_dict)
            self._model.eval()
            logger.info("SeasFireRunner: loaded checkpoint %s", self.checkpoint)
            return True
        except Exception as e:
            logger.warning("SeasFireRunner load failed: %s", e)
            return False

    def predict_batch(
        self,
        features: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Chạy inference. features có thể là:
        - torch.Tensor (batch, timesteps, input_size)
        - None → dùng heuristic từ params.get("met_data")
        """
        if self.artifact_store:
            cached = self.artifact_store.get(params)
            if cached:
                return cached

        # Có tensor features và model đã load
        if features is not None and self._model is not None:
            try:
                import torch
                with torch.no_grad():
                    x = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
                    if x.dim() == 2:
                        x = x.unsqueeze(0)
                    pred = self._model(x)
                    prob = torch.sigmoid(pred).squeeze().item()
                    if isinstance(prob, torch.Tensor):
                        prob = prob.item()
                risk_level = "High" if prob >= 0.5 else "Low"
                confidence = float(prob if prob >= 0.5 else 1 - prob)
                logit_val = pred.squeeze()
                if isinstance(logit_val, torch.Tensor):
                    logit_val = logit_val.item()
                out = {
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "result": {"probability": prob, "logit": logit_val},
                }
                if self.artifact_store:
                    self.artifact_store.put(params, out)
                return out
            except Exception as e:
                logger.warning("SeasFireRunner inference failed: %s", e)

        # Fallback: heuristic từ met_data
        return self._heuristic_predict(params)

    def _heuristic_predict(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Heuristic risk từ met_data (temperature, humidity).
        Khi không có seasfire model hoặc features.
        """
        met = params.get("met_data") or {}
        if isinstance(met, list):
            met = met[0] if met else {}
        temp = _safe_float(met.get("temperature") or met.get("t2m") or met.get("temp"))
        humidity = _safe_float(met.get("humidity") or met.get("q") or met.get("rh"))

        if temp is None and humidity is None:
            return {
                "risk_level": "Unknown",
                "confidence": 0.0,
                "result": None,
                "error": "No met_data; need seasfire checkpoint + features or GEE/CDS data",
            }

        # Heuristic đơn giản
        score = 0.5
        if temp is not None:
            if temp > 35:
                score += 0.3
            elif temp > 28:
                score += 0.15
        if humidity is not None:
            if humidity < 0.2:
                score += 0.25
            elif humidity < 0.35:
                score += 0.1
        score = min(1.0, score)
        risk_level = "High" if score >= 0.6 else "Medium" if score >= 0.4 else "Low"
        confidence = 0.6  # heuristic confidence

        out = {
            "risk_level": risk_level,
            "confidence": confidence,
            "result": {"heuristic_score": score, "temp": temp, "humidity": humidity},
        }
        if self.artifact_store:
            self.artifact_store.put(params, out)
        return out


def _safe_float(v: Any) -> float | None:
    """Chuyển giá trị sang float an toàn."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if not (f != f) else None  # NaN
    except (TypeError, ValueError):
        return None
