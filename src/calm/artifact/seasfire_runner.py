"""
SeasFireRunner — adapter tới seasfire-ml. Load checkpoint, batch inference.

Mục tiêu:
- Chuẩn hóa lỗi load model:
  - missing_checkpoint
  - architecture_mismatch
  - missing_seasfire_ml_path
- Chuẩn hóa output prediction:
  - probability
  - logit
  - risk_level
  - confidence
  - prediction_type
- Heuristic chỉ là fallback, không được ngụy trang như output model thật.
- Chuẩn bị chỗ mở rộng cho:
  - risk_map
  - spread_forecast
  - calibration / multi-output
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calm.artifact.prediction_store import PredictionArtifactStore

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> float | None:
    """Chuyển giá trị sang float an toàn."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _sigmoid(x: float) -> float:
    """Sigmoid an toàn cho scalar float."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class SeasFireRunner:
    """
    Adapter cho seasfire-ml. Load checkpoint, chạy inference.

    Khi có:
    - checkpoint hợp lệ
    - seasfire-ml import được
    - features hợp lệ
    => chạy model inference thật

    Khi không đủ điều kiện trên:
    - chỉ fallback heuristic từ met_data
    - prediction_type phải là heuristic_fallback hoặc unavailable
    - không được giả dạng output model
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
        self._load_attempted = False
        self._last_load_status = self._build_load_status(
            state="not_loaded",
            load_error=None,
            message="Model has not been loaded yet.",
        )

        # GRU architecture mặc định theo train_gru.py
        self._input_size = int(self.config.get("input_size", 59))
        self._hidden_size = int(self.config.get("hidden_size", 64))
        self._num_layers = int(self.config.get("num_layers", 2))
        self._output_size = int(self.config.get("output_size", 1))
        self._dropout = float(self.config.get("dropout", 0.1))

        # Calibration placeholder
        calibration_cfg = self.config.get("calibration") or {}
        self._calibration_temperature = _safe_float(
            calibration_cfg.get("temperature", self.config.get("calibration_temperature", 1.0))
        )
        if self._calibration_temperature is None or self._calibration_temperature <= 0:
            self._calibration_temperature = 1.0

        # Risk thresholds
        self._high_risk_threshold = float(self.config.get("high_risk_threshold", 0.6))
        self._medium_risk_threshold = float(self.config.get("medium_risk_threshold", 0.4))

    # ------------------------------------------------------------------
    # Load status / import helpers
    # ------------------------------------------------------------------

    def _build_load_status(
        self,
        state: str,
        load_error: str | None,
        message: str,
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "state": state,  # not_loaded | loaded | error
            "load_error": load_error,
            "message": message,
            "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
            "model_name": self.model_name,
            "architecture": {
                "input_size": self._input_size,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "output_size": self._output_size,
                "dropout": self._dropout,
            },
            **extra,
        }

    def get_load_status(self) -> dict[str, Any]:
        """Cho tầng trên đọc trạng thái load model hiện tại."""
        return dict(self._last_load_status)

    def _import_gru_class(self):
        """
        Import động seasfire-ml mỗi lần load để tránh phụ thuộc env ở thời điểm import module này.
        """
        seasfire_path = os.environ.get("SEASFIRE_ML_PATH")
        if not seasfire_path:
            self._last_load_status = self._build_load_status(
                state="error",
                load_error="missing_seasfire_ml_path",
                message="SEASFIRE_ML_PATH is not set.",
                seasfire_ml_path=None,
            )
            return None

        seasfire_repo = Path(seasfire_path)
        if not seasfire_repo.exists():
            self._last_load_status = self._build_load_status(
                state="error",
                load_error="missing_seasfire_ml_path",
                message=f"SEASFIRE_ML_PATH does not exist: {seasfire_repo}",
                seasfire_ml_path=str(seasfire_repo),
            )
            return None

        try:
            if str(seasfire_repo) not in sys.path:
                sys.path.insert(0, str(seasfire_repo))
            from models import GRUModel  # type: ignore
            return GRUModel
        except Exception as e:
            self._last_load_status = self._build_load_status(
                state="error",
                load_error="seasfire_import_failed",
                message=f"Failed to import seasfire-ml GRUModel: {e}",
                seasfire_ml_path=str(seasfire_repo),
            )
            logger.warning("SeasFireRunner import failed: %s", e)
            return None

    def _extract_state_dict(self, checkpoint_obj: Any) -> dict[str, Any]:
        """
        Hỗ trợ vài format checkpoint phổ biến:
        - raw state_dict
        - {'state_dict': ...}
        - {'model_state_dict': ...}
        - {'model': ...}
        """
        if isinstance(checkpoint_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint_obj.get(key)
                if isinstance(value, dict) and value:
                    return self._strip_module_prefix(value)

            # Nếu dict hiện tại đã là state_dict
            if checkpoint_obj and all(isinstance(k, str) for k in checkpoint_obj.keys()):
                return self._strip_module_prefix(checkpoint_obj)

        raise ValueError("Unsupported checkpoint format; cannot extract state_dict.")

    def _strip_module_prefix(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Xử lý checkpoint được lưu từ DataParallel."""
        if not state_dict:
            return state_dict
        if all(k.startswith("module.") for k in state_dict.keys()):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def load(self) -> bool:
        """
        Load checkpoint từ seasfire-ml.

        Trả về True nếu load thành công.
        Đồng thời cập nhật self._last_load_status để tầng trên biết lỗi cụ thể.
        """
        self._load_attempted = True

        if not self.checkpoint or not self.checkpoint.exists():
            self._model = None
            self._last_load_status = self._build_load_status(
                state="error",
                load_error="missing_checkpoint",
                message=f"Checkpoint not found: {self.checkpoint}",
            )
            logger.warning("SeasFireRunner: checkpoint not found %s", self.checkpoint)
            return False

        GRUModel = self._import_gru_class()
        if GRUModel is None:
            self._model = None
            return False

        try:
            import torch

            model = GRUModel(
                input_size=self._input_size,
                hidden_size=self._hidden_size,
                num_layers=self._num_layers,
                output_size=self._output_size,
                dropout=self._dropout,
            )

            checkpoint_obj = torch.load(str(self.checkpoint), map_location="cpu")
            state_dict = self._extract_state_dict(checkpoint_obj)

            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                self._model = None
                self._last_load_status = self._build_load_status(
                    state="error",
                    load_error="architecture_mismatch",
                    message=f"Checkpoint architecture mismatch: {e}",
                )
                logger.warning("SeasFireRunner architecture mismatch: %s", e)
                return False

            model.eval()
            self._model = model
            self._last_load_status = self._build_load_status(
                state="loaded",
                load_error=None,
                message=f"Loaded checkpoint successfully: {self.checkpoint}",
            )
            logger.info("SeasFireRunner: loaded checkpoint %s", self.checkpoint)
            return True

        except Exception as e:
            self._model = None
            self._last_load_status = self._build_load_status(
                state="error",
                load_error="load_failed",
                message=f"Failed to load model: {e}",
            )
            logger.warning("SeasFireRunner load failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _classify_risk(self, probability: float | None) -> str:
        if probability is None:
            return "Unknown"
        if probability >= self._high_risk_threshold:
            return "High"
        if probability >= self._medium_risk_threshold:
            return "Medium"
        return "Low"

    def _confidence_from_probability(self, probability: float | None) -> float:
        if probability is None:
            return 0.0
        return float(max(probability, 1.0 - probability))

    def _apply_calibration(self, logit: float) -> tuple[float, float, dict[str, Any]]:
        """
        Placeholder cho calibration.
        Hiện hỗ trợ temperature scaling đơn giản trên logit.
        """
        temperature = self._calibration_temperature or 1.0
        calibrated_logit = logit / temperature
        calibrated_probability = _sigmoid(calibrated_logit)
        return calibrated_logit, calibrated_probability, {
            "enabled": temperature != 1.0,
            "method": "temperature_scaling",
            "temperature": temperature,
        }

    def _to_scalar(self, value: Any) -> float:
        """Ép output model về scalar float."""
        try:
            return float(value.item())  # torch scalar
        except AttributeError:
            return float(value)

    def _extract_model_outputs(self, raw_output: Any) -> tuple[float, dict[str, Any]]:
        """
        Chuẩn bị chỗ mở rộng cho multi-output.

        Hiện tại:
        - nếu output là tensor / scalar => coi là primary logit
        - nếu output là tuple/list => lấy phần tử đầu làm primary logit
        - nếu output là dict => ưu tiên các key quen thuộc
        """
        aux_outputs: dict[str, Any] = {
            "risk_map": None,
            "spread_forecast": None,
            "raw_output_keys": None,
        }

        if isinstance(raw_output, dict):
            aux_outputs["raw_output_keys"] = list(raw_output.keys())

            if "risk_map" in raw_output:
                aux_outputs["risk_map"] = raw_output.get("risk_map")
            if "spread_forecast" in raw_output:
                aux_outputs["spread_forecast"] = raw_output.get("spread_forecast")

            for candidate_key in ("logit", "logits", "fire_probability_logit", "main", "primary"):
                if candidate_key in raw_output:
                    return self._to_scalar(raw_output[candidate_key]), aux_outputs

            # fallback: lấy phần value đầu tiên nếu dict không rỗng
            first_value = next(iter(raw_output.values()))
            return self._to_scalar(first_value), aux_outputs

        if isinstance(raw_output, (list, tuple)) and raw_output:
            return self._to_scalar(raw_output[0]), aux_outputs

        return self._to_scalar(raw_output), aux_outputs

    def _build_prediction_output(
        self,
        *,
        probability: float | None,
        logit: float | None,
        risk_level: str,
        confidence: float,
        prediction_type: str,
        error: str | None = None,
        prediction_metadata: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Chuẩn hóa output prediction dùng chung cho:
        - model inference thật
        - heuristic fallback
        - unavailable
        """
        output = {
            "probability": probability,
            "logit": logit,
            "risk_level": risk_level,
            "confidence": float(confidence),
            "prediction_type": prediction_type,  # model | heuristic_fallback | unavailable
            "error": error,
            "result": result
            or {
                "probability": probability,
                "logit": logit,
                "risk_map": None,
                "spread_forecast": None,
            },
            "prediction_metadata": prediction_metadata or {},
        }
        return output

    def _normalize_cached_prediction(self, cached: Any) -> dict[str, Any]:
        """
        Chuẩn hóa cached output cũ nếu artifact_store đang lưu format legacy.
        """
        if not isinstance(cached, dict):
            return self._build_prediction_output(
                probability=None,
                logit=None,
                risk_level="Unknown",
                confidence=0.0,
                prediction_type="unavailable",
                error="Cached prediction has unsupported format.",
                prediction_metadata={
                    "source": "artifact_store",
                    "normalized_from_legacy": True,
                    "load_status": self.get_load_status(),
                },
            )

        if {"probability", "logit", "risk_level", "confidence", "prediction_type"} <= set(cached.keys()):
            return cached

        result = cached.get("result")
        risk_level = cached.get("risk_level", "Unknown")
        confidence = float(cached.get("confidence", 0.0))

        if isinstance(result, dict) and "probability" in result:
            probability = _safe_float(result.get("probability"))
            logit = _safe_float(result.get("logit"))
            return self._build_prediction_output(
                probability=probability,
                logit=logit,
                risk_level=risk_level,
                confidence=confidence,
                prediction_type="model",
                prediction_metadata={
                    "source": "artifact_store",
                    "normalized_from_legacy": True,
                    "load_status": self.get_load_status(),
                },
                result={
                    "probability": probability,
                    "logit": logit,
                    "risk_map": None,
                    "spread_forecast": None,
                },
            )

        if isinstance(result, dict) and "heuristic_score" in result:
            return self._build_prediction_output(
                probability=None,
                logit=None,
                risk_level=risk_level,
                confidence=confidence,
                prediction_type="heuristic_fallback",
                prediction_metadata={
                    "source": "artifact_store",
                    "normalized_from_legacy": True,
                    "load_status": self.get_load_status(),
                    "heuristic_details": result,
                },
                result={
                    "probability": None,
                    "logit": None,
                    "risk_map": None,
                    "spread_forecast": None,
                },
            )

        return self._build_prediction_output(
            probability=None,
            logit=None,
            risk_level=risk_level,
            confidence=confidence,
            prediction_type="unavailable",
            error=cached.get("error"),
            prediction_metadata={
                "source": "artifact_store",
                "normalized_from_legacy": True,
                "load_status": self.get_load_status(),
            },
        )

    def _predict_with_model(
        self,
        features: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            x = features if isinstance(features, torch.Tensor) else torch.tensor(features, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            raw_output = self._model(x)
            raw_logit, aux_outputs = self._extract_model_outputs(raw_output)
            calibrated_logit, probability, calibration_meta = self._apply_calibration(raw_logit)

        risk_level = self._classify_risk(probability)
        confidence = self._confidence_from_probability(probability)

        out = self._build_prediction_output(
            probability=float(probability),
            logit=float(calibrated_logit),
            risk_level=risk_level,
            confidence=confidence,
            prediction_type="model",
            prediction_metadata={
                "source": "seasfire_model",
                "is_fallback": False,
                "load_status": self.get_load_status(),
                "calibration": calibration_meta,
                "has_features": features is not None,
                "has_met_data": bool(params.get("met_data")),
                "has_spatial_data": bool(params.get("spatial_data")),
                "output_heads": {
                    "primary": "fire_probability",
                    "risk_map_available": aux_outputs.get("risk_map") is not None,
                    "spread_forecast_available": aux_outputs.get("spread_forecast") is not None,
                    "raw_output_keys": aux_outputs.get("raw_output_keys"),
                },
            },
            result={
                "probability": float(probability),
                "logit": float(calibrated_logit),
                "risk_map": aux_outputs.get("risk_map"),
                "spread_forecast": aux_outputs.get("spread_forecast"),
            },
        )
        return out

    def predict_batch(
        self,
        features: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Chạy inference.

        features có thể là:
        - torch.Tensor (batch, timesteps, input_size)
        - array-like
        - None

        Rule:
        - Có model + có features => model prediction thật
        - Thiếu model hoặc thiếu features => heuristic fallback
        - Thiếu cả model condition lẫn met_data => unavailable
        """
        if self.artifact_store:
            cached = self.artifact_store.get(params)
            if cached:
                return self._normalize_cached_prediction(cached)

        if features is not None and self._model is None and not self._load_attempted:
            self.load()

        # Model inference thật
        if features is not None and self._model is not None:
            try:
                out = self._predict_with_model(features, params)
                if self.artifact_store:
                    self.artifact_store.put(params, out)
                return out
            except Exception as e:
                logger.warning("SeasFireRunner inference failed: %s", e)
                out = self._heuristic_predict(
                    params,
                    fallback_reason="model_inference_failed",
                    model_warning=f"Model inference failed, fallback to heuristic: {e}",
                )
                if self.artifact_store:
                    self.artifact_store.put(params, out)
                return out

        # Fallback rõ ràng, không giả là output model
        if features is None:
            fallback_reason = "missing_features"
        elif self._model is None:
            fallback_reason = "model_unavailable"
        else:
            fallback_reason = "unknown_fallback_reason"

        out = self._heuristic_predict(params, fallback_reason=fallback_reason)
        if self.artifact_store:
            self.artifact_store.put(params, out)
        return out

    def _heuristic_predict(
        self,
        params: dict[str, Any],
        fallback_reason: str = "missing_features",
        model_warning: str | None = None,
    ) -> dict[str, Any]:
        """
        Heuristic risk từ met_data (temperature, humidity).

        Quan trọng:
        - Đây chỉ là fallback.
        - probability/logit để None để tránh hiểu nhầm đây là output model thật.
        - prediction_type phải là heuristic_fallback hoặc unavailable.
        """
        met = params.get("met_data") or {}
        if isinstance(met, list):
            met = met[0] if met else {}

        temp = _safe_float(met.get("temperature") or met.get("t2m") or met.get("temp"))
        humidity = _safe_float(met.get("humidity") or met.get("q") or met.get("rh"))

        prediction_metadata = {
            "source": "met_data_heuristic",
            "is_fallback": True,
            "fallback_reason": fallback_reason,
            "load_status": self.get_load_status(),
            "heuristic_inputs": {
                "temperature": temp,
                "humidity": humidity,
            },
        }

        if model_warning:
            prediction_metadata["model_warning"] = model_warning

        if temp is None and humidity is None:
            return self._build_prediction_output(
                probability=None,
                logit=None,
                risk_level="Unknown",
                confidence=0.0,
                prediction_type="unavailable",
                error="No met_data available for heuristic fallback; need model features or weather data.",
                prediction_metadata=prediction_metadata,
                result={
                    "probability": None,
                    "logit": None,
                    "risk_map": None,
                    "spread_forecast": None,
                },
            )

        # Heuristic đơn giản, chỉ để fallback
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

        score = min(1.0, max(0.0, score))

        if score >= 0.6:
            risk_level = "High"
        elif score >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Confidence heuristic thấp hơn model để tránh hiểu nhầm
        confidence = 0.35 + abs(score - 0.5) * 0.5
        confidence = min(0.65, max(0.2, confidence))

        prediction_metadata["heuristic_details"] = {
            "heuristic_score": score,
            "note": "Fallback estimate from met_data only; not a model probability.",
        }

        return self._build_prediction_output(
            probability=None,
            logit=None,
            risk_level=risk_level,
            confidence=confidence,
            prediction_type="heuristic_fallback",
            prediction_metadata=prediction_metadata,
            result={
                "probability": None,
                "logit": None,
                "risk_map": None,
                "spread_forecast": None,
            },
        )