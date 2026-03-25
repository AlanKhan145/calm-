"""
File: prediction_reasoning_agent.py
Description: Prediction & Reasoning Agent — runs wildfire models
             (SeasFire GRU, LSTM, FireCastNet), feeds into RSEN.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PredictionReasoningAgent:
    """
    Chạy model dự đoán cháy rừng.

    Hỗ trợ 2 mode:
    - real_model_inference: gọi model_runner thật nếu có và feature đủ
    - fallback_inference: heuristic có kiểm soát, không bịa dữ liệu

    Tương thích các loại output:
    - fire_detection   (FR-R01)
    - risk_map         (FR-R02)
    - spread_forecast  (FR-R03)

    Không còn fail cứng kiểu "model unavailable" khi vẫn còn tín hiệu để fallback.
    """

    def __init__(
        self,
        model_runner: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_runner = model_runner
        self.memory = memory_store
        self.config = config or {}

        self.allow_fallback = self.config.get("allow_prediction_fallback", True)
        self.default_prediction_type = self.config.get("default_prediction_type", "risk_map")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def predict(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        rsen_context: Optional[Dict[str, Any]] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction with graceful degradation.

        Output luôn cố gắng có:
        - model_name
        - prediction_type
        - input_source
        - feature_status
        - used_fallback
        - reasoning_notes
        """
        params = parameters or {}

        prediction_type = self._infer_prediction_type(params)
        model_name = self._infer_model_name()
        model_inputs, input_source = self._extract_model_inputs(params)
        feature_status = self._assess_feature_status(model_inputs, prediction_type)

        local_rsen_context = rsen_context or params.get("rsen_context") or params.get("validation_context") or {}
        local_memory_context = memory_context or params.get("memory_context") or {}

        reasoning_notes: List[str] = []
        reasoning_notes.append("prediction_type=%s" % prediction_type)
        reasoning_notes.append("feature_readiness=%s" % feature_status.get("readiness", "unknown"))

        # 1) Thử real model nếu có thể
        if self.model_runner is not None:
            can_try_real_model = feature_status.get("can_run_real_model", False)
            if can_try_real_model:
                real_result = self._real_model_inference(
                    params=params,
                    model_inputs=model_inputs,
                    prediction_type=prediction_type,
                    model_name=model_name,
                )
                if not real_result.get("error"):
                    reasoning_notes.extend(real_result.get("reasoning_notes", []))
                    final_result = self._finalize_result(
                        base_result=real_result,
                        prediction_type=prediction_type,
                        model_name=model_name,
                        input_source=input_source,
                        feature_status=feature_status,
                        used_fallback=False,
                        inference_mode="real_model_inference",
                        reasoning_notes=reasoning_notes,
                        rsen_context=local_rsen_context,
                        memory_context=local_memory_context,
                    )
                    return final_result

                reasoning_notes.append(
                    "real_model_inference_failed: %s" % real_result.get("error", "unknown error")
                )
            else:
                reasoning_notes.append(
                    "real_model_inference_skipped_due_to_feature_status"
                )
        else:
            reasoning_notes.append("model_runner_unavailable")

        # 2) Fallback nếu được phép
        if self.allow_fallback:
            fallback_result = self._fallback_inference(
                prediction_type=prediction_type,
                model_inputs=model_inputs,
                feature_status=feature_status,
                params=params,
            )
            reasoning_notes.extend(fallback_result.get("reasoning_notes", []))
            final_result = self._finalize_result(
                base_result=fallback_result,
                prediction_type=prediction_type,
                model_name=model_name,
                input_source=input_source,
                feature_status=feature_status,
                used_fallback=True,
                inference_mode="fallback_inference",
                reasoning_notes=reasoning_notes,
                rsen_context=local_rsen_context,
                memory_context=local_memory_context,
            )
            return final_result

        # 3) Không fallback được
        return self._finalize_result(
            base_result={
                "error": "Real model unavailable or unsuitable, and fallback is disabled",
                "result": {
                    "status": "needs_retrieval",
                    "recommended_actions": self._build_recommended_actions(
                        feature_status=feature_status,
                        used_fallback=False,
                    ),
                },
                "risk_level": "Unknown",
                "confidence": 0.0,
                "reasoning_notes": [
                    "No usable inference path available",
                ],
            },
            prediction_type=prediction_type,
            model_name=model_name,
            input_source=input_source,
            feature_status=feature_status,
            used_fallback=False,
            inference_mode="unavailable",
            reasoning_notes=reasoning_notes,
            rsen_context=local_rsen_context,
            memory_context=local_memory_context,
        )

    # ─────────────────────────────────────────
    # Inference modes
    # ─────────────────────────────────────────

    def _real_model_inference(
        self,
        params: Dict[str, Any],
        model_inputs: Dict[str, Any],
        prediction_type: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Gọi model_runner thật.
        Chuẩn hóa output để downstream luôn đọc được.
        """
        try:
            payload = dict(params)
            payload.setdefault("model_inputs", model_inputs)
            payload.setdefault("prediction_type", prediction_type)

            raw = self.model_runner.predict(payload)

            if not isinstance(raw, dict):
                raw = {"result": raw}

            result = raw.get("result")
            risk_level = raw.get("risk_level")
            confidence = raw.get("confidence")

            # Cố suy ra nếu model trả schema không đồng nhất
            if risk_level is None:
                risk_level = self._derive_risk_level_from_result(raw)
            if confidence is None:
                confidence = self._derive_confidence_from_result(raw, default_value=0.75)

            return {
                "error": raw.get("error"),
                "result": self._normalize_model_result(
                    raw_result=result if result is not None else raw,
                    prediction_type=prediction_type,
                ),
                "risk_level": risk_level or "Unknown",
                "confidence": float(confidence or 0.0),
                "prediction_type": prediction_type,
                "model_name": raw.get("model_name", model_name),
                "reasoning_notes": [
                    "Used real model inference",
                ],
            }
        except Exception as e:
            logger.exception("Prediction failed in real_model_inference: %s", e)
            return {
                "error": str(e),
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
                "prediction_type": prediction_type,
                "model_name": model_name,
                "reasoning_notes": [
                    "real_model_inference raised exception",
                ],
            }

    def _fallback_inference(
        self,
        prediction_type: str,
        model_inputs: Dict[str, Any],
        feature_status: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Heuristic fallback có kiểm soát.
        Không sinh dữ liệu giả; chỉ suy luận từ tín hiệu thật hiện có.
        """
        met_timeseries = model_inputs.get("met_timeseries") or []
        satellite_features = model_inputs.get("satellite_features") or {}
        static_geo = model_inputs.get("static_geo") or {}
        textual_context = model_inputs.get("textual_context") or []

        latest_met = self._select_latest_timeseries_row(met_timeseries)
        heuristic_score = 0.0
        signals: List[str] = []
        warnings: List[str] = []

        # Meteorology
        temperature = self._to_float(latest_met.get("temperature"))
        humidity = self._to_float(latest_met.get("humidity"))
        wind_speed = self._to_float(latest_met.get("wind_speed"))
        precipitation = self._to_float(latest_met.get("precipitation"))

        if temperature is not None:
            if temperature >= 35:
                heuristic_score += 0.25
                signals.append("high_temperature")
            elif temperature >= 28:
                heuristic_score += 0.10
                signals.append("moderately_high_temperature")

        if humidity is not None:
            if humidity <= 25:
                heuristic_score += 0.25
                signals.append("very_low_humidity")
            elif humidity <= 40:
                heuristic_score += 0.10
                signals.append("low_humidity")

        if wind_speed is not None:
            if wind_speed >= 30:
                heuristic_score += 0.20
                signals.append("strong_wind")
            elif wind_speed >= 15:
                heuristic_score += 0.10
                signals.append("moderate_wind")

        if precipitation is not None:
            if precipitation <= 1:
                heuristic_score += 0.15
                signals.append("very_low_precipitation")
            elif precipitation <= 5:
                heuristic_score += 0.05
                signals.append("low_precipitation")

        # Satellite
        ndvi_mean = self._to_float(satellite_features.get("ndvi_mean"))
        burned_area = self._to_float(satellite_features.get("burned_area"))
        lst_mean = self._to_float(satellite_features.get("lst_mean"))

        if ndvi_mean is not None:
            if ndvi_mean <= 0.20:
                heuristic_score += 0.20
                signals.append("low_ndvi")
            elif ndvi_mean <= 0.35:
                heuristic_score += 0.10
                signals.append("moderately_low_ndvi")

        if burned_area is not None and burned_area > 0:
            heuristic_score += 0.30
            signals.append("burned_area_detected")

        if lst_mean is not None and lst_mean >= 320:
            heuristic_score += 0.10
            signals.append("high_surface_temperature")

        # Textual context
        if textual_context:
            signals.append("textual_context_available")

        risk_level = self._score_to_risk_level(heuristic_score)
        confidence = self._fallback_base_confidence(
            feature_status=feature_status,
            prediction_type=prediction_type,
            signal_count=len(signals),
        )

        if feature_status.get("readiness") in {"insufficient", "minimal"}:
            warnings.append("Fallback output is degraded because features are incomplete")

        reasoning_notes = [
            "Used fallback inference because real model was unavailable or unsuitable",
            "Fallback reasons: %s" % ", ".join(self._fallback_reasons(feature_status)),
            "Observed signals: %s" % (", ".join(signals) if signals else "none"),
        ]

        result = self._build_fallback_result(
            prediction_type=prediction_type,
            risk_level=risk_level,
            confidence=confidence,
            heuristic_score=heuristic_score,
            signals=signals,
            model_inputs=model_inputs,
            params=params,
            warnings=warnings,
        )

        return {
            "error": None,
            "result": result,
            "risk_level": risk_level,
            "confidence": confidence,
            "prediction_type": prediction_type,
            "reasoning_notes": reasoning_notes,
        }

    # ─────────────────────────────────────────
    # Output builders
    # ─────────────────────────────────────────

    def _build_fallback_result(
        self,
        prediction_type: str,
        risk_level: str,
        confidence: float,
        heuristic_score: float,
        signals: List[str],
        model_inputs: Dict[str, Any],
        params: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        geometry = (
            params.get("geometry")
            or params.get("bbox")
            or params.get("polygon")
            or params.get("location")
            or {}
        )

        if prediction_type == "fire_detection":
            detection_status = "possible_fire_activity" if heuristic_score >= 0.60 else "no_clear_detection"
            if not signals:
                detection_status = "insufficient_evidence"

            return {
                "status": "degraded" if warnings else "ok",
                "prediction_type": "fire_detection",
                "detection_status": detection_status,
                "risk_level": risk_level,
                "confidence": confidence,
                "supporting_signals": signals,
                "warnings": warnings,
                "recommended_actions": self._build_recommended_actions(
                    feature_status=self._assess_feature_status(model_inputs, prediction_type),
                    used_fallback=True,
                ),
            }

        if prediction_type == "spread_forecast":
            latest_met = self._select_latest_timeseries_row(model_inputs.get("met_timeseries") or [])
            wind_direction = latest_met.get("wind_direction")
            spread_potential = "high" if risk_level in {"High", "Very High"} else "moderate" if risk_level == "Moderate" else "low"

            return {
                "status": "degraded",
                "prediction_type": "spread_forecast",
                "forecast_generated": False,
                "downgraded_output_type": "spread_potential_assessment",
                "spread_potential": spread_potential,
                "risk_level": risk_level,
                "confidence": confidence,
                "direction_hint": wind_direction if wind_direction is not None else "unknown",
                "supporting_signals": signals,
                "warnings": warnings + [
                    "Fallback mode cannot produce a full spread forecast trajectory",
                ],
                "recommended_actions": self._build_recommended_actions(
                    feature_status=self._assess_feature_status(model_inputs, prediction_type),
                    used_fallback=True,
                ),
            }

        # Default: risk_map / risk assessment
        return {
            "status": "degraded" if warnings else "ok",
            "prediction_type": "risk_map",
            "map_generated": False,
            "downgraded_output_type": "area_risk_assessment",
            "area_risk_summary": risk_level,
            "heuristic_score": round(heuristic_score, 4),
            "confidence": confidence,
            "geometry": geometry,
            "supporting_signals": signals,
            "warnings": warnings + [
                "Fallback mode cannot generate a dense raster/grid risk map",
            ],
            "recommended_actions": self._build_recommended_actions(
                feature_status=self._assess_feature_status(model_inputs, prediction_type),
                used_fallback=True,
            ),
        }

    def _normalize_model_result(
        self,
        raw_result: Any,
        prediction_type: str,
    ) -> Any:
        """
        Chuẩn hóa nhẹ output từ model thật.
        Không ép schema quá mạnh để tránh làm mất dữ liệu gốc.
        """
        if isinstance(raw_result, dict):
            out = dict(raw_result)
            out.setdefault("prediction_type", prediction_type)
            return out
        return raw_result

    def _finalize_result(
        self,
        base_result: Dict[str, Any],
        prediction_type: str,
        model_name: str,
        input_source: List[str],
        feature_status: Dict[str, Any],
        used_fallback: bool,
        inference_mode: str,
        reasoning_notes: List[str],
        rsen_context: Optional[Dict[str, Any]],
        memory_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Làm giàu output cuối:
        - metadata
        - confidence recalibration
        - fallback explanation
        """
        result = dict(base_result)
        calibrated_confidence, calibration_notes = self._recalibrate_confidence(
            base_confidence=float(result.get("confidence", 0.0) or 0.0),
            used_fallback=used_fallback,
            feature_status=feature_status,
            prediction_type=prediction_type,
            rsen_context=rsen_context or {},
            memory_context=memory_context or {},
        )

        full_notes = list(reasoning_notes)
        full_notes.extend(result.get("reasoning_notes", []))
        full_notes.extend(calibration_notes)

        result["confidence"] = calibrated_confidence
        result["model_name"] = model_name
        result["prediction_type"] = prediction_type
        result["prediction_mode"] = inference_mode
        result["input_source"] = input_source
        result["feature_status"] = feature_status
        result["used_fallback"] = used_fallback
        result["reasoning_notes"] = full_notes
        result.setdefault("result", None)

        if used_fallback:
            result.setdefault("fallback_reason", self._fallback_reasons(feature_status))

        if result.get("result") is None:
            result["result"] = {
                "status": "needs_retrieval",
                "recommended_actions": self._build_recommended_actions(
                    feature_status=feature_status,
                    used_fallback=used_fallback,
                ),
            }

        return result

    # ─────────────────────────────────────────
    # Prediction type / model / inputs
    # ─────────────────────────────────────────

    def _infer_prediction_type(self, parameters: Dict[str, Any]) -> str:
        explicit = parameters.get("prediction_type")
        if explicit:
            return str(explicit)

        query = str(parameters.get("query", "") or "").lower()

        if any(k in query for k in ["detect", "detection", "hotspot", "active fire"]):
            return "fire_detection"
        if any(k in query for k in ["spread", "propagation", "trajectory", "forecast"]):
            return "spread_forecast"
        if any(k in query for k in ["map", "spatial risk", "risk map", "regional risk"]):
            return "risk_map"

        return self.default_prediction_type

    def _infer_model_name(self) -> str:
        if self.model_runner is None:
            return "fallback_only"

        for attr in ["model_name", "name", "__class__"]:
            value = getattr(self.model_runner, attr, None)
            if value is None:
                continue
            if attr == "__class__":
                return value.__name__
            if isinstance(value, str) and value.strip():
                return value.strip()

        return self.model_runner.__class__.__name__

    def _extract_model_inputs(
        self,
        parameters: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Tương thích cả luồng cũ lẫn luồng mới từ DataKnowledgeAgent.
        """
        provided = parameters.get("model_inputs") or {}
        model_inputs = {
            "met_timeseries": [],
            "satellite_features": {},
            "static_geo": {},
            "textual_context": [],
        }
        input_source: List[str] = []

        if isinstance(provided, dict):
            if provided.get("met_timeseries"):
                model_inputs["met_timeseries"] = provided["met_timeseries"]
                input_source.append("model_inputs.met_timeseries")
            if provided.get("satellite_features"):
                model_inputs["satellite_features"] = provided["satellite_features"]
                input_source.append("model_inputs.satellite_features")
            if provided.get("static_geo"):
                model_inputs["static_geo"] = provided["static_geo"]
                input_source.append("model_inputs.static_geo")
            if provided.get("textual_context"):
                model_inputs["textual_context"] = provided["textual_context"]
                input_source.append("model_inputs.textual_context")

        # backward compatibility: met_data
        met_data = parameters.get("met_data") or {}
        if not model_inputs["met_timeseries"] and isinstance(met_data, dict) and met_data:
            model_inputs["met_timeseries"] = [met_data]
            input_source.append("parameters.met_data")

        # backward compatibility: spatial_data -> satellite_features / static_geo
        spatial_data = parameters.get("spatial_data") or {}
        if isinstance(spatial_data, dict) and spatial_data:
            if not model_inputs["satellite_features"]:
                model_inputs["satellite_features"] = {
                    key: spatial_data.get(key)
                    for key in [
                        "ndvi_mean",
                        "ndvi_min",
                        "ndvi_max",
                        "burned_area",
                        "lst_mean",
                        "vegetation_index",
                    ]
                    if spatial_data.get(key) is not None
                }
                if model_inputs["satellite_features"]:
                    input_source.append("parameters.spatial_data->satellite_features")

            if not model_inputs["static_geo"]:
                model_inputs["static_geo"] = {
                    key: spatial_data.get(key)
                    for key in [
                        "elevation",
                        "slope",
                        "landcover",
                        "soil_type",
                        "aspect",
                    ]
                    if spatial_data.get(key) is not None
                }
                if not model_inputs["static_geo"]:
                    # fallback tối thiểu từ location/geometry
                    geometry = (
                        parameters.get("geometry")
                        or parameters.get("bbox")
                        or parameters.get("polygon")
                        or parameters.get("location")
                    )
                    if geometry:
                        model_inputs["static_geo"] = {"geometry": geometry}
                if model_inputs["static_geo"]:
                    input_source.append("parameters.spatial_data->static_geo")

        # textual_context direct
        if not model_inputs["textual_context"] and parameters.get("textual_context"):
            model_inputs["textual_context"] = parameters.get("textual_context")
            input_source.append("parameters.textual_context")

        # fallback geometry
        if not model_inputs["static_geo"]:
            geometry = (
                parameters.get("geometry")
                or parameters.get("bbox")
                or parameters.get("polygon")
                or parameters.get("location")
            )
            if geometry:
                model_inputs["static_geo"] = {"geometry": geometry}
                input_source.append("parameters.geometry")

        return model_inputs, input_source

    # ─────────────────────────────────────────
    # Feature status
    # ─────────────────────────────────────────

    def _assess_feature_status(
        self,
        model_inputs: Dict[str, Any],
        prediction_type: str,
    ) -> Dict[str, Any]:
        available: List[str] = []
        missing: List[str] = []

        met = model_inputs.get("met_timeseries") or []
        sat = model_inputs.get("satellite_features") or {}
        geo = model_inputs.get("static_geo") or {}
        text = model_inputs.get("textual_context") or []

        if met:
            available.append("met_timeseries")
        else:
            missing.append("met_timeseries")

        if sat:
            available.append("satellite_features")
        else:
            missing.append("satellite_features")

        if geo:
            available.append("static_geo")
        else:
            missing.append("static_geo")

        if text:
            available.append("textual_context")
        else:
            missing.append("textual_context")

        # requirement by type
        if prediction_type == "fire_detection":
            required = ["satellite_features"]
            optional = ["textual_context", "static_geo"]
        elif prediction_type == "spread_forecast":
            required = ["met_timeseries", "static_geo"]
            optional = ["satellite_features", "textual_context"]
        else:  # risk_map
            required = ["met_timeseries", "satellite_features", "static_geo"]
            optional = ["textual_context"]

        missing_required = [name for name in required if name not in available]

        if len(missing_required) == 0:
            readiness = "complete"
        elif len(missing_required) == 1 and len(available) >= 2:
            readiness = "partial"
        elif len(available) >= 1:
            readiness = "minimal"
        else:
            readiness = "insufficient"

        can_run_real_model = readiness in {"complete", "partial"}
        can_run_fallback = len(available) >= 1

        return {
            "readiness": readiness,
            "available": available,
            "missing": missing,
            "required": required,
            "optional": optional,
            "missing_required": missing_required,
            "can_run_real_model": can_run_real_model,
            "can_run_fallback": can_run_fallback,
        }

    # ─────────────────────────────────────────
    # Confidence recalibration
    # ─────────────────────────────────────────

    def _recalibrate_confidence(
        self,
        base_confidence: float,
        used_fallback: bool,
        feature_status: Dict[str, Any],
        prediction_type: str,
        rsen_context: Dict[str, Any],
        memory_context: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        confidence = float(base_confidence or 0.0)
        notes: List[str] = []

        if used_fallback:
            confidence = min(confidence, 0.65)
            notes.append("confidence_capped_due_to_fallback")

        readiness = feature_status.get("readiness")
        if readiness == "partial":
            confidence = min(confidence, 0.60)
            notes.append("confidence_capped_due_to_partial_features")
        elif readiness == "minimal":
            confidence = min(confidence, 0.40)
            notes.append("confidence_capped_due_to_minimal_features")
        elif readiness == "insufficient":
            confidence = min(confidence, 0.25)
            notes.append("confidence_capped_due_to_insufficient_features")

        if prediction_type == "spread_forecast" and used_fallback:
            confidence = min(confidence, 0.40)
            notes.append("spread_forecast_fallback_has_lower_confidence")

        # RSEN recalibration
        decision = str(
            rsen_context.get("validation_decision")
            or rsen_context.get("decision")
            or ""
        ).lower()

        if decision == "plausible":
            confidence = min(1.0, confidence + 0.05)
            notes.append("confidence_adjusted_up_by_rsen")
        elif decision == "implausible":
            confidence = max(0.0, confidence - 0.20)
            notes.append("confidence_adjusted_down_by_rsen")

        # Memory recalibration
        previous_failures = self._extract_previous_failure_count(memory_context)
        if previous_failures >= 2:
            confidence = max(0.0, confidence - 0.10)
            notes.append("confidence_adjusted_down_by_memory_context")

        confidence = round(confidence, 4)
        return confidence, notes

    def _extract_previous_failure_count(self, memory_context: Dict[str, Any]) -> int:
        for key in ["previous_failures", "failure_count", "recent_failures"]:
            value = memory_context.get(key)
            if isinstance(value, int):
                return value
        return 0

    # ─────────────────────────────────────────
    # Recommended actions
    # ─────────────────────────────────────────

    def _build_recommended_actions(
        self,
        feature_status: Dict[str, Any],
        used_fallback: bool,
    ) -> List[str]:
        actions: List[str] = []

        missing_required = feature_status.get("missing_required", [])
        for item in missing_required:
            if item == "met_timeseries":
                actions.append("re-retrieve meteorological timeseries or expand time window")
            elif item == "satellite_features":
                actions.append("re-retrieve satellite features or reduce spatial granularity")
            elif item == "static_geo":
                actions.append("re-retrieve static geo features or provide bbox/polygon/point geometry")

        if used_fallback:
            actions.append("run validation via RSEN and consider re-retrieval before finalizing")

        if not actions:
            actions.append("no additional action required")

        return actions

    def _fallback_reasons(self, feature_status: Dict[str, Any]) -> List[str]:
        reasons = []
        if self.model_runner is None:
            reasons.append("model_runner_unavailable")
        if not feature_status.get("can_run_real_model", False):
            reasons.append("feature_status_not_suitable_for_real_model")
        return reasons

    # ─────────────────────────────────────────
    # Heuristic helpers
    # ─────────────────────────────────────────

    def _fallback_base_confidence(
        self,
        feature_status: Dict[str, Any],
        prediction_type: str,
        signal_count: int,
    ) -> float:
        confidence = 0.45
        readiness = feature_status.get("readiness")

        if readiness == "complete":
            confidence = 0.60
        elif readiness == "partial":
            confidence = 0.50
        elif readiness == "minimal":
            confidence = 0.35
        else:
            confidence = 0.20

        if signal_count >= 4:
            confidence += 0.05
        elif signal_count == 0:
            confidence -= 0.10

        if prediction_type == "spread_forecast":
            confidence = min(confidence, 0.45)

        return round(max(0.0, min(1.0, confidence)), 4)

    def _score_to_risk_level(self, score: float) -> str:
        if score >= 0.80:
            return "Very High"
        if score >= 0.55:
            return "High"
        if score >= 0.30:
            return "Moderate"
        if score > 0.0:
            return "Low"
        return "Unknown"

    def _derive_risk_level_from_result(self, raw: Dict[str, Any]) -> str:
        score_candidates = [
            raw.get("score"),
            raw.get("risk_score"),
            raw.get("probability"),
            raw.get("confidence"),
        ]
        for candidate in score_candidates:
            value = self._to_float(candidate)
            if value is not None:
                return self._score_to_risk_level(value)
        return "Unknown"

    def _derive_confidence_from_result(
        self,
        raw: Dict[str, Any],
        default_value: float = 0.75,
    ) -> float:
        for key in ["confidence", "probability", "score", "risk_score"]:
            value = self._to_float(raw.get(key))
            if value is not None:
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        return default_value

    def _select_latest_timeseries_row(
        self,
        met_timeseries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not met_timeseries:
            return {}
        if isinstance(met_timeseries[-1], dict):
            return met_timeseries[-1]
        return {}

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None