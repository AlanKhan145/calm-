"""
FeedbackRefinerAgent — self-reflection + confidence refinement sau prediction.

Mục tiêu:
- Nhận prediction output + RSEN outputs + data quality + past reflections
- Chỉ refine ở mức reasoning / confidence / validation / re-query
- KHÔNG retrain model online bằng chính output của nó
- Phải lưu reflection vào memory

Input:
- prediction_output
- rsen_outputs
- data_quality
- past_reflections

Output:
- final_decision
- calibrated_confidence
- refinement_actions
- reflection_text
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackRefinerAgent:
    """
    Agent hậu xử lý cho prediction.

    Không thay đổi weights model.
    Không fit/retrain online.
    Chỉ:
    - đọc prediction hiện tại
    - đọc RSEN / uncertainty / disagreement
    - đọc data quality
    - đọc past reflections
    - refine confidence
    - sinh action để re-query / validate / human review
    - ghi reflection vào memory
    """

    def __init__(
        self,
        memory_store: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or {}

        self.min_confidence = float(self.config.get("min_confidence", 0.0))
        self.max_confidence = float(self.config.get("max_confidence", 1.0))
        self.low_conf_threshold = float(self.config.get("low_conf_threshold", 0.45))
        self.medium_conf_threshold = float(self.config.get("medium_conf_threshold", 0.65))
        self.high_conf_threshold = float(self.config.get("high_conf_threshold", 0.80))

        # trọng số refine
        self.weight_quality = float(self.config.get("weight_quality", 0.35))
        self.weight_rsen = float(self.config.get("weight_rsen", 0.35))
        self.weight_reflection = float(self.config.get("weight_reflection", 0.20))
        self.weight_prediction_type = float(self.config.get("weight_prediction_type", 0.10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prediction_output: dict[str, Any],
        rsen_outputs: dict[str, Any] | None = None,
        data_quality: dict[str, Any] | None = None,
        past_reflections: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.refine(
            prediction_output=prediction_output,
            rsen_outputs=rsen_outputs,
            data_quality=data_quality,
            past_reflections=past_reflections,
            context=context,
        )

    def refine(
        self,
        prediction_output: dict[str, Any],
        rsen_outputs: dict[str, Any] | None = None,
        data_quality: dict[str, Any] | None = None,
        past_reflections: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Refine prediction ở mức reasoning/confidence/action.
        Không retrain model online.
        """
        prediction_output = prediction_output or {}
        rsen_outputs = rsen_outputs or {}
        data_quality = data_quality or {}
        past_reflections = past_reflections or []
        context = context or {}

        pred = self._extract_prediction_summary(prediction_output)
        quality = self._extract_quality_summary(data_quality)
        rsen = self._extract_rsen_summary(rsen_outputs)
        reflection_signal = self._extract_reflection_signal(past_reflections)

        calibrated_confidence = self._calibrate_confidence(
            base_confidence=pred["base_confidence"],
            prediction_type=pred["prediction_type"],
            quality_score=quality["score"],
            rsen_score=rsen["score"],
            reflection_score=reflection_signal["score"],
        )

        refinement_actions = self._build_refinement_actions(
            prediction=pred,
            quality=quality,
            rsen=rsen,
            reflection_signal=reflection_signal,
            context=context,
        )

        final_decision = self._build_final_decision(
            prediction=pred,
            calibrated_confidence=calibrated_confidence,
            quality=quality,
            rsen=rsen,
            refinement_actions=refinement_actions,
        )

        reflection_text = self._build_reflection_text(
            prediction=pred,
            calibrated_confidence=calibrated_confidence,
            quality=quality,
            rsen=rsen,
            reflection_signal=reflection_signal,
            refinement_actions=refinement_actions,
            final_decision=final_decision,
        )

        memory_record = self._build_memory_record(
            prediction=pred,
            calibrated_confidence=calibrated_confidence,
            quality=quality,
            rsen=rsen,
            reflection_signal=reflection_signal,
            refinement_actions=refinement_actions,
            final_decision=final_decision,
            reflection_text=reflection_text,
            context=context,
        )
        memory_saved = self._persist_reflection(memory_record)

        return {
            "final_decision": final_decision,
            "calibrated_confidence": calibrated_confidence,
            "refinement_actions": refinement_actions,
            "reflection_text": reflection_text,
            "memory_saved": memory_saved,
            "memory_record": memory_record,
        }

    # ------------------------------------------------------------------
    # Extract / normalize
    # ------------------------------------------------------------------

    def _extract_prediction_summary(self, prediction_output: dict[str, Any]) -> dict[str, Any]:
        prediction_type = str(
            prediction_output.get("prediction_type")
            or "unavailable"
        )

        risk_level = str(
            prediction_output.get("risk_level")
            or "Unknown"
        )

        probability = self._safe_float(prediction_output.get("probability"))
        logit = self._safe_float(prediction_output.get("logit"))
        base_confidence = self._safe_float(prediction_output.get("confidence"))
        if base_confidence is None:
            base_confidence = 0.0

        error = prediction_output.get("error")
        metadata = prediction_output.get("prediction_metadata") or {}

        return {
            "prediction_type": prediction_type,  # model | heuristic_fallback | unavailable
            "risk_level": risk_level,
            "probability": probability,
            "logit": logit,
            "base_confidence": float(base_confidence),
            "error": error,
            "metadata": metadata,
        }

    def _extract_quality_summary(self, data_quality: dict[str, Any]) -> dict[str, Any]:
        """
        Chuẩn hóa data quality về score 0..1.

        Hỗ trợ các field phổ biến:
        - score / overall_score
        - completeness
        - observed_ratio
        - missing_ratio
        - feature_manifest.summary.observed_ratio
        - feature_manifest.summary.missing_features
        """
        feature_manifest = data_quality.get("feature_manifest") or {}
        manifest_summary = feature_manifest.get("summary") or {}

        score = (
            self._safe_float(data_quality.get("score"))
            or self._safe_float(data_quality.get("overall_score"))
            or self._safe_float(data_quality.get("completeness"))
            or self._safe_float(data_quality.get("observed_ratio"))
            or self._safe_float(manifest_summary.get("observed_ratio"))
        )

        missing_ratio = self._safe_float(data_quality.get("missing_ratio"))
        if score is None and missing_ratio is not None:
            score = max(0.0, 1.0 - missing_ratio)

        if score is None:
            missing_features = data_quality.get("missing_features") or manifest_summary.get("missing_features") or []
            total = self._safe_float(data_quality.get("total_features")) or self._safe_float(feature_manifest.get("input_size"))
            if total and total > 0:
                score = max(0.0, 1.0 - (len(missing_features) / total))

        if score is None:
            score = 0.5

        score = self._clamp(score, 0.0, 1.0)

        issues: list[str] = []
        if score < 0.5:
            issues.append("low_data_quality")
        if data_quality.get("missing_features") or manifest_summary.get("missing_features"):
            issues.append("missing_features_present")
        if data_quality.get("stale"):
            issues.append("stale_input_data")
        if data_quality.get("location_unresolved"):
            issues.append("location_unresolved")
        if data_quality.get("time_range_unresolved"):
            issues.append("time_range_unresolved")

        return {
            "score": score,
            "issues": issues,
            "feature_manifest": feature_manifest,
        }

    def _extract_rsen_summary(self, rsen_outputs: dict[str, Any]) -> dict[str, Any]:
        """
        Chuẩn hóa tín hiệu RSEN / self-reflection / ensemble reasoning.

        Hỗ trợ các field phổ biến:
        - confidence
        - uncertainty
        - disagreement
        - consensus
        - warnings / flags / suggested_actions
        """
        uncertainty = self._safe_float(rsen_outputs.get("uncertainty"))
        disagreement = self._safe_float(rsen_outputs.get("disagreement"))
        consensus = self._safe_float(rsen_outputs.get("consensus"))
        rsen_confidence = self._safe_float(rsen_outputs.get("confidence"))

        warnings = rsen_outputs.get("warnings") or rsen_outputs.get("flags") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        suggested_actions = rsen_outputs.get("suggested_actions") or []
        if not isinstance(suggested_actions, list):
            suggested_actions = [str(suggested_actions)]

        rationale = rsen_outputs.get("rationale") or rsen_outputs.get("explanation")

        # score càng cao càng đáng tin
        signals: list[float] = []

        if rsen_confidence is not None:
            signals.append(self._clamp(rsen_confidence, 0.0, 1.0))
        if consensus is not None:
            signals.append(self._clamp(consensus, 0.0, 1.0))
        if uncertainty is not None:
            signals.append(1.0 - self._clamp(uncertainty, 0.0, 1.0))
        if disagreement is not None:
            signals.append(1.0 - self._clamp(disagreement, 0.0, 1.0))

        score = sum(signals) / len(signals) if signals else 0.5

        issues: list[str] = []
        if uncertainty is not None and uncertainty > 0.5:
            issues.append("high_uncertainty")
        if disagreement is not None and disagreement > 0.4:
            issues.append("high_disagreement")
        if warnings:
            issues.append("rsen_warnings_present")

        return {
            "score": self._clamp(score, 0.0, 1.0),
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "consensus": consensus,
            "confidence": rsen_confidence,
            "warnings": warnings,
            "suggested_actions": suggested_actions,
            "rationale": rationale,
            "issues": issues,
        }

    def _extract_reflection_signal(self, past_reflections: list[Any]) -> dict[str, Any]:
        """
        Dùng past reflections như memory-based learning ở mức reasoning,
        không dùng để retrain weights.

        Heuristic:
        - Nếu nhiều reflection trước nói 'overconfident', 'missing data', 'fallback'
          thì giảm điểm.
        - Nếu nhiều reflection nói 'validated', 'confirmed', 'good coverage'
          thì tăng nhẹ.
        """
        if not past_reflections:
            return {
                "score": 0.5,
                "issues": [],
                "supporting_patterns": [],
            }

        negative_hits = 0
        positive_hits = 0
        supporting_patterns: list[str] = []

        negative_markers = [
            "overconfident",
            "missing data",
            "fallback",
            "low quality",
            "ambiguous location",
            "ambiguous time",
            "validation failed",
            "model unavailable",
            "heuristic",
        ]
        positive_markers = [
            "validated",
            "confirmed",
            "good coverage",
            "high quality",
            "consistent",
            "resolved",
        ]

        for item in past_reflections:
            text = self._reflection_to_text(item).lower()

            for marker in negative_markers:
                if marker in text:
                    negative_hits += 1
                    supporting_patterns.append(f"negative:{marker}")

            for marker in positive_markers:
                if marker in text:
                    positive_hits += 1
                    supporting_patterns.append(f"positive:{marker}")

        total_hits = negative_hits + positive_hits
        if total_hits == 0:
            score = 0.5
        else:
            score = (positive_hits + 0.5) / (total_hits + 1.0)

        issues: list[str] = []
        if negative_hits > positive_hits:
            issues.append("past_reflections_warn_against_overconfidence")
        if positive_hits > negative_hits:
            issues.append("past_reflections_support_consistency")

        return {
            "score": self._clamp(score, 0.0, 1.0),
            "issues": issues,
            "supporting_patterns": supporting_patterns[:20],
            "negative_hits": negative_hits,
            "positive_hits": positive_hits,
        }

    # ------------------------------------------------------------------
    # Confidence refinement
    # ------------------------------------------------------------------

    def _calibrate_confidence(
        self,
        base_confidence: float,
        prediction_type: str,
        quality_score: float,
        rsen_score: float,
        reflection_score: float,
    ) -> float:
        """
        Refine confidence bằng cách kết hợp:
        - base confidence từ prediction
        - quality
        - rsen/self-reflection signals
        - past reflections
        - prediction type penalty

        Không thay model.
        """
        pred_type_score = {
            "model": 1.0,
            "heuristic_fallback": 0.45,
            "unavailable": 0.10,
        }.get(prediction_type, 0.25)

        refined = (
            base_confidence * 0.45
            + quality_score * self.weight_quality
            + rsen_score * self.weight_rsen * 0.6
            + reflection_score * self.weight_reflection
            + pred_type_score * self.weight_prediction_type
        )

        # penalty riêng cho heuristic/unavailable để tránh hiểu nhầm
        if prediction_type == "heuristic_fallback":
            refined *= 0.75
        elif prediction_type == "unavailable":
            refined *= 0.35

        return self._clamp(refined, self.min_confidence, self.max_confidence)

    # ------------------------------------------------------------------
    # Decision / actions / reflection
    # ------------------------------------------------------------------

    def _build_refinement_actions(
        self,
        prediction: dict[str, Any],
        quality: dict[str, Any],
        rsen: dict[str, Any],
        reflection_signal: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []

        prediction_type = prediction["prediction_type"]
        risk_level = prediction["risk_level"]

        if prediction_type == "heuristic_fallback":
            actions.append(
                {
                    "action": "requery_online_features",
                    "priority": "high",
                    "reason": "Current output is heuristic fallback, not true model inference.",
                }
            )

        if prediction_type == "unavailable":
            actions.append(
                {
                    "action": "validate_model_and_feature_pipeline",
                    "priority": "high",
                    "reason": "Prediction pipeline unavailable; verify checkpoint, features, and upstream data.",
                }
            )

        if quality["score"] < 0.5:
            actions.append(
                {
                    "action": "validate_data_quality",
                    "priority": "high",
                    "reason": "Input data quality is low; refresh or complete missing feature sources.",
                }
            )

        if "missing_features_present" in quality["issues"]:
            actions.append(
                {
                    "action": "requery_missing_features",
                    "priority": "medium",
                    "reason": "Feature coverage is incomplete.",
                }
            )

        if "high_uncertainty" in rsen["issues"] or "high_disagreement" in rsen["issues"]:
            actions.append(
                {
                    "action": "cross_validate_prediction",
                    "priority": "high",
                    "reason": "RSEN indicates uncertainty or disagreement; validate with another evidence source.",
                }
            )

        for suggested in rsen.get("suggested_actions", []):
            actions.append(
                {
                    "action": "rsen_suggested_action",
                    "priority": "medium",
                    "reason": str(suggested),
                }
            )

        ambiguities = context.get("ambiguities") or []
        if ambiguities:
            fields = sorted({str(x.get("field")) for x in ambiguities if isinstance(x, dict) and x.get("field")})
            actions.append(
                {
                    "action": "resolve_query_ambiguities",
                    "priority": "high",
                    "reason": f"Ambiguous normalized query fields detected: {', '.join(fields)}.",
                }
            )

        if risk_level in {"High", "Very High"} and (
            prediction_type != "model" or quality["score"] < 0.6 or rsen["score"] < 0.6
        ):
            actions.append(
                {
                    "action": "human_review",
                    "priority": "high",
                    "reason": "High-risk outcome with insufficiently strong evidence requires human validation.",
                }
            )

        if reflection_signal["negative_hits"] > reflection_signal["positive_hits"]:
            actions.append(
                {
                    "action": "compare_with_past_failure_modes",
                    "priority": "medium",
                    "reason": "Past reflections indicate repeated failure patterns.",
                }
            )

        # dedupe
        deduped: list[dict[str, Any]] = []
        seen = set()
        for item in actions:
            key = (item.get("action"), item.get("reason"))
            if key not in seen:
                deduped.append(item)
                seen.add(key)

        return deduped

    def _build_final_decision(
        self,
        prediction: dict[str, Any],
        calibrated_confidence: float,
        quality: dict[str, Any],
        rsen: dict[str, Any],
        refinement_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prediction_type = prediction["prediction_type"]
        risk_level = prediction["risk_level"]

        action_names = {x.get("action") for x in refinement_actions}

        if prediction_type == "unavailable":
            decision = "insufficient_evidence"
            rationale = "Prediction pipeline did not produce a usable output."
        elif calibrated_confidence < self.low_conf_threshold:
            decision = "validate_before_use"
            rationale = "Confidence is low after refinement."
        elif "human_review" in action_names:
            decision = "review_required"
            rationale = "Risk is meaningful but evidence quality is not strong enough for autonomous acceptance."
        elif prediction_type == "heuristic_fallback":
            decision = "provisional_estimate"
            rationale = "Output is only a heuristic fallback, not model inference."
        elif calibrated_confidence < self.medium_conf_threshold:
            decision = "cautious_accept"
            rationale = "Prediction is usable with caution and should be cross-validated."
        else:
            decision = "accept_with_monitoring"
            rationale = "Prediction is supported well enough for downstream reasoning, with monitoring retained."

        return {
            "decision": decision,
            "risk_level": risk_level,
            "prediction_type": prediction_type,
            "rationale": rationale,
            "quality_score": quality["score"],
            "rsen_score": rsen["score"],
            "requires_validation": decision in {"validate_before_use", "review_required", "insufficient_evidence"},
        }

    def _build_reflection_text(
        self,
        prediction: dict[str, Any],
        calibrated_confidence: float,
        quality: dict[str, Any],
        rsen: dict[str, Any],
        reflection_signal: dict[str, Any],
        refinement_actions: list[dict[str, Any]],
        final_decision: dict[str, Any],
    ) -> str:
        action_text = ", ".join(action["action"] for action in refinement_actions) or "none"

        parts = [
            f"Prediction type={prediction['prediction_type']}",
            f"risk_level={prediction['risk_level']}",
            f"base_confidence={prediction['base_confidence']:.3f}",
            f"calibrated_confidence={calibrated_confidence:.3f}",
            f"quality_score={quality['score']:.3f}",
            f"rsen_score={rsen['score']:.3f}",
            f"reflection_score={reflection_signal['score']:.3f}",
            f"decision={final_decision['decision']}",
            f"actions={action_text}",
        ]

        if prediction.get("error"):
            parts.append(f"prediction_error={prediction['error']}")

        if quality["issues"]:
            parts.append(f"quality_issues={';'.join(quality['issues'])}")

        if rsen["issues"]:
            parts.append(f"rsen_issues={';'.join(rsen['issues'])}")

        if reflection_signal["issues"]:
            parts.append(f"memory_issues={';'.join(reflection_signal['issues'])}")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _build_memory_record(
        self,
        prediction: dict[str, Any],
        calibrated_confidence: float,
        quality: dict[str, Any],
        rsen: dict[str, Any],
        reflection_signal: dict[str, Any],
        refinement_actions: list[dict[str, Any]],
        final_decision: dict[str, Any],
        reflection_text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "prediction_reflection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction_type": prediction["prediction_type"],
            "risk_level": prediction["risk_level"],
            "probability": prediction["probability"],
            "logit": prediction["logit"],
            "base_confidence": prediction["base_confidence"],
            "calibrated_confidence": calibrated_confidence,
            "quality_score": quality["score"],
            "quality_issues": quality["issues"],
            "rsen_score": rsen["score"],
            "rsen_issues": rsen["issues"],
            "final_decision": final_decision,
            "refinement_actions": refinement_actions,
            "reflection_signal": reflection_signal,
            "reflection_text": reflection_text,
            "context": context,
        }

    def _persist_reflection(self, record: dict[str, Any]) -> bool:
        """
        Cố gắng lưu reflection vào memory store theo nhiều adapter phổ biến.
        """
        if self.memory_store is None:
            logger.info("FeedbackRefinerAgent: no memory_store configured, reflection not persisted.")
            return False

        method_candidates = [
            "store_reflection",
            "save_reflection",
            "append_reflection",
            "add_reflection",
            "put",
            "add",
            "append",
            "write",
            "save",
        ]

        for method_name in method_candidates:
            fn = getattr(self.memory_store, method_name, None)
            if callable(fn):
                try:
                    result = fn(record)
                    return bool(True if result is None else result)
                except TypeError:
                    try:
                        result = fn("prediction_reflections", record)
                        return bool(True if result is None else result)
                    except Exception as e:
                        logger.warning(
                            "FeedbackRefinerAgent memory_store.%s failed: %s",
                            method_name,
                            e,
                        )
                        continue
                except Exception as e:
                    logger.warning(
                        "FeedbackRefinerAgent memory_store.%s failed: %s",
                        method_name,
                        e,
                    )
                    continue

        logger.warning("FeedbackRefinerAgent: no supported method found on memory_store.")
        return False

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            x = float(value)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, float(value)))

    def _reflection_to_text(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return str(
                item.get("reflection_text")
                or item.get("text")
                or item.get("summary")
                or item.get("note")
                or item
            )
        return str(item)