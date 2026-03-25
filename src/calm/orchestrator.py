"""
CALMOrchestrator — bộ điều phối trung tâm của hệ thống CALM.

Luồng chuẩn:
    normalize -> plan -> route -> retrieve -> build_features -> predict
    -> validate -> refine -> memory/reflection

Mục tiêu của bản này:
- Không phụ thuộc planner mới có location/time_range.
- Không silent failure khi model runner build lỗi.
- Prediction pipeline có refinement rõ ràng.
- Output chuẩn hóa thống nhất cho cả QA và prediction.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.execution_agent import ExecutionAgent
from calm.agents.memory_agent import MemoryAgent
from calm.agents.planning_agent import PlanningAgent
from calm.agents.prediction_reasoning_agent import PredictionReasoningAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.agents.router_agent import RouterAgent
from calm.agents.rsen_module import RSENModule
from calm.tools.safety_check import SafetyChecker

logger = logging.getLogger(__name__)


class CALMOrchestrator:
    """
    Điểm vào duy nhất của CALM.

    Bản này dùng planner/router để quyết định luồng,
    nhưng prediction sẽ luôn đi theo pipeline tường minh để:
    - log đúng từng stage
    - không phụ thuộc planner mới có location/time_range
    - có refine sau validate
    """

    def __init__(
        self,
        planner: PlanningAgent,
        data_agent: DataKnowledgeAgent,
        qa_agent: WildfireQAAgent,
        prediction_agent: PredictionReasoningAgent,
        rsen: RSENModule,
        memory_agent: Optional[MemoryAgent] = None,
        router_agent: Optional[RouterAgent] = None,
        executor: Optional[ExecutionAgent] = None,
        geocoding_tool: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.planner = planner
        self.data_agent = data_agent
        self.qa_agent = qa_agent
        self.prediction_agent = prediction_agent
        self.rsen = rsen
        self.memory_agent = memory_agent
        self.router_agent = router_agent
        self.executor = executor
        self.geocoding_tool = geocoding_tool
        self.config = config or {}

        self._prediction_cfg = self._resolve_prediction_config(self.config)
        self._checkpoint_configured = bool(self._prediction_cfg.get("checkpoint"))
        self._model_runner_build_error = self.config.get("__model_runner_build_error")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def run(self, query: str) -> Dict[str, Any]:
        """
        Luồng chuẩn:
        1) Normalize query/context
        2) Planner sinh plan
        3) Router xác định task_type
        4) QA hoặc Prediction pipeline
        5) Memory / reflection
        """
        logger.info("[Orchestrator] Query: %s", query)

        normalized_context = self._normalize_query_context(query)
        normalized_query = normalized_context["normalized_query"]

        self._log_stage(
            "normalize",
            normalized_query=normalized_query,
            location=normalized_context.get("location"),
            coordinates=normalized_context.get("coordinates"),
            time_range=normalized_context.get("time_range"),
            source=normalized_context.get("source"),
        )

        plan_result: Dict[str, Any] = {}
        plan_steps: List[Dict[str, Any]] = []
        try:
            plan_result = self.planner.invoke(normalized_query)
            plan_steps = plan_result.get("final_output") or []
            if plan_result.get("error") and not plan_steps:
                logger.warning(
                    "[Orchestrator] Planning failed: %s",
                    plan_result.get("error"),
                )
        except Exception as e:
            logger.exception("[Orchestrator] Planner failed: %s", e)
            plan_result = {"error": str(e), "final_output": []}
            plan_steps = []

        routing = None
        if self.router_agent:
            try:
                routing = self.router_agent.route(normalized_query, plan_steps)
                task_type = routing.task_type
                if task_type == "hybrid":
                    # Cần cả bằng chứng và mô hình: chạy pipeline prediction (retrieve → predict → RSEN).
                    task_type = "prediction"
                logger.info(
                    "[Orchestrator] Router: task_type=%s, confidence=%.2f",
                    task_type,
                    routing.confidence,
                )
            except Exception as e:
                logger.warning("[Orchestrator] Router failed, fallback classify: %s", e)
                task_type = self._classify_intent_fallback(plan_steps, normalized_query)
        else:
            task_type = self._classify_intent_fallback(plan_steps, normalized_query)

        if task_type == "prediction":
            result = self._prediction_pipeline(
                original_query=query,
                normalized_context=normalized_context,
                plan_steps=plan_steps,
                plan_result=plan_result,
            )
        else:
            if self.executor and plan_steps:
                result = self._run_plan_driven(
                    original_query=query,
                    normalized_context=normalized_context,
                    plan_steps=plan_steps,
                    task_type=task_type,
                    plan_result=plan_result,
                )
            else:
                result = self._qa_pipeline(
                    original_query=query,
                    normalized_context=normalized_context,
                    plan_steps=plan_steps,
                    plan_result=plan_result,
                )

        result["memory_reflection"] = self._commit_memory(query, result, task_type)
        return result

    # ─────────────────────────────────────────
    # Routing / Plan
    # ─────────────────────────────────────────

    def _classify_intent_fallback(self, plan_steps: List[Dict[str, Any]], query: str) -> str:
        """Fallback keyword routing khi không có RouterAgent."""
        plan_pred = False
        plan_qa = False
        for step in plan_steps:
            action = str(step.get("action", "")).lower()
            agent = str(step.get("agent", "")).lower()
            if any(w in action or w in agent for w in ["predict", "forecast", "model", "run_model"]):
                plan_pred = True
            if any(w in action or w in agent for w in ["retrieve", "web_search", "qa", "compile_report"]):
                plan_qa = True
        if plan_pred:
            return "prediction"
        if plan_qa:
            return "qa"

        from calm.utils.intent_hints import infer_task_from_keywords

        return infer_task_from_keywords(query)

    def _run_plan_driven(
        self,
        original_query: str,
        normalized_context: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
        task_type: str,
        plan_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Giữ executor cho QA / generic execution.
        Prediction không dùng nhánh này để tránh thiếu log stage và thiếu refinement.
        """
        logger.info("[Orchestrator] Plan-driven execution: %d steps", len(plan_steps))

        params = self._merge_context_params(normalized_context, plan_steps)
        context: Dict[str, Any] = {
            "query": normalized_context["normalized_query"],
            "original_query": original_query,
            "normalized_query_context": normalized_context,
            "parameters": params,
        }

        for step in plan_steps:
            step_id = step.get("step_id", "unknown")
            agent_name = str(step.get("agent", "")).lower()

            if agent_name in ("data_knowledge",):
                self._log_stage("retrieve", step_id=step_id, parameters=params)
            elif agent_name in ("qa", "qa_agent", "question_answering"):
                self._log_stage("qa", step_id=step_id)

            try:
                result = self.executor.execute_step(step, context)
                context[step_id] = result

                if agent_name in ("data_knowledge",):
                    context["data_result"] = result
                    context["retrieved_data"] = result.get("retrieved_data", [])
                elif agent_name in ("qa", "qa_agent", "question_answering"):
                    context["final_output"] = result.get("final_output", {})
            except Exception as e:
                logger.warning("[Orchestrator] Step %s failed: %s", step_id, e)
                context[step_id] = {"error": str(e)}

        return self._format_plan_result(
            original_query=original_query,
            normalized_context=normalized_context,
            plan_steps=plan_steps,
            context=context,
            task_type=task_type,
            plan_result=plan_result or {},
        )

    def _format_plan_result(
        self,
        original_query: str,
        normalized_context: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
        context: Dict[str, Any],
        task_type: str,
        plan_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Định dạng kết quả sau khi chạy bằng executor."""
        data_result = context.get("data_result", {}) or {}
        data_quality = self._assess_data_quality(data_result, context.get("parameters", {}))

        if task_type == "prediction":
            pred = context.get("prediction", {}) or {}
            met = context.get("met_data", {}) or self._extract_met_data(data_result)
            spatial = context.get("spatial_data", {}) or self._extract_spatial_data(data_result)
            feature_status = self._build_feature_status(met, spatial)
            if not pred.get("error"):
                validation = self._safe_validate(pred, met, spatial)
            else:
                validation = {"validation_decision": "Unknown", "final_rationale": ""}
            return {
                "task_type": "prediction",
                "original_query": original_query,
                "normalized_query": normalized_context["normalized_query"],
                "normalized_query_context": normalized_context,
                "plan_steps": plan_steps,
                "plan_error": plan_result.get("error"),
                "parameters": context.get("parameters", {}),
                "prediction": pred,
                "validation": validation,
                "risk_level": pred.get("risk_level", "Unknown"),
                "confidence": pred.get("confidence", 0.0),
                "decision": validation.get("validation_decision", "Unknown"),
                "rationale": validation.get("final_rationale", ""),
                "data_quality": data_quality,
                "feature_status": feature_status,
                "used_fallback": bool(pred.get("used_fallback", False)),
                "refinement_trace": [],
                "approved": validation.get("validation_decision") == "Plausible",
                "error": pred.get("error") or validation.get("error"),
            }

        qa_out = context.get("final_output", {}) or {}
        for _, value in context.items():
            if isinstance(value, dict) and value.get("final_output"):
                qa_out = value.get("final_output", qa_out)
                break

        return {
            "task_type": "qa",
            "original_query": original_query,
            "normalized_query": normalized_context["normalized_query"],
            "normalized_query_context": normalized_context,
            "plan_steps": plan_steps,
            "plan_error": plan_result.get("error"),
            "parameters": context.get("parameters", {}),
            "data_collected": bool(data_result.get("retrieved_data")),
            "sources_count": len(data_result.get("retrieved_data", [])),
            "answer": qa_out.get("answer", ""),
            "reasoning_chain": qa_out.get("reasoning_chain", []),
            "citations": qa_out.get("citations", []),
            "confidence": qa_out.get("confidence", 0.0),
            "approved": qa_out.get("approved", True),
            "data_quality": data_quality,
            "feature_status": {},
            "used_fallback": False,
            "refinement_trace": [],
            "error": context.get("error"),
        }

    # ─────────────────────────────────────────
    # QA Pipeline
    # ─────────────────────────────────────────

    def _qa_pipeline(
        self,
        original_query: str,
        normalized_context: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
        plan_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("[Orchestrator] QA pipeline started")

        params = self._merge_context_params(normalized_context, plan_steps)

        self._log_stage(
            "retrieve",
            task_type="qa",
            normalized_query=normalized_context["normalized_query"],
            parameters=params,
        )

        data_result: Dict[str, Any] = {}
        try:
            data_result = self.data_agent.retrieve(normalized_context["normalized_query"], params)
            n_retrieved = len(data_result.get("retrieved_data", []))
            n_facts = len(
                data_result.get("extracted_knowledge", {}).get("factual_statements", [])
            )
            logger.info(
                "[Orchestrator] QA data: %d sources, %d facts",
                n_retrieved,
                n_facts,
            )
        except Exception as e:
            logger.warning("[Orchestrator] QA data collection failed: %s", e)
            data_result = {"error": str(e), "retrieved_data": []}

        qa_result: Dict[str, Any] = {}
        try:
            qa_result = self.qa_agent.invoke(
                normalized_context["normalized_query"],
                pre_retrieved=data_result,
            )
        except Exception as e:
            logger.exception("[Orchestrator] QA agent failed: %s", e)
            qa_result = {
                "final_output": {"answer": f"[ERROR] {e}", "citations": []},
                "approved": False,
                "error": str(e),
            }

        final_answer = qa_result.get("final_output") or {}
        data_quality = self._assess_data_quality(data_result, params)

        return {
            "task_type": "qa",
            "original_query": original_query,
            "normalized_query": normalized_context["normalized_query"],
            "normalized_query_context": normalized_context,
            "plan_steps": plan_steps,
            "plan_error": (plan_result or {}).get("error"),
            "parameters": params,
            "data_collected": bool(data_result.get("retrieved_data")),
            "sources_count": len(data_result.get("retrieved_data", [])),
            "answer": final_answer.get("answer", ""),
            "reasoning_chain": final_answer.get("reasoning_chain", []),
            "citations": final_answer.get("citations", []),
            "confidence": final_answer.get("confidence", 0.0),
            "approved": qa_result.get("approved", False),
            "data_quality": data_quality,
            "feature_status": {},
            "used_fallback": False,
            "refinement_trace": [],
            "error": qa_result.get("error") or data_result.get("error"),
        }

    # ─────────────────────────────────────────
    # Prediction Pipeline
    # ─────────────────────────────────────────

    def _prediction_pipeline(
        self,
        original_query: str,
        normalized_context: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
        plan_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("[Orchestrator] Prediction pipeline started")

        params = self._merge_context_params(normalized_context, plan_steps)
        refinement_trace: List[Dict[str, Any]] = []

        # Stage 1: retrieve
        self._log_stage(
            "retrieve",
            task_type="prediction",
            normalized_query=normalized_context["normalized_query"],
            parameters=params,
        )

        data_result: Dict[str, Any] = {}
        try:
            data_result = self.data_agent.retrieve(normalized_context["normalized_query"], params)
            logger.info(
                "[Orchestrator] Prediction data: %d sources",
                len(data_result.get("retrieved_data", [])),
            )
        except Exception as e:
            logger.warning("[Orchestrator] Prediction data collection failed: %s", e)
            data_result = {"error": str(e), "retrieved_data": []}

        data_quality = self._assess_data_quality(data_result, params)

        # Stage 2: build features
        met_data = self._extract_met_data(data_result)
        spatial_data = self._extract_spatial_data(data_result)
        feature_status = self._build_feature_status(met_data, spatial_data)

        params["met_data"] = met_data
        params["spatial_data"] = spatial_data

        self._log_stage(
            "build_features",
            met_keys=sorted(list(met_data.keys())),
            spatial_keys=sorted(list(spatial_data.keys())),
            feature_status=feature_status,
        )

        # Stage 3: predict
        prediction, used_fallback = self._safe_predict(params)

        self._log_stage(
            "predict",
            risk_level=prediction.get("risk_level"),
            confidence=prediction.get("confidence"),
            used_fallback=used_fallback,
            error=prediction.get("error"),
        )

        # Stage 4: validate
        validation = self._safe_validate(prediction, met_data, spatial_data)

        self._log_stage(
            "validate",
            decision=validation.get("validation_decision", validation.get("decision")),
            rationale=validation.get("final_rationale", ""),
            error=validation.get("error"),
        )

        # Stage 5: refine
        prediction, validation, data_result, data_quality, feature_status, used_fallback, refinement_trace = (
            self._refine_prediction_if_needed(
                normalized_query=normalized_context["normalized_query"],
                params=params,
                data_result=data_result,
                data_quality=data_quality,
                feature_status=feature_status,
                prediction=prediction,
                validation=validation,
                used_fallback=used_fallback,
            )
        )

        decision = validation.get("validation_decision", validation.get("decision", "Unknown"))
        rationale = validation.get("final_rationale", "")

        return {
            "task_type": "prediction",
            "original_query": original_query,
            "normalized_query": normalized_context["normalized_query"],
            "normalized_query_context": normalized_context,
            "plan_steps": plan_steps,
            "plan_error": (plan_result or {}).get("error"),
            "parameters": params,
            "prediction": prediction,
            "validation": validation,
            "risk_level": prediction.get("risk_level", "Unknown"),
            "confidence": prediction.get("confidence", 0.0),
            "decision": decision,
            "rationale": rationale,
            "data_quality": data_quality,
            "feature_status": feature_status,
            "used_fallback": used_fallback,
            "refinement_trace": refinement_trace,
            "approved": decision == "Plausible",
            "error": prediction.get("error") or validation.get("error") or data_result.get("error"),
        }

    def _safe_predict(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Chạy prediction an toàn:
        - log rõ nếu checkpoint có nhưng model_runner không build được
        - vẫn cho phép heuristic fallback nếu PredictionAgent hỗ trợ
        """
        prediction: Dict[str, Any] = {}
        model_runner = getattr(self.prediction_agent, "model_runner", None)
        used_fallback = model_runner is None

        if self._checkpoint_configured and model_runner is None:
            msg = self._model_runner_build_error or (
                "Prediction checkpoint được cấu hình nhưng model runner không build được."
            )
            logger.error("[Orchestrator] %s", msg)

        try:
            prediction = self.prediction_agent.predict(params)
        except Exception as e:
            logger.exception("[Orchestrator] Prediction model failed: %s", e)
            prediction = {
                "error": str(e),
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }

        if self._checkpoint_configured and model_runner is None:
            prediction.setdefault("warnings", [])
            prediction["warnings"].append(
                self._model_runner_build_error
                or "Configured checkpoint exists but model runner is unavailable."
            )
            prediction["model_status"] = "runner_unavailable"

        prediction["used_fallback"] = bool(prediction.get("used_fallback", used_fallback))
        return prediction, bool(prediction["used_fallback"])

    def _safe_validate(
        self,
        prediction: Dict[str, Any],
        met_data: Dict[str, Any],
        spatial_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Chạy RSEN validation an toàn."""
        if prediction.get("error"):
            return {
                "validation_decision": "Unknown",
                "final_rationale": "",
                "error": prediction.get("error"),
            }

        try:
            return self.rsen.validate(prediction, met_data, spatial_data)
        except Exception as e:
            logger.exception("[Orchestrator] RSEN validation failed: %s", e)
            return {
                "error": str(e),
                "validation_decision": "Unknown",
                "final_rationale": "",
            }

    def _refine_prediction_if_needed(
        self,
        normalized_query: str,
        params: Dict[str, Any],
        data_result: Dict[str, Any],
        data_quality: Dict[str, Any],
        feature_status: Dict[str, Any],
        prediction: Dict[str, Any],
        validation: Dict[str, Any],
        used_fallback: bool,
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        bool,
        List[Dict[str, Any]],
    ]:
        """
        Nếu prediction implausible hoặc dữ liệu chưa đủ:
        - re-query data
        - mở rộng window thời gian nếu được
        - chạy predict/validate lại
        - nếu vẫn không ổn thì hạ confidence
        """
        reasons = self._collect_refinement_reasons(
            data_quality=data_quality,
            feature_status=feature_status,
            prediction=prediction,
            validation=validation,
        )
        if not reasons:
            return (
                prediction,
                validation,
                data_result,
                data_quality,
                feature_status,
                used_fallback,
                [],
            )

        refinement_trace: List[Dict[str, Any]] = [
            {
                "stage": "refine",
                "action": "trigger_refinement",
                "reasons": reasons,
            }
        ]

        refined_params = copy.deepcopy(params)

        if any(
            reason in reasons
            for reason in ["implausible", "low_data_quality", "missing_features"]
        ):
            expanded_time_range = self._expand_time_range(refined_params.get("time_range"), days=3)
            if expanded_time_range != refined_params.get("time_range"):
                refinement_trace.append(
                    {
                        "stage": "refine",
                        "action": "expand_time_window",
                        "old_time_range": refined_params.get("time_range"),
                        "new_time_range": expanded_time_range,
                    }
                )
                refined_params["time_range"] = expanded_time_range

        self._log_stage(
            "refine",
            reasons=reasons,
            parameters=refined_params,
        )

        refined_data_result: Dict[str, Any] = {}
        try:
            refined_data_result = self.data_agent.retrieve(normalized_query, refined_params)
            refinement_trace.append(
                {
                    "stage": "refine",
                    "action": "requery_data",
                    "sources_count": len(refined_data_result.get("retrieved_data", [])),
                }
            )
        except Exception as e:
            logger.warning("[Orchestrator] Refinement re-query failed: %s", e)
            refinement_trace.append(
                {
                    "stage": "refine",
                    "action": "requery_data_failed",
                    "error": str(e),
                }
            )
            refined_data_result = {"error": str(e), "retrieved_data": []}

        refined_data_quality = self._assess_data_quality(refined_data_result, refined_params)
        refined_met = self._extract_met_data(refined_data_result)
        refined_spatial = self._extract_spatial_data(refined_data_result)
        refined_feature_status = self._build_feature_status(refined_met, refined_spatial)

        refined_params["met_data"] = refined_met
        refined_params["spatial_data"] = refined_spatial

        refined_prediction, refined_used_fallback = self._safe_predict(refined_params)
        refined_validation = self._safe_validate(refined_prediction, refined_met, refined_spatial)

        refinement_trace.append(
            {
                "stage": "refine",
                "action": "rerun_predict_validate",
                "refined_risk_level": refined_prediction.get("risk_level"),
                "refined_confidence": refined_prediction.get("confidence"),
                "refined_decision": refined_validation.get("validation_decision", refined_validation.get("decision")),
            }
        )

        if self._should_use_refined_result(
            old_prediction=prediction,
            old_validation=validation,
            old_data_quality=data_quality,
            old_feature_status=feature_status,
            new_prediction=refined_prediction,
            new_validation=refined_validation,
            new_data_quality=refined_data_quality,
            new_feature_status=refined_feature_status,
        ):
            refinement_trace.append(
                {
                    "stage": "refine",
                    "action": "use_refined_result",
                }
            )
            return (
                refined_prediction,
                refined_validation,
                refined_data_result,
                refined_data_quality,
                refined_feature_status,
                refined_used_fallback,
                refinement_trace,
            )

        downgraded_prediction = self._downgrade_prediction_confidence(prediction, reasons)
        refinement_trace.append(
            {
                "stage": "refine",
                "action": "keep_original_result_and_downgrade_confidence",
                "old_confidence": prediction.get("confidence", 0.0),
                "new_confidence": downgraded_prediction.get("confidence", 0.0),
            }
        )

        return (
            downgraded_prediction,
            validation,
            data_result,
            data_quality,
            feature_status,
            used_fallback,
            refinement_trace,
        )

    # ─────────────────────────────────────────
    # Normalize Query Context
    # ─────────────────────────────────────────

    def _normalize_query_context(self, query: str) -> Dict[str, Any]:
        """
        Normalize query trước planner:
        - chuẩn hóa text
        - cố gắng tách location / coordinates / time_range
        - geocode nếu có thể
        """
        from calm.utils.time_utils import resolve_time_range

        cleaned_query = " ".join(str(query).strip().split())
        location_hint = self._extract_location_hint(cleaned_query)
        coordinates = self._extract_coordinates(cleaned_query)
        time_text = self._extract_time_expression(cleaned_query)

        geocoded: Dict[str, Any] = {}
        if location_hint:
            geocoded = self._try_geocode_location(location_hint)

        canonical_location = (
            geocoded.get("location")
            or geocoded.get("name")
            or location_hint
        )

        if not coordinates and geocoded:
            lat = geocoded.get("latitude") or geocoded.get("lat")
            lon = geocoded.get("longitude") or geocoded.get("lon") or geocoded.get("lng")
            if lat is not None and lon is not None:
                coordinates = {"latitude": lat, "longitude": lon}

        resolved_time_range = resolve_time_range(time_text, default_today=True)

        metadata_tokens: List[str] = []
        if canonical_location:
            metadata_tokens.append("location=%s" % canonical_location)
        if coordinates:
            metadata_tokens.append(
                "coordinates=(%s,%s)"
                % (coordinates.get("latitude"), coordinates.get("longitude"))
            )
        if resolved_time_range:
            metadata_tokens.append("time_range=%s" % self._stringify_time_range(resolved_time_range))

        normalized_query = cleaned_query
        if metadata_tokens:
            normalized_query = "%s [%s]" % (cleaned_query, "; ".join(metadata_tokens))

        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "location": canonical_location,
            "coordinates": coordinates,
            "time_text": time_text,
            "time_range": resolved_time_range,
            "source": {
                "location": "geocoder" if geocoded else ("query_regex" if location_hint else None),
                "time_range": "query_regex_or_default",
            },
        }

    def _extract_location_hint(self, query: str) -> Optional[str]:
        """
        Heuristic nhẹ để lấy location nếu người dùng ghi trong câu.
        Không phụ thuộc planner.
        """
        patterns = [
            r"\b(?:for|in|at|near|around|over)\s+([A-Za-z][A-Za-z0-9\s,\-]+?)(?=\s+(?:today|tomorrow|this|next|from|between|during|on)\b|$)",
            r"\b(?:risk for|wildfire risk for|predict for)\s+([A-Za-z][A-Za-z0-9\s,\-]+?)(?=\s+(?:today|tomorrow|this|next|from|between|during|on)\b|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ,.-")
                if value:
                    return value
        return None

    def _extract_coordinates(self, query: str) -> Dict[str, float]:
        """Tách tọa độ dạng lat/lon nếu có trong query."""
        lat_match = re.search(r"(?:lat|latitude)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", query, flags=re.IGNORECASE)
        lon_match = re.search(
            r"(?:lon|lng|longitude)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
            query,
            flags=re.IGNORECASE,
        )
        if lat_match and lon_match:
            try:
                return {
                    "latitude": float(lat_match.group(1)),
                    "longitude": float(lon_match.group(1)),
                }
            except Exception:
                return {}
        return {}

    def _extract_time_expression(self, query: str) -> Optional[str]:
        """
        Lấy time phrase để đưa qua resolve_time_range().
        """
        patterns = [
            r"\bnext\s+\d+\s+days?\b",
            r"\bnext\s+\d+\s+weeks?\b",
            r"\bnext\s+week\b",
            r"\bthis\s+week\b",
            r"\btoday\b",
            r"\btomorrow\b",
            r"\byesterday\b",
            r"\bbetween\s+.+?\s+and\s+.+?\b",
            r"\bfrom\s+.+?\s+to\s+.+?\b",
            r"\bon\s+[A-Za-z0-9,\-\s]+\b",
            r"\bduring\s+[A-Za-z0-9,\-\s]+\b",
        ]
        lowered = query.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _try_geocode_location(self, location_text: str) -> Dict[str, Any]:
        """
        Cố gắng geocode nếu geocoding tool có sẵn.
        Không ràng buộc chặt vào một method name cụ thể.
        """
        tool = self.geocoding_tool
        if tool is None or not location_text:
            return {}

        candidate_methods = ["geocode", "resolve", "lookup", "search", "infer"]
        for method_name in candidate_methods:
            method = getattr(tool, method_name, None)
            if not callable(method):
                continue

            for payload in (location_text, {"query": location_text}, {"location": location_text}):
                try:
                    result = method(payload)
                    parsed = self._parse_geocoding_result(result)
                    if parsed:
                        return parsed
                except TypeError:
                    continue
                except Exception as e:
                    logger.debug("[Orchestrator] geocoding %s failed: %s", method_name, e)
        return {}

    @staticmethod
    def _parse_geocoding_result(result: Any) -> Dict[str, Any]:
        """Chuẩn hóa output geocoder về dict chung."""
        if isinstance(result, dict):
            if any(k in result for k in ["latitude", "lat", "location", "name"]):
                return result
            for key in ["result", "data", "payload"]:
                inner = result.get(key)
                if isinstance(inner, dict) and any(
                    k in inner for k in ["latitude", "lat", "location", "name"]
                ):
                    return inner
        return {}

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _merge_context_params(
        self,
        normalized_context: Dict[str, Any],
        plan_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Ưu tiên normalized_query_context trước.
        Planner chỉ là nguồn bổ sung, không phải nguồn bắt buộc.
        """
        from calm.utils.time_utils import resolve_time_range

        params: Dict[str, Any] = {}
        plan_params = self._extract_plan_parameters(plan_steps)

        # Ưu tiên normalized context
        if normalized_context.get("location"):
            params["location"] = normalized_context["location"]
        if normalized_context.get("coordinates"):
            coords = normalized_context["coordinates"] or {}
            if coords.get("latitude") is not None:
                params["latitude"] = coords["latitude"]
            if coords.get("longitude") is not None:
                params["longitude"] = coords["longitude"]
        if normalized_context.get("time_range"):
            params["time_range"] = normalized_context["time_range"]

        # Bổ sung từ plan nếu thiếu
        for key, value in plan_params.items():
            if value is None:
                continue
            if key == "location" and not params.get("location"):
                params["location"] = value
            elif key == "area" and not params.get("area"):
                params["area"] = value
            elif key == "time_range" and not params.get("time_range"):
                params["time_range"] = value
            elif key in ("latitude", "longitude") and key not in params:
                params[key] = value
            elif key not in params:
                params[key] = value

        params["time_range"] = resolve_time_range(params.get("time_range"), default_today=True)
        return params

    @staticmethod
    def _extract_plan_parameters(plan_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gộp parameters từ plan; lấy first-non-empty cho các trường chính.
        """
        merged: Dict[str, Any] = {}
        for step in plan_steps:
            params = dict(step.get("parameters") or {})
            if not params:
                continue
            for key, value in params.items():
                if value is None:
                    continue
                if key not in merged or merged.get(key) in (None, "", {}, []):
                    merged[key] = value
        return merged

    def _assess_data_quality(
        self,
        data_result: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Đánh giá chất lượng dữ liệu để quyết định refine hay không."""
        retrieved = data_result.get("retrieved_data", []) or []
        met = self._extract_met_data(data_result)
        spatial = self._extract_spatial_data(data_result)

        missing: List[str] = []
        if not retrieved:
            missing.append("retrieved_data")
        if not met:
            missing.append("meteorology")
        if not spatial:
            missing.append("spatial")
        if not (params.get("location") or (params.get("latitude") is not None and params.get("longitude") is not None)):
            missing.append("location_or_coordinates")

        if not retrieved or len(missing) >= 3:
            level = "low"
        elif missing:
            level = "medium"
        else:
            level = "high"

        return {
            "level": level,
            "sources_count": len(retrieved),
            "has_meteorology": bool(met),
            "has_spatial": bool(spatial),
            "missing": missing,
        }

    @staticmethod
    def _build_feature_status(
        met_data: Dict[str, Any],
        spatial_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Đánh giá feature readiness trước khi predict."""
        met_required = ["temperature", "humidity", "wind_speed", "precipitation"]
        spatial_required = ["ndvi_mean"]

        met_available = [k for k in met_required if met_data.get(k) is not None]
        spatial_available = [k for k in spatial_required if spatial_data.get(k) is not None]

        missing = [k for k in met_required if k not in met_available] + [
            k for k in spatial_required if k not in spatial_available
        ]

        if len(missing) == 0:
            status = "complete"
        elif len(met_available) + len(spatial_available) > 0:
            status = "partial"
        else:
            status = "missing"

        return {
            "status": status,
            "met_available": met_available,
            "spatial_available": spatial_available,
            "missing": missing,
        }

    def _collect_refinement_reasons(
        self,
        data_quality: Dict[str, Any],
        feature_status: Dict[str, Any],
        prediction: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> List[str]:
        reasons: List[str] = []

        decision = str(
            validation.get("validation_decision", validation.get("decision", ""))
        ).lower()
        if decision == "implausible":
            reasons.append("implausible")

        if data_quality.get("level") == "low":
            reasons.append("low_data_quality")

        if feature_status.get("status") in {"partial", "missing"}:
            reasons.append("missing_features")

        if prediction.get("error"):
            reasons.append("prediction_error")

        return reasons

    def _should_use_refined_result(
        self,
        old_prediction: Dict[str, Any],
        old_validation: Dict[str, Any],
        old_data_quality: Dict[str, Any],
        old_feature_status: Dict[str, Any],
        new_prediction: Dict[str, Any],
        new_validation: Dict[str, Any],
        new_data_quality: Dict[str, Any],
        new_feature_status: Dict[str, Any],
    ) -> bool:
        old_decision = str(
            old_validation.get("validation_decision", old_validation.get("decision", "Unknown"))
        )
        new_decision = str(
            new_validation.get("validation_decision", new_validation.get("decision", "Unknown"))
        )

        if old_decision != "Plausible" and new_decision == "Plausible":
            return True
        if old_prediction.get("error") and not new_prediction.get("error"):
            return True
        if old_data_quality.get("level") != "high" and new_data_quality.get("level") == "high":
            return True
        if old_feature_status.get("status") != "complete" and new_feature_status.get("status") == "complete":
            return True
        if (new_prediction.get("confidence", 0.0) or 0.0) > (old_prediction.get("confidence", 0.0) or 0.0):
            return True
        return False

    @staticmethod
    def _downgrade_prediction_confidence(
        prediction: Dict[str, Any],
        reasons: List[str],
    ) -> Dict[str, Any]:
        """Hạ confidence khi refine không cải thiện được."""
        downgraded = dict(prediction)
        old_conf = float(downgraded.get("confidence", 0.0) or 0.0)
        penalty = 0.15
        if "implausible" in reasons:
            penalty += 0.10
        if "low_data_quality" in reasons:
            penalty += 0.10
        if "missing_features" in reasons:
            penalty += 0.05
        downgraded["confidence"] = max(0.0, round(old_conf - penalty, 4))
        downgraded.setdefault("warnings", [])
        downgraded["warnings"].append(
            "Confidence downgraded after refinement because validation/data quality remained weak."
        )
        return downgraded

    @staticmethod
    def _expand_time_range(time_range: Any, days: int = 3) -> Any:
        """
        Mở rộng cửa sổ thời gian để re-query.
        Hỗ trợ các key phổ biến: start/end hoặc start_date/end_date.
        """
        if not isinstance(time_range, dict):
            return time_range

        start_key = "start" if "start" in time_range else "start_date" if "start_date" in time_range else None
        end_key = "end" if "end" in time_range else "end_date" if "end_date" in time_range else None
        if not start_key or not end_key:
            return time_range

        try:
            start_val = time_range.get(start_key)
            end_val = time_range.get(end_key)
            if not start_val or not end_val:
                return time_range

            start_dt = CALMOrchestrator._parse_dt(start_val)
            end_dt = CALMOrchestrator._parse_dt(end_val)
            if not start_dt or not end_dt:
                return time_range

            expanded = dict(time_range)
            new_start = start_dt - timedelta(days=days)
            new_end = end_dt + timedelta(days=days)

            expanded[start_key] = CALMOrchestrator._format_dt_like(start_val, new_start)
            expanded[end_key] = CALMOrchestrator._format_dt_like(end_val, new_end)
            return expanded
        except Exception:
            return time_range

    @staticmethod
    def _parse_dt(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(text, "%Y-%m-%d")
            except Exception:
                return None

    @staticmethod
    def _format_dt_like(original: Any, value: datetime) -> str:
        text = str(original)
        if "T" in text:
            return value.isoformat()
        return value.strftime("%Y-%m-%d")

    @staticmethod
    def _stringify_time_range(time_range: Any) -> str:
        if isinstance(time_range, dict):
            start = time_range.get("start") or time_range.get("start_date")
            end = time_range.get("end") or time_range.get("end_date")
            if start or end:
                return "%s -> %s" % (start, end)
        return str(time_range)

    def _log_stage(self, stage: str, **kwargs: Any) -> None:
        logger.info("[Orchestrator][Stage] %s | %s", stage, kwargs)

    def _commit_memory(
        self,
        query: str,
        result: Dict[str, Any],
        task_type: str,
    ) -> Dict[str, Any]:
        """
        Memory / reflection cuối pipeline.
        Đáp ứng đúng paper: plan -> retrieve -> predict -> validate -> memory/reflection
        """
        if not self.memory_agent:
            return {"stored": False, "reason": "memory_agent_unavailable"}

        try:
            snapshot = dict(result)
            self.memory_agent.add_episode(query, snapshot, task_type)
            self.memory_agent.add_short_term(query, snapshot)
            logger.info("[Orchestrator][Stage] memory_reflection | stored=True")
            return {"stored": True, "mode": "episode+short_term"}
        except Exception as e:
            logger.warning("[Orchestrator] Memory/reflection failed: %s", e)
            return {"stored": False, "error": str(e)}

    # ─────────────────────────────────────────
    # Data extractors
    # ─────────────────────────────────────────

    @staticmethod
    def _extract_met_data(data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Lấy dữ liệu khí tượng từ kết quả retrieve."""
        for item in data_result.get("retrieved_data", []):
            if item.get("source") in {"Copernicus CDS", "ERA5"}:
                met = item.get("data_content") or {}
                if not isinstance(met, dict):
                    return {}
                summary = met.get("summary")
                if isinstance(summary, dict):
                    return {
                        "temperature": met.get("temperature", summary.get("temperature")),
                        "humidity": met.get("humidity", summary.get("humidity")),
                        "wind_speed": met.get("wind_speed", summary.get("wind_speed")),
                        "precipitation": met.get("precipitation", summary.get("precipitation")),
                    }
                return met
        return {}

    @staticmethod
    def _extract_spatial_data(data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Lấy dữ liệu địa lý từ kết quả retrieve."""
        for item in data_result.get("retrieved_data", []):
            if item.get("source") in {"GEE", "Google Earth Engine"}:
                spatial = item.get("data_content") or {}
                if not isinstance(spatial, dict):
                    return {}
                stats = spatial.get("stats")
                if isinstance(stats, dict):
                    return {
                        **spatial,
                        "ndvi_mean": stats.get("ndvi_mean"),
                        "ndvi_min": stats.get("ndvi_min"),
                        "ndvi_max": stats.get("ndvi_max"),
                    }
                return spatial
        return {}

    # ─────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────

    @staticmethod
    def _resolve_prediction_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Đọc config prediction từ cả 2 nhánh."""
        return (
            config.get("prediction")
            or config.get("agent_config", {}).get("prediction")
            or {}
        )

    @staticmethod
    def _build_seasfire_model_runner(config: Dict[str, Any]):
        """
        Tạo SeasFireModelRunner từ config nếu có checkpoint.
        Đọc được cả:
        - cfg["prediction"]
        - cfg["agent_config"]["prediction"]

        Không silent failure:
        - ghi __model_runner_build_error vào config
        - log exception rõ ràng
        """
        pred_cfg = CALMOrchestrator._resolve_prediction_config(config)
        checkpoint = pred_cfg.get("checkpoint", "")
        if not checkpoint:
            return None

        try:
            from calm.artifact.feature_builder import SeasFireFeatureBuilder
            from calm.artifact.model_runner import SeasFireModelRunner
            from calm.artifact.seasfire_runner import SeasFireRunner

            ckpt = Path(os.path.expandvars(str(checkpoint)))
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt

            if not ckpt.exists():
                raise FileNotFoundError("Checkpoint not found: %s" % ckpt)

            seasfire = SeasFireRunner(
                checkpoint_path=ckpt,
                config={
                    "input_size": pred_cfg.get("seasfire_variables", 59),
                    "hidden_size": pred_cfg.get("hidden_size", 64),
                    "num_layers": pred_cfg.get("num_layers", 2),
                },
            )
            seasfire.load()

            dataset_path = (
                os.path.expandvars(str(pred_cfg.get("dataset_path", "") or ""))
                or os.environ.get("SEASFIRE_DATASET_PATH")
                or ""
            )

            feature_builder = None
            if dataset_path:
                ds = Path(dataset_path)
                if ds.exists():
                    feature_builder = SeasFireFeatureBuilder(
                        dataset_path=ds,
                        timesteps=pred_cfg.get("timesteps", 6),
                        target_week=pred_cfg.get("target_week", 4),
                    )
                else:
                    logger.warning(
                        "[Orchestrator] dataset_path configured but not found: %s",
                        ds,
                    )

            config["__model_runner_build_error"] = None
            logger.info("[Orchestrator] SeasFireModelRunner built successfully")
            return SeasFireModelRunner(seasfire, feature_builder)

        except Exception as e:
            msg = "Could not build SeasFireModelRunner: %s" % e
            config["__model_runner_build_error"] = msg
            logger.exception("[Orchestrator] %s", msg)
            return None

    @classmethod
    def from_llm(
        cls,
        llm,
        memory_store,
        tools: Optional[Dict[str, Any]] = None,
        model_runner=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "CALMOrchestrator":
        """
        Khởi tạo nhanh orchestrator từ LLM + memory store.
        """
        cfg = copy.deepcopy(config or {})
        _tools = dict(tools or {})

        safety = SafetyChecker(llm=llm)

        if "earth_engine" not in _tools:
            from calm.tools.earth_engine import EarthEngineTool
            _tools["earth_engine"] = EarthEngineTool(safety_checker=safety, config=cfg)

        if "copernicus" not in _tools:
            from calm.tools.copernicus import CopernicusTool
            _tools["copernicus"] = CopernicusTool(safety_checker=safety, config=cfg)

        if "web_search" not in _tools:
            from calm.tools.web_search import WebSearchTool
            _tools["web_search"] = WebSearchTool(safety_checker=safety, config=cfg)

        if "arxiv" not in _tools:
            from calm.tools.arxiv_tool import ArXivTool
            _tools["arxiv"] = ArXivTool(safety_checker=safety, config=cfg)

        if "geocoding" not in _tools:
            from calm.tools.geocoding import GeocodingTool
            _tools["geocoding"] = GeocodingTool(safety_checker=safety, config=cfg)

        memory_agent = MemoryAgent(
            long_term_store=memory_store,
            short_term_size=cfg.get("memory_short_term_size", 5),
            episodic_max=cfg.get("memory_episodic_max", 100),
        )

        planner = PlanningAgent(
            llm=llm,
            config=cfg,
            n_max=cfg.get("planner_n_max", 3),
            f_max=cfg.get("planner_f_max", 3),
        )

        data_agent = DataKnowledgeAgent(
            llm=llm,
            tools=_tools,
            memory_store=memory_agent,
            config=cfg,
        )

        web_search = _tools.get("web_search")
        qa_agent = WildfireQAAgent(
            llm=llm,
            data_agent=data_agent,
            web_search_tool=web_search,
            memory_store=memory_agent,
            config=cfg,
            n_max=cfg.get("qa_n_max", 3),
            f_max=cfg.get("qa_f_max", 3),
        )

        # Model runner:
        # 1) ưu tiên runner truyền vào
        # 2) nếu không có thì auto-build từ prediction hoặc agent_config.prediction
        _model_runner = model_runner
        pred_cfg = cls._resolve_prediction_config(cfg)

        if _model_runner is None and pred_cfg.get("checkpoint"):
            _model_runner = cls._build_seasfire_model_runner(cfg)
            if _model_runner is None:
                logger.error(
                    "[Orchestrator] checkpoint configured but model runner could not be built: %s",
                    cfg.get("__model_runner_build_error"),
                )
        elif _model_runner is None:
            logger.warning(
                "[Orchestrator] No model_runner provided and no prediction checkpoint found. "
                "PredictionAgent may fall back to heuristic mode."
            )

        prediction_agent = PredictionReasoningAgent(
            model_runner=_model_runner,
            config=cfg,
        )

        rsen = RSENModule(
            llm=llm,
            memory_store=memory_agent,
            k=cfg.get("rsen_k", 3),
        )

        router_agent = RouterAgent(llm=llm, config=cfg)
        exec_tools = {
            "data_knowledge": data_agent,
            "prediction": prediction_agent,
            "qa": qa_agent,
            "rsen": rsen,
            "web_search": _tools.get("web_search"),
        }
        executor = ExecutionAgent(
            llm=llm,
            tools=exec_tools,
            safety_checker=safety,
            config=cfg,
        )

        return cls(
            planner=planner,
            data_agent=data_agent,
            qa_agent=qa_agent,
            prediction_agent=prediction_agent,
            rsen=rsen,
            memory_agent=memory_agent,
            router_agent=router_agent,
            executor=executor,
            geocoding_tool=_tools.get("geocoding"),
            config=cfg,
        )