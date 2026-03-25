from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from calm.schemas.contracts import TaskRouting

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """
You are a task router for a wildfire monitoring system.
Given a user query and the plan steps, determine the primary task type
and what artifacts are needed.

Task types:
- qa: question answering, information retrieval, explain, describe, what/why/how
- prediction: risk assessment, forecast, predict, detect, likelihood, next days/weeks
- hybrid: query clearly needs both (e.g. "explain and predict")

Required artifacts:
- evidence: for QA (retrieved data, citations)
- prediction: for prediction (model output)
- met_data: meteorological data for RSEN validation
- spatial_data: geospatial data for RSEN validation

Respond with JSON only:
{
  "task_type": "qa" | "prediction" | "hybrid",
  "confidence": 0.0-1.0,
  "required_artifacts": ["evidence", "prediction", ...],
  "next_steps": ["brief step description"],
  "reasoning": "one sentence"
}
"""


class RouterAgent:
    """
    Router dựa trên plan + query.
    Trả về TaskRouting, fallback về keyword nếu LLM fail.
    """

    def __init__(self, llm, config: dict | None = None) -> None:
        self.llm = llm
        self.config = config or {}
        # Tín hiệu chi tiết nằm ở calm.utils.intent_hints (tránh substring "risk" → prediction nhầm).
        self._fallback_keywords = {
            "qa": {
                "what",
                "why",
                "how",
                "explain",
                "describe",
                "causes",
                "information",
            },
        }

    def route(
        self,
        query: str,
        plan_steps: list[dict[str, Any]],
    ) -> TaskRouting:
        """
        Xác định task_type, confidence, required_artifacts từ plan + query.
        Fallback về keyword nếu LLM fail.
        """
        if not plan_steps:
            return self._keyword_fallback(query)

        prompt = (
            ROUTER_SYSTEM_PROMPT
            + f"\n\nQuery: {query}\n\nPlan steps:\n{json.dumps(plan_steps[:5], default=str, ensure_ascii=False)}"
        )

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = self._strip_code_fence(str(content).strip())

            data = json.loads(content)

            next_steps = self._ensure_str_list(data.get("next_steps", []))
            required_artifacts = self._ensure_str_list(
                data.get("required_artifacts", [])
            )

            confidence_raw = data.get("confidence", 0.5)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            task_type = str(data.get("task_type", "qa")).strip().lower()
            if task_type not in {"qa", "prediction", "hybrid"}:
                task_type = "qa"

            return TaskRouting(
                task_type=task_type,
                confidence=confidence,
                required_artifacts=required_artifacts,
                next_steps=next_steps,
                reasoning=str(data.get("reasoning", "")).strip(),
            )

        except Exception as e:
            logger.warning("RouterAgent LLM failed, using keyword fallback: %s", e)
            return self._keyword_fallback(query, plan_steps)

    def _keyword_fallback(
        self,
        query: str,
        plan_steps: list[dict[str, Any]] | None = None,
    ) -> TaskRouting:
        """Fallback dựa trên plan action/agent và query keywords."""
        q_lower = query.lower()

        # Duyệt toàn bộ plan: retrieve thường đứng trước predict — không được trả QA sớm.
        plan_pred = False
        plan_qa = False
        for step in (plan_steps or []):
            action = str(step.get("action", "")).lower()
            agent = str(step.get("agent", "")).lower()
            if any(w in action or w in agent for w in ["predict", "forecast", "model", "run_model"]):
                plan_pred = True
            if any(w in action or w in agent for w in ["retrieve", "web_search", "qa", "compile_report"]):
                plan_qa = True

        if plan_pred:
            return TaskRouting(
                task_type="prediction",
                confidence=0.7,
                required_artifacts=["prediction"],
                next_steps=["Run prediction workflow based on planned model step."],
                reasoning="From plan action/agent",
            )
        if plan_qa:
            return TaskRouting(
                task_type="qa",
                confidence=0.7,
                required_artifacts=["evidence"],
                next_steps=["Retrieve evidence and answer the question."],
                reasoning="From plan action/agent",
            )

        from calm.utils.intent_hints import query_suggests_prediction

        if query_suggests_prediction(query):
            return TaskRouting(
                task_type="prediction",
                confidence=0.65,
                required_artifacts=["prediction"],
                next_steps=["Run prediction pipeline."],
                reasoning="Heuristic: forecast/predict/risk-window patterns in query",
            )

        for w in self._fallback_keywords["qa"]:
            if w in q_lower:
                return TaskRouting(
                    task_type="qa",
                    confidence=0.6,
                    required_artifacts=["evidence"],
                    next_steps=["Retrieve evidence and answer the question."],
                    reasoning=f"Keyword: {w}",
                )

        return TaskRouting(
            task_type="qa",
            confidence=0.5,
            required_artifacts=["evidence"],
            next_steps=["Retrieve evidence and answer the question."],
            reasoning="Default to QA",
        )

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        """Chuẩn hóa value thành list[str]."""
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Bỏ ```json ... ``` nếu LLM trả về fenced code block."""
        if text.startswith("```"):
            lines = text.splitlines()
            cleaned = [
                line
                for line in lines
                if not line.strip().startswith("```") and line.strip().lower() != "json"
            ]
            return "\n".join(cleaned).strip()
        return text