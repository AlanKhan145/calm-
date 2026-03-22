"""
RouterAgent — xác định task_type, confidence, required_artifacts từ plan.

Thay thế keyword fallback trong _classify_intent bằng LLM-based routing
trả về TaskRouting (task_type, confidence, required_artifacts, next_steps).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from calm.schemas.contracts import TaskRouting

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """
You are a task router for a wildfire monitoring system. Given a user query and the plan steps,
determine the primary task type and what artifacts are needed.

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
    Router dựa trên plan + query. Trả về TaskRouting thay vì keyword fallback.
    """

    def __init__(self, llm, config: dict | None = None) -> None:
        self.llm = llm
        self.config = config or {}
        self._fallback_keywords = {
            "prediction": {"predict", "forecast", "risk", "detect", "likelihood", "next week", "next days"},
            "qa": {"what", "why", "how", "explain", "describe", "causes", "information"},
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
            + f"\n\nQuery: {query}\n\nPlan steps:\n{json.dumps(plan_steps[:5], default=str)}"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln for ln in lines
                    if not ln.strip().startswith("```") and ln.strip() != "json"
                )
            data = json.loads(content)
            return TaskRouting(
                task_type=data.get("task_type", "qa"),
                confidence=float(data.get("confidence", 0.5)),
                required_artifacts=data.get("required_artifacts", []),
                next_steps=data.get("next_steps", []),
                reasoning=data.get("reasoning", ""),
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
        for step in (plan_steps or []):
            action = str(step.get("action", "")).lower()
            agent = str(step.get("agent", "")).lower()
            if any(w in action or w in agent for w in ["predict", "forecast", "model", "run_model"]):
                return TaskRouting(task_type="prediction", confidence=0.7, reasoning="From plan action/agent")
            if any(w in action or w in agent for w in ["retrieve", "web_search", "qa", "compile_report"]):
                return TaskRouting(task_type="qa", confidence=0.7, reasoning="From plan action/agent")

        for w in self._fallback_keywords["prediction"]:
            if w in q_lower:
                return TaskRouting(task_type="prediction", confidence=0.6, reasoning=f"Keyword: {w}")
        for w in self._fallback_keywords["qa"]:
            if w in q_lower:
                return TaskRouting(task_type="qa", confidence=0.6, reasoning=f"Keyword: {w}")

        return TaskRouting(task_type="qa", confidence=0.5, reasoning="Default to QA")
