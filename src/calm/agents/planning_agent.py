"""
Mô-đun Planning Agent — phân rã câu truy vấn cháy rừng thành kế hoạch JSON.

Theo chuẩn URSA 3 node: generator (tạo bước) → reflector (rà soát, [APPROVED])
→ formalizer (xuất JSON plan_steps). Dùng cho bảng A.1 trong tài liệu CALM.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from calm.agents.base_agent import AgentState, BaseCALMAgent
from calm.prompt_library.planning_prompts import (
    PLANNER_FORMALIZE_PROMPT,
    PLANNER_REFLECTION_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Mapping action keywords -> agent (khi LLM không trả về agent)
_ACTION_TO_AGENT = {
    "retrieve_knowledge": "data_knowledge",
    "retrieve_data": "data_knowledge",
    "collect_data": "data_knowledge",
    "retrieve_historical": "data_knowledge",
    "satellite": "data_knowledge",
    "web_search": "qa",
    "search": "qa",
    "prediction_reasoning": "prediction",
    "invoke_prediction": "prediction",
    "run_model": "prediction",
    "predict": "prediction",
    "compile_report": "qa",
    "evaluate": "rsen",
    "validate": "rsen",
    "validate_prediction": "rsen",
}


def _infer_agent(step: dict[str, Any]) -> str:
    """Suy luận agent từ action khi agent thiếu. Default: execution."""
    agent = step.get("agent")
    if agent and str(agent).strip():
        return str(agent).strip()
    action = str(step.get("action", "")).lower().replace("-", "_")
    for kw, ag in _ACTION_TO_AGENT.items():
        if kw in action:
            return ag
    return "execution"


class PlanningAgent(BaseCALMAgent):
    """Agent lập kế hoạch cấp cao: phân rã query → phản ánh → formalize thành JSON."""

    def _generator_node(self, state: AgentState) -> dict[str, Any]:
        """Tạo kế hoạch ban đầu từ câu truy vấn (bước đầu tiên trong 3 node)."""
        msgs = [
            HumanMessage(
                content=PLANNER_SYSTEM_PROMPT + "\n\nTask: " + state["query"]
            )
        ]
        msgs += state.get("conversation") or []
        try:
            resp = self.llm.invoke(msgs)
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("Planning generator failed: %s", e)
            content = f"[ERROR] Generator failed: {e}"
        return {
            "conversation": [AIMessage(content=content)],
            "iteration": state.get("iteration", 0) + 1,
        }

    def _reflector_node(self, state: AgentState) -> dict[str, Any]:
        """Rà soát kế hoạch: rõ ràng, đầy đủ, khả thi, an toàn; trả về [APPROVED] nếu ổn."""
        msgs = [
            HumanMessage(
                content=(
                    PLANNER_REFLECTION_PROMPT
                    + "\n\nOriginal task: "
                    + state["query"]
                )
            )
        ]
        msgs += state.get("conversation") or []
        try:
            resp = self.llm.invoke(msgs)
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("Planning reflector failed: %s", e)
            content = f"[ERROR] Reflection failed: {e}"
        return {"conversation": [AIMessage(content=content)]}

    def _formalizer_node(self, state: AgentState) -> dict[str, Any]:
        """Chuyển nội dung đã duyệt sang JSON hợp lệ; tối đa f_max lần thử."""
        conv = list(state.get("conversation") or [])
        last_content = ""
        for attempt in range(self.f_max):
            msgs = [HumanMessage(content=PLANNER_FORMALIZE_PROMPT)] + conv
            try:
                resp = self.llm.invoke(msgs)
                content = (
                    resp.content.strip()
                    if hasattr(resp, "content")
                    else str(resp).strip()
                )
                last_content = content
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(
                        ln
                        for ln in lines
                        if not ln.startswith("```") and ln.strip() != "json"
                    )
                plan = json.loads(content)
                steps = plan if isinstance(plan, list) else plan.get("plan_steps", plan)
                if not isinstance(steps, list):
                    steps = [plan]
                # Điền agent nếu thiếu (fallback từ action)
                for s in steps:
                    if isinstance(s, dict) and not s.get("agent"):
                        s["agent"] = _infer_agent(s)
                if isinstance(plan, list):
                    return {"final_output": steps, "approved": True, "error": None}
                if isinstance(plan, dict) and "plan_steps" in plan:
                    return {
                        "final_output": steps,
                        "approved": True,
                        "error": None,
                    }
                return {"final_output": steps, "approved": True, "error": None}
            except json.JSONDecodeError as e:
                conv.append(AIMessage(content=last_content))
                conv.append(
                    HumanMessage(
                        content=f"Your response was not valid JSON. "
                        f"Error: {e}. Try again."
                    )
                )
            except Exception as e:
                logger.exception("Formalizer attempt %s failed: %s", attempt + 1, e)
                conv.append(AIMessage(content=last_content))
                conv.append(
                    HumanMessage(content=f"JSON parsing failed: {e}. Try again.")
                )
        return {
            "error": "JSON formalization failed after f_max attempts",
            "final_output": None,
            "approved": False,
        }
