"""
File: planning_agent.py
Description: Planning agent — decompose wildfire query into JSON plan
             (CALM Table A.1, URSA 3-node structure).
Author: CALM Team
Created: 2026-03-13
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


class PlanningAgent(BaseCALMAgent):
    """High-level planner: decompose, reflect, formalize into JSON plan."""

    def _generator_node(self, state: AgentState) -> dict[str, Any]:
        """Generate initial plan from query."""
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
        """Review plan for clarity, completeness, feasibility, safety."""
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
        """URSA: f_max retries for valid JSON."""
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
                if isinstance(plan, list):
                    return {"final_output": plan, "approved": True, "error": None}
                if isinstance(plan, dict) and "plan_steps" in plan:
                    return {
                        "final_output": plan["plan_steps"],
                        "approved": True,
                        "error": None,
                    }
                return {"final_output": plan, "approved": True, "error": None}
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
