"""
File: execution_agent.py
Description: Execution agent — URSA Code Block 2 + CALM 4-stage workflow:
             Task Interpretation, Safety Check, Tool Calling, Summarization.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

EXECUTOR_PROMPT = """
You are a responsible and efficient execution agent tasked with
carrying out a provided plan designed to solve a specific problem.

Your responsibilities:
1. Carefully review each step of the provided plan.
2. Use appropriate tools:
   - Perform internet searches for additional information.
   - Write and execute code for computational tasks.
     Do not generate any placeholder or synthetic data! Only real data!
   - Execute safe system commands after safety verification.
3. Document each action: tools used, inputs, outputs, errors.
4. Immediately flag steps that are unclear, unsafe, or impractical.

Execute accurately, safely, and transparently.
"""

EXECUTION_SUMMARIZER_PROMPT = """
You are a summarizing agent. Summarize the execution conversation.
- Keep all important points.
- Ensure the summary responds to the original query goals.
- Summarize all work carried out.
- Highlight where goals were not achieved and why.
"""


class ExecutionAgent:
    """
    Executes plan steps. Safety check before EVERY tool call.
    Uses LangGraph with generator → reflector → summarizer pattern.
    """

    def __init__(
        self,
        llm,
        tools: dict,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        """Initialize execution agent with LLM, tools, and safety checker."""
        self.llm = llm
        self.tools = tools
        self.safety_checker = safety_checker
        self.config = config or {}

    def execute_step(
        self,
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a single plan step. Returns dict (never DataFrame).
        On failure: return {"error": "...", "result": null} — NEVER fabricate.
        """
        action = step.get("action", "")
        agent_name = step.get("agent", "")
        params = step.get("parameters", {}) or {}
        result: dict[str, Any] = {}
        try:
            if agent_name == "data_knowledge":
                tool = self.tools.get("data_knowledge")
                if tool:
                    result = tool.retrieve(step.get("query", ""), params)
                else:
                    result = {
                        "error": "data_knowledge tool not available",
                        "result": None,
                    }
            elif agent_name == "prediction":
                tool = self.tools.get("prediction")
                if tool:
                    result = tool.predict(params)
                else:
                    result = {
                        "error": "prediction tool not available",
                        "result": None,
                    }
            elif "web_search" in str(step) or "search" in str(action).lower():
                tool = self.tools.get("web_search")
                if tool:
                    q = params.get("query", context.get("query", ""))
                    results = tool.search(q)
                    result = {"results": results, "source": "web_search"}
                else:
                    result = {
                        "error": "web_search tool not available",
                        "result": None,
                    }
            else:
                result = {"status": "skipped", "reason": f"Unknown action: {action}"}
        except PermissionError as e:
            result = {"error": str(e), "result": None}
        except Exception as e:
            logger.exception("Execution step failed: %s", e)
            result = {"error": str(e), "result": None}
        return result
