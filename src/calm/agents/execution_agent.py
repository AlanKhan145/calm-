"""
Mô-đun Execution Agent — thực thi từng bước kế hoạch.

Luồng: Diễn giải bước → Kiểm tra an toàn → Gọi tool (data_knowledge, prediction,
web_search) → Trả về kết quả. Không bịa dữ liệu; lỗi trả về {"error": "...", "result": null}.
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
    Routes to: data_knowledge | prediction | qa | web_search.
    On failure: return {"error": "...", "result": null} — NEVER fabricate.
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

        Routing table (agent_name → tool key):
          "data_knowledge" → tools["data_knowledge"].retrieve()
          "prediction"     → tools["prediction"].predict()
          "qa"             → tools["qa"].invoke()
          "rsen"           → tools["rsen"].validate()
          (web_search)     → tools["web_search"].search()
        """
        action = step.get("action", "")
        agent_name = str(step.get("agent", "")).lower().replace("-", "_")
        params = step.get("parameters", {}) or {}
        result: dict[str, Any] = {}
        try:
            if agent_name == "data_knowledge":
                tool = self.tools.get("data_knowledge")
                if tool:
                    query_for_retrieve = (
                        step.get("prompt")
                        or params.get("query")
                        or step.get("query")
                        or context.get("query", "")
                    )
                    result = tool.retrieve(query_for_retrieve, params)
                else:
                    result = {
                        "error": "data_knowledge tool not available",
                        "result": None,
                    }
            elif agent_name in {"prediction", "predict_agent", "fire_prediction"}:
                tool = self.tools.get("prediction")
                if tool:
                    # Enrich params: met_data, spatial_data từ context (data_knowledge step trước đó)
                    pred_params = dict(params)
                    if not pred_params.get("location"):
                        pred_params["location"] = (context.get("parameters") or {}).get("location")
                    if not pred_params.get("time_range"):
                        pred_params["time_range"] = (context.get("parameters") or {}).get("time_range")
                    pred_params["met_data"] = context.get("met_data")
                    pred_params["spatial_data"] = context.get("spatial_data")
                    result = tool.predict(pred_params)
                else:
                    result = {
                        "error": "prediction tool not available",
                        "result": None,
                    }
            elif agent_name in {"qa", "qa_agent", "question_answering"}:
                tool = self.tools.get("qa")
                if tool:
                    q = (
                        step.get("prompt")
                        or params.get("query")
                        or step.get("query")
                        or context.get("query", "")
                    )
                    pre = context.get("data_result")
                    try:
                        result = tool.invoke(q, pre_retrieved=pre) if pre is not None else tool.invoke(q)
                    except TypeError:
                        result = tool.invoke(q)
                else:
                    result = {
                        "error": "qa tool not available",
                        "result": None,
                    }
            elif agent_name == "rsen":
                tool = self.tools.get("rsen")
                if tool:
                    prediction = context.get("prediction", {})
                    met = context.get("met_data", {})
                    spatial = context.get("spatial_data", {})
                    result = tool.validate(prediction, met, spatial)
                else:
                    result = {
                        "error": "rsen tool not available",
                        "result": None,
                    }
            elif "web_search" in str(step) or "search" in str(action).lower():
                tool = self.tools.get("web_search")
                if tool:
                    q = (
                        step.get("prompt")
                        or params.get("query")
                        or context.get("query", "")
                    )
                    results = tool.search(q)
                    result = {"results": results, "source": "web_search"}
                else:
                    result = {
                        "error": "web_search tool not available",
                        "result": None,
                    }
            else:
                result = {
                    "status": "skipped",
                    "reason": f"Unknown action: {action}",
                }
        except PermissionError as e:
            result = {"error": str(e), "result": None}
        except Exception as e:
            logger.exception("Execution step failed: %s", e)
            result = {"error": str(e), "result": None}
        return result
