"""
Mô-đun Planning Agent — phân rã câu truy vấn cháy rừng thành kế hoạch JSON.

Theo chuẩn URSA 3 node:
    generator -> reflector -> formalizer

Bản này siết chặt:
- Prompt nhận thêm normalized_query_context.
- Formalizer ép mọi step có đủ:
    step_id, agent, action, parameters, expected_output, success_criteria
- Với task prediction, plan tối thiểu phải có đủ chuỗi:
    resolve query context
    retrieve environmental data
    build features
    run model
    validate via RSEN
    refine/finalize
- Không cho final plan thiếu agent.
- Không cho prediction step có parameters rỗng.
- Reflection lưu lỗi plan vào memory theo best-effort.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage

from calm.agents.base_agent import AgentState, BaseCALMAgent
from calm.prompt_library.planning_prompts import (
    PLANNER_FORMALIZE_PROMPT,
    PLANNER_REFLECTION_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Mapping action keywords -> agent
_ACTION_TO_AGENT = {
    "resolve query context": "execution",
    "normalize query": "execution",
    "retrieve_knowledge": "data_knowledge",
    "retrieve_data": "data_knowledge",
    "collect_data": "data_knowledge",
    "retrieve_historical": "data_knowledge",
    "retrieve environmental data": "data_knowledge",
    "satellite": "data_knowledge",
    "web_search": "qa",
    "search": "qa",
    "prediction_reasoning": "prediction",
    "invoke_prediction": "prediction",
    "run_model": "prediction",
    "run model": "prediction",
    "predict": "prediction",
    "build features": "prediction",
    "feature": "prediction",
    "compile_report": "qa",
    "evaluate": "rsen",
    "validate": "rsen",
    "validate_prediction": "rsen",
    "validate via rsen": "rsen",
    "rsen": "rsen",
    "refine": "execution",
    "finalize": "execution",
    "refine/finalize": "execution",
}

_REQUIRED_STEP_KEYS = [
    "step_id",
    "agent",
    "action",
    "parameters",
    "expected_output",
    "success_criteria",
]

_REQUIRED_PREDICTION_CHAIN = [
    "resolve query context",
    "retrieve environmental data",
    "build features",
    "run model",
    "validate via RSEN",
    "refine/finalize",
]


def _infer_agent(step: Dict[str, Any]) -> str:
    """Suy luận agent từ action nếu thiếu."""
    agent = step.get("agent")
    if agent and str(agent).strip():
        return str(agent).strip()

    action = str(step.get("action", "")).lower().replace("-", "_")
    for keyword, agent_name in _ACTION_TO_AGENT.items():
        if keyword.replace("-", "_") in action:
            return agent_name
    return ""


def _safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _strip_code_fence(text: str) -> str:
    content = (text or "").strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    cleaned: List[str] = []
    for line in lines:
        striped = line.strip().lower()
        if striped.startswith("```"):
            continue
        if striped == "json":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class PlanningAgent(BaseCALMAgent):
    """Agent lập kế hoạch cấp cao: query -> reflection -> JSON plan."""

    # ─────────────────────────────────────────
    # Node 1: Generator
    # ─────────────────────────────────────────

    def _generator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Tạo kế hoạch nháp ban đầu.
        Prompt nhận cả original query và normalized_query_context.
        """
        prompt = self._build_generator_prompt(state)
        msgs = [HumanMessage(content=prompt)]
        msgs += state.get("conversation") or []

        try:
            resp = self.llm.invoke(msgs)
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("Planning generator failed: %s", e)
            content = "[ERROR] Generator failed: %s" % e

        return {
            "conversation": [AIMessage(content=content)],
            "iteration": state.get("iteration", 0) + 1,
        }

    # ─────────────────────────────────────────
    # Node 2: Reflector
    # ─────────────────────────────────────────

    def _reflector_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Rà soát kế hoạch:
        - đủ bước chưa
        - có agent chưa
        - prediction chain đã đủ chưa
        - có dùng normalized_query_context hay đang tự đoán lại time/location
        """
        prompt = self._build_reflection_prompt(state)
        msgs = [HumanMessage(content=prompt)]
        msgs += state.get("conversation") or []

        reflection_text = ""
        reflection_errors: List[str] = []
        approved = False

        try:
            resp = self.llm.invoke(msgs)
            reflection_text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("Planning reflector failed: %s", e)
            reflection_text = "[ERROR] Reflection failed: %s" % e

        approved = "[APPROVED]" in reflection_text
        if not approved:
            reflection_errors = self._extract_reflection_errors(reflection_text)
            self._store_reflection_issue_to_memory(
                query=state.get("query", ""),
                normalized_query_context=self._get_normalized_query_context(state),
                reflection_text=reflection_text,
                reflection_errors=reflection_errors,
            )

        return {
            "conversation": [AIMessage(content=reflection_text)],
            "reflection_approved": approved,
            "reflection_errors": reflection_errors,
        }

    # ─────────────────────────────────────────
    # Node 3: Formalizer
    # ─────────────────────────────────────────

    def _formalizer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Chuyển nội dung đã duyệt sang JSON hợp lệ; tối đa f_max lần thử.

        Siết chặt:
        - mọi step phải có đủ schema
        - prediction plan phải có đủ chain tối thiểu
        - prediction step không được có parameters rỗng
        """
        conv = list(state.get("conversation") or [])
        normalized_ctx = self._get_normalized_query_context(state)
        task_type = self._infer_task_type(state.get("query", ""), normalized_ctx)
        last_content = ""
        last_errors: List[str] = []

        for attempt in range(self.f_max):
            prompt = self._build_formalizer_prompt(state, task_type=task_type)
            msgs = [HumanMessage(content=prompt)] + conv

            try:
                resp = self.llm.invoke(msgs)
                content = resp.content if hasattr(resp, "content") else str(resp)
                last_content = (content or "").strip()

                parsed = json.loads(_strip_code_fence(last_content))
                steps = self._extract_steps_from_json(parsed)

                steps = self._normalize_steps(
                    steps=steps,
                    task_type=task_type,
                    normalized_query_context=normalized_ctx,
                )

                validation_errors = self._validate_steps(
                    steps=steps,
                    task_type=task_type,
                    normalized_query_context=normalized_ctx,
                )

                if validation_errors:
                    last_errors = validation_errors
                    self._store_reflection_issue_to_memory(
                        query=state.get("query", ""),
                        normalized_query_context=normalized_ctx,
                        reflection_text="Formalizer validation failed",
                        reflection_errors=validation_errors,
                    )
                    conv.append(AIMessage(content=last_content))
                    conv.append(
                        HumanMessage(
                            content=(
                                "Your JSON plan is invalid.\n"
                                "Fix these issues and return ONLY valid JSON.\n"
                                + "\n".join("- " + e for e in validation_errors)
                            )
                        )
                    )
                    continue

                return {
                    "final_output": steps,
                    "approved": True,
                    "error": None,
                    "reflection_errors": state.get("reflection_errors", []),
                }

            except json.JSONDecodeError as e:
                last_errors = ["Invalid JSON: %s" % e]
                logger.warning("Planning formalizer JSON decode failed: %s", e)
                conv.append(AIMessage(content=last_content))
                conv.append(
                    HumanMessage(
                        content=(
                            "Your response was not valid JSON.\n"
                            "Error: %s\n"
                            "Return ONLY valid JSON." % e
                        )
                    )
                )
            except Exception as e:
                last_errors = ["Formalizer exception: %s" % e]
                logger.exception("Formalizer attempt %s failed: %s", attempt + 1, e)
                conv.append(AIMessage(content=last_content))
                conv.append(
                    HumanMessage(
                        content=(
                            "JSON parsing/normalization failed: %s\n"
                            "Try again and return ONLY valid JSON." % e
                        )
                    )
                )

        self._store_reflection_issue_to_memory(
            query=state.get("query", ""),
            normalized_query_context=normalized_ctx,
            reflection_text="Formalizer failed after retries",
            reflection_errors=last_errors or ["JSON formalization failed after f_max attempts"],
        )

        return {
            "error": "JSON formalization failed after f_max attempts",
            "final_output": None,
            "approved": False,
            "reflection_errors": state.get("reflection_errors", []) + last_errors,
        }

    # ─────────────────────────────────────────
    # Prompt builders
    # ─────────────────────────────────────────

    def _build_generator_prompt(self, state: AgentState) -> str:
        query = state.get("query", "")
        normalized_ctx = self._get_normalized_query_context(state)
        task_type = self._infer_task_type(query, normalized_ctx)

        extra_rules = """
Planner requirements:
1) You must extract or reuse:
   - location
   - time_range
   - task_type
2) You must use normalized_query_context as source of truth for ambiguous time/location.
   Do NOT guess again if normalized context already contains them.
3) For each step, specify:
   - step_id
   - agent
   - action
   - parameters
   - expected_output
   - success_criteria
4) If task_type is prediction, your plan must include at least these stages:
   - resolve query context
   - retrieve environmental data
   - build features
   - run model
   - validate via RSEN
   - refine/finalize
5) Parameters must be concrete and non-empty for prediction-related steps.
6) Output should be easy to formalize into strict JSON.
"""

        return (
            PLANNER_SYSTEM_PROMPT
            + "\n\n"
            + extra_rules.strip()
            + "\n\nOriginal query:\n"
            + str(query)
            + "\n\nNormalized query context:\n"
            + _safe_json_dumps(normalized_ctx)
            + "\n\nDetected task type:\n"
            + task_type
            + "\n\nGenerate a high-quality step-by-step plan draft."
        )

    def _build_reflection_prompt(self, state: AgentState) -> str:
        query = state.get("query", "")
        normalized_ctx = self._get_normalized_query_context(state)
        task_type = self._infer_task_type(query, normalized_ctx)

        extra_rules = """
Review the draft plan strictly.

Check:
- every step has a clear agent and action
- parameters are concrete enough to execute
- expected_output and success_criteria are present and measurable
- if query time is ambiguous (e.g. "next 7 days", "next week"), the plan uses normalized_query_context instead of re-guessing
- if task_type is prediction, the plan contains ALL required stages:
  1. resolve query context
  2. retrieve environmental data
  3. build features
  4. run model
  5. validate via RSEN
  6. refine/finalize

Response rules:
- If fully valid, include [APPROVED].
- If invalid, list concrete issues as bullet points.
- Mention missing step fields explicitly.
- Mention if any agent is missing.
- Mention if any prediction step has empty parameters.
"""

        return (
            PLANNER_REFLECTION_PROMPT
            + "\n\n"
            + extra_rules.strip()
            + "\n\nOriginal task:\n"
            + str(query)
            + "\n\nNormalized query context:\n"
            + _safe_json_dumps(normalized_ctx)
            + "\n\nDetected task type:\n"
            + task_type
        )

    def _build_formalizer_prompt(self, state: AgentState, task_type: str) -> str:
        normalized_ctx = self._get_normalized_query_context(state)

        extra_rules = """
Return ONLY valid JSON.

Schema:
{
  "plan_steps": [
    {
      "step_id": "step_1",
      "agent": "execution|data_knowledge|prediction|rsen|qa",
      "action": "string",
      "parameters": { "any": "structured values" },
      "expected_output": "string or structured summary",
      "success_criteria": "string or list"
    }
  ]
}

Strict requirements:
- Every step MUST contain:
  step_id, agent, action, parameters, expected_output, success_criteria
- agent MUST NOT be empty
- parameters MUST be a non-empty object for prediction steps
- Use normalized_query_context as the source of truth for time/location if provided
- Do not omit required stages for prediction plans

For prediction tasks, include at least these stages in order:
1. resolve query context
2. retrieve environmental data
3. build features
4. run model
5. validate via RSEN
6. refine/finalize
"""

        return (
            PLANNER_FORMALIZE_PROMPT
            + "\n\n"
            + extra_rules.strip()
            + "\n\nDetected task type:\n"
            + task_type
            + "\n\nNormalized query context:\n"
            + _safe_json_dumps(normalized_ctx)
        )

    # ─────────────────────────────────────────
    # Normalization / validation
    # ─────────────────────────────────────────

    def _extract_steps_from_json(self, parsed: Any) -> List[Dict[str, Any]]:
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]

        if isinstance(parsed, dict):
            if isinstance(parsed.get("plan_steps"), list):
                return [s for s in parsed["plan_steps"] if isinstance(s, dict)]
            return [parsed]

        raise ValueError("Formalizer output must be a JSON object or list")

    def _normalize_steps(
        self,
        steps: List[Dict[str, Any]],
        task_type: str,
        normalized_query_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa plan:
        - ép đủ field
        - bổ sung agent nếu suy luận được
        - prediction: ép đủ chain tối thiểu và đúng thứ tự
        """
        normalized: List[Dict[str, Any]] = []

        for idx, raw_step in enumerate(steps, start=1):
            step = dict(raw_step)

            if not step.get("step_id"):
                step["step_id"] = "step_%d" % idx

            if not step.get("agent"):
                inferred = _infer_agent(step)
                if inferred:
                    step["agent"] = inferred

            step["action"] = str(step.get("action", "")).strip()

            if not isinstance(step.get("parameters"), dict):
                step["parameters"] = {}

            if not step["parameters"]:
                step["parameters"] = self._default_parameters_for_step(
                    action=step["action"],
                    task_type=task_type,
                    normalized_query_context=normalized_query_context,
                )

            if not step.get("expected_output"):
                step["expected_output"] = self._default_expected_output(
                    action=step["action"],
                    agent=step.get("agent", ""),
                )

            if not step.get("success_criteria"):
                step["success_criteria"] = self._default_success_criteria(
                    action=step["action"],
                    agent=step.get("agent", ""),
                )

            normalized.append(step)

        if task_type == "prediction":
            normalized = self._ensure_prediction_chain(
                steps=normalized,
                normalized_query_context=normalized_query_context,
            )

        # Reindex step_id sau khi chèn/sắp xếp lại
        for idx, step in enumerate(normalized, start=1):
            step["step_id"] = "step_%d" % idx

        return normalized

    def _validate_steps(
        self,
        steps: List[Dict[str, Any]],
        task_type: str,
        normalized_query_context: Dict[str, Any],
    ) -> List[str]:
        errors: List[str] = []

        if not isinstance(steps, list) or not steps:
            return ["Plan must contain at least one step"]

        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                errors.append("Step %d is not a JSON object" % idx)
                continue

            for key in _REQUIRED_STEP_KEYS:
                if key not in step:
                    errors.append("Step %d missing required field: %s" % (idx, key))
                    continue

                if key == "parameters":
                    if not isinstance(step.get("parameters"), dict):
                        errors.append("Step %d parameters must be a JSON object" % idx)
                else:
                    value = step.get(key)
                    if value is None or not str(value).strip():
                        errors.append("Step %d field '%s' must not be empty" % (idx, key))

            if not step.get("agent"):
                errors.append("Step %d missing agent and could not infer one" % idx)

            if task_type == "prediction":
                if not isinstance(step.get("parameters"), dict) or not step.get("parameters"):
                    errors.append("Prediction plan step %d must have non-empty parameters" % idx)

        if task_type == "prediction":
            prediction_chain_errors = self._validate_prediction_chain(steps)
            errors.extend(prediction_chain_errors)

            # Nếu normalized context đã có time/location thì plan không được bỏ mất hoàn toàn
            if normalized_query_context:
                has_context_step = any(
                    self._action_matches(str(step.get("action", "")), "resolve query context")
                    for step in steps
                )
                if not has_context_step:
                    errors.append(
                        "Prediction plan must include 'resolve query context' step using normalized_query_context"
                    )

        return errors

    def _validate_prediction_chain(self, steps: List[Dict[str, Any]]) -> List[str]:
        errors: List[str] = []
        actions = [str(step.get("action", "")) for step in steps]

        for required_action in _REQUIRED_PREDICTION_CHAIN:
            if not any(self._action_matches(action, required_action) for action in actions):
                errors.append(
                    "Prediction plan missing required stage: %s" % required_action
                )

        return errors

    # ─────────────────────────────────────────
    # Prediction chain enforcement
    # ─────────────────────────────────────────

    def _ensure_prediction_chain(
        self,
        steps: List[Dict[str, Any]],
        normalized_query_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Với prediction, ép đủ chain tối thiểu và đúng thứ tự.
        Nếu thiếu step, thêm step mặc định có parameters không rỗng.
        """
        ordered_steps: List[Dict[str, Any]] = []
        used_indexes = set()

        for required_action in _REQUIRED_PREDICTION_CHAIN:
            matched_index = None
            matched_step = None

            for idx, step in enumerate(steps):
                if idx in used_indexes:
                    continue
                if self._action_matches(str(step.get("action", "")), required_action):
                    matched_index = idx
                    matched_step = dict(step)
                    break

            if matched_step is None:
                matched_step = self._build_required_prediction_step(
                    action=required_action,
                    normalized_query_context=normalized_query_context,
                )
            else:
                used_indexes.add(matched_index)
                matched_step["action"] = required_action
                if not matched_step.get("agent"):
                    matched_step["agent"] = _infer_agent(matched_step)
                if not matched_step.get("parameters"):
                    matched_step["parameters"] = self._default_parameters_for_step(
                        action=required_action,
                        task_type="prediction",
                        normalized_query_context=normalized_query_context,
                    )
                if not matched_step.get("expected_output"):
                    matched_step["expected_output"] = self._default_expected_output(
                        action=required_action,
                        agent=matched_step.get("agent", ""),
                    )
                if not matched_step.get("success_criteria"):
                    matched_step["success_criteria"] = self._default_success_criteria(
                        action=required_action,
                        agent=matched_step.get("agent", ""),
                    )

            ordered_steps.append(matched_step)

        # Thêm các step phụ còn lại xuống cuối
        for idx, step in enumerate(steps):
            if idx not in used_indexes:
                ordered_steps.append(step)

        return ordered_steps

    def _build_required_prediction_step(
        self,
        action: str,
        normalized_query_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent = _infer_agent({"action": action}) or "execution"
        return {
            "step_id": "",
            "agent": agent,
            "action": action,
            "parameters": self._default_parameters_for_step(
                action=action,
                task_type="prediction",
                normalized_query_context=normalized_query_context,
            ),
            "expected_output": self._default_expected_output(action=action, agent=agent),
            "success_criteria": self._default_success_criteria(action=action, agent=agent),
        }

    def _action_matches(self, action_text: str, canonical_action: str) -> bool:
        text = str(action_text or "").strip().lower()
        target = canonical_action.strip().lower()

        keyword_groups = {
            "resolve query context": ["resolve", "context", "normalize", "query context"],
            "retrieve environmental data": ["retrieve", "environment", "data", "meteorology", "satellite"],
            "build features": ["build", "feature"],
            "run model": ["run", "model", "predict"],
            "validate via rsen": ["validate", "rsen", "evaluate"],
            "refine/finalize": ["refine", "finalize", "finalise", "postprocess"],
        }

        if target not in keyword_groups:
            return target in text

        keywords = keyword_groups[target]
        return any(k in text for k in keywords)

    # ─────────────────────────────────────────
    # Defaults
    # ─────────────────────────────────────────

    def _default_parameters_for_step(
        self,
        action: str,
        task_type: str,
        normalized_query_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_context = {
            "original_query": normalized_query_context.get("original_query"),
            "normalized_query": normalized_query_context.get("normalized_query"),
            "normalized_query_context": normalized_query_context,
        }

        location = normalized_query_context.get("location")
        coordinates = normalized_query_context.get("coordinates") or {}
        time_range = normalized_query_context.get("time_range")

        action_lower = str(action or "").strip().lower()

        if "resolve" in action_lower and "context" in action_lower:
            return base_context

        if "retrieve" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "data_sources": ["earth_engine", "copernicus", "web_search"],
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {})}

        if "build" in action_lower and "feature" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "input_refs": ["retrieved_environmental_data"],
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {})}

        if "run" in action_lower and "model" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "model_input_ref": "built_features",
                "task_type": task_type,
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {})}

        if "validate" in action_lower or "rsen" in action_lower:
            return {
                "prediction_ref": "model_output",
                "met_data_ref": "retrieved_environmental_data",
                "spatial_data_ref": "retrieved_environmental_data",
                "validator": "RSEN",
                "query_context": base_context,
            }

        if "refine" in action_lower or "finalize" in action_lower:
            return {
                "prediction_ref": "validated_prediction",
                "refinement_policy": {
                    "requery_if_implausible": True,
                    "expand_time_window_if_needed": True,
                    "downgrade_confidence_if_data_weak": True,
                },
                "query_context": base_context,
            }

        return {
            "query_context": base_context,
            "task_type": task_type,
        }

    @staticmethod
    def _default_expected_output(action: str, agent: str) -> str:
        action_lower = str(action or "").lower()

        if "resolve" in action_lower and "context" in action_lower:
            return "Resolved query context with location, coordinates, time_range, and execution-ready parameters"
        if "retrieve" in action_lower:
            return "Retrieved environmental data package including meteorological and spatial inputs"
        if "build" in action_lower and "feature" in action_lower:
            return "Feature set ready for wildfire prediction model"
        if "run" in action_lower and "model" in action_lower:
            return "Raw wildfire prediction output with risk_level, confidence, and model rationale"
        if "validate" in action_lower or "rsen" in action_lower:
            return "RSEN validation result with plausibility decision and rationale"
        if "refine" in action_lower or "finalize" in action_lower:
            return "Final refined result with fallback/refinement trace and confidence adjustment"

        return "Execution result produced by agent '%s'" % agent

    @staticmethod
    def _default_success_criteria(action: str, agent: str) -> Any:
        action_lower = str(action or "").lower()

        if "resolve" in action_lower and "context" in action_lower:
            return [
                "location/time_range extracted or reused from normalized_query_context",
                "no ambiguous time/location guessing remains",
            ]
        if "retrieve" in action_lower:
            return [
                "at least one relevant environmental source retrieved",
                "data payload is structured and usable downstream",
            ]
        if "build" in action_lower and "feature" in action_lower:
            return [
                "feature payload is non-empty",
                "feature schema is compatible with prediction step",
            ]
        if "run" in action_lower and "model" in action_lower:
            return [
                "prediction output contains risk_level and confidence",
                "no model invocation error",
            ]
        if "validate" in action_lower or "rsen" in action_lower:
            return [
                "RSEN returns decision and rationale",
                "prediction plausibility is assessed",
            ]
        if "refine" in action_lower or "finalize" in action_lower:
            return [
                "final output includes refinement outcome",
                "confidence adjusted if validation/data quality is weak",
            ]

        return "Step executed successfully by agent '%s'" % agent

    # ─────────────────────────────────────────
    # Reflection memory
    # ─────────────────────────────────────────

    def _extract_reflection_errors(self, reflection_text: str) -> List[str]:
        errors: List[str] = []
        for line in (reflection_text or "").splitlines():
            striped = line.strip()
            if striped.startswith("- "):
                errors.append(striped[2:].strip())
            elif striped.startswith("* "):
                errors.append(striped[2:].strip())
        if not errors and reflection_text and "[APPROVED]" not in reflection_text:
            errors.append(reflection_text.strip())
        return errors

    def _store_reflection_issue_to_memory(
        self,
        query: str,
        normalized_query_context: Dict[str, Any],
        reflection_text: str,
        reflection_errors: List[str],
    ) -> None:
        """
        Best-effort persistence cho lỗi planning/reflection.
        Để bám FR-P04 -> FR-P09.
        """
        payload = {
            "type": "planning_reflection_issue",
            "query": query,
            "normalized_query_context": normalized_query_context,
            "reflection_errors": reflection_errors,
            "reflection_text": reflection_text,
            "tags": ["planning", "reflection", "FR-P04", "FR-P05", "FR-P06", "FR-P07", "FR-P08", "FR-P09"],
        }

        stored = False

        # Các kiểu memory có thể tồn tại tùy hệ thống
        memory_candidates = [
            getattr(self, "memory_store", None),
            getattr(self, "memory_agent", None),
            getattr(self, "memory", None),
        ]

        for memory_obj in memory_candidates:
            if memory_obj is None:
                continue

            try:
                if hasattr(memory_obj, "add_episode"):
                    memory_obj.add_episode(query, payload, "planning")
                    stored = True
                if hasattr(memory_obj, "add_short_term"):
                    memory_obj.add_short_term(query, payload)
                    stored = True
                if hasattr(memory_obj, "add"):
                    memory_obj.add(payload)
                    stored = True
                if hasattr(memory_obj, "store"):
                    memory_obj.store(payload)
                    stored = True
                if hasattr(memory_obj, "save"):
                    memory_obj.save(payload)
                    stored = True
            except Exception as e:
                logger.debug("Could not store planning reflection issue in memory: %s", e)

        if stored:
            logger.info("Stored planning reflection issue to memory")
        else:
            logger.info("Planning reflection issue detected but memory store unavailable")

    # ─────────────────────────────────────────
    # Context helpers
    # ─────────────────────────────────────────

    def _get_normalized_query_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Ưu tiên lấy từ state.
        Hỗ trợ nhiều key để dễ tích hợp với orchestrator mới.
        """
        candidates = [
            state.get("normalized_query_context"),
            state.get("context", {}).get("normalized_query_context") if isinstance(state.get("context"), dict) else None,
            state.get("parameters", {}).get("normalized_query_context") if isinstance(state.get("parameters"), dict) else None,
        ]

        for item in candidates:
            if isinstance(item, dict) and item:
                return item

        # fallback rất nhẹ: tạo context tối thiểu từ query
        query = state.get("query", "")
        return {
            "original_query": query,
            "normalized_query": query,
            "location": None,
            "coordinates": {},
            "time_range": None,
        }

    def _infer_task_type(self, query: str, normalized_query_context: Optional[Dict[str, Any]] = None) -> str:
        text = " ".join(
            str(x)
            for x in [
                query or "",
                _safe_json_dumps(normalized_query_context or {}),
            ]
        ).lower()

        prediction_keywords = [
            "predict",
            "prediction",
            "forecast",
            "risk",
            "wildfire risk",
            "fire danger",
            "next week",
            "next 7 days",
            "next days",
        ]
        if any(keyword in text for keyword in prediction_keywords):
            return "prediction"
        return "qa"