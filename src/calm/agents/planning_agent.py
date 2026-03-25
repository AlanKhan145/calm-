"""
Mô-đun Planning Agent — phân rã câu truy vấn thành kế hoạch JSON.

Thiết kế:
    generator -> reflector -> formalizer

Điểm chính của bản viết lại:
- Parse LLM response robust hơn:
  - hỗ trợ content là string / list block / object
  - xử lý empty response
  - bóc JSON từ prose, markdown fence, object/array lẫn trong text
- Siết schema từng step:
  - step_id, agent, action, parameters, expected_output, success_criteria
- Với task prediction:
  - ép đủ chain tối thiểu
  - kiểm tra thứ tự chain
  - prediction step phải có parameters không rỗng
- Tận dụng normalized_query_context làm source of truth
- Reflection/formalizer failures lưu vào memory theo best-effort
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

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
    "retrieve data": "data_knowledge",
    "retrieve_data": "data_knowledge",
    "collect_data": "data_knowledge",
    "retrieve_historical": "data_knowledge",
    "retrieve environmental data": "data_knowledge",
    "satellite": "data_knowledge",
    "meteorology": "data_knowledge",
    "weather": "data_knowledge",
    "web_search": "qa",
    "search": "qa",
    "qa": "qa",
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


def _safe_json_dumps(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _infer_agent(step: Dict[str, Any]) -> str:
    agent = step.get("agent")
    if agent and str(agent).strip():
        return str(agent).strip()

    action = str(step.get("action", "")).lower().replace("-", "_")
    for keyword, agent_name in _ACTION_TO_AGENT.items():
        if keyword.replace("-", "_") in action:
            return agent_name
    return ""


def _strip_code_fence(text: str) -> str:
    content = (text or "").strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("```"):
            continue
        if stripped == "json":
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class PlanningAgent(BaseCALMAgent):
    """Agent lập kế hoạch cấp cao: query -> reflection -> JSON plan."""

    # ─────────────────────────────────────────
    # Public / orchestration helpers
    # ─────────────────────────────────────────

    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Driver tối giản nếu orchestrator gọi trực tiếp agent.
        Nếu hệ thống của bạn đã có graph riêng thì có thể không dùng hàm này.
        """
        generated = self._generator_node(state)
        reflect_state = dict(state)
        reflect_state["conversation"] = list(state.get("conversation") or []) + list(
            generated.get("conversation") or []
        )

        reflected = self._reflector_node(reflect_state)

        formalize_state = dict(reflect_state)
        formalize_state["conversation"] = list(reflect_state.get("conversation") or []) + list(
            reflected.get("conversation") or []
        )
        formalize_state["reflection_errors"] = reflected.get("reflection_errors", [])
        formalize_state["reflection_approved"] = reflected.get("reflection_approved", False)

        return self._formalizer_node(formalize_state)

    # ─────────────────────────────────────────
    # Node 1: Generator
    # ─────────────────────────────────────────

    def _generator_node(self, state: AgentState) -> Dict[str, Any]:
        prompt = self._build_generator_prompt(state)
        msgs = [HumanMessage(content=prompt)]
        msgs += state.get("conversation") or []

        try:
            resp = self.llm.invoke(msgs)
            content = self._extract_llm_text(resp)
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
        prompt = self._build_reflection_prompt(state)
        msgs = [HumanMessage(content=prompt)]
        msgs += state.get("conversation") or []

        reflection_text = ""
        reflection_errors: List[str] = []
        approved = False

        try:
            resp = self.llm.invoke(msgs)
            reflection_text = self._extract_llm_text(resp)
        except Exception as e:
            logger.exception("Planning reflector failed: %s", e)
            reflection_text = "[ERROR] Reflection failed: %s" % e

        approved = "[APPROVED]" in (reflection_text or "")
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

        Bản robust:
        - không crash nếu content rỗng
        - không crash nếu content là list block
        - bóc JSON từ prose / code fence
        - validate chặt hơn prediction chain
        """
        conv = list(state.get("conversation") or [])
        normalized_ctx = self._get_normalized_query_context(state)
        task_type = self._infer_task_type(state.get("query", ""), normalized_ctx)

        last_content = ""
        last_errors: List[str] = []
        max_attempts = getattr(self, "f_max", 3) or 3

        for attempt in range(max_attempts):
            prompt = self._build_formalizer_prompt(state, task_type=task_type)
            msgs = [HumanMessage(content=prompt)] + conv

            try:
                resp = self.llm.invoke(msgs)
                last_content = self._extract_llm_text(resp).strip()

                if not last_content:
                    raise json.JSONDecodeError("Empty response from LLM", "", 0)

                parsed = self._parse_json_loose(last_content)
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
                    conv.append(AIMessage(content=last_content or "[EMPTY RESPONSE]"))
                    conv.append(
                        HumanMessage(
                            content=(
                                "Your JSON plan is invalid.\n"
                                "Fix these issues and return ONLY valid JSON.\n"
                                "Do not use markdown. Do not add explanations.\n"
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
                logger.warning(
                    "Planning formalizer JSON decode failed: %s | raw=%r",
                    e,
                    (last_content or "")[:1000],
                )
                conv.append(AIMessage(content=last_content or "[EMPTY RESPONSE]"))
                conv.append(
                    HumanMessage(
                        content=(
                            "Your response was not valid JSON.\n"
                            "Error: %s\n"
                            "Return ONLY valid JSON matching schema {\"plan_steps\": [...]}.\n"
                            "Do not wrap in markdown. Do not include explanations.\n"
                            "If unsure, return {\"plan_steps\": []}." % e
                        )
                    )
                )
            except Exception as e:
                last_errors = ["Formalizer exception: %s" % e]
                logger.exception("Formalizer attempt %s failed: %s", attempt + 1, e)
                conv.append(AIMessage(content=last_content or "[NO CONTENT]"))
                conv.append(
                    HumanMessage(
                        content=(
                            "JSON parsing/normalization failed: %s\n"
                            "Try again and return ONLY valid JSON.\n"
                            "Do not wrap in markdown. Do not add explanations." % e
                        )
                    )
                )

        self._store_reflection_issue_to_memory(
            query=state.get("query", ""),
            normalized_query_context=normalized_ctx,
            reflection_text="Formalizer failed after retries",
            reflection_errors=last_errors or ["JSON formalization failed after retries"],
        )

        return {
            "error": "JSON formalization failed after retries",
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
7) Prefer concise, execution-ready actions over vague prose.
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
- if task_type is prediction, the stages should appear in that logical order

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
Do not wrap in markdown.
Do not include explanations.
Do not include leading or trailing text.
If you are unsure, still return valid JSON.

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
    # LLM output extraction / JSON parsing
    # ─────────────────────────────────────────

    def _extract_llm_text(self, resp: Any) -> str:
        if resp is None:
            return ""

        if isinstance(resp, str):
            return resp.strip()

        content = getattr(resp, "content", None)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue

                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                    continue

                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    parts.append(text_attr.strip())
                    continue

                content_attr = getattr(item, "content", None)
                if isinstance(content_attr, str) and content_attr.strip():
                    parts.append(content_attr.strip())

            return "\n".join(parts).strip()

        text = getattr(resp, "text", None)
        if isinstance(text, str):
            return text.strip()

        generations = getattr(resp, "generations", None)
        if generations:
            try:
                first = generations[0]
                if isinstance(first, list) and first:
                    item = first[0]
                    item_text = getattr(item, "text", None)
                    if isinstance(item_text, str):
                        return item_text.strip()
            except Exception:
                pass

        return str(resp).strip()

    def _parse_json_loose(self, raw: str) -> Any:
        text = (raw or "").strip()
        if not text:
            raise json.JSONDecodeError("Empty response from LLM", "", 0)

        cleaned = _strip_code_fence(text).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        fenced_candidates = self._extract_fenced_blocks(text)
        for candidate in fenced_candidates:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        json_candidate = self._find_first_json_substring(cleaned)
        if json_candidate:
            return json.loads(json_candidate)

        raise json.JSONDecodeError("No JSON object/array found in model response", text, 0)

    @staticmethod
    def _extract_fenced_blocks(raw: str) -> List[str]:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw or "", flags=re.DOTALL | re.IGNORECASE)
        return [b.strip() for b in blocks if isinstance(b, str) and b.strip()]

    @staticmethod
    def _find_first_json_substring(raw: str) -> Optional[str]:
        """
        Tìm object/array JSON đầu tiên bằng bracket matching,
        tránh regex tham lam gây bắt sai khi text dài.
        """
        if not raw:
            return None

        for opener, closer in (("{", "}"), ("[", "]")):
            start = raw.find(opener)
            while start != -1:
                depth = 0
                in_string = False
                escape = False

                for idx in range(start, len(raw)):
                    ch = raw[idx]

                    if in_string:
                        if escape:
                            escape = False
                        elif ch == "\\":
                            escape = True
                        elif ch == '"':
                            in_string = False
                        continue

                    if ch == '"':
                        in_string = True
                        continue

                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = raw[start : idx + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                break

                start = raw.find(opener, start + 1)

        return None

    # ─────────────────────────────────────────
    # Normalization / validation
    # ─────────────────────────────────────────

    def _extract_steps_from_json(self, parsed: Any) -> List[Dict[str, Any]]:
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]

        if isinstance(parsed, dict):
            if isinstance(parsed.get("plan_steps"), list):
                return [s for s in parsed["plan_steps"] if isinstance(s, dict)]

            # hỗ trợ model trả {"steps": [...]}
            if isinstance(parsed.get("steps"), list):
                return [s for s in parsed["steps"] if isinstance(s, dict)]

            # fallback: coi cả object là 1 step nếu có shape phù hợp
            return [parsed]

        raise ValueError("Formalizer output must be a JSON object or list")

    def _normalize_steps(
        self,
        steps: List[Dict[str, Any]],
        task_type: str,
        normalized_query_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for idx, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                continue

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
            errors.extend(self._validate_prediction_chain(steps))

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

        positions: Dict[str, int] = {}
        for required_action in _REQUIRED_PREDICTION_CHAIN:
            found_index = None
            for idx, action in enumerate(actions):
                if self._action_matches(action, required_action):
                    found_index = idx
                    break
            if found_index is None:
                errors.append("Prediction plan missing required stage: %s" % required_action)
            else:
                positions[required_action] = found_index

        if len(positions) == len(_REQUIRED_PREDICTION_CHAIN):
            ordered = [positions[a] for a in _REQUIRED_PREDICTION_CHAIN]
            if ordered != sorted(ordered):
                errors.append(
                    "Prediction plan stages are present but not in the required order: "
                    + " -> ".join(_REQUIRED_PREDICTION_CHAIN)
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

                if not isinstance(matched_step.get("parameters"), dict) or not matched_step.get("parameters"):
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

        synonym_groups = {
            "resolve query context": [
                "resolve query context",
                "resolve context",
                "normalize query",
                "query normalization",
                "query context",
                "resolve location and time",
            ],
            "retrieve environmental data": [
                "retrieve environmental data",
                "retrieve data",
                "collect data",
                "environmental data",
                "meteorology",
                "satellite",
                "weather",
            ],
            "build features": [
                "build features",
                "feature engineering",
                "construct features",
                "prepare features",
            ],
            "run model": [
                "run model",
                "predict",
                "prediction",
                "invoke model",
                "invoke prediction",
            ],
            "validate via rsen": [
                "validate via rsen",
                "validate",
                "rsen",
                "evaluate plausibility",
            ],
            "refine/finalize": [
                "refine/finalize",
                "refine",
                "finalize",
                "finalise",
                "postprocess",
            ],
        }

        variants = synonym_groups.get(target)
        if not variants:
            return target in text

        return any(variant in text for variant in variants)

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

        location = (
            normalized_query_context.get("location")
            or normalized_query_context.get("location_text")
            or normalized_query_context.get("location_resolved", {}).get("name")
            if isinstance(normalized_query_context.get("location_resolved"), dict)
            else normalized_query_context.get("location")
        )

        coordinates = (
            normalized_query_context.get("coordinates")
            or {
                "lat": normalized_query_context.get("lat"),
                "lon": normalized_query_context.get("lon"),
            }
        )
        if coordinates == {"lat": None, "lon": None}:
            coordinates = {}

        time_range = normalized_query_context.get("time_range")
        prediction_target = normalized_query_context.get("prediction_target")
        requested_output = normalized_query_context.get("requested_output") or []

        action_lower = str(action or "").strip().lower()

        if "resolve" in action_lower and "context" in action_lower:
            return base_context

        if "retrieve" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "prediction_target": prediction_target,
                "requested_output": requested_output,
                "data_sources": ["earth_engine", "copernicus", "web_search"],
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {}, [])}

        if "build" in action_lower and "feature" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "prediction_target": prediction_target,
                "input_refs": ["retrieved_environmental_data"],
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {}, [])}

        if "run" in action_lower and "model" in action_lower:
            params = {
                "location": location,
                "coordinates": coordinates,
                "time_range": time_range,
                "prediction_target": prediction_target,
                "model_input_ref": "built_features",
                "task_type": task_type,
                "requested_output": requested_output,
                "query_context": base_context,
            }
            return {k: v for k, v in params.items() if v not in (None, "", {}, [])}

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
                "requested_output": requested_output,
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
            return "Raw prediction output with risk_level, confidence, and model rationale"
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
            stripped = line.strip()
            if stripped.startswith("- "):
                errors.append(stripped[2:].strip())
            elif stripped.startswith("* "):
                errors.append(stripped[2:].strip())

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
        payload = {
            "type": "planning_reflection_issue",
            "query": query,
            "normalized_query_context": normalized_query_context,
            "reflection_errors": reflection_errors,
            "reflection_text": reflection_text,
            "tags": [
                "planning",
                "reflection",
                "FR-P04",
                "FR-P05",
                "FR-P06",
                "FR-P07",
                "FR-P08",
                "FR-P09",
            ],
        }

        stored = False
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
        candidates = [
            state.get("normalized_query_context"),
            state.get("context", {}).get("normalized_query_context")
            if isinstance(state.get("context"), dict)
            else None,
            state.get("parameters", {}).get("normalized_query_context")
            if isinstance(state.get("parameters"), dict)
            else None,
        ]

        for item in candidates:
            if isinstance(item, dict) and item:
                return item

        query = state.get("query", "")
        return {
            "original_query": query,
            "normalized_query": query,
            "location": None,
            "coordinates": {},
            "time_range": None,
        }

    def _infer_task_type(
        self,
        query: str,
        normalized_query_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = normalized_query_context or {}

        direct = context.get("task_type")
        if isinstance(direct, str) and direct.strip():
            return direct.strip().lower()

        text = " ".join(
            str(x)
            for x in [
                query or "",
                _safe_json_dumps(context),
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
            "dự đoán",
            "nguy cơ",
            "rủi ro",
        ]

        if any(keyword in text for keyword in prediction_keywords):
            return "prediction"

        return "qa"