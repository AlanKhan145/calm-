"""
Mô-đun Wildfire QA Agent — truy xuất, đánh giá bằng chứng, trả lời hoặc kích hoạt tìm kiếm.

Evidence Evaluator là cơ chế chống hallucination: chỉ trả lời khi bằng chứng đủ
theo evidence_threshold; nếu không thì kích hoạt web search rồi lặp.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from calm.agents.base_agent import AgentState, BaseCALMAgent
from calm.prompt_library.qa_prompts import (
    EVIDENCE_EVALUATOR_SYSTEM_PROMPT,
    QA_SELF_REFLECTION_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class WildfireQAAgent(BaseCALMAgent):
    """
    QA pipeline: decompose → retrieve → evidence evaluate → answer or search.
    Evidence Evaluator is the primary anti-hallucination mechanism.
    """

    def __init__(
        self,
        llm,
        data_agent,
        web_search_tool,
        memory_store,
        config: dict | None = None,
        n_max: int = 3,
        f_max: int = 3,
    ) -> None:
        """Initialize QA agent with data agent, web search, memory."""
        super().__init__(llm=llm, config=config or {}, n_max=n_max, f_max=f_max)
        self.data_agent = data_agent
        self.web_search_tool = web_search_tool
        self.memory_store = memory_store
        self.evidence_threshold = (config or {}).get("evidence_threshold", 0.65)

    def invoke(self, query: str, pre_retrieved: dict[str, Any] | None = None) -> dict[str, Any]:
        """Chạy QA. Nếu orchestrator đã gọi retrieve, truyền pre_retrieved để tránh gọi lại."""
        self._pre_retrieved = pre_retrieved
        try:
            return super().invoke(query)
        finally:
            self._pre_retrieved = None

    def _generator_node(self, state: AgentState) -> dict[str, Any]:
        """Retrieve data and run evidence evaluation. Dùng pre_retrieved nếu có (tránh duplicate)."""
        query = state["query"]
        conv = state.get("conversation") or []
        if not conv:
            pre = getattr(self, "_pre_retrieved", None)
            if pre is not None:
                retrieved = pre
                if pre.get("dedup") and hasattr(self.memory_store, "similarity_search"):
                    docs = self.memory_store.similarity_search(query, k=5)
                    if docs:
                        ctx = [getattr(d, "page_content", str(d)) for d in docs]
                        retrieved = {**pre, "cached_context": ctx, "retrieved_data": [{"data_content": c} for c in ctx]}
            else:
                retrieved = self.data_agent.retrieve(query)
            retrieved_str = json.dumps(retrieved, default=str)[:8000]
            short_term_ctx = ""
            if hasattr(self.memory_store, "get_short_term_context"):
                short_term_ctx = self.memory_store.get_short_term_context(
                    max_chars=800
                )
            ctx_extra = ""
            if short_term_ctx:
                ctx_extra = f"\n\nRecent session context:\n{short_term_ctx}"
            eval_prompt = (
                EVIDENCE_EVALUATOR_SYSTEM_PROMPT
                + f"\n\nOriginal query: {query}\n\nRetrieved data:\n"
                + retrieved_str
                + ctx_extra
            )
        else:
            last_str = str(conv[-1].content)[:8000]
            eval_prompt = (
                EVIDENCE_EVALUATOR_SYSTEM_PROMPT
                + f"\n\nOriginal query: {query}\n\nEvidence to evaluate:\n"
                + last_str
            )
        try:
            resp = self.llm.invoke([HumanMessage(content=eval_prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.exception("QA generator failed: %s", e)
            content = f"[ERROR] {e}"
        return {
            "conversation": [AIMessage(content=content)],
            "iteration": state.get("iteration", 0) + 1,
        }

    def _reflector_node(self, state: AgentState) -> dict[str, Any]:
        """Self-reflect and trigger online search if evidence fails."""
        conv = state.get("conversation") or []
        last = str(conv[-1].content) if conv else ""
        if "[APPROVED]" in last:
            return {"conversation": []}
        prompt = (
            QA_SELF_REFLECTION_SYSTEM_PROMPT
            + f"\n\nQuery: {state['query']}\n\nResponse: {last}"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            refined = {}
            if content and "{" in content:
                try:
                    refined = json.loads(content)
                except json.JSONDecodeError:
                    pass
            refined_query = refined.get("refined_query", state["query"])
            if self.web_search_tool:
                results = self.web_search_tool.search(refined_query)
                content += (
                    f"\n\n[ONLINE_SEARCH]\n"
                    f"{json.dumps(results[:5], default=str)}"
                )
        except Exception as e:
            logger.exception("QA reflector failed: %s", e)
            content = str(e)
        return {"conversation": [AIMessage(content=content)]}

    def _formalizer_node(self, state: AgentState) -> dict[str, Any]:
        """Produce final JSON answer with f_max retries."""
        conv = state.get("conversation") or []
        last = str(conv[-1].content) if conv else ""
        json_schema = (
            '{"answer": "...", "reasoning_chain": ["..."], '
            '"citations": ["..."], "confidence": 0.0}'
        )
        answer_prompt = (
            QA_SYSTEM_PROMPT
            + f"\n\nQuery: {state['query']}\n\nEvidence and context:\n{last}"
            + f"\n\nProduce final answer as JSON: {json_schema}"
        )
        for attempt in range(self.f_max):
            try:
                resp = self.llm.invoke([HumanMessage(content=answer_prompt)])
                content = (
                    resp.content.strip()
                    if hasattr(resp, "content")
                    else str(resp).strip()
                )
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(
                        ln
                        for ln in lines
                        if not ln.strip().startswith("```")
                        and ln.strip() != "json"
                    )
                out = json.loads(content)
                return {
                    "final_output": out,
                    "approved": True,
                    "error": None,
                }
            except json.JSONDecodeError as e:
                answer_prompt += f"\n\nInvalid JSON: {e}. Try again."
        return {
            "error": "QA formalization failed",
            "final_output": {
                "answer": last,
                "reasoning_chain": [],
                "citations": [],
                "confidence": 0.0,
            },
            "approved": False,
        }
