"""
File: base_agent.py
Description: Base CALM agent — mirrors URSA 3-node structure
             (generator→reflector→formalizer) using LangGraph StateGraph.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for every CALM agent graph (URSA pattern)."""

    query: str
    conversation: Annotated[list[BaseMessage], operator.add]
    approved: bool
    iteration: int
    final_output: dict[str, Any] | list[dict[str, Any]] | None
    error: str | None


class BaseCALMAgent:
    """
    Mirrors URSA agent structure exactly.
    3 mandatory nodes: generator → reflector → formalizer.
    """

    def __init__(
        self,
        llm,
        config: dict | None = None,
        n_max: int = 3,
        f_max: int = 3,
    ) -> None:
        """Initialize base agent with LLM and config."""
        self.llm = llm
        self.config = config or {}
        self.n_max = n_max
        self.f_max = f_max
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph with 3 nodes: generator, reflector, formalizer."""
        g = StateGraph(AgentState)
        g.add_node("generator", self._generator_node)
        g.add_node("reflector", self._reflector_node)
        g.add_node("formalizer", self._formalizer_node)
        g.set_entry_point("generator")
        g.add_edge("generator", "reflector")
        g.add_conditional_edges(
            "reflector",
            self._route,
            {"generator": "generator", "formalizer": "formalizer"},
        )
        g.add_edge("formalizer", END)
        return g.compile()

    def _route(self, state: AgentState) -> str:
        """URSA: [APPROVED] or n_max → formalizer."""
        conv = state.get("conversation") or []
        if not conv:
            return "formalizer"
        last = str(conv[-1].content)
        if "[APPROVED]" in last:
            return "formalizer"
        if state.get("iteration", 0) >= self.n_max:
            return "formalizer"
        return "generator"

    def _generator_node(self, state: AgentState) -> dict:
        """Generate initial response. Override in subclasses."""
        raise NotImplementedError

    def _reflector_node(self, state: AgentState) -> dict:
        """Critically review output. Override in subclasses."""
        raise NotImplementedError

    def _formalizer_node(self, state: AgentState) -> dict:
        """Convert to validated JSON. Override in subclasses."""
        raise NotImplementedError

    def invoke(self, query: str) -> dict:
        """Run the agent on a query."""
        return self.graph.invoke({
            "query": query,
            "conversation": [],
            "approved": False,
            "iteration": 0,
            "final_output": None,
            "error": None,
        })
