"""
Mô-đun agent cơ sở — kiến trúc URSA 3 node.

Áp dụng luồng generator → reflector → formalizer (LangGraph StateGraph).
Mọi agent CALM kế thừa BaseCALMAgent và triển khai _generator_node,
_reflector_node, _formalizer_node.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Trạng thái đồ thị cho mọi agent CALM (chuẩn URSA).

    query: câu truy vấn đầu vào.
    conversation: lịch sử tin nhắn (accumulator).
    approved: đã được reflector chấp nhận hay chưa.
    iteration: số lần reflector đã lặp.
    final_output: kết quả cuối (dict hoặc list bước).
    error: thông báo lỗi nếu có.
    """

    query: str
    conversation: Annotated[list[BaseMessage], operator.add]
    approved: bool
    iteration: int
    final_output: dict[str, Any] | list[dict[str, Any]] | None
    error: str | None


class BaseCALMAgent:
    """
    Agent cơ sở theo đúng kiến trúc URSA.

    Ba node bắt buộc: generator (tạo output) → reflector (phản ánh, [APPROVED])
    → formalizer (chuyển sang JSON hợp lệ). n_max: số lần tối đa reflector
    lặp; f_max: số lần tối đa formalizer thử parse JSON.
    """

    def __init__(
        self,
        llm,
        config: dict | None = None,
        n_max: int = 3,
        f_max: int = 3,
    ) -> None:
        """Khởi tạo agent với LLM và cấu hình; dựng đồ thị LangGraph."""
        self.llm = llm
        self.config = config or {}
        self.n_max = n_max
        self.f_max = f_max
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Dựng LangGraph với 3 node: generator → reflector → formalizer."""
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
        """Điều hướng: nếu [APPROVED] hoặc đạt n_max thì sang formalizer."""
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
        """Tạo phản hồi ban đầu. Phải override ở lớp con."""
        raise NotImplementedError

    def _reflector_node(self, state: AgentState) -> dict:
        """Rà soát output và quyết định [APPROVED] hay lặp. Override ở lớp con."""
        raise NotImplementedError

    def _formalizer_node(self, state: AgentState) -> dict:
        """Chuyển output sang JSON hợp lệ. Override ở lớp con."""
        raise NotImplementedError

    def invoke(self, query: str) -> dict:
        """Chạy agent với câu truy vấn; trả về state cuối (final_output, approved, error)."""
        return self.graph.invoke({
            "query": query,
            "conversation": [],
            "approved": False,
            "iteration": 0,
            "final_output": None,
            "error": None,
        })
