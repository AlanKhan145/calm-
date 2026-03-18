"""
Example 04 — Full Pipeline với CALMOrchestrator.

Mục đích: minh hoạ khái niệm AGENT thực sự:
  - Người dùng chỉ gọi orchestrator.run(query).
  - Hệ thống TỰ ĐỘNG nhận dạng loại yêu cầu (QA / Prediction)
    và định tuyến sang đúng pipeline — không cần viết use-case riêng.
  - Sau đó EvaluatorAgent chấm điểm (LLM-as-a-Judge) kết quả.

Luồng:
  Query
    → PlanningAgent (JSON plan)
    → _classify_intent (đọc plan + từ khoá)
    ├── "qa"         → DataAgent → ChromaDB → WildfireQAAgent
    └── "prediction" → DataAgent → PredictionAgent → RSENModule
    → EvaluatorAgent (scores 0–100)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.utils.env_loader import load_env

load_env()

from calm.agents.evaluator_agent import EvaluatorAgent
from calm.memory.chroma_store import ChromaMemoryStore
from calm.orchestrator import CALMOrchestrator
from calm.tools.web_search import WebSearchTool


# ── Khởi tạo LLM ────────────────────────────────────────────────────────────
def _build_llm():
    if os.environ.get("OPENROUTER_API_KEY"):
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            temperature=0.0,
        )
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.0,
        )
    raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")


def main() -> None:
    """
    Chạy hai truy vấn khác loại qua một orchestrator duy nhất;
    in kết quả và điểm Evaluator cho mỗi truy vấn.
    """
    llm = _build_llm()

    memory = ChromaMemoryStore(
        collection_name="calm_pipeline",
        persist_directory=os.path.join(os.path.dirname(__file__), "..", ".chroma"),
        use_openai_embeddings=bool(os.environ.get("OPENAI_API_KEY")),
    )

    tools: dict = {}
    try:
        tools["web_search"] = WebSearchTool()
    except Exception:
        pass

    orchestrator = CALMOrchestrator.from_llm(
        llm=llm,
        memory_store=memory,
        tools=tools,
        config={"planner_n_max": 2, "qa_n_max": 2},
    )

    evaluator = EvaluatorAgent(
        llm=llm,
        config={"passing_score": 70.0},
    )

    queries = [
        # ── QA: sẽ tự route sang QA Pipeline ───────────────────────────
        "What are the primary environmental factors contributing to the "
        "increased wildfire frequency in the Amazon rainforest?",
        # ── Prediction: sẽ tự route sang Prediction Pipeline ────────────
        "Predict wildfire risk for California Central Valley over the next 7 days",
    ]

    for query in queries:
        print(f"\n{'─'*64}")
        print(f"[Query]  {query}")
        print(f"{'─'*64}")

        # Orchestrator tự định tuyến
        result = orchestrator.run(query)
        task_type = result.get("task_type", "?")
        print(f"  → Routed to : {task_type.upper()} pipeline")
        print(f"  → Plan steps: {len(result.get('plan_steps', []))}")

        if task_type == "qa":
            print(f"  → Answer    : {str(result.get('answer', ''))[:200]}")
            print(f"  → Confidence: {result.get('confidence', 0.0):.2f}")
            citations = result.get("citations", [])
            if citations:
                print(f"  → Citations : {citations[0]}")
        else:
            print(f"  → Risk level: {result.get('risk_level', '?')}")
            print(f"  → RSEN      : {result.get('decision', '?')}")
            rationale = str(result.get("rationale", ""))
            if rationale:
                print(f"  → Rationale : {rationale[:200]}")

        if result.get("error"):
            print(f"  [!] Error   : {result['error']}")

        # EvaluatorAgent chấm điểm
        eval_result = evaluator.evaluate(response=result, query=query)
        print(
            f"  → Eval score: {eval_result.get('average_score', 0.0):.1f}/100  "
            f"(passed={eval_result.get('passed', False)})"
        )


if __name__ == "__main__":
    main()
