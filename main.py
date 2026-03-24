#!/usr/bin/env python3
"""
CALM — Điểm vào chính với CALMOrchestrator.

Hệ thống tự động định tuyến câu truy vấn sang đúng pipeline:
  • Câu hỏi  → QA Pipeline   (DataAgent → ChromaDB → WildfireQAAgent)
  • Dự đoán  → Prediction Pipeline (DataAgent → PredictionAgent → RSEN)

Không cần viết use-case riêng cho từng loại task. Chỉ cần gọi:
    orchestrator.run(query)

Đảm bảo đã cài đặt:  pip install -e .
Và đặt API key trong .env: OPENAI_API_KEY hoặc OPENROUTER_API_KEY
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import os

from calm.utils.env_loader import load_env

load_env(_root / ".env")
load_env(_root.parent / ".env")

from calm.memory.chroma_store import ChromaMemoryStore
from calm.orchestrator import CALMOrchestrator


def _build_llm():
    """Khởi tạo LLM từ biến môi trường."""
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
    raise ValueError(
        "Chưa có API key. Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env"
    )


def _print_result(label: str, result: dict) -> None:
    """In kết quả ra stdout theo loại task."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    task_type = result.get("task_type", "?")
    print(f"  Task type  : {task_type}")
    print(f"  Plan steps : {len(result.get('plan_steps', []))}")
    print(f"  Error      : {result.get('error') or 'None'}")

    if task_type == "qa":
        print(f"  Answer     : {str(result.get('answer', ''))[:300]}")
        print(f"  Confidence : {result.get('confidence', 0.0):.2f}")
        citations = result.get("citations", [])
        if citations:
            print(f"  Citations  : {citations[0]}")
        print(f"  Approved   : {result.get('approved', False)}")
    elif task_type == "prediction":
        print(f"  Risk level : {result.get('risk_level', '?')}")
        print(f"  Confidence : {result.get('confidence', 0.0):.2f}")
        print(f"  RSEN decision : {result.get('decision', '?')}")
        rationale = str(result.get("rationale", ""))
        if rationale:
            print(f"  Rationale  : {rationale[:300]}")


def main() -> None:
    """
    Demo: hai câu truy vấn khác loại → orchestrator tự định tuyến.

    Query 1 (QA)         → route sang QA Pipeline
    Query 2 (Prediction) → route sang Prediction Pipeline
    """
    try:
        llm = _build_llm()
    except (ImportError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ── Khởi tạo ChromaDB memory store ──────────────────────────────────
    memory_store = ChromaMemoryStore(
        collection_name="calm_main",
        persist_directory=str(_root / ".chroma"),
        use_openai_embeddings=bool(os.environ.get("OPENAI_API_KEY")),
    )

    # ── Tạo Orchestrator ─────────────────────────────────────────────────
    orchestrator = CALMOrchestrator.from_llm(
        llm=llm,
        memory_store=memory_store,
        tools={},
        config={"planner_n_max": 2, "qa_n_max": 2},
    )

    # ════════════════════════════════════════════════════════════════════
    # Demo 1 — Câu hỏi: tự route sang QA Pipeline
    # ════════════════════════════════════════════════════════════════════
    qa_query = (
        "What are the primary environmental factors contributing to the "
        "increased wildfire frequency in the Amazon rainforest?"
    )
    print(f"\n[Query 1 — QA] {qa_query}")
    qa_result = orchestrator.run(qa_query)
    _print_result("QA Pipeline Result", qa_result)

    # ════════════════════════════════════════════════════════════════════
    # Demo 2 — Dự đoán: tự route sang Prediction Pipeline
    # ════════════════════════════════════════════════════════════════════
    pred_query = (
        "Predict wildfire risk for California Central Valley over the next 7 days"
    )
    print(f"\n[Query 2 — Prediction] {pred_query}")
    pred_result = orchestrator.run(pred_query)
    _print_result("Prediction Pipeline Result", pred_result)


if __name__ == "__main__":
    main()
