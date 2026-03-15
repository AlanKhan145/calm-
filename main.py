#!/usr/bin/env python3
"""
Điểm vào chính để chạy CALM demo.

Nạp biến môi trường từ .env, khởi tạo LLM và chạy Planning Agent với một
câu truy vấn mẫu. Đảm bảo đã cài đặt package (pip install -e .) và
đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo có thể import calm khi chạy từ thư mục gốc (chưa cài package)
_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import os

from calm.utils.env_loader import load_env

load_env(_root / ".env")
load_env(_root.parent / ".env")

from calm.agents.planning_agent import PlanningAgent


def main() -> None:
    """Chạy demo Planning Agent: phân rã câu hỏi cháy rừng thành kế hoạch JSON."""
    try:
        if os.environ.get("OPENROUTER_API_KEY"):
            from langchain_openrouter import ChatOpenRouter
            llm = ChatOpenRouter(
                model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o"),
                api_key=os.environ["OPENROUTER_API_KEY"],
                temperature=0.0,
            )
        elif os.environ.get("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                openai_api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.0,
            )
        else:
            raise ValueError("Đặt OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env")
    except ImportError as e:
        print("Lỗi thiếu thư viện: pip install -e . hoặc pip install -r requirements.txt")
        print("Chi tiết:", e)
        sys.exit(1)
    except ValueError as e:
        print("Lỗi cấu hình:", e)
        sys.exit(1)

    agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
    query = "Wildfire risk assessment for Amazon region next 7 days"
    print("Query:", query)
    print("Đang chạy Planning Agent...")
    result = agent.invoke(query)
    plan = result.get("final_output") or []
    print("Kết quả:")
    print("  Approved:", result.get("approved"))
    print("  Số bước kế hoạch:", len(plan) if isinstance(plan, list) else 0)
    for i, step in enumerate(plan if isinstance(plan, list) else [], 1):
        print(f"  Bước {i}: {step.get('step_id', '?')} - {step.get('action', '?')}")


if __name__ == "__main__":
    main()
