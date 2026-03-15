"""
Example — Planning Agent: phân rã câu truy vấn cháy rừng thành kế hoạch JSON.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.utils.env_loader import load_env
load_env()

from calm.agents.planning_agent import PlanningAgent

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


def main() -> None:
    """Chạy Planning Agent với query mẫu."""
    agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
    query = "Wildfire risk assessment for Amazon region next 7 days"
    result = agent.invoke(query)
    print("Plan:", result.get("final_output"))
    print("Approved:", result.get("approved"))


if __name__ == "__main__":
    main()
