"""
Example — Full pipeline: plan → evaluate (LLM-as-a-Judge).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.utils.env_loader import load_env
load_env()

from calm.agents.evaluator_agent import EvaluatorAgent
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
    """Chạy pipeline: plan rồi evaluator."""
    planner = PlanningAgent(llm=llm, config={})
    evaluator = EvaluatorAgent(
        llm=llm,
        config={"passing_score": 75.0},
    )
    query = "Assess wildfire risk for California Central Valley next 14 days"
    plan_result = planner.invoke(query)
    plan = plan_result.get("final_output") or []
    print("Plan steps:", len(plan))
    eval_result = evaluator.evaluate(
        response={"plan": plan, "query": query},
        query=query,
    )
    print("Evaluation passed:", eval_result.get("passed"))
    print("Average score:", eval_result.get("average_score"))


if __name__ == "__main__":
    main()
