"""
File: 04_full_pipeline.py
Description: Example — Full CALM pipeline: plan → execute → evaluate.
Author: CALM Team
Created: 2026-03-13
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

from calm.agents.evaluator_agent import EvaluatorAgent
from calm.agents.planning_agent import PlanningAgent


def main() -> None:
    """Run full pipeline: plan, then evaluate."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
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
