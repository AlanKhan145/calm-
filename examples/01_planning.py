"""
File: 01_planning.py
Description: Example — Planning Agent decomposes wildfire query into JSON plan.
Author: CALM Team
Created: 2026-03-13
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.agents.planning_agent import PlanningAgent
from calm.llm_factory import get_llm


def main() -> None:
    """Run planning agent on wildfire query."""
    llm = get_llm()
    agent = PlanningAgent(llm=llm, config={}, n_max=3, f_max=3)
    query = "Wildfire risk assessment for Amazon region next 7 days"
    result = agent.invoke(query)
    print("Plan:", result.get("final_output"))
    print("Approved:", result.get("approved"))


if __name__ == "__main__":
    main()
