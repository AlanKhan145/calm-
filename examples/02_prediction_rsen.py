"""
File: 02_prediction_rsen.py
Description: Example — RSEN validation: parallel weather + geo analysts.
Author: CALM Team
Created: 2026-03-13
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

from calm.agents.rsen_module import RSENModule
from calm.memory.chroma_store import ChromaMemoryStore


def main() -> None:
    """Run RSEN validation on prediction with met and spatial data."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    memory = ChromaMemoryStore(
        collection_name="calm_rsen_memory",
        persist_directory=".chroma",
        k=3,
    )
    rsen = RSENModule(llm=llm, memory_store=memory, k=3)
    result = rsen.validate(
        prediction={"risk_level": "High", "confidence": 0.8},
        met_data={
            "temperature": 35.0,
            "humidity": 0.2,
            "wind_speed": 15,
        },
        spatial_data={
            "fuel_type": "Shrubland",
            "slope": 25,
            "elevation": 500,
        },
    )
    print("Validation:", result.get("validation_decision"))
    rationale = result.get("final_rationale", "")[:200]
    print("Rationale:", rationale)


if __name__ == "__main__":
    main()
