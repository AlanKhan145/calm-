"""
Example — RSEN: xác thực dự đoán với Weather Analyst và Geo Analyst song song.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.utils.env_loader import load_env
load_env()

from calm.agents.rsen_module import RSENModule
from calm.memory.chroma_store import ChromaMemoryStore

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
    """Chạy RSEN validation với dữ liệu mẫu."""
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
