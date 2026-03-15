"""
Example — Wildfire QA Agent: retrieve, evidence evaluator, trả lời.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calm.utils.env_loader import load_env
load_env()

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.memory.chroma_store import ChromaMemoryStore
from calm.tools.safety_check import SafetyChecker
from calm.tools.web_search import WebSearchTool

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
    """Chạy QA agent với câu hỏi mẫu."""
    safety = SafetyChecker(llm=llm)
    web = WebSearchTool(safety_checker=safety, config={"max_news_results": 5})
    memory = ChromaMemoryStore(
        collection_name="calm_qa_memory",
        persist_directory=".chroma",
        k=3,
    )
    tools = {
        "earth_engine": None,
        "copernicus": None,
        "web_search": web,
        "arxiv": None,
    }
    data_agent = DataKnowledgeAgent(
        llm=llm,
        tools=tools,
        memory_store=memory,
        config={"dedup_check": True},
    )
    qa = WildfireQAAgent(
        llm=llm,
        data_agent=data_agent,
        web_search_tool=web,
        memory_store=memory,
        config={},
    )
    result = qa.invoke("What caused the 2023 Canadian wildfires?")
    print("Answer:", result.get("final_output"))


if __name__ == "__main__":
    main()
