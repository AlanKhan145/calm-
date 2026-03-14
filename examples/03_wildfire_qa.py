"""
File: 03_wildfire_qa.py
Description: Example — Wildfire QA Agent: retrieve, evidence eval, answer.
Author: CALM Team
Created: 2026-03-13
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_openai import ChatOpenAI

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.memory.chroma_store import ChromaMemoryStore
from calm.tools.safety_check import SafetyChecker
from calm.tools.web_search import WebSearchTool


def main() -> None:
    """Run QA agent on wildfire question."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
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
