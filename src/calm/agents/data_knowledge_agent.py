"""
Mô-đun Data & Knowledge Management Agent — Thu thập, Trích xuất, Truy xuất.

Nguồn: GEE, Copernicus CDS, DuckDuckGo, ArXiv. FR-D05: kiểm tra trùng lặp (dedup)
trước khi crawl để tránh thu thập dư thừa.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from calm.prompt_library.data_prompts import KNOWLEDGE_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class DataKnowledgeAgent:
    """
    CALM §4.2: Collection → Extraction → Retrieval.
    FR-D05: dedup check BEFORE crawling.
    """

    def __init__(
        self,
        llm,
        tools: dict,
        memory_store,
        config: dict | None = None,
    ) -> None:
        """Initialize with LLM, tools dict, memory store, and config."""
        self.llm = llm
        self.tools = tools
        self.memory = memory_store
        self.config = config or {}
        self.dedup_check = self.config.get("dedup_check", True)

    def collect(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Collection: C(q, A, T; S) → D.
        FR-D05: Check deduplication before crawling.
        """
        params = parameters or {}
        location = params.get("location", "")
        time_range = params.get("time_range", {})

        if self.dedup_check:
            existing = self.memory.similarity_search(query, k=1)
            if existing and existing[0]:
                logger.info(
                    "Dedup: similar query found, avoiding redundant crawl"
                )
                return {"source": "cache", "dedup": True}

        data: dict[str, Any] = {
            "retrieval_summary": {"original_query": query},
            "retrieved_data": [],
        }

        gee = self.tools.get("earth_engine")
        cds = self.tools.get("copernicus")
        web = self.tools.get("web_search")
        arxiv_tool = self.tools.get("arxiv")

        if gee and location:
            try:
                stats = gee.fetch_satellite_stats(
                    location=location,
                    time_range=time_range,
                )
                data["retrieved_data"].append({
                    "sub_question_id": "satellite",
                    "data_content": stats,
                    "source": "GEE",
                    "citation": "Google Earth Engine",
                    "confidence_score": 0.9,
                })
            except Exception as e:
                logger.warning("GEE collection failed: %s", e)

        if cds and location:
            try:
                lat = (
                    location.get("lat", 0.0)
                    if isinstance(location, dict)
                    else 0.0
                )
                lon = (
                    location.get("lon", 0.0)
                    if isinstance(location, dict)
                    else 0.0
                )
                met = cds.fetch_era5(
                    lat=lat, lon=lon, time_range=time_range
                )
                data["retrieved_data"].append({
                    "sub_question_id": "met",
                    "data_content": met,
                    "source": "Copernicus CDS",
                    "citation": "ERA5",
                    "confidence_score": 0.9,
                })
            except Exception as e:
                logger.warning("CDS collection failed: %s", e)

        if web:
            try:
                max_news = self.config.get("max_news_results", 10)
                results = web.search(query, max_results=max_news)
                for i, r in enumerate(results):
                    data["retrieved_data"].append({
                        "sub_question_id": f"news-{i}",
                        "data_content": r,
                        "source": "DuckDuckGo",
                        "citation": r.get("url", ""),
                        "confidence_score": 0.7,
                    })
            except Exception as e:
                logger.warning("Web search collection failed: %s", e)

        if arxiv_tool:
            try:
                max_papers = self.config.get("max_arxiv_papers", 3)
                papers = arxiv_tool.search(query, max_results=max_papers)
                for i, p in enumerate(papers):
                    data["retrieved_data"].append({
                        "sub_question_id": f"arxiv-{i}",
                        "data_content": p,
                        "source": "ArXiv",
                        "citation": p.get("url", ""),
                        "confidence_score": 0.85,
                    })
            except Exception as e:
                logger.warning("ArXiv collection failed: %s", e)

        return data

    def extract_knowledge(self, text: str) -> dict[str, list[str]]:
        """Extract factual_statements and causal_relationships from text."""
        prompt = KNOWLEDGE_EXTRACTION_PROMPT + f"\n\nText:\n{text}"
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            if content.strip().startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln
                    for ln in lines
                    if not ln.strip().startswith("```")
                    and ln.strip() != "json"
                )
            out = json.loads(content)
            return {
                "factual_statements": out.get("factual_statements", []),
                "causal_relationships": out.get("causal_relationships", []),
            }
        except json.JSONDecodeError:
            return {"factual_statements": [], "causal_relationships": []}

    def retrieve(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full pipeline: collect, extract, store. Returns dict (not DataFrame)."""
        collected = self.collect(query, parameters)
        if isinstance(collected, dict) and collected.get("dedup"):
            return collected
        texts = []
        for item in collected.get("retrieved_data", []):
            dc = item.get("data_content")
            if isinstance(dc, str):
                texts.append(dc)
            elif isinstance(dc, dict):
                texts.append(json.dumps(dc, default=str))
        knowledge = {"factual_statements": [], "causal_relationships": []}
        for t in texts[:5]:
            k = self.extract_knowledge(t)
            knowledge["factual_statements"].extend(
                k.get("factual_statements", [])
            )
            knowledge["causal_relationships"].extend(
                k.get("causal_relationships", [])
            )
        if knowledge["factual_statements"] or knowledge["causal_relationships"]:
            self.memory.add_texts(
                knowledge["factual_statements"]
                + knowledge["causal_relationships"]
            )
        return {
            "retrieval_summary": collected.get("retrieval_summary", {}),
            "retrieved_data": collected.get("retrieved_data", []),
            "extracted_knowledge": knowledge,
        }
