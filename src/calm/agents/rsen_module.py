"""
File: rsen_module.py
Description: RSEN — Reflexive Structured Experts Network. Parallel isolated
             weather + geo analysts. Equations 17–22.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from calm.prompt_library.rsen_prompts import (
    GEO_ANALYST_SYSTEM_PROMPT,
    OPS_COORDINATOR_SYSTEM_PROMPT,
    WEATHER_ANALYST_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class RSENModule:
    """
    Reflexive Structured Experts Network (CALM §4.3).
    Parallel ISOLATED structure — each specialist independent.
    """

    def __init__(self, llm, memory_store, k: int = 3) -> None:
        """Initialize with LLM, memory store, and top-k retrieval."""
        self.llm = llm
        self.memory = memory_store
        self.k = k

    def validate(
        self,
        prediction: dict[str, Any],
        met_data: dict[str, Any],
        spatial_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Equations 17–22: M_weather, M_geo, R_weather, R_geo (parallel)."""
        mem_w = self.memory.similarity_search(str(met_data), k=self.k) or []
        mem_g = (
            self.memory.similarity_search(str(spatial_data), k=self.k) or []
        )
        mem_c = self.memory.similarity_search(str(prediction), k=self.k) or []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
            fut_w = exe.submit(
                self._run_specialist,
                WEATHER_ANALYST_SYSTEM_PROMPT,
                prediction,
                met_data,
                mem_w,
            )
            fut_g = exe.submit(
                self._run_specialist,
                GEO_ANALYST_SYSTEM_PROMPT,
                prediction,
                spatial_data,
                mem_g,
            )
            weather_report = fut_w.result()
            geo_report = fut_g.result()

        validation = self._run_coordinator(
            prediction, weather_report, geo_report, mem_c
        )

        self.memory.add_texts([
            f"Prediction: {prediction.get('risk_level', '?')}. "
            f"Validation: {validation.get('validation_decision', '?')}. "
            f"Rationale: {validation.get('final_rationale', '')}",
        ])
        return validation

    def _run_specialist(
        self,
        system_prompt: str,
        prediction: dict,
        data: dict,
        memories: list,
    ) -> dict[str, Any]:
        """Run specialist with ONLY its domain data (NP-1.5)."""
        context = "\n".join(
            m.page_content if hasattr(m, "page_content") else str(m)
            for m in memories
        )
        full = (
            system_prompt
            + f"\n\nPast memory reflections:\n{context}"
            + f"\n\nPrediction: {json.dumps(prediction)}"
            + f"\n\nData: {json.dumps(data, default=str)}"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=full)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            if content.strip().startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln
                    for ln in lines
                    if not ln.strip().startswith("```")
                    and ln.strip() != "json"
                )
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("RSEN specialist JSON decode failed: %s", e)
            return {"error": str(e), "validation_decision": "Implausible"}

    def _run_coordinator(
        self,
        pred: dict,
        weather_rpt: dict,
        geo_rpt: dict,
        memories: list,
    ) -> dict[str, Any]:
        """NP-5.5: Both R_weather and R_geo required."""
        if "error" in weather_rpt or "error" in geo_rpt:
            return {
                "final_prediction": pred,
                "validation_decision": "Implausible",
                "reasoning_summary": {
                    "synthesis": "incomplete specialist reports"
                },
                "final_rationale": "One or more specialist reports failed.",
            }
        context = "\n".join(
            m.page_content if hasattr(m, "page_content") else str(m)
            for m in memories
        )
        full = (
            OPS_COORDINATOR_SYSTEM_PROMPT
            + f"\n\nPast memory:\n{context}"
            + f"\n\nInitial Prediction: {json.dumps(pred)}"
            + f"\n\nWeather Report: {json.dumps(weather_rpt)}"
            + f"\n\nGeo-Spatial Report: {json.dumps(geo_rpt)}"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=full)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            if content.strip().startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln
                    for ln in lines
                    if not ln.strip().startswith("```")
                    and ln.strip() != "json"
                )
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning("RSEN coordinator JSON decode failed: %s", e)
            return {
                "final_prediction": pred,
                "validation_decision": "Implausible",
                "reasoning_summary": {},
                "final_rationale": f"Coordinator parse error: {e}",
            }
