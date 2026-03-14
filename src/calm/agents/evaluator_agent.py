"""
File: evaluator_agent.py
Description: Evaluator Agent — LLM-as-a-Judge with 5 criteria:
             Data-Accuracy, Explainability, Jargon-Avoidance,
             Redundancy-Avoidance, Citation-Quality.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """
You are the CALM Evaluator Agent using LLM-as-a-Judge methodology.
Assess the quality of CALM system outputs on a 0–100 scale across
5 criteria (from CALM paper §5.2):

1. Data-Accuracy (0-100):
   How well does the response provide facts matching verified evidence?
   - 90-100: all claims verifiable, proper citations
   - 70-89: mostly accurate, minor unsupported claims
   - <70: significant factual errors or unsupported assertions

2. Explainability (0-100):
   How clearly and logically does the response explain its reasoning?
   - 90-100: step-by-step causal reasoning, clear chain-of-thought
   - 70-89: mostly logical but some gaps in reasoning
   - <70: unclear or missing causal explanations

3. Jargon-Avoidance (0-100):
   Are technical terms explained simply for non-experts?
   - 90-100: complex terms defined, accessible to non-specialists
   - 70-89: mostly accessible with occasional unexplained jargon
   - <70: heavy unexplained technical language

4. Redundancy-Avoidance (0-100):
   Does the response provide direct answers without repetition?
   - 90-100: concise, no repeated information
   - 70-89: minor repetition
   - <70: significant redundancy

5. Citation-Quality (0-100):
   How accurately does the response link information to sources?
   - 90-100: all claims cited, sources credible and relevant
   - 70-89: most claims cited, sources generally reliable
   - <70: missing or low-quality citations

Output ONLY valid JSON:
{
  "scores": {
    "data_accuracy": 0,
    "explainability": 0,
    "jargon_avoidance": 0,
    "redundancy_avoidance": 0,
    "citation_quality": 0
  },
  "average_score": 0.0,
  "assessment_summary": "...",
  "recommendations": ["..."]
}
"""


class EvaluatorAgent:
    """LLM-as-a-Judge on 5 criteria. NFR-AC03: passing_score >= 75."""

    def __init__(self, llm, config: dict | None = None) -> None:
        """Initialize with LLM and config (passing_score, criteria)."""
        self.llm = llm
        self.config = config or {}
        self.passing_score = self.config.get("passing_score", 75.0)
        self.criteria = self.config.get(
            "criteria",
            [
                "data_accuracy",
                "explainability",
                "jargon_avoidance",
                "redundancy_avoidance",
                "citation_quality",
            ],
        )

    def evaluate(
        self,
        response: dict[str, Any],
        query: str = "",
    ) -> dict[str, Any]:
        """Score response on all 5 criteria."""
        prompt = (
            EVALUATOR_SYSTEM_PROMPT
            + f"\n\nQuery: {query}\n\nResponse to evaluate:\n"
            + json.dumps(response, default=str)
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = (
                resp.content.strip()
                if hasattr(resp, "content")
                else str(resp).strip()
            )
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    ln
                    for ln in lines
                    if not ln.strip().startswith("```")
                    and ln.strip() != "json"
                )
            out = json.loads(content)
            passed = out.get("average_score", 0) >= self.passing_score
            out["passed"] = passed
            return out
        except json.JSONDecodeError as e:
            logger.warning("Evaluator JSON failed: %s", e)
            return {
                "scores": {c: 0 for c in self.criteria},
                "average_score": 0.0,
                "assessment_summary": str(e),
                "recommendations": ["Re-evaluate manually"],
                "passed": False,
            }
