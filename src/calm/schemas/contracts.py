"""
CALM contracts — Pydantic models cho plan, task routing, retrieval, prediction, QA.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """Một bước trong kế hoạch. Schema chuẩn theo tài liệu CALM."""

    step_id: str = Field(..., description="step-1, step-2, ...")
    action: str = Field(..., description="retrieve_knowledge, web_search, prediction_reasoning, compile_report, ...")
    agent: str = Field(..., description="data_knowledge, qa, prediction, rsen, execution")
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_output: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    """Kế hoạch đầy đủ từ PlanningAgent."""

    query_summary: str = ""
    plan_steps: list[PlanStep] = Field(default_factory=list)
    overall_goal: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRouting(BaseModel):
    """Kết quả RouterAgent — task_type, confidence, artifacts cần dùng."""

    task_type: str = Field(..., description="qa | prediction | hybrid")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    required_artifacts: list[str] = Field(default_factory=list, description="evidence, prediction, met_data, spatial_data")
    next_steps: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RetrievedEvidence(BaseModel):
    """Kết quả thu thập/trích xuất từ DataKnowledgeAgent."""

    retrieval_summary: dict[str, Any] = Field(default_factory=dict)
    retrieved_data: list[dict[str, Any]] = Field(default_factory=list)
    extracted_knowledge: dict[str, list[str]] = Field(
        default_factory=lambda: {"factual_statements": [], "causal_relationships": []}
    )


class PredictionOutput(BaseModel):
    """Output chuẩn từ model / PredictionReasoningAgent."""

    risk_level: str = "Unknown"
    confidence: float = 0.0
    result: Any = None
    error: str | None = None


class ValidationOutput(BaseModel):
    """Output chuẩn từ RSEN — dùng validation_decision thống nhất."""

    validation_decision: str = Field(..., description="Plausible | Implausible")
    final_prediction: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: dict[str, Any] = Field(default_factory=dict)
    final_rationale: str = ""
    error: str | None = None


class QAResponse(BaseModel):
    """Output chuẩn từ WildfireQAAgent."""

    answer: str = ""
    reasoning_chain: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
