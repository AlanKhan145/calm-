"""
Tests for CALMOrchestrator — kiểm tra auto-routing QA / Prediction.

Tất cả LLM, tool, memory đều được mock (NP-7.2 — no real API calls).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

src = Path(__file__).resolve().parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from calm.orchestrator import CALMOrchestrator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_mock_llm_for_plan(plan_steps: list[dict]):
    """LLM mock trả về JSON plan đã được duyệt."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content="Step-by-step plan drafted."),
        MagicMock(content="Plan is complete and feasible. [APPROVED]"),
        MagicMock(content=json.dumps(plan_steps)),
    ]
    return llm


def _make_memory():
    mem = MagicMock()
    mem.similarity_search.return_value = []
    mem.add_texts.return_value = None
    return mem


def _make_data_agent(memory):
    da = MagicMock()
    da.retrieve.return_value = {
        "retrieval_summary": {"original_query": "test"},
        "retrieved_data": [
            {
                "sub_question_id": "news-0",
                "data_content": "Wildfire facts",
                "source": "DuckDuckGo",
                "citation": "https://example.com",
                "confidence_score": 0.7,
            }
        ],
        "extracted_knowledge": {
            "factual_statements": ["Fact A"],
            "causal_relationships": [],
        },
    }
    return da


def _make_qa_agent():
    qa = MagicMock()
    qa.invoke.return_value = {
        "final_output": {
            "answer": "Amazon wildfires are caused by deforestation and drought.",
            "reasoning_chain": ["Step 1", "Step 2"],
            "citations": ["https://source.com"],
            "confidence": 0.85,
        },
        "approved": True,
        "error": None,
    }
    return qa


def _make_prediction_agent():
    pa = MagicMock()
    pa.predict.return_value = {
        "risk_level": "High",
        "confidence": 0.82,
        "result": {"fire_prob": 0.82},
        "error": None,
    }
    return pa


def _make_rsen():
    rsen = MagicMock()
    rsen.validate.return_value = {
        "validation_decision": "Plausible",
        "decision": "Plausible",
        "final_rationale": "High temp, low humidity, dry fuel.",
        "error": None,
    }
    return rsen


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Intent Classification
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifyIntent:
    def _make_orchestrator(self):
        return CALMOrchestrator(
            planner=MagicMock(),
            data_agent=MagicMock(),
            qa_agent=MagicMock(),
            prediction_agent=MagicMock(),
            rsen=MagicMock(),
        )

    def test_prediction_from_plan_action(self):
        orc = self._make_orchestrator()
        steps = [{"action": "predict_wildfire", "agent": "prediction"}]
        assert orc._classify_intent_fallback(steps, "tell me") == "prediction"

    def test_qa_from_plan_action(self):
        orc = self._make_orchestrator()
        steps = [{"action": "qa", "agent": "qa_agent"}]
        assert orc._classify_intent_fallback(steps, "predict fire") == "qa"

    def test_prediction_from_query_keyword(self):
        orc = self._make_orchestrator()
        assert orc._classify_intent_fallback([], "predict wildfire risk next 7 days") == "prediction"

    def test_qa_from_query_keyword(self):
        orc = self._make_orchestrator()
        assert orc._classify_intent_fallback([], "What causes wildfires in the Amazon?") == "qa"

    def test_default_qa_when_no_match(self):
        orc = self._make_orchestrator()
        assert orc._classify_intent_fallback([], "climate change impacts forests") == "qa"

    def test_plan_takes_priority_over_query(self):
        """Plan action beats query keywords."""
        orc = self._make_orchestrator()
        steps = [{"action": "call_prediction_agent", "agent": "prediction"}]
        # query says "what" but plan says prediction
        assert orc._classify_intent_fallback(steps, "what is the fire risk") == "prediction"


# ──────────────────────────────────────────────────────────────────────────────
# Tests — QA Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestQAPipeline:
    def _make_orchestrator(self, llm=None):
        mem = _make_memory()
        da = _make_data_agent(mem)
        qa = _make_qa_agent()
        plan_steps = [{"step_id": "1", "action": "retrieve_knowledge", "agent": "qa"}]
        _llm = llm or _make_mock_llm_for_plan(plan_steps)
        planner = MagicMock()
        planner.invoke.return_value = {
            "final_output": plan_steps,
            "approved": True,
            "error": None,
        }
        return CALMOrchestrator(
            planner=planner,
            data_agent=da,
            qa_agent=qa,
            prediction_agent=_make_prediction_agent(),
            rsen=_make_rsen(),
        )

    def test_qa_query_routes_to_qa(self):
        orc = self._make_orchestrator()
        result = orc.run("What causes wildfires in the Amazon?")
        assert result["task_type"] == "qa"

    def test_qa_result_has_answer(self):
        orc = self._make_orchestrator()
        result = orc.run("What causes wildfires in the Amazon?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_qa_result_has_citations(self):
        orc = self._make_orchestrator()
        result = orc.run("What caused the 2023 Canadian wildfires?")
        assert isinstance(result.get("citations"), list)

    def test_qa_data_agent_called(self):
        orc = self._make_orchestrator()
        orc.run("What are the fire causes?")
        orc.data_agent.retrieve.assert_called_once()

    def test_qa_result_no_error_on_success(self):
        orc = self._make_orchestrator()
        result = orc.run("Tell me about wildfire prevention")
        assert result.get("error") is None

    def test_qa_result_approved_flag(self):
        orc = self._make_orchestrator()
        result = orc.run("Explain NDVI")
        assert result.get("approved") is True

    def test_qa_graceful_on_data_agent_failure(self):
        """QA should still run if data collection fails."""
        mem = _make_memory()
        da = _make_data_agent(mem)
        da.retrieve.side_effect = RuntimeError("GEE down")
        qa = _make_qa_agent()
        planner = MagicMock()
        planner.invoke.return_value = {
            "final_output": [{"action": "qa", "agent": "qa"}],
            "approved": True,
            "error": None,
        }
        orc = CALMOrchestrator(
            planner=planner,
            data_agent=da,
            qa_agent=qa,
            prediction_agent=_make_prediction_agent(),
            rsen=_make_rsen(),
        )
        result = orc.run("What causes fires?")
        # QA still runs even if data collection fails
        assert result["task_type"] == "qa"
        qa.invoke.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Prediction Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionPipeline:
    def _make_orchestrator(self):
        plan_steps = [
            {"step_id": "1", "action": "run_prediction_model", "agent": "prediction"}
        ]
        planner = MagicMock()
        planner.invoke.return_value = {
            "final_output": plan_steps,
            "approved": True,
            "error": None,
        }
        return CALMOrchestrator(
            planner=planner,
            data_agent=_make_data_agent(_make_memory()),
            qa_agent=_make_qa_agent(),
            prediction_agent=_make_prediction_agent(),
            rsen=_make_rsen(),
        )

    def test_prediction_query_routes_to_prediction(self):
        orc = self._make_orchestrator()
        result = orc.run("Predict wildfire risk for California next 7 days")
        assert result["task_type"] == "prediction"

    def test_prediction_result_has_risk_level(self):
        orc = self._make_orchestrator()
        result = orc.run("Predict wildfire risk California")
        assert "risk_level" in result

    def test_prediction_result_has_rsen_decision(self):
        orc = self._make_orchestrator()
        result = orc.run("Forecast fire California next week")
        assert "decision" in result

    def test_prediction_calls_model(self):
        orc = self._make_orchestrator()
        orc.run("Predict wildfire risk for Amazon")
        orc.prediction_agent.predict.assert_called_once()

    def test_rsen_called_when_model_succeeds(self):
        orc = self._make_orchestrator()
        orc.run("Predict wildfire California")
        orc.rsen.validate.assert_called_once()

    def test_rsen_skipped_when_model_errors(self):
        """RSEN should NOT be called if prediction model returns an error."""
        planner = MagicMock()
        planner.invoke.return_value = {
            "final_output": [{"action": "predict", "agent": "prediction"}],
            "approved": True,
            "error": None,
        }
        pa = MagicMock()
        pa.predict.return_value = {
            "error": "model unavailable",
            "result": None,
            "risk_level": "Unknown",
            "confidence": 0.0,
        }
        rsen = _make_rsen()
        orc = CALMOrchestrator(
            planner=planner,
            data_agent=_make_data_agent(_make_memory()),
            qa_agent=_make_qa_agent(),
            prediction_agent=pa,
            rsen=rsen,
        )
        orc.run("predict fire risk")
        rsen.validate.assert_not_called()

    def test_prediction_result_contains_plan_steps(self):
        orc = self._make_orchestrator()
        result = orc.run("Forecast fire California")
        assert isinstance(result.get("plan_steps"), list)


# ──────────────────────────────────────────────────────────────────────────────
# Tests — from_llm factory
# ──────────────────────────────────────────────────────────────────────────────

class TestFromLLMFactory:
    def test_factory_creates_orchestrator(self):
        llm = MagicMock()
        memory = _make_memory()
        orc = CALMOrchestrator.from_llm(llm=llm, memory_store=memory)
        assert isinstance(orc, CALMOrchestrator)

    def test_factory_with_tools(self):
        llm = MagicMock()
        memory = _make_memory()
        web = MagicMock()
        orc = CALMOrchestrator.from_llm(
            llm=llm, memory_store=memory, tools={"web_search": web}
        )
        assert orc.qa_agent.web_search_tool is web

    def test_factory_planner_config(self):
        llm = MagicMock()
        memory = _make_memory()
        orc = CALMOrchestrator.from_llm(
            llm=llm,
            memory_store=memory,
            config={"planner_n_max": 5, "qa_n_max": 2},
        )
        assert orc.planner.n_max == 5
        assert orc.qa_agent.n_max == 2

    def test_factory_with_model_runner(self):
        llm = MagicMock()
        memory = _make_memory()
        runner = MagicMock()
        orc = CALMOrchestrator.from_llm(
            llm=llm, memory_store=memory, model_runner=runner
        )
        assert orc.prediction_agent.model_runner is runner
