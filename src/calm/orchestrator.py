"""
CALMOrchestrator — bộ định tuyến trung tâm của hệ thống CALM.

Nhận BẤT KỲ câu truy vấn nào từ người dùng, dùng PlanningAgent để phân rã
thành kế hoạch JSON, rồi TỰ ĐỘNG định tuyến sang đúng pipeline:

  • QA Pipeline    : DataAgent (thu thập → lưu ChromaDB) → WildfireQAAgent
  • Prediction Pipeline : DataAgent (lấy dữ liệu môi trường)
                          → PredictionReasoningAgent (chạy model)
                          → RSENModule (xác thực + giải thích vật lý)

Người dùng CHỈ cần gọi orchestrator.run(query). Hệ thống tự nhận biết yêu
cầu là "hỏi đáp" hay "dự đoán" dựa trên kế hoạch do PlanningAgent tạo ra.
"""

from __future__ import annotations

import logging
from typing import Any

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.planning_agent import PlanningAgent
from calm.agents.prediction_reasoning_agent import PredictionReasoningAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.agents.rsen_module import RSENModule

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Từ khoá nhận dạng intent trong kế hoạch / query
# ─────────────────────────────────────────────
_PREDICT_ACTIONS = {
    "predict", "forecast", "detection", "risk_assessment",
    "fire_prediction", "run_model", "model_inference",
}
_PREDICT_AGENTS = {"prediction", "model", "fire_prediction", "predict_agent"}
_PREDICT_QUERY_WORDS = {
    "predict", "forecast", "risk", "detect", "identify fire",
    "fire probability", "next week", "next days", "likelihood",
    "will there be", "assess risk", "fire risk",
}

_QA_ACTIONS = {
    "qa", "question", "answer", "retrieve_knowledge",
    "web_search", "information", "explain",
}
_QA_AGENTS = {"qa", "question_answering", "qa_agent"}
_QA_QUERY_WORDS = {
    "what", "why", "how", "explain", "describe", "tell me",
    "information", "facts", "causes", "history", "recent",
    "which", "when",
}


class CALMOrchestrator:
    """
    Điểm vào duy nhất của CALM. Tự động định tuyến truy vấn sang đúng pipeline.

    Ví dụ:
        orchestrator = CALMOrchestrator.from_llm(llm)
        # Hỏi đáp — tự route sang QAAgent
        result = orchestrator.run("What causes wildfires in the Amazon?")
        # Dự đoán — tự route sang PredictionAgent + RSEN
        result = orchestrator.run("Predict wildfire risk for California next 7 days")
    """

    def __init__(
        self,
        planner: PlanningAgent,
        data_agent: DataKnowledgeAgent,
        qa_agent: WildfireQAAgent,
        prediction_agent: PredictionReasoningAgent,
        rsen: RSENModule,
        config: dict | None = None,
    ) -> None:
        self.planner = planner
        self.data_agent = data_agent
        self.qa_agent = qa_agent
        self.prediction_agent = prediction_agent
        self.rsen = rsen
        self.config = config or {}

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def run(self, query: str) -> dict[str, Any]:
        """
        Xử lý truy vấn từ đầu đến cuối.

        1. Gọi PlanningAgent → JSON plan
        2. Phân loại intent từ plan (QA / Prediction)
        3. Định tuyến sang đúng pipeline và trả kết quả
        """
        logger.info("[Orchestrator] Query: %s", query)

        # ── Bước 1: Lập kế hoạch ──────────────────────────────────────────
        plan_result = self.planner.invoke(query)
        plan_steps: list[dict] = plan_result.get("final_output") or []
        plan_error = plan_result.get("error")
        if plan_error and not plan_steps:
            logger.warning("[Orchestrator] Planning failed: %s", plan_error)

        # ── Bước 2: Phân loại intent ──────────────────────────────────────
        task_type = self._classify_intent(plan_steps, query)
        logger.info("[Orchestrator] Classified task_type=%s", task_type)

        # ── Bước 3: Định tuyến sang pipeline ─────────────────────────────
        if task_type == "prediction":
            return self._prediction_pipeline(query, plan_steps)
        return self._qa_pipeline(query, plan_steps)

    # ─────────────────────────────────────────
    # Intent Classification
    # ─────────────────────────────────────────

    def _classify_intent(self, plan_steps: list[dict], query: str) -> str:
        """
        Xác định loại nhiệm vụ từ plan steps và từ khoá trong query.

        Ưu tiên (theo thứ tự):
          1. action / agent trong plan steps
          2. Từ khoá trong câu query
          3. Mặc định: "qa"
        """
        for step in plan_steps:
            action = str(step.get("action", "")).lower().replace("-", "_")
            agent = str(step.get("agent", "")).lower().replace("-", "_")
            if any(w in action for w in _PREDICT_ACTIONS) or any(
                w in agent for w in _PREDICT_AGENTS
            ):
                return "prediction"
            if any(w in action for w in _QA_ACTIONS) or any(
                w in agent for w in _QA_AGENTS
            ):
                return "qa"

        q_lower = query.lower()
        if any(w in q_lower for w in _PREDICT_QUERY_WORDS):
            return "prediction"
        if any(w in q_lower for w in _QA_QUERY_WORDS):
            return "qa"

        return "qa"

    # ─────────────────────────────────────────
    # QA Pipeline
    # ─────────────────────────────────────────

    def _qa_pipeline(
        self, query: str, plan_steps: list[dict]
    ) -> dict[str, Any]:
        """
        Pipeline trả lời câu hỏi:
          DataAgent.retrieve() → lưu ChromaDB → WildfireQAAgent.invoke()

        DataAgent thu thập dữ liệu từ web/arxiv/GEE, trích xuất tri thức và
        lưu vào ChromaDB. WildfireQAAgent sau đó truy xuất ChromaDB và tổng
        hợp câu trả lời có trích dẫn.
        """
        logger.info("[Orchestrator] QA pipeline started")

        # Step 1: Thu thập dữ liệu & lưu ChromaDB
        params = self._location_params_from_plan(plan_steps)
        data_result: dict[str, Any] = {}
        try:
            data_result = self.data_agent.retrieve(query, params)
            n_retrieved = len(data_result.get("retrieved_data", []))
            n_facts = len(
                data_result.get("extracted_knowledge", {}).get(
                    "factual_statements", []
                )
            )
            logger.info(
                "[Orchestrator] QA data: %d sources, %d facts → ChromaDB",
                n_retrieved,
                n_facts,
            )
        except Exception as e:
            logger.warning("[Orchestrator] QA data collection failed: %s", e)

        # Step 2: WildfireQAAgent trả lời
        qa_result: dict[str, Any] = {}
        try:
            qa_result = self.qa_agent.invoke(query)
        except Exception as e:
            logger.exception("[Orchestrator] QA agent failed: %s", e)
            qa_result = {
                "final_output": {"answer": f"[ERROR] {e}", "citations": []},
                "approved": False,
                "error": str(e),
            }

        final_answer = qa_result.get("final_output") or {}
        return {
            "task_type": "qa",
            "plan_steps": plan_steps,
            "data_collected": bool(data_result.get("retrieved_data")),
            "sources_count": len(data_result.get("retrieved_data", [])),
            "answer": final_answer.get("answer", ""),
            "reasoning_chain": final_answer.get("reasoning_chain", []),
            "citations": final_answer.get("citations", []),
            "confidence": final_answer.get("confidence", 0.0),
            "approved": qa_result.get("approved", False),
            "error": qa_result.get("error"),
        }

    # ─────────────────────────────────────────
    # Prediction Pipeline
    # ─────────────────────────────────────────

    def _prediction_pipeline(
        self, query: str, plan_steps: list[dict]
    ) -> dict[str, Any]:
        """
        Pipeline dự đoán cháy rừng:
          DataAgent.retrieve() → PredictionReasoningAgent.predict()
          → RSENModule.validate() (Weather + Geo analysts song song)

        Kết quả gồm: mức rủi ro, độ tin cậy, lý luận từ RSEN,
        và quyết định Plausible/Implausible.
        """
        logger.info("[Orchestrator] Prediction pipeline started")

        # Step 1: Thu thập dữ liệu môi trường
        params = self._location_params_from_plan(plan_steps)
        data_result: dict[str, Any] = {}
        try:
            data_result = self.data_agent.retrieve(query, params)
            logger.info(
                "[Orchestrator] Prediction data: %d sources",
                len(data_result.get("retrieved_data", [])),
            )
        except Exception as e:
            logger.warning(
                "[Orchestrator] Prediction data collection failed: %s", e
            )

        # Step 2: Chạy model dự đoán
        prediction: dict[str, Any] = {}
        try:
            prediction = self.prediction_agent.predict(params)
            logger.info(
                "[Orchestrator] Model output: risk=%s, conf=%.2f",
                prediction.get("risk_level", "?"),
                prediction.get("confidence", 0.0),
            )
        except Exception as e:
            logger.exception(
                "[Orchestrator] Prediction model failed: %s", e
            )
            prediction = {
                "error": str(e),
                "result": None,
                "risk_level": "Unknown",
                "confidence": 0.0,
            }

        # Step 3: RSEN — xác thực song song (Weather + Geo)
        validation: dict[str, Any] = {}
        if not prediction.get("error"):
            try:
                met_data = self._extract_met_data(data_result)
                spatial_data = self._extract_spatial_data(data_result)
                validation = self.rsen.validate(prediction, met_data, spatial_data)
                logger.info(
                    "[Orchestrator] RSEN decision: %s",
                    validation.get("decision", "?"),
                )
            except Exception as e:
                logger.exception(
                    "[Orchestrator] RSEN validation failed: %s", e
                )
                validation = {
                    "error": str(e),
                    "decision": "Unknown",
                    "final_rationale": "",
                }

        return {
            "task_type": "prediction",
            "plan_steps": plan_steps,
            "prediction": prediction,
            "validation": validation,
            "risk_level": prediction.get("risk_level", "Unknown"),
            "confidence": prediction.get("confidence", 0.0),
            "decision": validation.get("decision", "Unknown"),
            "rationale": validation.get("final_rationale", ""),
            "error": prediction.get("error") or validation.get("error"),
        }

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _location_params_from_plan(
        self, plan_steps: list[dict]
    ) -> dict[str, Any]:
        """Trích xuất tham số location/time_range từ plan steps."""
        for step in plan_steps:
            params = step.get("parameters") or {}
            if params.get("location") or params.get("area"):
                return params
        return {}

    @staticmethod
    def _extract_met_data(data_result: dict[str, Any]) -> dict[str, Any]:
        """Lấy dữ liệu khí tượng từ kết quả thu thập."""
        for item in data_result.get("retrieved_data", []):
            if item.get("source") in {"Copernicus CDS", "ERA5"}:
                return item.get("data_content") or {}
        return {}

    @staticmethod
    def _extract_spatial_data(data_result: dict[str, Any]) -> dict[str, Any]:
        """Lấy dữ liệu địa lý từ kết quả thu thập."""
        for item in data_result.get("retrieved_data", []):
            if item.get("source") in {"GEE", "Google Earth Engine"}:
                return item.get("data_content") or {}
        return {}

    # ─────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────

    @classmethod
    def from_llm(
        cls,
        llm,
        memory_store,
        tools: dict | None = None,
        model_runner=None,
        config: dict | None = None,
    ) -> "CALMOrchestrator":
        """
        Khởi tạo nhanh orchestrator từ LLM + memory store.

        Args:
            llm: LangChain LLM (ChatOpenAI, ChatOpenRouter, ...).
            memory_store: ChromaMemoryStore instance.
            tools: dict gồm earth_engine, copernicus, web_search, arxiv.
            model_runner: Runner cho ML model (tùy chọn).
            config: Cấu hình tổng thể.
        """
        cfg = config or {}
        _tools = tools or {}

        planner = PlanningAgent(
            llm=llm,
            config=cfg,
            n_max=cfg.get("planner_n_max", 3),
            f_max=cfg.get("planner_f_max", 3),
        )

        data_agent = DataKnowledgeAgent(
            llm=llm,
            tools=_tools,
            memory_store=memory_store,
            config=cfg,
        )

        web_search = _tools.get("web_search")
        qa_agent = WildfireQAAgent(
            llm=llm,
            data_agent=data_agent,
            web_search_tool=web_search,
            memory_store=memory_store,
            config=cfg,
            n_max=cfg.get("qa_n_max", 3),
            f_max=cfg.get("qa_f_max", 3),
        )

        prediction_agent = PredictionReasoningAgent(
            model_runner=model_runner,
            config=cfg,
        )

        rsen = RSENModule(
            llm=llm,
            memory_store=memory_store,
            k=cfg.get("rsen_k", 3),
        )

        return cls(
            planner=planner,
            data_agent=data_agent,
            qa_agent=qa_agent,
            prediction_agent=prediction_agent,
            rsen=rsen,
            config=cfg,
        )
