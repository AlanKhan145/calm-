"""
CALMOrchestrator — bộ định tuyến trung tâm của hệ thống CALM.

Plan-driven execution: PlanningAgent → RouterAgent → ExecutionAgent chạy từng step.
Không còn 2 pipeline cứng; mọi bước đi theo plan JSON.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from calm.agents.data_knowledge_agent import DataKnowledgeAgent
from calm.agents.execution_agent import ExecutionAgent
from calm.agents.memory_agent import MemoryAgent
from calm.agents.planning_agent import PlanningAgent
from calm.agents.prediction_reasoning_agent import PredictionReasoningAgent
from calm.agents.qa_agent import WildfireQAAgent
from calm.agents.router_agent import RouterAgent
from calm.agents.rsen_module import RSENModule
from calm.tools.safety_check import SafetyChecker

logger = logging.getLogger(__name__)


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
        memory_agent: MemoryAgent | None = None,
        router_agent: RouterAgent | None = None,
        executor: ExecutionAgent | None = None,
        config: dict | None = None,
    ) -> None:
        self.planner = planner
        self.data_agent = data_agent
        self.qa_agent = qa_agent
        self.prediction_agent = prediction_agent
        self.rsen = rsen
        self.memory_agent = memory_agent
        self.router_agent = router_agent
        self.executor = executor
        self.config = config or {}

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def run(self, query: str) -> dict[str, Any]:
        """
        Plan-driven execution:
        1. PlanningAgent → plan JSON
        2. RouterAgent → task_type, confidence
        3. ExecutionAgent chạy từng step (hoặc pipeline cũ nếu không có executor)
        """
        logger.info("[Orchestrator] Query: %s", query)

        plan_result = self.planner.invoke(query)
        plan_steps: list[dict] = plan_result.get("final_output") or []
        if plan_result.get("error") and not plan_steps:
            logger.warning("[Orchestrator] Planning failed: %s", plan_result.get("error"))

        routing = None
        if self.router_agent:
            routing = self.router_agent.route(query, plan_steps)
            task_type = routing.task_type
            logger.info("[Orchestrator] Router: task_type=%s, confidence=%.2f", task_type, routing.confidence)
        else:
            task_type = self._classify_intent_fallback(plan_steps, query)

        if self.executor and plan_steps:
            return self._run_plan_driven(query, plan_steps, task_type)
        if task_type == "prediction":
            return self._prediction_pipeline(query, plan_steps)
        return self._qa_pipeline(query, plan_steps)

    def _classify_intent_fallback(self, plan_steps: list[dict], query: str) -> str:
        """Fallback keyword routing khi không có RouterAgent."""
        for step in plan_steps:
            action = str(step.get("action", "")).lower()
            agent = str(step.get("agent", "")).lower()
            if any(w in action or w in agent for w in ["predict", "forecast", "model", "run_model"]):
                return "prediction"
            if any(w in action or w in agent for w in ["retrieve", "web_search", "qa", "compile_report"]):
                return "qa"
        q_lower = query.lower()
        if any(w in q_lower for w in ["predict", "forecast", "risk", "next week", "next days"]):
            return "prediction"
        return "qa"

    def _run_plan_driven(
        self, query: str, plan_steps: list[dict], task_type: str
    ) -> dict[str, Any]:
        """Thực thi từng step theo plan; QA và Prediction dùng chung cơ chế context."""
        logger.info("[Orchestrator] Plan-driven execution: %d steps", len(plan_steps))
        params = self._location_params_from_plan(plan_steps)
        context: dict[str, Any] = {"query": query, "parameters": params}

        for step in plan_steps:
            step_id = step.get("step_id", "unknown")
            agent_name = str(step.get("agent", "")).lower()
            try:
                result = self.executor.execute_step(step, context)
                context[step_id] = result
                if agent_name in ("data_knowledge",):
                    context["data_result"] = result
                    context["retrieved_data"] = result.get("retrieved_data", [])
                    context["met_data"] = self._extract_met_data(result)
                    context["spatial_data"] = self._extract_spatial_data(result)
                elif agent_name in ("prediction", "predict_agent", "fire_prediction"):
                    context["prediction"] = result
                elif agent_name in ("qa", "qa_agent", "question_answering"):
                    context["final_output"] = result.get("final_output", {})
            except Exception as e:
                logger.warning("[Orchestrator] Step %s failed: %s", step_id, e)
                context[step_id] = {"error": str(e)}

        return self._format_plan_result(query, plan_steps, context, task_type)

    def _format_plan_result(
        self, query: str, plan_steps: list[dict], context: dict[str, Any], task_type: str
    ) -> dict[str, Any]:
        """Định dạng kết quả từ context sau khi chạy plan."""
        if task_type == "prediction":
            pred = context.get("prediction", {})
            if not pred.get("error"):
                met = context.get("met_data", {})
                spatial = context.get("spatial_data", {})
                validation = self.rsen.validate(pred, met, spatial)
            else:
                validation = {"validation_decision": "Unknown", "final_rationale": ""}
            result = {
                "task_type": "prediction",
                "plan_steps": plan_steps,
                "prediction": pred,
                "validation": validation,
                "risk_level": pred.get("risk_level", "Unknown"),
                "confidence": pred.get("confidence", 0.0),
                "decision": validation.get("validation_decision", "Unknown"),
                "rationale": validation.get("final_rationale", ""),
                "error": pred.get("error"),
            }
        else:
            qa_out = context.get("final_output", {})
            for sid, v in context.items():
                if isinstance(v, dict) and v.get("final_output"):
                    qa_out = v.get("final_output", qa_out)
                    break
            data_result = context.get("data_result", {})
            result = {
                "task_type": "qa",
                "plan_steps": plan_steps,
                "data_collected": bool(data_result.get("retrieved_data")),
                "sources_count": len(data_result.get("retrieved_data", [])),
                "answer": qa_out.get("answer", ""),
                "reasoning_chain": qa_out.get("reasoning_chain", []),
                "citations": qa_out.get("citations", []),
                "confidence": qa_out.get("confidence", 0.0),
                "approved": True,
                "error": None,
            }
        if self.memory_agent:
            self.memory_agent.add_episode(query, result, task_type)
            self.memory_agent.add_short_term(query, result)
        return result

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

        # Step 2: WildfireQAAgent trả lời (truyền pre_retrieved để tránh gọi retrieve lại)
        qa_result: dict[str, Any] = {}
        try:
            qa_result = self.qa_agent.invoke(query, pre_retrieved=data_result)
        except Exception as e:
            logger.exception("[Orchestrator] QA agent failed: %s", e)
            qa_result = {
                "final_output": {"answer": f"[ERROR] {e}", "citations": []},
                "approved": False,
                "error": str(e),
            }

        final_answer = qa_result.get("final_output") or {}
        result = {
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
        if self.memory_agent:
            self.memory_agent.add_episode(query, result, "qa")
            self.memory_agent.add_short_term(query, result)
        return result

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

        # Step 2: Chạy model dự đoán (truyền met_data, spatial_data cho heuristic/model)
        prediction: dict[str, Any] = {}
        params["met_data"] = self._extract_met_data(data_result)
        params["spatial_data"] = self._extract_spatial_data(data_result)
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
                met_data = params.get("met_data") or self._extract_met_data(data_result)
                spatial_data = params.get("spatial_data") or self._extract_spatial_data(data_result)
                validation = self.rsen.validate(prediction, met_data, spatial_data)
                logger.info(
                    "[Orchestrator] RSEN decision: %s",
                    validation.get("validation_decision", validation.get("decision", "?")),
                )
            except Exception as e:
                logger.exception(
                    "[Orchestrator] RSEN validation failed: %s", e
                )
                validation = {
                    "error": str(e),
                    "validation_decision": "Unknown",
                    "final_rationale": "",
                }

        result = {
            "task_type": "prediction",
            "plan_steps": plan_steps,
            "prediction": prediction,
            "validation": validation,
            "risk_level": prediction.get("risk_level", "Unknown"),
            "confidence": prediction.get("confidence", 0.0),
            "decision": validation.get("validation_decision", validation.get("decision", "Unknown")),
            "rationale": validation.get("final_rationale", ""),
            "error": prediction.get("error") or validation.get("error"),
        }
        if self.memory_agent:
            self.memory_agent.add_episode(query, result, "prediction")
            self.memory_agent.add_short_term(query, result)
        return result

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _location_params_from_plan(
        self, plan_steps: list[dict]
    ) -> dict[str, Any]:
        """Trích xuất tham số location/time_range từ plan steps. Luôn chuẩn hóa time_range, mặc định ngày hôm nay."""
        from calm.utils.time_utils import resolve_time_range

        for step in plan_steps:
            params = dict(step.get("parameters") or {})
            if params.get("location") or params.get("area"):
                params["time_range"] = resolve_time_range(
                    params.get("time_range"), default_today=True
                )
                return params
        return {"time_range": resolve_time_range(None, default_today=True)}

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

    @staticmethod
    def _build_seasfire_model_runner(config: dict):
        """Tạo SeasFireModelRunner từ config nếu có checkpoint."""
        pred_cfg = (
            config.get("prediction")
            or config.get("agent_config", {}).get("prediction")
            or {}
        )
        checkpoint = pred_cfg.get("checkpoint", "")
        if not checkpoint:
            return None
        try:
            from calm.artifact.seasfire_runner import SeasFireRunner
            from calm.artifact.model_runner import SeasFireModelRunner
            from calm.artifact.feature_builder import SeasFireFeatureBuilder

            ckpt = Path(checkpoint)
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            seasfire = SeasFireRunner(
                checkpoint_path=ckpt,
                config={
                    "input_size": pred_cfg.get("seasfire_variables", 59),
                    "hidden_size": pred_cfg.get("hidden_size", 64),
                    "num_layers": pred_cfg.get("num_layers", 2),
                },
            )
            seasfire.load()
            dataset_path = (
                os.path.expandvars(str(pred_cfg.get("dataset_path", "") or ""))
                or os.environ.get("SEASFIRE_DATASET_PATH")
                or ""
            )
            feature_builder = None
            if dataset_path and Path(dataset_path).exists():
                feature_builder = SeasFireFeatureBuilder(
                    dataset_path=dataset_path,
                    timesteps=pred_cfg.get("timesteps", 6),
                    target_week=pred_cfg.get("target_week", 4),
                )
            return SeasFireModelRunner(seasfire, feature_builder)
        except Exception as e:
            logger.warning("Could not build SeasFireModelRunner: %s", e)
            return None

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
        _tools = dict(tools or {})
        safety = SafetyChecker(llm=llm)
        if "geocoding" not in _tools:
            from calm.tools.geocoding import GeocodingTool
            _tools["geocoding"] = GeocodingTool(safety_checker=safety, config=cfg)

        memory_agent = MemoryAgent(
            long_term_store=memory_store,
            short_term_size=cfg.get("memory_short_term_size", 5),
            episodic_max=cfg.get("memory_episodic_max", 100),
        )

        planner = PlanningAgent(
            llm=llm,
            config=cfg,
            n_max=cfg.get("planner_n_max", 3),
            f_max=cfg.get("planner_f_max", 3),
        )

        data_agent = DataKnowledgeAgent(
            llm=llm,
            tools=_tools,
            memory_store=memory_agent,
            config=cfg,
        )

        web_search = _tools.get("web_search")
        qa_agent = WildfireQAAgent(
            llm=llm,
            data_agent=data_agent,
            web_search_tool=web_search,
            memory_store=memory_agent,
            config=cfg,
            n_max=cfg.get("qa_n_max", 3),
            f_max=cfg.get("qa_f_max", 3),
        )

        # Model runner: từ tham số hoặc từ config (seasfire checkpoint)
        _model_runner = model_runner
        if _model_runner is None and cfg.get("prediction", {}).get("checkpoint"):
            _model_runner = _build_seasfire_model_runner(cfg)

        prediction_agent = PredictionReasoningAgent(
            model_runner=_model_runner,
            config=cfg,
        )

        rsen = RSENModule(
            llm=llm,
            memory_store=memory_agent,
            k=cfg.get("rsen_k", 3),
        )

        router_agent = RouterAgent(llm=llm, config=cfg)
        exec_tools = {
            "data_knowledge": data_agent,
            "prediction": prediction_agent,
            "qa": qa_agent,
            "rsen": rsen,
            "web_search": _tools.get("web_search"),
        }
        executor = ExecutionAgent(
            llm=llm,
            tools=exec_tools,
            safety_checker=safety,
            config=cfg,
        )

        return cls(
            planner=planner,
            data_agent=data_agent,
            qa_agent=qa_agent,
            prediction_agent=prediction_agent,
            rsen=rsen,
            memory_agent=memory_agent,
            router_agent=router_agent,
            executor=executor,
            config=cfg,
        )
