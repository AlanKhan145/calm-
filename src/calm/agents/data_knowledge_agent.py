"""
Mô-đun Data & Knowledge Management Agent — Collection / Extraction / Retrieval.

Mục tiêu bản này:
- Hỗ trợ retrieval riêng cho QA và prediction.
- Trả dữ liệu usable cho model thay vì chỉ summary.
- Chuẩn hóa output prediction thành:
    model_inputs.met_timeseries
    model_inputs.satellite_features
    model_inputs.static_geo
    model_inputs.textual_context
- Có data_quality:
    - nguồn nào ok / lỗi
    - biến nào thiếu
    - có fallback hay không
- Hỗ trợ point / bbox / polygon.
- Có sub-request JSON, reflection/retry nhẹ.
- Dedup không làm mất truy vấn khác vùng / thời gian.
- Geocoding fail phải báo rõ, không silent None.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from calm.prompt_library.data_prompts import KNOWLEDGE_EXTRACTION_PROMPT
from calm.utils.time_utils import resolve_time_range

logger = logging.getLogger(__name__)


class DataKnowledgeAgent:
    """
    CALM §4.2: Collection → Extraction → Retrieval.

    FR-D01 → FR-D12 (mức best-effort trong phạm vi agent này):
    - đa nguồn: GEE, CDS, web, ArXiv
    - query normalization
    - sub-request decomposition
    - retry / reflection
    - dedup an toàn theo query + vùng + thời gian + task_type
    - output usable cho QA và prediction
    """

    def __init__(
        self,
        llm,
        tools: dict,
        memory_store,
        config: dict | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or {}
        self.memory = memory_store
        self.config = config or {}

        self.dedup_check = self.config.get("dedup_check", True)
        self.max_news_results = self.config.get("max_news_results", 10)
        self.max_arxiv_papers = self.config.get("max_arxiv_papers", 3)
        self.max_retry_rounds = self.config.get("data_retry_rounds", 2)
        self.default_today = self.config.get("default_today", True)

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Full pipeline:
        1) normalize query/context
        2) sub-request JSON
        3) collect by task type
        4) extract knowledge
        5) store memory
        """
        params = parameters or {}
        normalized_request = self._normalize_request(query, params)
        task_type = normalized_request["task_type"]

        if normalized_request["errors"]:
            return self._build_error_result(
                query=query,
                normalized_request=normalized_request,
                error="; ".join(normalized_request["errors"]),
            )

        dedup_hit = self._check_dedup(query, normalized_request)
        if dedup_hit is not None:
            return dedup_hit

        sub_requests = self._build_sub_requests(query, normalized_request)

        if task_type == "prediction":
            collected = self._retrieve_for_prediction(
                query=query,
                normalized_request=normalized_request,
                sub_requests=sub_requests,
            )
        else:
            collected = self._retrieve_for_qa(
                query=query,
                normalized_request=normalized_request,
                sub_requests=sub_requests,
            )

        extracted_knowledge = self._extract_knowledge_from_collected(collected)

        try:
            all_knowledge = (
                extracted_knowledge.get("factual_statements", [])
                + extracted_knowledge.get("causal_relationships", [])
            )
            if all_knowledge:
                self.memory.add_texts(all_knowledge)
        except Exception as e:
            logger.debug("Could not persist extracted knowledge: %s", e)

        result = {
            "task_type": task_type,
            "retrieval_summary": collected.get("retrieval_summary", {}),
            "normalized_query": normalized_request.get("normalized_query"),
            "normalized_query_context": normalized_request.get("normalized_query_context", {}),
            "sub_requests": sub_requests,
            "retrieved_data": collected.get("retrieved_data", []),
            "model_inputs": collected.get("model_inputs", self._empty_model_inputs()),
            "data_quality": collected.get("data_quality", self._empty_data_quality()),
            "extracted_knowledge": extracted_knowledge,
            "reflection_trace": collected.get("reflection_trace", []),
            "error": collected.get("error"),
        }
        return result

    def collect(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Backward-compatible wrapper.
        """
        out = self.retrieve(query, parameters)
        return {
            "retrieval_summary": out.get("retrieval_summary", {}),
            "retrieved_data": out.get("retrieved_data", []),
            "model_inputs": out.get("model_inputs", self._empty_model_inputs()),
            "data_quality": out.get("data_quality", self._empty_data_quality()),
            "reflection_trace": out.get("reflection_trace", []),
            "error": out.get("error"),
        }

    def extract_knowledge(self, text: str) -> dict[str, list[str]]:
        """Extract factual_statements and causal_relationships from text."""
        prompt = KNOWLEDGE_EXTRACTION_PROMPT + f"\n\nText:\n{text}"
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = self._strip_code_fence(content)
            out = json.loads(content)
            return {
                "factual_statements": out.get("factual_statements", []),
                "causal_relationships": out.get("causal_relationships", []),
            }
        except json.JSONDecodeError:
            return {"factual_statements": [], "causal_relationships": []}
        except Exception as e:
            logger.debug("Knowledge extraction failed: %s", e)
            return {"factual_statements": [], "causal_relationships": []}

    # ─────────────────────────────────────────
    # Normalize request
    # ─────────────────────────────────────────

    def _normalize_request(
        self,
        query: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Chuẩn hóa request dùng cho cả QA và prediction.

        Nguồn có thể đến từ:
        - parameters["normalized_query_context"]
        - parameters["location"] / bbox / polygon / coordinates
        - parameters["time_range"]
        - query
        """
        normalized_ctx = deepcopy(parameters.get("normalized_query_context") or {})
        normalized_query = (
            parameters.get("normalized_query")
            or normalized_ctx.get("normalized_query")
            or query
        )

        task_type = (
            parameters.get("task_type")
            or normalized_ctx.get("task_type")
            or self._infer_task_type(query, parameters)
        )

        time_range = resolve_time_range(
            parameters.get("time_range") or normalized_ctx.get("time_range"),
            default_today=self.default_today,
        )

        geometry, geo_errors = self._resolve_spatial_context(parameters, normalized_ctx)

        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "normalized_query_context": normalized_ctx,
            "task_type": task_type,
            "time_range": time_range,
            "geometry": geometry,
            "errors": geo_errors,
        }

    def _resolve_spatial_context(
        self,
        parameters: dict[str, Any],
        normalized_ctx: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """
        Hỗ trợ:
        - point: location / lat lon / coordinates
        - bbox: {min_lat, min_lon, max_lat, max_lon}
        - polygon: GeoJSON-like
        """
        errors: list[str] = []

        # 1) polygon
        polygon = (
            parameters.get("polygon")
            or parameters.get("geometry", {}).get("polygon")
            if isinstance(parameters.get("geometry"), dict)
            else None
        )
        if polygon:
            return {"type": "polygon", "value": polygon}, errors

        # 2) bbox
        bbox = parameters.get("bbox")
        if isinstance(bbox, dict):
            keys = {"min_lat", "min_lon", "max_lat", "max_lon"}
            if keys.issubset(set(bbox.keys())):
                return {"type": "bbox", "value": bbox}, errors

        # 3) point from explicit coordinates
        coords = (
            parameters.get("coordinates")
            or normalized_ctx.get("coordinates")
            or {}
        )
        lat = (
            parameters.get("lat")
            or parameters.get("latitude")
            or coords.get("lat")
            or coords.get("latitude")
        )
        lon = (
            parameters.get("lon")
            or parameters.get("lng")
            or parameters.get("longitude")
            or coords.get("lon")
            or coords.get("lng")
            or coords.get("longitude")
        )
        if lat is not None and lon is not None:
            try:
                return {
                    "type": "point",
                    "value": {"lat": float(lat), "lon": float(lon)},
                }, errors
            except Exception:
                errors.append(f"Invalid coordinates: lat={lat}, lon={lon}")
                return None, errors

        # 4) point from location string
        location = parameters.get("location") or normalized_ctx.get("location")
        if isinstance(location, dict):
            lat = location.get("lat") or location.get("latitude")
            lon = location.get("lon") or location.get("lng") or location.get("longitude")
            if lat is not None and lon is not None:
                try:
                    return {
                        "type": "point",
                        "value": {"lat": float(lat), "lon": float(lon)},
                    }, errors
                except Exception:
                    errors.append(f"Invalid location dict coordinates: {location}")
                    return None, errors

        if isinstance(location, str) and location.strip():
            resolved, geo_error = self._geocode_location(location.strip())
            if geo_error:
                errors.append(geo_error)
                return None, errors
            if resolved:
                return {"type": "point", "value": resolved}, errors

        # QA có thể không cần geometry; prediction thường cần.
        return None, errors

    def _geocode_location(self, location_text: str) -> tuple[dict[str, float] | None, str | None]:
        geocoding = self.tools.get("geocoding")
        if not geocoding:
            return None, f"Geocoding tool unavailable for location='{location_text}'"

        methods = ["geocode", "resolve", "lookup", "search"]
        last_error = None

        for method_name in methods:
            fn = getattr(geocoding, method_name, None)
            if not callable(fn):
                continue
            try:
                result = fn(location_text)
                if isinstance(result, dict):
                    if result.get("error"):
                        last_error = str(result["error"])
                        continue
                    lat = result.get("lat") or result.get("latitude")
                    lon = result.get("lon") or result.get("lng") or result.get("longitude")
                    if lat is not None and lon is not None:
                        return {"lat": float(lat), "lon": float(lon)}, None
            except Exception as e:
                last_error = str(e)

        return None, (
            f"Geocoding failed for location='{location_text}'"
            + (f": {last_error}" if last_error else "")
        )

    def _infer_task_type(self, query: str, parameters: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(query or ""),
                str(parameters.get("task_type") or ""),
            ]
        ).lower()

        keywords = [
            "predict",
            "prediction",
            "forecast",
            "risk",
            "wildfire risk",
            "fire danger",
            "next week",
            "next 7 days",
            "next days",
        ]
        return "prediction" if any(k in text for k in keywords) else "qa"

    # ─────────────────────────────────────────
    # Dedup
    # ─────────────────────────────────────────

    def _dedup_key(
        self,
        query: str,
        normalized_request: dict[str, Any],
    ) -> str:
        """
        Dedup phải giữ được khác biệt query/vùng/thời gian/task_type.
        Không dedup mù theo query thuần.
        """
        geometry = normalized_request.get("geometry")
        time_range = normalized_request.get("time_range") or {}
        task_type = normalized_request.get("task_type", "qa")

        geo_str = self._serialize_geometry(geometry)
        start = time_range.get("start") or time_range.get("start_date") or ""
        end = time_range.get("end") or time_range.get("end_date") or ""

        return f"{task_type} | {query} | {geo_str} | {start} | {end}".strip()

    def _check_dedup(
        self,
        query: str,
        normalized_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.dedup_check:
            return None

        try:
            dedup_query = self._dedup_key(query, normalized_request)
            existing = self.memory.similarity_search(dedup_query, k=1)
            if existing and existing[0]:
                logger.info("Dedup hit: similar query+geometry+time already exists")
                return {
                    "task_type": normalized_request["task_type"],
                    "retrieval_summary": {
                        "original_query": query,
                        "normalized_query": normalized_request["normalized_query"],
                        "dedup": True,
                        "source": "cache",
                    },
                    "retrieved_data": [],
                    "model_inputs": self._empty_model_inputs(),
                    "data_quality": self._empty_data_quality(),
                    "reflection_trace": [],
                    "error": None,
                }
        except Exception as e:
            logger.debug("Dedup check skipped due to error: %s", e)

        return None

    # ─────────────────────────────────────────
    # Sub-request JSON
    # ─────────────────────────────────────────

    def _build_sub_requests(
        self,
        query: str,
        normalized_request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Tạo sub-request JSON để tách retrieval logic.
        Nếu LLM fail thì fallback cứng.
        """
        prompt = (
            "You are decomposing a wildfire data retrieval request.\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "sub_requests": [\n'
            "    {\n"
            '      "id": "sr_1",\n'
            '      "source": "earth_engine|copernicus|web_search|arxiv",\n'
            '      "purpose": "short text",\n'
            '      "priority": 1\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Task type:\n"
            f"{normalized_request['task_type']}\n\n"
            "Normalized request:\n"
            f"{json.dumps(normalized_request, ensure_ascii=False, default=str)}\n\n"
            "Original query:\n"
            f"{query}"
        )

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, "content") else str(resp)
            content = self._strip_code_fence(content)
            parsed = json.loads(content)
            srs = parsed.get("sub_requests", [])
            if isinstance(srs, list) and srs:
                out = []
                for i, sr in enumerate(srs, start=1):
                    if not isinstance(sr, dict):
                        continue
                    out.append(
                        {
                            "id": sr.get("id") or f"sr_{i}",
                            "source": str(sr.get("source") or "").strip(),
                            "purpose": str(sr.get("purpose") or "").strip(),
                            "priority": int(sr.get("priority") or i),
                        }
                    )
                if out:
                    return out
        except Exception as e:
            logger.debug("Sub-request LLM generation failed, using fallback: %s", e)

        task_type = normalized_request["task_type"]
        if task_type == "prediction":
            return [
                {"id": "sr_1", "source": "earth_engine", "purpose": "satellite features", "priority": 1},
                {"id": "sr_2", "source": "copernicus", "purpose": "meteorological timeseries", "priority": 2},
                {"id": "sr_3", "source": "web_search", "purpose": "recent textual context", "priority": 3},
                {"id": "sr_4", "source": "arxiv", "purpose": "supporting wildfire literature", "priority": 4},
            ]

        return [
            {"id": "sr_1", "source": "web_search", "purpose": "recent web evidence", "priority": 1},
            {"id": "sr_2", "source": "arxiv", "purpose": "scientific evidence", "priority": 2},
            {"id": "sr_3", "source": "earth_engine", "purpose": "satellite evidence", "priority": 3},
            {"id": "sr_4", "source": "copernicus", "purpose": "meteorological evidence", "priority": 4},
        ]

    # ─────────────────────────────────────────
    # Retrieval flows
    # ─────────────────────────────────────────

    def _retrieve_for_prediction(
        self,
        query: str,
        normalized_request: dict[str, Any],
        sub_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Prediction retrieval:
        - cần data usable cho model
        - có retry chiến lược
        """
        attempts: list[dict[str, Any]] = []
        current_request = deepcopy(normalized_request)

        for retry_idx in range(self.max_retry_rounds + 1):
            attempt = self._collect_prediction_once(query, current_request, sub_requests)
            attempts.append(attempt)

            if self._prediction_attempt_is_usable(attempt):
                return attempt

            if retry_idx < self.max_retry_rounds:
                current_request = self._next_retry_request(current_request, attempt, retry_idx)

        best = self._choose_best_prediction_attempt(attempts)
        return best

    def _retrieve_for_qa(
        self,
        query: str,
        normalized_request: dict[str, Any],
        sub_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        QA retrieval:
        - giữ retrieved_data đầy đủ cho trả lời
        - vẫn trả model_inputs nhưng nhẹ hơn
        """
        geometry = normalized_request.get("geometry")
        time_range = normalized_request.get("time_range")

        retrieved_data: list[dict[str, Any]] = []
        source_status = self._init_source_status()
        reflection_trace: list[dict[str, Any]] = []

        # web
        if self.tools.get("web_search"):
            ok, items, error = self._fetch_web(query)
            if ok:
                retrieved_data.extend(items)
                source_status["web_search"] = {"ok": True, "error": None, "count": len(items)}
            else:
                source_status["web_search"] = {"ok": False, "error": error, "count": 0}

        # arxiv
        if self.tools.get("arxiv"):
            ok, items, error = self._fetch_arxiv(query)
            if ok:
                retrieved_data.extend(items)
                source_status["arxiv"] = {"ok": True, "error": None, "count": len(items)}
            else:
                source_status["arxiv"] = {"ok": False, "error": error, "count": 0}

        # optionally add geo sources if geometry exists
        if geometry:
            if self.tools.get("earth_engine"):
                ok, item, error = self._fetch_gee(geometry, time_range, mode="qa")
                if ok:
                    retrieved_data.append(item)
                    source_status["earth_engine"] = {"ok": True, "error": None, "count": 1}
                else:
                    source_status["earth_engine"] = {"ok": False, "error": error, "count": 0}

            if self.tools.get("copernicus"):
                ok, item, error = self._fetch_copernicus(geometry, time_range, mode="qa")
                if ok:
                    retrieved_data.append(item)
                    source_status["copernicus"] = {"ok": True, "error": None, "count": 1}
                else:
                    source_status["copernicus"] = {"ok": False, "error": error, "count": 0}

        model_inputs = self._empty_model_inputs()
        model_inputs["textual_context"] = self._build_textual_context(retrieved_data)

        data_quality = self._build_data_quality(
            task_type="qa",
            source_status=source_status,
            model_inputs=model_inputs,
            used_fallback=False,
        )

        return {
            "retrieval_summary": {
                "original_query": query,
                "normalized_query": normalized_request["normalized_query"],
                "task_type": "qa",
                "geometry": geometry,
                "time_range": time_range,
            },
            "retrieved_data": retrieved_data,
            "model_inputs": model_inputs,
            "data_quality": data_quality,
            "reflection_trace": reflection_trace,
            "error": self._coalesce_source_errors(source_status),
        }

    def _collect_prediction_once(
        self,
        query: str,
        normalized_request: dict[str, Any],
        sub_requests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        geometry = normalized_request.get("geometry")
        time_range = normalized_request.get("time_range")

        retrieved_data: list[dict[str, Any]] = []
        model_inputs = self._empty_model_inputs()
        source_status = self._init_source_status()
        reflection_trace: list[dict[str, Any]] = []

        # GEE / satellite
        if self.tools.get("earth_engine") and geometry:
            ok, item, error = self._fetch_gee(geometry, time_range, mode="prediction")
            if ok and item:
                retrieved_data.append(item)
                model_inputs["satellite_features"] = self._parse_satellite_features(item.get("data_content"))
                model_inputs["static_geo"] = self._extract_static_geo(item.get("data_content"), geometry)
                source_status["earth_engine"] = {"ok": True, "error": None, "count": 1}
            else:
                source_status["earth_engine"] = {"ok": False, "error": error, "count": 0}

        # Copernicus / met
        if self.tools.get("copernicus") and geometry:
            ok, item, error = self._fetch_copernicus(geometry, time_range, mode="prediction")
            if ok and item:
                retrieved_data.append(item)
                model_inputs["met_timeseries"] = self._parse_met_timeseries(item.get("data_content"))
                source_status["copernicus"] = {"ok": True, "error": None, "count": 1}
            else:
                source_status["copernicus"] = {"ok": False, "error": error, "count": 0}

        # Web textual context
        if self.tools.get("web_search"):
            ok, items, error = self._fetch_web(query)
            if ok:
                retrieved_data.extend(items)
                source_status["web_search"] = {"ok": True, "error": None, "count": len(items)}
            else:
                source_status["web_search"] = {"ok": False, "error": error, "count": 0}

        # Arxiv textual support
        if self.tools.get("arxiv"):
            ok, items, error = self._fetch_arxiv(query)
            if ok:
                retrieved_data.extend(items)
                source_status["arxiv"] = {"ok": True, "error": None, "count": len(items)}
            else:
                source_status["arxiv"] = {"ok": False, "error": error, "count": 0}

        model_inputs["textual_context"] = self._build_textual_context(retrieved_data)

        reflection = self._reflect_prediction_bundle(
            query=query,
            normalized_request=normalized_request,
            model_inputs=model_inputs,
            source_status=source_status,
        )
        reflection_trace.append(reflection)

        used_fallback = reflection.get("needs_retry", False)
        data_quality = self._build_data_quality(
            task_type="prediction",
            source_status=source_status,
            model_inputs=model_inputs,
            used_fallback=used_fallback,
        )

        return {
            "retrieval_summary": {
                "original_query": query,
                "normalized_query": normalized_request["normalized_query"],
                "task_type": "prediction",
                "geometry": geometry,
                "time_range": time_range,
            },
            "retrieved_data": retrieved_data,
            "model_inputs": model_inputs,
            "data_quality": data_quality,
            "reflection_trace": reflection_trace,
            "error": self._coalesce_source_errors(source_status),
        }

    # ─────────────────────────────────────────
    # Retry / reflection
    # ─────────────────────────────────────────

    def _reflect_prediction_bundle(
        self,
        query: str,
        normalized_request: dict[str, Any],
        model_inputs: dict[str, Any],
        source_status: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Reflection nhẹ:
        - thiếu met_timeseries?
        - thiếu satellite_features?
        - thiếu textual_context?
        """
        missing = []
        if not model_inputs.get("met_timeseries"):
            missing.append("met_timeseries")
        if not model_inputs.get("satellite_features"):
            missing.append("satellite_features")
        if not model_inputs.get("static_geo"):
            missing.append("static_geo")
        if not model_inputs.get("textual_context"):
            missing.append("textual_context")

        needs_retry = bool(missing)
        strategy = []

        if "satellite_features" in missing:
            strategy.append("switch_source_or_reduce_spatial_granularity")
        if "met_timeseries" in missing:
            strategy.append("expand_time_window_or_switch_source")
        if "textual_context" in missing:
            strategy.append("switch_source_to_web_or_arxiv")
        if "static_geo" in missing:
            strategy.append("fallback_to_geometry_derived_static_geo")

        return {
            "query": query,
            "task_type": "prediction",
            "missing_components": missing,
            "source_status": source_status,
            "needs_retry": needs_retry,
            "recommended_retry_strategy": strategy,
        }

    def _prediction_attempt_is_usable(self, attempt: dict[str, Any]) -> bool:
        mi = attempt.get("model_inputs", {})
        return bool(
            mi.get("met_timeseries")
            and mi.get("satellite_features")
            and mi.get("static_geo")
        )

    def _next_retry_request(
        self,
        current_request: dict[str, Any],
        attempt: dict[str, Any],
        retry_idx: int,
    ) -> dict[str, Any]:
        """
        Retry chiến lược:
        1) mở rộng time window
        2) giảm spatial granularity
        """
        new_request = deepcopy(current_request)
        strategy_notes = []

        if retry_idx == 0:
            new_request["time_range"] = self._expand_time_window(new_request.get("time_range"), days=3)
            strategy_notes.append("expand_time_window")
        else:
            new_request["geometry"] = self._reduce_spatial_granularity(new_request.get("geometry"))
            strategy_notes.append("reduce_spatial_granularity")

        logger.info("Prediction retrieval retry strategy: %s", strategy_notes)
        return new_request

    def _choose_best_prediction_attempt(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Chọn attempt tốt nhất dựa trên số thành phần model_inputs có đủ.
        """
        def score(attempt: dict[str, Any]) -> int:
            mi = attempt.get("model_inputs", {})
            s = 0
            for key in ["met_timeseries", "satellite_features", "static_geo", "textual_context"]:
                if mi.get(key):
                    s += 1
            return s

        best = max(attempts, key=score)
        dq = best.get("data_quality", {})
        dq["used_fallback"] = True
        best["data_quality"] = dq
        return best

    def _expand_time_window(self, time_range: dict[str, Any] | None, days: int = 3) -> dict[str, Any] | None:
        if not isinstance(time_range, dict):
            return time_range

        start_key = "start" if "start" in time_range else "start_date" if "start_date" in time_range else None
        end_key = "end" if "end" in time_range else "end_date" if "end_date" in time_range else None
        if not start_key or not end_key:
            return time_range

        try:
            start_dt = self._parse_dt(time_range.get(start_key))
            end_dt = self._parse_dt(time_range.get(end_key))
            if not start_dt or not end_dt:
                return time_range

            expanded = dict(time_range)
            expanded[start_key] = self._format_dt_like(time_range[start_key], start_dt - timedelta(days=days))
            expanded[end_key] = self._format_dt_like(time_range[end_key], end_dt + timedelta(days=days))
            return expanded
        except Exception:
            return time_range

    def _reduce_spatial_granularity(self, geometry: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Nếu bbox/polygon quá chi tiết thì giảm về point centroid.
        """
        if not isinstance(geometry, dict):
            return geometry

        gtype = geometry.get("type")
        value = geometry.get("value")

        if gtype == "bbox" and isinstance(value, dict):
            try:
                lat = (float(value["min_lat"]) + float(value["max_lat"])) / 2.0
                lon = (float(value["min_lon"]) + float(value["max_lon"])) / 2.0
                return {"type": "point", "value": {"lat": lat, "lon": lon}}
            except Exception:
                return geometry

        if gtype == "polygon":
            centroid = self._polygon_centroid(value)
            if centroid:
                return {"type": "point", "value": centroid}

        return geometry

    # ─────────────────────────────────────────
    # Source fetchers
    # ─────────────────────────────────────────

    def _fetch_gee(
        self,
        geometry: dict[str, Any],
        time_range: dict[str, Any] | None,
        mode: str = "prediction",
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        gee = self.tools.get("earth_engine")
        if not gee:
            return False, None, "Earth Engine tool unavailable"

        try:
            if geometry["type"] == "point":
                payload = geometry["value"]
                result = self._call_first_available(
                    gee,
                    [
                        ("fetch_satellite_stats", {"location": payload, "time_range": time_range}),
                        ("fetch_stats", {"location": payload, "time_range": time_range}),
                        ("retrieve", {"location": payload, "time_range": time_range}),
                    ],
                )
            elif geometry["type"] == "bbox":
                payload = geometry["value"]
                result = self._call_first_available(
                    gee,
                    [
                        ("fetch_satellite_stats", {"bbox": payload, "time_range": time_range}),
                        ("fetch_stats", {"bbox": payload, "time_range": time_range}),
                        ("retrieve", {"bbox": payload, "time_range": time_range}),
                    ],
                )
            else:  # polygon
                payload = geometry["value"]
                result = self._call_first_available(
                    gee,
                    [
                        ("fetch_satellite_stats", {"polygon": payload, "time_range": time_range}),
                        ("fetch_stats", {"polygon": payload, "time_range": time_range}),
                        ("retrieve", {"polygon": payload, "time_range": time_range}),
                    ],
                )

            return True, {
                "sub_question_id": "satellite",
                "data_content": result,
                "source": "GEE",
                "citation": "Google Earth Engine",
                "confidence_score": 0.90 if mode == "prediction" else 0.80,
            }, None
        except Exception as e:
            logger.warning("GEE collection failed: %s", e)
            return False, None, str(e)

    def _fetch_copernicus(
        self,
        geometry: dict[str, Any],
        time_range: dict[str, Any] | None,
        mode: str = "prediction",
    ) -> tuple[bool, dict[str, Any] | None, str | None]:
        cds = self.tools.get("copernicus")
        if not cds:
            return False, None, "Copernicus tool unavailable"

        try:
            if geometry["type"] == "point":
                pt = geometry["value"]
                result = self._call_first_available(
                    cds,
                    [
                        ("fetch_era5", {"lat": pt["lat"], "lon": pt["lon"], "time_range": time_range}),
                        ("retrieve", {"lat": pt["lat"], "lon": pt["lon"], "time_range": time_range}),
                    ],
                )
            elif geometry["type"] == "bbox":
                bbox = geometry["value"]
                result = self._call_first_available(
                    cds,
                    [
                        ("fetch_era5", {"bbox": bbox, "time_range": time_range}),
                        ("retrieve", {"bbox": bbox, "time_range": time_range}),
                    ],
                )
            else:
                poly = geometry["value"]
                result = self._call_first_available(
                    cds,
                    [
                        ("fetch_era5", {"polygon": poly, "time_range": time_range}),
                        ("retrieve", {"polygon": poly, "time_range": time_range}),
                    ],
                )

            return True, {
                "sub_question_id": "met",
                "data_content": result,
                "source": "Copernicus CDS",
                "citation": "ERA5",
                "confidence_score": 0.90 if mode == "prediction" else 0.80,
            }, None
        except Exception as e:
            logger.warning("Copernicus collection failed: %s", e)
            return False, None, str(e)

    def _fetch_web(
        self,
        query: str,
    ) -> tuple[bool, list[dict[str, Any]], str | None]:
        web = self.tools.get("web_search")
        if not web:
            return False, [], "Web search tool unavailable"

        try:
            results = self._call_first_available(
                web,
                [
                    ("search", {"query": query, "max_results": self.max_news_results}),
                    ("search", {"q": query, "max_results": self.max_news_results}),
                    ("retrieve", {"query": query, "max_results": self.max_news_results}),
                ],
            )
            items = []
            for i, r in enumerate(results or []):
                items.append(
                    {
                        "sub_question_id": f"news-{i}",
                        "data_content": r,
                        "source": "DuckDuckGo",
                        "citation": r.get("url", "") if isinstance(r, dict) else "",
                        "confidence_score": 0.70,
                    }
                )
            return True, items, None
        except Exception as e:
            logger.warning("Web search collection failed: %s", e)
            return False, [], str(e)

    def _fetch_arxiv(
        self,
        query: str,
    ) -> tuple[bool, list[dict[str, Any]], str | None]:
        arxiv_tool = self.tools.get("arxiv")
        if not arxiv_tool:
            return False, [], "ArXiv tool unavailable"

        try:
            papers = self._call_first_available(
                arxiv_tool,
                [
                    ("search", {"query": query, "max_results": self.max_arxiv_papers}),
                    ("retrieve", {"query": query, "max_results": self.max_arxiv_papers}),
                ],
            )
            items = []
            for i, p in enumerate(papers or []):
                items.append(
                    {
                        "sub_question_id": f"arxiv-{i}",
                        "data_content": p,
                        "source": "ArXiv",
                        "citation": p.get("url", "") if isinstance(p, dict) else "",
                        "confidence_score": 0.85,
                    }
                )
            return True, items, None
        except Exception as e:
            logger.warning("ArXiv collection failed: %s", e)
            return False, [], str(e)

    def _call_first_available(
        self,
        tool_obj: Any,
        candidates: list[tuple[str, dict[str, Any]]],
    ) -> Any:
        last_error = None
        for method_name, kwargs in candidates:
            fn = getattr(tool_obj, method_name, None)
            if not callable(fn):
                continue
            try:
                return fn(**kwargs)
            except TypeError:
                try:
                    return fn(kwargs)
                except Exception as e:
                    last_error = e
            except Exception as e:
                last_error = e
        raise RuntimeError(last_error or f"No compatible method found for {tool_obj}")

    # ─────────────────────────────────────────
    # Parse model inputs
    # ─────────────────────────────────────────

    def _parse_met_timeseries(self, raw: Any) -> list[dict[str, Any]]:
        """
        Chuẩn hóa met_timeseries.
        Ưu tiên field có sẵn, fallback từ summary nếu cần.
        """
        if not isinstance(raw, dict):
            return []

        # direct timeseries
        for key in ["timeseries", "time_series", "series", "hourly", "daily"]:
            value = raw.get(key)
            if isinstance(value, list) and value:
                return value

        summary = raw.get("summary", {}) if isinstance(raw.get("summary"), dict) else {}
        flattened = {
            "temperature": raw.get("temperature", summary.get("temperature")),
            "humidity": raw.get("humidity", summary.get("humidity")),
            "wind_speed": raw.get("wind_speed", summary.get("wind_speed")),
            "precipitation": raw.get("precipitation", summary.get("precipitation")),
        }
        if any(v is not None for v in flattened.values()):
            return [flattened]

        return []

    def _parse_satellite_features(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}

        stats = raw.get("stats", {}) if isinstance(raw.get("stats"), dict) else {}
        out = {
            "ndvi_mean": raw.get("ndvi_mean", stats.get("ndvi_mean")),
            "ndvi_min": raw.get("ndvi_min", stats.get("ndvi_min")),
            "ndvi_max": raw.get("ndvi_max", stats.get("ndvi_max")),
            "lst_mean": raw.get("lst_mean", stats.get("lst_mean")),
            "burned_area": raw.get("burned_area", stats.get("burned_area")),
            "vegetation_index": raw.get("vegetation_index", stats.get("vegetation_index")),
        }
        return {k: v for k, v in out.items() if v is not None}

    def _extract_static_geo(
        self,
        raw: Any,
        geometry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        if isinstance(raw, dict):
            for key in ["elevation", "slope", "landcover", "soil_type", "aspect"]:
                if raw.get(key) is not None:
                    out[key] = raw.get(key)

        if not out and geometry:
            if geometry.get("type") == "point":
                pt = geometry.get("value", {})
                out = {
                    "lat": pt.get("lat"),
                    "lon": pt.get("lon"),
                }
            elif geometry.get("type") == "bbox":
                out = {"bbox": geometry.get("value")}
            elif geometry.get("type") == "polygon":
                out = {"polygon": geometry.get("value")}

        return {k: v for k, v in out.items() if v is not None}

    def _build_textual_context(self, retrieved_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Chuẩn hóa textual_context từ web/arxiv cho prediction hoặc QA.
        """
        out = []
        for item in retrieved_data:
            source = item.get("source")
            if source not in {"DuckDuckGo", "ArXiv"}:
                continue

            dc = item.get("data_content")
            if isinstance(dc, str):
                out.append({"source": source, "text": dc})
                continue

            if isinstance(dc, dict):
                title = dc.get("title") or dc.get("name") or ""
                body = dc.get("body") or dc.get("snippet") or dc.get("summary") or dc.get("abstract") or dc.get("content") or ""
                text = "\n".join(part for part in [title, body] if part).strip()
                if text:
                    out.append(
                        {
                            "source": source,
                            "title": title,
                            "text": text,
                            "url": dc.get("url") or dc.get("link"),
                        }
                    )
        return out

    # ─────────────────────────────────────────
    # Knowledge extraction from collected items
    # ─────────────────────────────────────────

    def _extract_knowledge_from_collected(
        self,
        collected: dict[str, Any],
    ) -> dict[str, list[str]]:
        texts = []
        for item in collected.get("retrieved_data", []):
            dc = item.get("data_content")
            if isinstance(dc, str):
                texts.append(dc)
            elif isinstance(dc, dict):
                title = dc.get("title") or dc.get("name") or ""
                body = dc.get("body") or dc.get("snippet") or dc.get("content") or dc.get("summary") or dc.get("abstract") or ""
                if title or body:
                    texts.append(f"Title: {title}\nContent: {body}".strip())
                else:
                    texts.append(json.dumps(dc, default=str))

        knowledge = {"factual_statements": [], "causal_relationships": []}
        for text in texts[:5]:
            k = self.extract_knowledge(text)
            knowledge["factual_statements"].extend(k.get("factual_statements", []))
            knowledge["causal_relationships"].extend(k.get("causal_relationships", []))

        return knowledge

    # ─────────────────────────────────────────
    # Data quality
    # ─────────────────────────────────────────

    def _build_data_quality(
        self,
        task_type: str,
        source_status: dict[str, Any],
        model_inputs: dict[str, Any],
        used_fallback: bool,
    ) -> dict[str, Any]:
        missing_variables: list[str] = []

        if task_type == "prediction":
            if not model_inputs.get("met_timeseries"):
                missing_variables.append("model_inputs.met_timeseries")
            if not model_inputs.get("satellite_features"):
                missing_variables.append("model_inputs.satellite_features")
            if not model_inputs.get("static_geo"):
                missing_variables.append("model_inputs.static_geo")
            if not model_inputs.get("textual_context"):
                missing_variables.append("model_inputs.textual_context")

        ok_sources = [k for k, v in source_status.items() if v.get("ok")]
        error_sources = {k: v.get("error") for k, v in source_status.items() if not v.get("ok") and v.get("error")}

        level = "high"
        if missing_variables:
            level = "medium"
        if len(missing_variables) >= 2:
            level = "low"

        return {
            "level": level,
            "sources_ok": ok_sources,
            "sources_error": error_sources,
            "missing_variables": missing_variables,
            "used_fallback": used_fallback,
            "source_status": source_status,
        }

    def _coalesce_source_errors(self, source_status: dict[str, Any]) -> str | None:
        errors = []
        for source_name, status in source_status.items():
            if status.get("error"):
                errors.append(f"{source_name}: {status['error']}")
        return "; ".join(errors) if errors else None

    # ─────────────────────────────────────────
    # Utils
    # ─────────────────────────────────────────

    def _build_error_result(
        self,
        query: str,
        normalized_request: dict[str, Any],
        error: str,
    ) -> dict[str, Any]:
        return {
            "task_type": normalized_request.get("task_type", "qa"),
            "retrieval_summary": {
                "original_query": query,
                "normalized_query": normalized_request.get("normalized_query"),
            },
            "normalized_query": normalized_request.get("normalized_query"),
            "normalized_query_context": normalized_request.get("normalized_query_context", {}),
            "sub_requests": [],
            "retrieved_data": [],
            "model_inputs": self._empty_model_inputs(),
            "data_quality": {
                **self._empty_data_quality(),
                "sources_error": {"normalization": error},
                "level": "low",
            },
            "extracted_knowledge": {"factual_statements": [], "causal_relationships": []},
            "reflection_trace": [],
            "error": error,
        }

    def _init_source_status(self) -> dict[str, Any]:
        return {
            "earth_engine": {"ok": False, "error": None, "count": 0},
            "copernicus": {"ok": False, "error": None, "count": 0},
            "web_search": {"ok": False, "error": None, "count": 0},
            "arxiv": {"ok": False, "error": None, "count": 0},
        }

    def _empty_model_inputs(self) -> dict[str, Any]:
        return {
            "met_timeseries": [],
            "satellite_features": {},
            "static_geo": {},
            "textual_context": [],
        }

    def _empty_data_quality(self) -> dict[str, Any]:
        return {
            "level": "unknown",
            "sources_ok": [],
            "sources_error": {},
            "missing_variables": [],
            "used_fallback": False,
            "source_status": self._init_source_status(),
        }

    def _serialize_geometry(self, geometry: dict[str, Any] | None) -> str:
        if not geometry:
            return "none"
        try:
            return json.dumps(geometry, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return str(geometry)

    def _polygon_centroid(self, polygon: Any) -> dict[str, float] | None:
        """
        Best-effort centroid cho polygon GeoJSON-like.
        """
        try:
            coords = None
            if isinstance(polygon, dict):
                if polygon.get("type") == "Polygon":
                    coords = polygon.get("coordinates", [[]])[0]
                elif polygon.get("coordinates"):
                    coords = polygon.get("coordinates")[0]
            elif isinstance(polygon, list):
                coords = polygon

            if not coords:
                return None

            xs, ys = [], []
            for pt in coords:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    lon, lat = pt[0], pt[1]
                    xs.append(float(lon))
                    ys.append(float(lat))

            if not xs or not ys:
                return None

            return {"lat": sum(ys) / len(ys), "lon": sum(xs) / len(xs)}
        except Exception:
            return None

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(text, "%Y-%m-%d")
            except Exception:
                return None

    def _format_dt_like(self, original: Any, value: datetime) -> str:
        text = str(original)
        if "T" in text:
            return value.isoformat()
        return value.strftime("%Y-%m-%d")

    def _strip_code_fence(self, text: str) -> str:
        content = (text or "").strip()
        if not content.startswith("```"):
            return content
        lines = content.splitlines()
        cleaned = []
        for line in lines:
            striped = line.strip().lower()
            if striped.startswith("```") or striped == "json":
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()