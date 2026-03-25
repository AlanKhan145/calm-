"""
QueryNormalizerAgent — chuẩn hóa truy vấn thô thành context JSON chuẩn cho planner.

Agent này nên chạy TRƯỚC planner.

Input:
- query text thô từ user, có thể là tiếng Việt / tiếng Anh / pha trộn tự nhiên

Output JSON chuẩn:
- task_type
- location_text
- location_kind
- location_resolved
- time_expression
- time_range
- prediction_target
- requested_output
- ambiguities

Nguyên tắc:
- Không hallucinate lat/lon hay time_range.
- LLM chỉ dùng để hiểu câu và tách trường.
- Tọa độ thật vẫn phải đến từ geocoder hoặc từ query coordinates.
- Nếu mơ hồ hoặc resolve thất bại thì ghi vào ambiguities.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class QueryNormalizerAgent:
    """
    Chuẩn hóa truy vấn trước planner.

    geocoder:
        object optional, có thể hỗ trợ một trong các method:
        - resolve(text)
        - geocode(text)
        - resolve_location(text)
        - lookup(text)

    time_utils:
        object optional, có thể hỗ trợ một trong các method:
        - extract_time_expression(query)
        - find_time_expression(query)
        - resolve_time_expression(expression, ...)
        - parse_time_expression(expression, ...)
        - to_time_range(expression, ...)
        - parse_query_time(expression, ...)

    llm:
        object optional, có thể hỗ trợ một trong các method:
        - invoke(prompt)
        - run(prompt)
        - complete(prompt)
        - generate(prompt)
    """

    _COORD_PATTERN = re.compile(
        r"(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(?P<lon>-?\d{1,3}(?:\.\d+)?)"
    )
    _ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    _DMY_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

    def __init__(
        self,
        geocoder: Any | None = None,
        time_utils: Any | None = None,
        llm: Any | None = None,
        timezone: str = "Asia/Bangkok",
        now_fn=None,
    ) -> None:
        self.geocoder = geocoder
        self.time_utils = time_utils
        self.llm = llm
        self.timezone = timezone
        self.now_fn = now_fn or self._default_now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query_text: str) -> dict[str, Any]:
        return self.normalize(query_text)

    def normalize(self, query_text: str) -> dict[str, Any]:
        query = (query_text or "").strip()
        ambiguities: list[dict[str, Any]] = []

        if not query:
            return self._build_empty_result()

        task_type = self._infer_task_type(query)
        prediction_target = self._infer_prediction_target(query)
        requested_output = self._infer_requested_output(query)

        time_expression, time_expr_ambiguities = self._extract_time_expression(query)
        ambiguities.extend(time_expr_ambiguities)

        llm_parse = self._llm_extract_semantics(query)
        if llm_parse is not None:
            llm_ambiguities = llm_parse.pop("_ambiguities", None)
            if isinstance(llm_ambiguities, list):
                ambiguities.extend(llm_ambiguities)

        if llm_parse:
            task_type = self._coalesce_valid(
                self._normalize_task_type(llm_parse.get("task_type")),
                task_type,
            )
            prediction_target = self._coalesce_valid(
                self._normalize_prediction_target(llm_parse.get("prediction_target")),
                prediction_target,
            )
            requested_output = self._merge_unique(
                requested_output,
                self._normalize_requested_outputs(llm_parse.get("requested_output")),
            )

            llm_time_expression = llm_parse.get("time_expression")
            if isinstance(llm_time_expression, str) and llm_time_expression.strip():
                time_expression = llm_time_expression.strip()

        location_text, location_kind, location_ambiguities = self._extract_location(
            query=query,
            llm_parse=llm_parse,
            known_time_expression=time_expression,
        )
        ambiguities.extend(location_ambiguities)

        location_resolved = None
        if location_text is not None:
            location_resolved, resolve_location_ambiguities = self._resolve_location(
                location_text=location_text,
                location_kind=location_kind,
            )
            ambiguities.extend(resolve_location_ambiguities)

        time_range = None
        if time_expression is not None:
            time_range, time_range_ambiguities = self._resolve_time_expression(time_expression)
            ambiguities.extend(time_range_ambiguities)

        if task_type == "prediction" and prediction_target is None:
            ambiguities.append(
                {
                    "field": "prediction_target",
                    "reason": "unspecified_prediction_target",
                    "message": "Prediction intent detected but target is not explicit enough.",
                }
            )

        return {
            "task_type": task_type,
            "location_text": location_text,
            "location_kind": location_kind,
            "location_resolved": location_resolved,
            "time_expression": time_expression,
            "time_range": time_range,
            "prediction_target": prediction_target,
            "requested_output": requested_output,
            "ambiguities": ambiguities,
        }

    # ------------------------------------------------------------------
    # Empty result
    # ------------------------------------------------------------------

    def _build_empty_result(self) -> dict[str, Any]:
        return {
            "task_type": "unknown",
            "location_text": None,
            "location_kind": "unknown",
            "location_resolved": None,
            "time_expression": None,
            "time_range": None,
            "prediction_target": None,
            "requested_output": [],
            "ambiguities": [
                {
                    "field": "query",
                    "reason": "empty_query",
                    "message": "Query text is empty.",
                }
            ],
        }

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer_task_type(self, query: str) -> str:
        q = query.lower()

        prediction_keywords = [
            "predict", "prediction", "forecast", "estimate", "project",
            "dự đoán", "dự báo", "ước tính",
            "risk", "rủi ro", "nguy cơ",
            "spread", "lan rộng",
        ]
        monitoring_keywords = [
            "monitor", "monitoring", "alert", "watch",
            "cảnh báo", "theo dõi", "giám sát",
        ]
        qa_keywords = [
            "what", "why", "how", "when", "where",
            "giải thích", "là gì", "tại sao", "như thế nào", "ở đâu", "khi nào",
        ]
        analysis_keywords = [
            "analyze", "analysis", "compare", "evaluation", "assess",
            "phân tích", "so sánh", "đánh giá",
        ]

        if any(k in q for k in monitoring_keywords):
            return "monitoring"
        if any(k in q for k in prediction_keywords):
            return "prediction"
        if any(k in q for k in analysis_keywords):
            return "analysis"
        if any(k in q for k in qa_keywords):
            return "qa"
        return "unknown"

    def _infer_prediction_target(self, query: str) -> str | None:
        q = query.lower()

        if any(k in q for k in ["spread forecast", "fire spread", "lan rộng", "cháy lan"]):
            return "fire_spread"
        if any(k in q for k in ["risk map", "bản đồ rủi ro", "heatmap"]):
            return "risk_map"
        if any(k in q for k in ["wildfire", "cháy rừng", "fire risk", "nguy cơ cháy", "rủi ro cháy"]):
            return "wildfire_risk"
        return None

    def _infer_requested_output(self, query: str) -> list[str]:
        q = query.lower()
        outputs: list[str] = []

        rules = [
            ("probability", ["probability", "xác suất"]),
            ("logit", ["logit"]),
            ("risk_level", ["risk level", "mức rủi ro"]),
            ("confidence", ["confidence", "độ tin cậy"]),
            ("risk_map", ["risk map", "bản đồ rủi ro", "heatmap"]),
            ("spread_forecast", ["spread forecast", "fire spread", "lan rộng", "cháy lan"]),
        ]

        for output_name, keywords in rules:
            if any(k in q for k in keywords):
                outputs.append(output_name)

        if "risk" in q or "rủi ro" in q or "nguy cơ" in q:
            outputs.append("risk_level")

        return self._unique_keep_order(outputs)

    # ------------------------------------------------------------------
    # LLM semantic extraction
    # ------------------------------------------------------------------

    def _llm_extract_semantics(self, query: str) -> dict[str, Any] | None:
        if self.llm is None:
            return None

        prompt = self._build_llm_prompt(query)

        try:
            raw = self._call_llm(prompt)
            payload = self._extract_json_object(raw)
            if not isinstance(payload, dict):
                return None

            normalized = {
                "task_type": self._normalize_task_type(payload.get("task_type")),
                "prediction_target": self._normalize_prediction_target(payload.get("prediction_target")),
                "location_text": self._normalize_llm_location_text(payload.get("location_text")),
                "time_expression": self._normalize_optional_string(payload.get("time_expression")),
                "requested_output": self._normalize_requested_outputs(payload.get("requested_output")),
                "confidence": self._safe_float(payload.get("confidence")),
            }

            ambiguities: list[dict[str, Any]] = []

            loc = normalized.get("location_text")
            if isinstance(loc, str) and loc and len(loc.split()) > 12:
                ambiguities.append(
                    {
                        "field": "location",
                        "reason": "llm_location_too_long",
                        "message": "LLM returned an overlong location candidate; falling back to rules if needed.",
                        "raw": loc,
                    }
                )

            normalized["_ambiguities"] = ambiguities
            return normalized
        except Exception as e:
            logger.warning("QueryNormalizerAgent llm extraction failed: %s", e)
            return None

    def _build_llm_prompt(self, query: str) -> str:
        return f"""
You are a multilingual query normalizer for a wildfire and geospatial prediction system.

The user may write in English, Vietnamese, or mixed language.
Your job is to extract structured fields from the query.

Return ONLY valid JSON. No markdown. No explanation.

Schema:
{{
  "task_type": "prediction|monitoring|analysis|qa|unknown|null",
  "prediction_target": "wildfire_risk|fire_spread|risk_map|unknown|null",
  "location_text": "string|null",
  "time_expression": "string|null",
  "requested_output": ["risk_level"|"probability"|"confidence"|"risk_map"|"spread_forecast"|"logit"],
  "confidence": 0.0
}}

Rules:
- Understand English, Vietnamese, and mixed-language queries.
- Separate location from time.
- Never include time words inside location_text.
- Keep location_text concise but complete.
- Do NOT invent coordinates.
- If uncertain, use null.
- If the query asks for wildfire/fire risk prediction, prefer prediction target "wildfire_risk".
- If the query asks about spread/lan rộng/cháy lan, prefer prediction target "fire_spread".
- If the query asks for a map/heatmap/bản đồ rủi ro, include "risk_map" in requested_output.

Examples:
Query: "Predict wildfire risk for Central Valley California next 7 days"
JSON:
{{
  "task_type": "prediction",
  "prediction_target": "wildfire_risk",
  "location_text": "Central Valley California",
  "time_expression": "next 7 days",
  "requested_output": ["risk_level"],
  "confidence": 0.95
}}

Query: "Dự đoán nguy cơ cháy rừng ở Tây Nguyên 7 ngày tới"
JSON:
{{
  "task_type": "prediction",
  "prediction_target": "wildfire_risk",
  "location_text": "Tây Nguyên",
  "time_expression": "7 ngày tới",
  "requested_output": ["risk_level"],
  "confidence": 0.95
}}

Query:
{query}
""".strip()

    def _call_llm(self, prompt: str) -> str:
        method_names = ["invoke", "run", "complete", "generate"]
        for name in method_names:
            fn = getattr(self.llm, name, None)
            if not callable(fn):
                continue

            result = fn(prompt)

            if isinstance(result, str):
                return result

            content = getattr(result, "content", None)
            if isinstance(content, str):
                return content

            text = getattr(result, "text", None)
            if isinstance(text, str):
                return text

            generations = getattr(result, "generations", None)
            if generations:
                try:
                    first = generations[0]
                    if isinstance(first, list) and first:
                        item = first[0]
                        item_text = getattr(item, "text", None)
                        if isinstance(item_text, str):
                            return item_text
                except Exception:
                    pass

        raise RuntimeError("No compatible LLM method found on injected llm object.")

    def _extract_json_object(self, raw: str) -> dict[str, Any] | None:
        if not isinstance(raw, str):
            return None

        raw = raw.strip()

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None

        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

        return None

    # ------------------------------------------------------------------
    # Location parsing + geocoding
    # ------------------------------------------------------------------

    def _extract_location(
        self,
        query: str,
        llm_parse: dict[str, Any] | None = None,
        known_time_expression: str | None = None,
    ) -> tuple[str | None, str, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        coord_matches = list(self._COORD_PATTERN.finditer(query))
        valid_coords = []
        for match in coord_matches:
            lat = self._safe_float(match.group("lat"))
            lon = self._safe_float(match.group("lon"))
            if lat is None or lon is None:
                continue
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                valid_coords.append(match.group(0).strip())

        if len(valid_coords) == 1:
            return valid_coords[0], "coordinates", ambiguities

        if len(valid_coords) > 1:
            ambiguities.append(
                {
                    "field": "location",
                    "reason": "multiple_coordinate_candidates",
                    "message": "Multiple coordinate candidates found in query.",
                    "candidates": valid_coords,
                }
            )
            return None, "unknown", ambiguities

        llm_location = None
        if isinstance(llm_parse, dict):
            llm_location = llm_parse.get("location_text")

        if isinstance(llm_location, str) and llm_location.strip():
            cleaned = self._sanitize_location_text(
                llm_location,
                known_time_expression=known_time_expression,
            )
            if cleaned:
                return cleaned, "place_name", ambiguities

        query_wo_time = self._remove_time_phrases_for_location(
            query,
            known_time_expression=known_time_expression,
        )

        candidate_patterns = [
            r"\b(?:ở|tại|khu vực|vùng|quanh|gần|cho|cho khu vực)\s+(.+?)(?=$|[.;?!])",
            r"\b(?:in|at|around|near|for|over|across)\s+(.+?)(?=$|[.;?!])",
        ]

        candidates: list[str] = []
        for pattern in candidate_patterns:
            for m in re.finditer(pattern, query_wo_time, flags=re.IGNORECASE):
                raw = (m.group(1) or "").strip()
                cleaned = self._trim_location_candidate(raw)
                if cleaned:
                    candidates.append(cleaned)

        candidates = self._unique_keep_order(candidates)

        if len(candidates) == 1:
            return candidates[0], "place_name", ambiguities

        if len(candidates) > 1:
            ambiguities.append(
                {
                    "field": "location",
                    "reason": "multiple_location_candidates",
                    "message": "Multiple location text candidates found in query.",
                    "candidates": candidates,
                }
            )
            return None, "unknown", ambiguities

        direct = self._extract_direct_place_candidate(query_wo_time)
        if direct:
            return direct, "place_name", ambiguities

        return None, "unknown", ambiguities

    def _remove_time_phrases_for_location(
        self,
        query: str,
        known_time_expression: str | None = None,
    ) -> str:
        q = f" {query} "

        if known_time_expression:
            q = re.sub(re.escape(known_time_expression), " ", q, flags=re.IGNORECASE)

        time_patterns = [
            r"\bnext\s+\d+\s+(?:day|days|hour|hours|week|weeks|month|months|year|years)\b",
            r"\bthis\s+(?:week|month|year)\b",
            r"\bnext\s+(?:week|month|year)\b",
            r"\btoday\b",
            r"\btomorrow\b",
            r"\byesterday\b",
            r"\btonight\b",
            r"\bthis\s+evening\b",
            r"\bthis\s+afternoon\b",
            r"\b\d+\s+ngày\s+tới\b",
            r"\b\d+\s+giờ\s+tới\b",
            r"\btuần\s+này\b",
            r"\btuần\s+tới\b",
            r"\btháng\s+này\b",
            r"\btháng\s+tới\b",
            r"\bnăm\s+này\b",
            r"\bnăm\s+tới\b",
            r"\bhôm\s+nay\b",
            r"\bngày\s+mai\b",
            r"\bhôm\s+qua\b",
            r"\btối\s+nay\b",
            r"\bchiều\s+nay\b",
        ]

        for pattern in time_patterns:
            q = re.sub(pattern, " ", q, flags=re.IGNORECASE)

        return re.sub(r"\s+", " ", q).strip()

    def _extract_direct_place_candidate(self, query: str) -> str | None:
        q = query.strip(" \t\n\"'“”‘’,:-")
        if not q:
            return None

        q = re.sub(
            r"^(predict|forecast|estimate|analyze|show|give|compute|calculate|monitor)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^(dự đoán|dự báo|ước tính|phân tích|cho biết|theo dõi|giám sát)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^(wildfire risk|fire risk|risk map|spread forecast|wildfire|fire spread)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^(nguy cơ cháy rừng|rủi ro cháy rừng|bản đồ rủi ro|cháy rừng|cháy lan)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )

        q = q.strip(" \t\n\"'“”‘’,:-")
        q = self._trim_location_candidate(q)

        if not q:
            return None

        banned_fragments = {
            "wildfire risk",
            "fire risk",
            "risk map",
            "spread forecast",
            "prediction",
            "forecast",
            "dự đoán",
            "dự báo",
            "nguy cơ cháy",
            "rủi ro cháy",
        }
        if q.lower() in banned_fragments:
            return None

        if len(q.split()) > 8:
            return None

        return q

    def _trim_location_candidate(self, text: str) -> str | None:
        if not text:
            return None

        candidate = text.strip(" \t\n\"'“”‘’,:-")

        stop_markers = [
            " hôm nay", " ngày mai", " hôm qua",
            " tuần này", " tuần tới", " tháng này", " tháng tới", " năm nay", " năm tới",
            " 24 giờ tới", " 48 giờ tới", " 7 ngày tới",
            " this week", " next week", " this month", " next month", " this year", " next year",
            " today", " tomorrow", " yesterday", " tonight",
            " next 24 hours", " next 48 hours", " next 7 days",
            " from ", " to ", " between ", " and ", " during ",
            " từ ", " đến ", " vào ", " lúc ", " trong ",
        ]

        lowered = candidate.lower()
        cut_index = None
        for marker in stop_markers:
            idx = lowered.find(marker)
            if idx != -1:
                cut_index = idx if cut_index is None else min(cut_index, idx)

        if cut_index is not None:
            candidate = candidate[:cut_index].strip(" \t\n\"'“”‘’,:-")

        if not candidate:
            return None

        if candidate.lower() in {"đó", "đây", "there", "here", "khu vực đó"}:
            return None

        return candidate

    def _sanitize_location_text(
        self,
        text: str,
        known_time_expression: str | None = None,
    ) -> str | None:
        if not text or not isinstance(text, str):
            return None

        candidate = text.strip(" \t\n\"'“”‘’,:-")

        if known_time_expression:
            candidate = re.sub(
                re.escape(known_time_expression),
                " ",
                candidate,
                flags=re.IGNORECASE,
            )

        candidate = self._trim_location_candidate(candidate)
        if not candidate:
            return None

        banned = {
            "wildfire risk",
            "fire risk",
            "next 7 days",
            "this week",
            "today",
            "tomorrow",
            "dự đoán",
            "dự báo",
            "7 ngày tới",
            "tuần này",
            "tuần tới",
        }
        if candidate.lower() in banned:
            return None

        if len(candidate.split()) > 8:
            return None

        return candidate

    def _resolve_location(
        self,
        location_text: str,
        location_kind: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        if location_kind == "coordinates":
            m = self._COORD_PATTERN.search(location_text)
            if not m:
                ambiguities.append(
                    {
                        "field": "location",
                        "reason": "invalid_coordinates",
                        "message": "Coordinate text found but could not be parsed.",
                        "raw": location_text,
                    }
                )
                return None, ambiguities

            lat = self._safe_float(m.group("lat"))
            lon = self._safe_float(m.group("lon"))
            if lat is None or lon is None:
                ambiguities.append(
                    {
                        "field": "location",
                        "reason": "invalid_coordinates",
                        "message": "Coordinate values are invalid.",
                        "raw": location_text,
                    }
                )
                return None, ambiguities

            return {
                "source": "query_coordinates",
                "raw_text": location_text,
                "name": None,
                "lat": lat,
                "lon": lon,
                "bbox": None,
                "resolution_kind": "exact_coordinates",
            }, ambiguities

        if location_kind != "place_name":
            return None, ambiguities

        if self.geocoder is None:
            ambiguities.append(
                {
                    "field": "location",
                    "reason": "geocoder_unavailable",
                    "message": "Location text was found but geocoder is not available.",
                    "raw": location_text,
                }
            )
            return None, ambiguities

        geocode_result = self._call_geocoder(location_text)

        if geocode_result is None:
            ambiguities.append(
                {
                    "field": "location",
                    "reason": "geocoding_failed",
                    "message": "Geocoder could not resolve the location text.",
                    "raw": location_text,
                }
            )
            return None, ambiguities

        if isinstance(geocode_result, list):
            normalized_candidates = [
                c for c in (self._normalize_geocode_candidate(x) for x in geocode_result)
                if c is not None
            ]

            if len(normalized_candidates) == 1:
                normalized = normalized_candidates[0]
                ambiguities.extend(self._post_validate_resolved_location(location_text, normalized))
                return normalized, ambiguities

            if len(normalized_candidates) > 1:
                ambiguities.append(
                    {
                        "field": "location",
                        "reason": "multiple_geocoding_candidates",
                        "message": "Geocoder returned multiple valid candidates.",
                        "raw": location_text,
                        "candidates": normalized_candidates,
                    }
                )
                return None, ambiguities

            ambiguities.append(
                {
                    "field": "location",
                    "reason": "geocoding_failed",
                    "message": "Geocoder returned candidates but none could be normalized.",
                    "raw": location_text,
                }
            )
            return None, ambiguities

        normalized = self._normalize_geocode_candidate(geocode_result)
        if normalized is None:
            ambiguities.append(
                {
                    "field": "location",
                    "reason": "geocoding_unrecognized_format",
                    "message": "Geocoder returned an unsupported format.",
                    "raw": location_text,
                }
            )
            return None, ambiguities

        ambiguities.extend(self._post_validate_resolved_location(location_text, normalized))
        return normalized, ambiguities

    def _post_validate_resolved_location(
        self,
        location_text: str,
        resolved: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        ambiguities: list[dict[str, Any]] = []

        if not resolved or not isinstance(resolved, dict):
            return ambiguities

        if self._is_region_like_location_text(location_text):
            if self._is_point_sized_bbox(resolved.get("bbox")):
                resolved["resolution_kind"] = "region_centroid_fallback"
                resolved["needs_region_bbox"] = True

                ambiguities.append(
                    {
                        "field": "location",
                        "reason": "region_resolved_as_point",
                        "message": (
                            "Location appears to describe a large region, but geocoder returned "
                            "a point-sized result or extremely small bbox."
                        ),
                        "raw": location_text,
                        "resolved_name": resolved.get("name"),
                        "bbox": resolved.get("bbox"),
                    }
                )
            else:
                resolved["resolution_kind"] = "region_bbox"

        return ambiguities

    def _is_region_like_location_text(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False

        t = text.lower().strip()

        region_markers = [
            "valley", "delta", "basin", "plateau", "region", "area", "highlands", "lowlands",
            "cao nguyên", "đồng bằng", "thung lũng", "vùng", "khu vực", "miền", "lưu vực",
            "tây nguyên",
        ]
        return any(marker in t for marker in region_markers)

    def _is_point_sized_bbox(self, bbox: Any, max_span_deg: float = 0.05) -> bool:
        if not isinstance(bbox, dict):
            return True

        min_lat = self._safe_float(bbox.get("min_lat"))
        max_lat = self._safe_float(bbox.get("max_lat"))
        min_lon = self._safe_float(bbox.get("min_lon"))
        max_lon = self._safe_float(bbox.get("max_lon"))

        if None in (min_lat, max_lat, min_lon, max_lon):
            return True

        lat_span = abs(max_lat - min_lat)
        lon_span = abs(max_lon - min_lon)

        return lat_span <= max_span_deg and lon_span <= max_span_deg

    def _call_geocoder(self, location_text: str) -> Any:
        method_names = ["resolve", "geocode", "resolve_location", "lookup"]
        for name in method_names:
            fn = getattr(self.geocoder, name, None)
            if callable(fn):
                try:
                    return fn(location_text)
                except TypeError:
                    try:
                        return fn(text=location_text)
                    except Exception:
                        continue
                except Exception as e:
                    logger.warning("QueryNormalizerAgent geocoder.%s failed: %s", name, e)
                    return None

        if callable(self.geocoder):
            try:
                return self.geocoder(location_text)
            except Exception as e:
                logger.warning("QueryNormalizerAgent geocoder callable failed: %s", e)
                return None

        return None

    def _normalize_geocode_candidate(self, candidate: Any) -> dict[str, Any] | None:
        if candidate is None:
            return None

        if isinstance(candidate, dict):
            lat = self._safe_float(
                candidate.get("lat")
                or candidate.get("latitude")
                or (candidate.get("coords") or {}).get("lat")
                or (candidate.get("center") or {}).get("lat")
            )
            lon = self._safe_float(
                candidate.get("lon")
                or candidate.get("lng")
                or candidate.get("longitude")
                or (candidate.get("coords") or {}).get("lon")
                or (candidate.get("coords") or {}).get("lng")
                or (candidate.get("center") or {}).get("lon")
                or (candidate.get("center") or {}).get("lng")
            )

            if lat is None or lon is None:
                return None

            bbox = self._extract_bbox(candidate)

            return {
                "source": "geocoder",
                "raw_text": candidate.get("raw_text"),
                "name": candidate.get("name") or candidate.get("display_name") or candidate.get("label"),
                "lat": lat,
                "lon": lon,
                "bbox": bbox,
                "country": candidate.get("country"),
                "admin1": candidate.get("admin1") or candidate.get("state") or candidate.get("province"),
                "admin2": candidate.get("admin2") or candidate.get("county") or candidate.get("district"),
                "resolution_kind": "point_or_bbox",
                "needs_region_bbox": False,
            }

        lat = self._safe_float(getattr(candidate, "lat", None) or getattr(candidate, "latitude", None))
        lon = self._safe_float(
            getattr(candidate, "lon", None)
            or getattr(candidate, "lng", None)
            or getattr(candidate, "longitude", None)
        )
        if lat is None or lon is None:
            return None

        bbox = self._extract_bbox(candidate)

        return {
            "source": "geocoder",
            "raw_text": None,
            "name": getattr(candidate, "name", None) or getattr(candidate, "display_name", None),
            "lat": lat,
            "lon": lon,
            "bbox": bbox,
            "country": getattr(candidate, "country", None),
            "admin1": getattr(candidate, "admin1", None) or getattr(candidate, "state", None),
            "admin2": getattr(candidate, "admin2", None) or getattr(candidate, "district", None),
            "resolution_kind": "point_or_bbox",
            "needs_region_bbox": False,
        }

    def _extract_bbox(self, candidate: Any) -> dict[str, float] | None:
        if isinstance(candidate, dict):
            bbox = candidate.get("bbox") or candidate.get("boundingbox") or candidate.get("bounds")
            return self._normalize_bbox(bbox)

        bbox = (
            getattr(candidate, "bbox", None)
            or getattr(candidate, "boundingbox", None)
            or getattr(candidate, "bounds", None)
        )
        return self._normalize_bbox(bbox)

    def _normalize_bbox(self, bbox: Any) -> dict[str, float] | None:
        if bbox is None:
            return None

        if isinstance(bbox, dict):
            min_lat = self._safe_float(bbox.get("min_lat") or bbox.get("south") or bbox.get("miny"))
            max_lat = self._safe_float(bbox.get("max_lat") or bbox.get("north") or bbox.get("maxy"))
            min_lon = self._safe_float(bbox.get("min_lon") or bbox.get("west") or bbox.get("minx"))
            max_lon = self._safe_float(bbox.get("max_lon") or bbox.get("east") or bbox.get("maxx"))
            if None not in (min_lat, max_lat, min_lon, max_lon):
                return {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon,
                }

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            values = [self._safe_float(x) for x in bbox[:4]]
            if all(v is not None for v in values):
                south, north, west, east = values
                return {
                    "min_lat": south,
                    "max_lat": north,
                    "min_lon": west,
                    "max_lon": east,
                }

        return None

    # ------------------------------------------------------------------
    # Time parsing + time_utils
    # ------------------------------------------------------------------

    def _extract_time_expression(self, query: str) -> tuple[str | None, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        if self.time_utils is not None:
            extracted = self._call_time_expression_extractor(query)
            if isinstance(extracted, str) and extracted.strip():
                return extracted.strip(), ambiguities

        range_patterns = [
            r"\btừ\s+(.+?)\s+đến\s+(.+?)(?=$|[.;?!])",
            r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=$|[.;?!])",
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=$|[.;?!])",
        ]

        range_candidates: list[str] = []
        for pattern in range_patterns:
            for m in re.finditer(pattern, query, flags=re.IGNORECASE):
                raw = m.group(0).strip()
                if raw:
                    range_candidates.append(raw)

        range_candidates = self._unique_keep_order(range_candidates)
        if len(range_candidates) == 1:
            return range_candidates[0], ambiguities
        if len(range_candidates) > 1:
            ambiguities.append(
                {
                    "field": "time",
                    "reason": "multiple_time_expressions",
                    "message": "Multiple time range candidates found in query.",
                    "candidates": range_candidates,
                }
            )
            return None, ambiguities

        candidates: list[str] = []

        relative_markers = [
            "hôm nay", "ngày mai", "hôm qua",
            "tuần này", "tuần tới", "tháng này", "tháng tới", "năm nay", "năm tới",
            "24 giờ tới", "48 giờ tới", "7 ngày tới",
            "today", "tomorrow", "yesterday",
            "this week", "next week", "this month", "next month", "this year", "next year",
            "next 24 hours", "next 48 hours", "next 7 days",
        ]
        lowered = query.lower()
        for marker in relative_markers:
            if marker in lowered:
                candidates.append(marker)

        candidates.extend(self._ISO_DATE_PATTERN.findall(query))
        for d, m, y in self._DMY_DATE_PATTERN.findall(query):
            candidates.append(f"{int(y):04d}-{int(m):02d}-{int(d):02d}")

        candidates = self._unique_keep_order(candidates)

        if len(candidates) == 1:
            return candidates[0], ambiguities
        if len(candidates) > 1:
            ambiguities.append(
                {
                    "field": "time",
                    "reason": "multiple_time_expressions",
                    "message": "Multiple time expressions found in query.",
                    "candidates": candidates,
                }
            )
            return None, ambiguities

        return None, ambiguities

    def _resolve_time_expression(
        self,
        time_expression: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        if not time_expression:
            return None, ambiguities

        if self.time_utils is not None:
            resolved = self._call_time_resolver(time_expression)
            normalized = self._normalize_time_range(resolved)
            if normalized is not None:
                return normalized, ambiguities

        parsed = self._parse_time_expression_locally(time_expression)
        if parsed is not None:
            return parsed, ambiguities

        ambiguities.append(
            {
                "field": "time",
                "reason": "time_resolution_failed",
                "message": "Time expression was detected but could not be resolved safely.",
                "raw": time_expression,
            }
        )
        return None, ambiguities

    def _call_time_expression_extractor(self, query: str) -> Any:
        method_names = ["extract_time_expression", "find_time_expression", "extract"]
        for name in method_names:
            fn = getattr(self.time_utils, name, None)
            if callable(fn):
                try:
                    return fn(query)
                except Exception as e:
                    logger.warning("QueryNormalizerAgent time_utils.%s failed: %s", name, e)
                    return None
        return None

    def _call_time_resolver(self, expression: str) -> Any:
        method_names = [
            "resolve_time_expression",
            "parse_time_expression",
            "to_time_range",
            "parse_query_time",
            "resolve",
        ]
        now = self.now_fn()

        for name in method_names:
            fn = getattr(self.time_utils, name, None)
            if not callable(fn):
                continue

            for call in (
                lambda: fn(expression, reference_time=now, timezone=self.timezone),
                lambda: fn(expression, now=now, timezone=self.timezone),
                lambda: fn(expression, now=now),
                lambda: fn(expression),
            ):
                try:
                    return call()
                except TypeError:
                    continue
                except Exception as e:
                    logger.warning("QueryNormalizerAgent time_utils.%s failed: %s", name, e)
                    return None

        return None

    def _normalize_time_range(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            start = value.get("start") or value.get("from")
            end = value.get("end") or value.get("to")
            if start is None or end is None:
                return None
            return {
                "start": self._to_iso_string(start),
                "end": self._to_iso_string(end),
                "timezone": value.get("timezone") or self.timezone,
                "source": value.get("source") or "time_utils",
            }

        if isinstance(value, (tuple, list)) and len(value) == 2:
            return {
                "start": self._to_iso_string(value[0]),
                "end": self._to_iso_string(value[1]),
                "timezone": self.timezone,
                "source": "time_utils",
            }

        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        if start is not None and end is not None:
            return {
                "start": self._to_iso_string(start),
                "end": self._to_iso_string(end),
                "timezone": getattr(value, "timezone", self.timezone),
                "source": "time_utils",
            }

        return None

    def _parse_time_expression_locally(self, expression: str) -> dict[str, Any] | None:
        expr = expression.strip().lower()
        now = self.now_fn()

        explicit_range = self._parse_explicit_range(expression)
        if explicit_range is not None:
            return {
                "start": self._to_iso_string(explicit_range[0]),
                "end": self._to_iso_string(explicit_range[1]),
                "timezone": self.timezone,
                "source": "local_parser",
            }

        single_date = self._parse_single_date_token(expression)
        if single_date is not None:
            start = datetime.combine(single_date.date(), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=1)
            return {
                "start": self._to_iso_string(start),
                "end": self._to_iso_string(end),
                "timezone": self.timezone,
                "source": "local_parser",
            }

        if expr in {"hôm nay", "today"}:
            start = datetime.combine(now.date(), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=1)
            return self._build_time_range(start, end)

        if expr in {"ngày mai", "tomorrow"}:
            start = datetime.combine(now.date() + timedelta(days=1), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=1)
            return self._build_time_range(start, end)

        if expr in {"hôm qua", "yesterday"}:
            start = datetime.combine(now.date() - timedelta(days=1), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=1)
            return self._build_time_range(start, end)

        if expr in {"tuần này", "this week"}:
            start = datetime.combine(now.date() - timedelta(days=now.weekday()), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=7)
            return self._build_time_range(start, end)

        if expr in {"tuần tới", "next week"}:
            this_monday = now.date() - timedelta(days=now.weekday())
            start = datetime.combine(this_monday + timedelta(days=7), time.min, tzinfo=now.tzinfo)
            end = start + timedelta(days=7)
            return self._build_time_range(start, end)

        if expr in {"tháng này", "this month"}:
            start = datetime(now.year, now.month, 1, tzinfo=now.tzinfo)
            if now.month == 12:
                end = datetime(now.year + 1, 1, 1, tzinfo=now.tzinfo)
            else:
                end = datetime(now.year, now.month + 1, 1, tzinfo=now.tzinfo)
            return self._build_time_range(start, end)

        if expr in {"tháng tới", "next month"}:
            if now.month == 12:
                start = datetime(now.year + 1, 1, 1, tzinfo=now.tzinfo)
                end = datetime(now.year + 1, 2, 1, tzinfo=now.tzinfo)
            elif now.month == 11:
                start = datetime(now.year, 12, 1, tzinfo=now.tzinfo)
                end = datetime(now.year + 1, 1, 1, tzinfo=now.tzinfo)
            else:
                start = datetime(now.year, now.month + 1, 1, tzinfo=now.tzinfo)
                end = datetime(now.year, now.month + 2, 1, tzinfo=now.tzinfo)
            return self._build_time_range(start, end)

        if expr in {"24 giờ tới", "next 24 hours"}:
            return self._build_time_range(now, now + timedelta(hours=24))

        if expr in {"48 giờ tới", "next 48 hours"}:
            return self._build_time_range(now, now + timedelta(hours=48))

        if expr in {"7 ngày tới", "next 7 days"}:
            return self._build_time_range(now, now + timedelta(days=7))

        return None

    def _parse_explicit_range(self, expression: str) -> tuple[datetime, datetime] | None:
        patterns = [
            r"\btừ\s+(.+?)\s+đến\s+(.+)$",
            r"\bfrom\s+(.+?)\s+to\s+(.+)$",
            r"\bbetween\s+(.+?)\s+and\s+(.+)$",
        ]
        for pattern in patterns:
            m = re.search(pattern, expression, flags=re.IGNORECASE)
            if not m:
                continue
            start_expr = m.group(1).strip()
            end_expr = m.group(2).strip()

            start_dt = self._parse_single_date_token(start_expr)
            end_dt = self._parse_single_date_token(end_expr)

            if start_dt is not None and end_dt is not None:
                if end_dt <= start_dt:
                    end_dt = end_dt + timedelta(days=1)
                return start_dt, end_dt
        return None

    def _parse_single_date_token(self, text: str) -> datetime | None:
        tz = self.now_fn().tzinfo

        iso_match = self._ISO_DATE_PATTERN.search(text)
        if iso_match:
            value = iso_match.group(1)
            try:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.replace(tzinfo=tz)
            except ValueError:
                return None

        dmy_match = self._DMY_DATE_PATTERN.search(text)
        if dmy_match:
            d, m, y = dmy_match.groups()
            try:
                return datetime(int(y), int(m), int(d), tzinfo=tz)
            except ValueError:
                return None

        return None

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_task_type(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        allowed = {"prediction", "monitoring", "analysis", "qa", "unknown"}
        return value if value in allowed else None

    def _normalize_prediction_target(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        aliases = {
            "wildfire_risk": "wildfire_risk",
            "fire_risk": "wildfire_risk",
            "risk": "wildfire_risk",
            "fire_spread": "fire_spread",
            "spread": "fire_spread",
            "risk_map": "risk_map",
            "map": "risk_map",
            "unknown": None,
        }
        return aliases.get(value, None)

    def _normalize_requested_outputs(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = [x for x in value if isinstance(x, str)]
        else:
            return []

        aliases = {
            "risk_level": "risk_level",
            "risk": "risk_level",
            "probability": "probability",
            "confidence": "confidence",
            "risk_map": "risk_map",
            "map": "risk_map",
            "spread_forecast": "spread_forecast",
            "fire_spread": "spread_forecast",
            "logit": "logit",
        }

        out: list[str] = []
        for item in items:
            key = item.strip().lower()
            normalized = aliases.get(key)
            if normalized:
                out.append(normalized)

        return self._unique_keep_order(out)

    def _normalize_llm_location_text(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def _normalize_optional_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    def _coalesce_valid(self, preferred: Any, fallback: Any) -> Any:
        return preferred if preferred is not None else fallback

    def _merge_unique(self, base: list[str], extra: list[str]) -> list[str]:
        return self._unique_keep_order((base or []) + (extra or []))

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _build_time_range(self, start: datetime, end: datetime) -> dict[str, Any]:
        return {
            "start": self._to_iso_string(start),
            "end": self._to_iso_string(end),
            "timezone": self.timezone,
            "source": "local_parser",
        }

    def _default_now(self) -> datetime:
        return datetime.now(ZoneInfo(self.timezone))

    def _to_iso_string(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _unique_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        out = []
        for item in items:
            if not isinstance(item, str):
                continue
            norm = item.strip()
            if not norm:
                continue
            key = norm.lower()
            if key not in seen:
                out.append(norm)
                seen.add(key)
        return out