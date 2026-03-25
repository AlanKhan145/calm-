"""
QueryNormalizerAgent — chuẩn hóa truy vấn thô thành context JSON chuẩn cho planner.

Agent này phải chạy TRƯỚC planner.

Input:
- query text thô từ user

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
- Không hallucinate location/time.
- Nếu mơ hồ hoặc resolve thất bại thì ghi vào ambiguities.
- Có gọi geocoding + time_utils nếu được inject vào.
"""

from __future__ import annotations

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

    Lưu ý:
    - Nếu geocoder/time_utils không có hoặc không resolve được,
      agent KHÔNG tự bịa ra location/time_range.
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
        timezone: str = "Asia/Bangkok",
        now_fn=None,
    ) -> None:
        self.geocoder = geocoder
        self.time_utils = time_utils
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

        task_type = self._infer_task_type(query)
        prediction_target = self._infer_prediction_target(query)
        requested_output = self._infer_requested_output(query)

        location_text, location_kind, location_ambiguities = self._extract_location(query)
        ambiguities.extend(location_ambiguities)

        location_resolved = None
        if location_text is not None:
            location_resolved, resolve_location_ambiguities = self._resolve_location(
                location_text=location_text,
                location_kind=location_kind,
            )
            ambiguities.extend(resolve_location_ambiguities)

        time_expression, time_expr_ambiguities = self._extract_time_expression(query)
        ambiguities.extend(time_expr_ambiguities)

        time_range = None
        if time_expression is not None:
            time_range, time_range_ambiguities = self._resolve_time_expression(time_expression)
            ambiguities.extend(time_range_ambiguities)

        # Không hallucinate target nếu query chưa rõ là prediction gì
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
    # Core inference
    # ------------------------------------------------------------------

    def _infer_task_type(self, query: str) -> str:
        q = query.lower()

        prediction_keywords = [
            "predict", "prediction", "forecast", "dự đoán", "dự báo", "ước tính",
            "risk", "rủi ro", "nguy cơ", "spread", "lan rộng",
        ]
        monitoring_keywords = [
            "monitor", "monitoring", "alert", "cảnh báo", "theo dõi",
        ]
        qa_keywords = [
            "what", "why", "how", "giải thích", "là gì", "tại sao", "như thế nào",
        ]
        analysis_keywords = [
            "analyze", "analysis", "phân tích", "compare", "so sánh", "đánh giá",
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
        if any(k in q for k in ["risk map", "bản đồ rủi ro", "heatmap", "map"]):
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
            ("risk_level", ["risk level", "mức rủi ro", "risk", "rủi ro"]),
            ("confidence", ["confidence", "độ tin cậy"]),
            ("risk_map", ["risk map", "bản đồ rủi ro", "heatmap", "map"]),
            ("spread_forecast", ["spread forecast", "fire spread", "lan rộng", "cháy lan"]),
        ]

        for output_name, keywords in rules:
            if any(k in q for k in keywords):
                outputs.append(output_name)

        # tránh duplicate, giữ thứ tự
        seen = set()
        unique_outputs = []
        for item in outputs:
            if item not in seen:
                unique_outputs.append(item)
                seen.add(item)
        return unique_outputs

    # ------------------------------------------------------------------
    # Location parsing + geocoding
    # ------------------------------------------------------------------

    def _extract_location(self, query: str) -> tuple[str | None, str, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        # 1) Ưu tiên coordinates rõ ràng
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

        # 2) Thử bắt location text bằng pattern bảo thủ
        candidate_patterns = [
            r"\b(?:ở|tại|khu vực|vùng|quanh|gần)\s+(.+?)(?=$|[.;?!])",
            r"\b(?:in|at|around|near|for)\s+(.+?)(?=$|[.;?!])",
        ]

        candidates: list[str] = []
        for pattern in candidate_patterns:
            for m in re.finditer(pattern, query, flags=re.IGNORECASE):
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

        return None, "unknown", ambiguities

    def _trim_location_candidate(self, text: str) -> str | None:
        if not text:
            return None

        candidate = text.strip(" \t\n\"'“”‘’,:-")

        # cắt đuôi nếu location bị dính time phrase
        stop_markers = [
            " hôm nay", " ngày mai", " hôm qua",
            " tuần này", " tuần tới", " tháng này", " tháng tới",
            " this week", " next week", " today", " tomorrow", " yesterday",
            " từ ", " đến ", " between ", " from ", " vào ", " lúc ", " during ",
        ]
        lowered = candidate.lower()
        cut_index = None
        for marker in stop_markers:
            idx = lowered.find(marker)
            if idx != -1:
                cut_index = idx if cut_index is None else min(cut_index, idx)

        if cut_index is not None:
            candidate = candidate[:cut_index].strip(" \t\n\"'“”‘’,:-")

        # loại bỏ chuỗi quá chung chung
        if not candidate:
            return None
        if candidate.lower() in {"đó", "đây", "there", "here", "khu vực đó"}:
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
                c for c in (self._normalize_geocode_candidate(x) for x in geocode_result) if c is not None
            ]
            if len(normalized_candidates) == 1:
                return normalized_candidates[0], ambiguities
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

        return normalized, ambiguities

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
            )
            lon = self._safe_float(
                candidate.get("lon")
                or candidate.get("lng")
                or candidate.get("longitude")
                or (candidate.get("coords") or {}).get("lon")
                or (candidate.get("coords") or {}).get("lng")
            )

            if lat is None or lon is None:
                return None

            return {
                "source": "geocoder",
                "raw_text": candidate.get("raw_text"),
                "name": candidate.get("name") or candidate.get("display_name") or candidate.get("label"),
                "lat": lat,
                "lon": lon,
                "country": candidate.get("country"),
                "admin1": candidate.get("admin1") or candidate.get("state") or candidate.get("province"),
                "admin2": candidate.get("admin2") or candidate.get("county") or candidate.get("district"),
            }

        # object style
        lat = self._safe_float(getattr(candidate, "lat", None) or getattr(candidate, "latitude", None))
        lon = self._safe_float(
            getattr(candidate, "lon", None)
            or getattr(candidate, "lng", None)
            or getattr(candidate, "longitude", None)
        )
        if lat is None or lon is None:
            return None

        return {
            "source": "geocoder",
            "raw_text": None,
            "name": getattr(candidate, "name", None) or getattr(candidate, "display_name", None),
            "lat": lat,
            "lon": lon,
            "country": getattr(candidate, "country", None),
            "admin1": getattr(candidate, "admin1", None) or getattr(candidate, "state", None),
            "admin2": getattr(candidate, "admin2", None) or getattr(candidate, "district", None),
        }

    # ------------------------------------------------------------------
    # Time parsing + time_utils
    # ------------------------------------------------------------------

    def _extract_time_expression(self, query: str) -> tuple[str | None, list[dict[str, Any]]]:
        ambiguities: list[dict[str, Any]] = []

        # Ưu tiên gọi time_utils.extract_* nếu có
        if self.time_utils is not None:
            extracted = self._call_time_expression_extractor(query)
            if isinstance(extracted, str) and extracted.strip():
                return extracted.strip(), ambiguities

        # 1) range tường minh: từ ... đến ... / from ... to ... / between ... and ...
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

        # 2) relative / explicit đơn
        candidates: list[str] = []

        relative_markers = [
            "hôm nay", "ngày mai", "hôm qua",
            "tuần này", "tuần tới", "tháng này", "tháng tới",
            "24 giờ tới", "48 giờ tới", "7 ngày tới",
            "today", "tomorrow", "yesterday",
            "this week", "next week", "this month", "next month",
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

        # Ưu tiên gọi time_utils
        if self.time_utils is not None:
            resolved = self._call_time_resolver(time_expression)
            normalized = self._normalize_time_range(resolved)
            if normalized is not None:
                return normalized, ambiguities

        # fallback nội bộ: chỉ parse những gì chắc chắn
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

            # thử một vài signature phổ biến
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

        # explicit range
        explicit_range = self._parse_explicit_range(expression)
        if explicit_range is not None:
            return {
                "start": self._to_iso_string(explicit_range[0]),
                "end": self._to_iso_string(explicit_range[1]),
                "timezone": self.timezone,
                "source": "local_parser",
            }

        # single explicit date
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

        # relative
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
            norm = item.strip()
            if not norm:
                continue
            key = norm.lower()
            if key not in seen:
                out.append(norm)
                seen.add(key)
        return out