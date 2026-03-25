"""
GeocodingTool — chuyển địa chỉ/vùng văn bản thành tọa độ + metadata không gian.

Thiết kế mới:
- dùng geopy + Nominatim
- hỗ trợ point query và area query
- lấy nhiều candidates rồi tự xếp hạng
- ưu tiên bbox thật cho region queries
- không dùng safety checker kiểu LLM để chặn public place names
- output phù hợp cho QueryNormalizerAgent / prediction pipeline
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GeocodingTool:
    """Geocoding tool cho point/area query, tối ưu cho downstream planning + prediction."""

    def __init__(
        self,
        safety_checker=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.safety_checker = safety_checker
        self.config = config or {}
        self.user_agent = self.config.get("user_agent", "calm-wildfire-monitoring")
        self.timeout = int(self.config.get("timeout", 10))
        self.language = self.config.get("language", "en")
        self.country_bias = self.config.get("country_bias")
        self.max_candidates = int(self.config.get("max_candidates", 5))
        self.enable_safety_for_geocoding = bool(
            self.config.get("enable_safety_for_geocoding", False)
        )
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def geocode(self, address: str) -> Dict[str, Any]:
        """
        Output chuẩn:

        {
            "query": "...",
            "normalized_query": "...",
            "name": "...",
            "display_name": "...",
            "query_type": "point_query|area_query|unknown",
            "geometry_type": "point|bbox|unknown",
            "lat": ...,
            "lon": ...,
            "latitude": ...,
            "longitude": ...,
            "bbox": {...} | None,
            "admin_level": "...",
            "country": "...",
            "state": "...",
            "region": "...",
            "county": "...",
            "city": "...",
            "district": "...",
            "confidence": 0.0-1.0,
            "resolution_kind": "exact|ranked_candidate|region_bbox|centroid_fallback|ambiguous|error",
            "needs_region_bbox": bool,
            "candidates": [...],
            "area_metadata": {...},
            "raw_type": "...",
            "raw_class": "...",
            "addresstype": "...",
            "error": None | "..."
        }
        """
        if not address or not str(address).strip():
            return self._error_result("", "empty address")

        query = str(address).strip()
        normalized_query = self._normalize_place_name(query)

        cached = self._cache_get(query, normalized_query)
        if cached is not None:
            return dict(cached)

        if self.enable_safety_for_geocoding and self.safety_checker is not None:
            if not self._is_safe_geocoding_text(query):
                result = self._error_result(query, "unsafe geocoding query")
                self._cache_store(query, normalized_query, result)
                return result

        try:
            from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(user_agent=self.user_agent)
            query_type_hint = self._infer_query_type_from_text(query)

            results = geolocator.geocode(
                query,
                exactly_one=False,
                timeout=self.timeout,
                addressdetails=True,
                namedetails=True,
                extratags=True,
                geometry="geojson",
                language=self.language,
                country_codes=self.country_bias,
            )

            if not results:
                result = self._error_result(query, f"Không tìm thấy: {query}")
                self._cache_store(query, normalized_query, result)
                return result

            normalized_candidates: List[Dict[str, Any]] = []
            for loc in results[: self.max_candidates]:
                raw = getattr(loc, "raw", {}) or {}
                cand = self._build_candidate(
                    query=query,
                    normalized_query=normalized_query,
                    loc=loc,
                    raw=raw,
                    query_type_hint=query_type_hint,
                )
                if cand is not None:
                    normalized_candidates.append(cand)

            if not normalized_candidates:
                result = self._error_result(query, f"Không chuẩn hóa được kết quả geocoding cho: {query}")
                self._cache_store(query, normalized_query, result)
                return result

            ranked = sorted(
                normalized_candidates,
                key=lambda x: x.get("confidence", 0.0),
                reverse=True,
            )

            best = ranked[0]
            best = self._post_process_best_candidate(
                query=query,
                query_type_hint=query_type_hint,
                candidate=best,
            )

            if len(ranked) > 1:
                gap = float(best.get("confidence", 0.0)) - float(ranked[1].get("confidence", 0.0))
                ambiguous = gap < 0.12
            else:
                ambiguous = False

            final = dict(best)
            final["candidates"] = [
                {
                    "name": c.get("name"),
                    "display_name": c.get("display_name"),
                    "lat": c.get("lat"),
                    "lon": c.get("lon"),
                    "bbox": c.get("bbox"),
                    "confidence": c.get("confidence"),
                    "admin_level": c.get("admin_level"),
                    "query_type": c.get("query_type"),
                    "raw_type": c.get("raw_type"),
                    "addresstype": c.get("addresstype"),
                }
                for c in ranked[: self.max_candidates]
            ]

            if ambiguous:
                final["resolution_kind"] = "ambiguous"
                final["ambiguity_note"] = "Multiple similarly plausible geocoding candidates found."

            self._cache_store(query, normalized_query, final)
            self._cache_store_aliases(final)
            return dict(final)

        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
            logger.warning("Geocoding failed for %s: %s", query, e)
            result = self._error_result(query, str(e))
            self._cache_store(query, normalized_query, result)
            return result
        except ImportError:
            logger.warning("geopy not installed: pip install geopy")
            result = self._error_result(query, "geopy not installed")
            self._cache_store(query, normalized_query, result)
            return result
        except Exception as e:
            logger.warning("Geocoding error for %s: %s", query, e)
            result = self._error_result(query, str(e))
            self._cache_store(query, normalized_query, result)
            return result

    def resolve(self, address: str) -> Dict[str, Any]:
        return self.geocode(address)

    def lookup(self, address: str) -> Dict[str, Any]:
        return self.geocode(address)

    def search(self, address: str) -> Dict[str, Any]:
        return self.geocode(address)

    # ─────────────────────────────────────────
    # Candidate builders
    # ─────────────────────────────────────────

    def _build_candidate(
        self,
        query: str,
        normalized_query: str,
        loc: Any,
        raw: Dict[str, Any],
        query_type_hint: str,
    ) -> Optional[Dict[str, Any]]:
        address = raw.get("address", {}) if isinstance(raw.get("address"), dict) else {}
        bbox = self._extract_bbox(raw)
        raw_type = str(raw.get("type") or "")
        raw_class = str(raw.get("class") or "")
        addresstype = str(raw.get("addresstype") or "")

        lat = self._to_float(getattr(loc, "latitude", None))
        lon = self._to_float(getattr(loc, "longitude", None))

        admin = self._extract_admin_metadata(address, raw, bbox)

        resolved_query_type = self._resolve_query_type(
            query=query,
            query_type_hint=query_type_hint,
            raw_type=raw_type,
            addresstype=addresstype,
            bbox=bbox,
        )

        geometry_type = "bbox" if self._is_area_like(
            query_type=resolved_query_type,
            bbox=bbox,
            raw_type=raw_type,
            addresstype=addresstype,
        ) else "point"

        confidence = self._estimate_confidence(
            query=query,
            display_name=str(getattr(loc, "address", "") or query),
            name=self._best_name(raw, loc, query),
            raw_type=raw_type,
            addresstype=addresstype,
            bbox=bbox,
            query_type=resolved_query_type,
            admin_level=admin.get("admin_level"),
        )

        needs_region_bbox = False
        resolution_kind = "ranked_candidate"

        if resolved_query_type == "area_query":
            if bbox is None:
                needs_region_bbox = True
                resolution_kind = "centroid_fallback"
            elif self._is_point_sized_bbox(bbox):
                needs_region_bbox = True
                resolution_kind = "centroid_fallback"
            else:
                resolution_kind = "region_bbox"

        return {
            "query": query,
            "normalized_query": normalized_query,
            "name": self._best_name(raw, loc, query),
            "display_name": str(getattr(loc, "address", "") or query),
            "query_type": resolved_query_type,
            "geometry_type": geometry_type,
            "lat": lat,
            "lon": lon,
            "latitude": lat,
            "longitude": lon,
            "bbox": bbox,
            "admin_level": admin.get("admin_level"),
            "country": admin.get("country"),
            "country_code": admin.get("country_code"),
            "state": admin.get("state"),
            "region": admin.get("region"),
            "county": admin.get("county"),
            "city": admin.get("city"),
            "district": admin.get("district"),
            "area_metadata": {
                "is_area": geometry_type == "bbox" or resolved_query_type == "area_query",
                "bbox": bbox,
                "admin_level": admin.get("admin_level"),
                "country": admin.get("country"),
                "state": admin.get("state"),
                "region": admin.get("region"),
                "county": admin.get("county"),
                "city": admin.get("city"),
                "raw_type": raw_type,
                "raw_class": raw_class,
                "addresstype": addresstype,
            },
            "confidence": confidence,
            "resolution_kind": resolution_kind,
            "needs_region_bbox": needs_region_bbox,
            "raw_type": raw_type,
            "raw_class": raw_class,
            "addresstype": addresstype,
            "error": None,
        }

    def _post_process_best_candidate(
        self,
        query: str,
        query_type_hint: str,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(candidate)

        # Nếu query giống vùng rộng mà bbox lại quá nhỏ, đánh dấu để downstream xử lý theo vùng.
        if query_type_hint == "area_query":
            if result.get("bbox") is None or self._is_point_sized_bbox(result.get("bbox")):
                result["needs_region_bbox"] = True
                result["resolution_kind"] = "centroid_fallback"
                result["geometry_type"] = "point"

        # Nếu bbox đủ lớn thì coi là area chuẩn.
        if result.get("bbox") and self._is_large_bbox(result["bbox"]):
            result["geometry_type"] = "bbox"
            result["resolution_kind"] = "region_bbox"
            result["needs_region_bbox"] = False

        return result

    # ─────────────────────────────────────────
    # Admin / query type inference
    # ─────────────────────────────────────────

    def _extract_admin_metadata(
        self,
        address: Dict[str, Any],
        raw: Dict[str, Any],
        bbox: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        country = address.get("country")
        state = address.get("state") or address.get("state_district") or address.get("province")
        region = (
            address.get("region")
            or address.get("state_district")
            or address.get("province")
            or address.get("territory")
        )
        county = address.get("county")
        city = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
        )
        district = address.get("district") or address.get("borough") or address.get("suburb")
        country_code = address.get("country_code")

        raw_type = str(raw.get("type") or "")
        addresstype = str(raw.get("addresstype") or "")

        admin_level = self._infer_admin_level(
            raw_type=raw_type,
            addresstype=addresstype,
            country=country,
            state=state,
            region=region,
            county=county,
            city=city,
            bbox=bbox,
        )

        return {
            "admin_level": admin_level,
            "country": country,
            "country_code": country_code,
            "state": state,
            "region": region,
            "county": county,
            "city": city,
            "district": district,
        }

    def _infer_query_type_from_text(self, query: str) -> str:
        q = self._normalize_place_name(query)

        area_markers = [
            "region", "area", "forest", "basin", "delta", "valley",
            "northern", "southern", "eastern", "western", "central",
            "province", "state", "county", "district", "territory",
            "national park", "amazon", "plateau", "highlands", "plain",
        ]
        point_markers = [
            "street", " st ", " avenue", " ave", "road", " rd", "building",
            "campus", "airport", "station", "peak", "volcano",
        ]

        if any(marker in q for marker in area_markers):
            return "area_query"
        if any(marker in q for marker in point_markers):
            return "point_query"
        if re.search(r"\d{1,5}\s+\w+", q):
            return "point_query"

        return "unknown"

    def _resolve_query_type(
        self,
        query: str,
        query_type_hint: str,
        raw_type: str,
        addresstype: str,
        bbox: Optional[Dict[str, float]],
    ) -> str:
        q = self._normalize_place_name(query)

        if query_type_hint == "area_query":
            return "area_query"

        area_types = {
            "administrative", "state", "region", "county", "forest",
            "nature_reserve", "national_park", "protected_area", "archipelago",
            "suburb", "valley", "basin", "delta",
        }
        area_addresstypes = {
            "state", "region", "county", "administrative",
            "suburb", "district", "city_district", "province",
        }

        if raw_type in area_types or addresstype in area_addresstypes:
            return "area_query"

        if self._is_large_bbox(bbox):
            return "area_query"

        if "valley" in q or "delta" in q or "basin" in q:
            return "area_query"

        return "point_query"

    def _infer_admin_level(
        self,
        raw_type: str,
        addresstype: str,
        country: Optional[str],
        state: Optional[str],
        region: Optional[str],
        county: Optional[str],
        city: Optional[str],
        bbox: Optional[Dict[str, float]],
    ) -> str:
        addresstype = str(addresstype or "").lower()
        raw_type = str(raw_type or "").lower()

        if addresstype == "country":
            return "country"
        if addresstype in {"state", "province"}:
            return "state"
        if addresstype in {"region", "state_district", "territory"}:
            return "region"
        if addresstype in {"county", "district"}:
            return "county"
        if addresstype in {"city", "town", "municipality", "village"}:
            return "city"

        if raw_type in {"forest", "nature_reserve", "national_park", "protected_area", "valley", "delta"}:
            return "region"

        if state and not county and not city:
            return "state"
        if region and not county and not city:
            return "region"
        if county and not city:
            return "county"
        if city:
            return "city"
        if country and self._is_large_bbox(bbox):
            return "region"

        return "unknown"

    # ─────────────────────────────────────────
    # Confidence / bbox helpers
    # ─────────────────────────────────────────

    def _extract_bbox(self, raw: Dict[str, Any]) -> Optional[Dict[str, float]]:
        bb = raw.get("boundingbox")
        if not isinstance(bb, list) or len(bb) != 4:
            return None

        try:
            south = float(bb[0])
            north = float(bb[1])
            west = float(bb[2])
            east = float(bb[3])
            return {
                "min_lat": south,
                "max_lat": north,
                "min_lon": west,
                "max_lon": east,
            }
        except Exception:
            return None

    def _estimate_confidence(
        self,
        query: str,
        display_name: str,
        name: str,
        raw_type: str,
        addresstype: str,
        bbox: Optional[Dict[str, float]],
        query_type: str,
        admin_level: Optional[str],
    ) -> float:
        score = 0.45

        q = self._normalize_place_name(query)
        d = self._normalize_place_name(display_name)
        n = self._normalize_place_name(name)

        if q and d:
            if q == d or q == n:
                score += 0.25
            elif q in d or q in n:
                score += 0.18

        if raw_type or addresstype:
            score += 0.08

        if admin_level in {"region", "state", "county"}:
            score += 0.06

        if query_type == "area_query" and bbox is not None:
            score += 0.12

        if query_type == "point_query" and not self._is_large_bbox(bbox):
            score += 0.06

        if self._is_large_bbox(bbox):
            score += 0.05

        if self._is_suspicious_match(q, d):
            score -= 0.20

        return round(max(0.0, min(1.0, score)), 4)

    def _is_large_bbox(self, bbox: Optional[Dict[str, float]]) -> bool:
        if not bbox:
            return False
        try:
            lat_span = abs(float(bbox["max_lat"]) - float(bbox["min_lat"]))
            lon_span = abs(float(bbox["max_lon"]) - float(bbox["min_lon"]))
            return lat_span >= 0.10 or lon_span >= 0.10
        except Exception:
            return False

    def _is_point_sized_bbox(self, bbox: Optional[Dict[str, float]]) -> bool:
        if not bbox:
            return False
        try:
            lat_span = abs(float(bbox["max_lat"]) - float(bbox["min_lat"]))
            lon_span = abs(float(bbox["max_lon"]) - float(bbox["min_lon"]))
            return lat_span <= 0.005 and lon_span <= 0.005
        except Exception:
            return False

    def _is_area_like(
        self,
        query_type: str,
        bbox: Optional[Dict[str, float]],
        raw_type: str,
        addresstype: str,
    ) -> bool:
        if query_type == "area_query":
            return True

        if self._is_large_bbox(bbox):
            return True

        area_types = {
            "administrative", "state", "region", "county", "forest",
            "nature_reserve", "national_park", "protected_area", "valley", "delta",
        }
        area_addresstypes = {
            "state", "region", "county", "district", "administrative", "province",
        }

        return raw_type in area_types or addresstype in area_addresstypes

    # ─────────────────────────────────────────
    # Cache helpers
    # ─────────────────────────────────────────

    def _cache_key(self, value: str) -> str:
        return self._normalize_place_name(value)

    def _cache_get(self, original_query: str, normalized_query: str) -> Optional[Dict[str, Any]]:
        for key in [self._cache_key(original_query), normalized_query]:
            if key and key in self._cache:
                return self._cache[key]
        return None

    def _cache_store(self, original_query: str, normalized_query: str, result: Dict[str, Any]) -> None:
        for key in [self._cache_key(original_query), normalized_query]:
            if key:
                self._cache[key] = dict(result)

    def _cache_store_aliases(self, result: Dict[str, Any]) -> None:
        aliases = [
            result.get("name"),
            result.get("display_name"),
            result.get("country"),
            result.get("state"),
            result.get("region"),
            result.get("county"),
            result.get("city"),
        ]
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                key = self._cache_key(alias)
                if key:
                    self._cache[key] = dict(result)

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    def _best_name(self, raw: Dict[str, Any], loc: Any, fallback: str) -> str:
        namedetails = raw.get("namedetails", {}) if isinstance(raw.get("namedetails"), dict) else {}
        for key in ["name", "official_name", "short_name"]:
            value = namedetails.get(key)
            if value:
                return str(value)

        address = raw.get("address", {}) if isinstance(raw.get("address"), dict) else {}
        for key in ["city", "county", "state", "region", "country"]:
            value = address.get(key)
            if value:
                return str(value)

        return str(getattr(loc, "address", "") or fallback)

    def _normalize_place_name(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[,_\-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_suspicious_match(self, normalized_query: str, normalized_display: str) -> bool:
        if not normalized_query or not normalized_display:
            return False

        q_tokens = set(normalized_query.split())
        d_tokens = set(normalized_display.split())
        if not q_tokens or not d_tokens:
            return False

        overlap = q_tokens.intersection(d_tokens)
        return len(overlap) == 0

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _is_safe_geocoding_text(self, text: str) -> bool:
        q = self._normalize_place_name(text)
        if not q:
            return False
        if len(q) > 160:
            return False

        blocked = [
            "password",
            "api key",
            "token",
            "private address",
            "home address",
            "exact current location",
        ]
        return not any(x in q for x in blocked)

    def _error_result(self, query: str, error: str) -> Dict[str, Any]:
        normalized_query = self._normalize_place_name(query)
        return {
            "query": query,
            "normalized_query": normalized_query,
            "name": query,
            "display_name": query,
            "query_type": "unknown",
            "geometry_type": "unknown",
            "lat": None,
            "lon": None,
            "latitude": None,
            "longitude": None,
            "bbox": None,
            "admin_level": None,
            "country": None,
            "country_code": None,
            "state": None,
            "region": None,
            "county": None,
            "city": None,
            "district": None,
            "area_metadata": {
                "is_area": False,
                "bbox": None,
                "admin_level": None,
                "country": None,
                "state": None,
                "region": None,
                "county": None,
                "city": None,
                "raw_type": None,
                "raw_class": None,
                "addresstype": None,
            },
            "confidence": 0.0,
            "resolution_kind": "error",
            "needs_region_bbox": False,
            "candidates": [],
            "raw_type": None,
            "raw_class": None,
            "addresstype": None,
            "error": error,
        }