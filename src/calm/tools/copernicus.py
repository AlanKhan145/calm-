"""
File: copernicus.py
Description: Copernicus CDS wrapper (FR-D02). ERA5 meteorological data.
             Safety check before every call.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class CopernicusTool:
    """
    Copernicus Climate Data Store (ERA5) tool.

    Bản này tách rõ:
    - fetch_era5_cds(): cố gắng lấy từ CDS thật
    - fetch_era5_fallback_openmeteo(): fallback minh bạch sang Open-Meteo
    - fetch_era5(): điều phối giữa 2 nhánh

    Output được chuẩn hóa để dùng cho:
    - prediction
    - RSEN weather analyst
    """

    DEFAULT_VARS = [
        "2m_temperature",
        "relative_humidity_2m",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "10m_wind_speed",
    ]

    CDS_DATASET = "reanalysis-era5-single-levels"

    # map logical vars -> CDS variable names
    CDS_VAR_MAP = {
        "2m_temperature": "2m_temperature",
        "relative_humidity_2m": "relative_humidity",
        "total_precipitation": "total_precipitation",
        "10m_u_component_of_wind": "10m_u_component_of_wind",
        "10m_v_component_of_wind": "10m_v_component_of_wind",
        "surface_pressure": "surface_pressure",
        "soil_temperature_level_1": "soil_temperature_level_1",
    }

    # map logical vars -> Open-Meteo daily fields
    OPENMETEO_DAILY_MAP = {
        "2m_temperature": "temperature_2m_mean",
        "relative_humidity_2m": "relative_humidity_2m_mean",
        "total_precipitation": "precipitation_sum",
        "10m_wind_speed": "wind_speed_10m_max",
    }

    def __init__(
        self,
        safety_checker,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.safety_checker = safety_checker
        self.config = config or {}
        self.timeout = int(self.config.get("timeout", 20))
        self.force_openmeteo = bool(self.config.get("force_openmeteo", False))
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def fetch_era5(
        self,
        lat: float,
        lon: float,
        time_range: Optional[Dict[str, Any]],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Unified API:
        - dùng CDS khi phù hợp và có cấu hình
        - fallback minh bạch sang Open-Meteo khi cần

        Không còn để tên là CDS nhưng dữ liệu thật từ nguồn khác mà không báo.
        """
        action = f"Copernicus fetch_era5 lat={lat} lon={lon} time_range={time_range}"
        self.safety_checker.check_or_raise(action)

        start, end, horizon_days, is_forecast_window = self._normalize_time_range(time_range)
        requested_vars = self._normalize_requested_vars(variables)

        cache_key = self._cache_key(
            lat=lat,
            lon=lon,
            start=start,
            end=end,
            requested_vars=requested_vars,
        )
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        cds_result = None
        fallback_reason = None

        if self.force_openmeteo:
            fallback_reason = "force_openmeteo enabled"
        elif is_forecast_window:
            fallback_reason = "forecast window requested; ERA5 CDS is not suitable for future forecast retrieval"
        else:
            cds_result = self.fetch_era5_cds(
                lat=lat,
                lon=lon,
                time_range={"start": start, "end": end},
                variables=requested_vars,
            )

            if not cds_result.get("error"):
                self._cache[cache_key] = dict(cds_result)
                return cds_result

            fallback_reason = cds_result.get("error") or "CDS retrieval failed"

        fallback_result = self.fetch_era5_fallback_openmeteo(
            lat=lat,
            lon=lon,
            time_range={"start": start, "end": end},
            variables=requested_vars,
            fallback_reason=fallback_reason,
        )
        self._cache[cache_key] = dict(fallback_result)
        return fallback_result

    def fetch_era5_cds(
        self,
        lat: float,
        lon: float,
        time_range: Dict[str, Any],
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Lấy ERA5 thật từ CDS nếu môi trường đã cấu hình được cdsapi.

        Lưu ý:
        - Hàm này dành cho historical/reanalysis windows.
        - Nếu CDS chưa cấu hình hoặc parse thất bại, trả error rõ ràng.
        """
        requested_vars = self._normalize_requested_vars(variables)
        start, end, horizon_days, is_forecast_window = self._normalize_time_range(time_range)

        if is_forecast_window:
            return self._build_error_result(
                provider="cds",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error="CDS ERA5 direct retrieval does not support forecast window in this tool",
                fallback_used=False,
            )

        cds_vars = [self.CDS_VAR_MAP[v] for v in requested_vars if v in self.CDS_VAR_MAP]
        if not cds_vars:
            return self._build_error_result(
                provider="cds",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error="No requested variables are supported by CDS mapping",
                fallback_used=False,
            )

        try:
            import cdsapi  # type: ignore
        except Exception:
            return self._build_error_result(
                provider="cds",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error="cdsapi not installed or unavailable",
                fallback_used=False,
            )

        try:
            client = cdsapi.Client(
                quiet=True,
                debug=False,
            )
        except Exception as e:
            return self._build_error_result(
                provider="cds",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error=f"CDS client initialization failed: {e}",
                fallback_used=False,
            )

        years, months, days = self._expand_dates_for_cds(start, end)
        hours = ["00:00"]

        request_payload = {
            "product_type": "reanalysis",
            "variable": cds_vars,
            "year": years,
            "month": months,
            "day": days,
            "time": hours,
            "format": "json",
            "area": [lat, lon, lat, lon],  # north, west, south, east
        }

        try:
            # Nhiều môi trường cdsapi không hỗ trợ JSON trực tiếp ổn định,
            # nên dùng temp file và thử parse nhiều kiểu.
            with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
                client.retrieve(self.CDS_DATASET, request_payload, tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    payload = json.load(f)

            timeseries = self._parse_cds_json_payload(
                payload=payload,
                requested_vars=requested_vars,
            )
            available_vars = self._extract_available_vars_from_timeseries(timeseries)

            result = self._build_success_result(
                provider="cds",
                fallback_used=False,
                requested_vars=requested_vars,
                available_vars=available_vars,
                time_range={"start": start, "end": end},
                timeseries=timeseries,
                raw_source_metadata={
                    "dataset": self.CDS_DATASET,
                    "cds_variables": cds_vars,
                    "request_payload": request_payload,
                },
            )
            return result

        except Exception as e:
            logger.warning("CDS retrieval failed: %s", e)
            return self._build_error_result(
                provider="cds",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error=f"CDS retrieval failed: {e}",
                fallback_used=False,
            )

    def fetch_era5_fallback_openmeteo(
        self,
        lat: float,
        lon: float,
        time_range: Dict[str, Any],
        variables: Optional[List[str]] = None,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fallback minh bạch sang Open-Meteo.

        Hỗ trợ:
        - past/archive windows
        - future/forecast windows
        - mixed windows: tự chia phần quá khứ và tương lai, rồi merge
        """
        requested_vars = self._normalize_requested_vars(variables)
        start, end, horizon_days, is_forecast_window = self._normalize_time_range(time_range)

        today = date.today()
        start_d = self._parse_date(start)
        end_d = self._parse_date(end)

        timeseries: List[Dict[str, Any]] = []
        provider_parts: List[str] = []

        try:
            if start_d <= today - timedelta(days=1):
                archive_end = min(end_d, today - timedelta(days=1))
                archive_rows = self._fetch_openmeteo_archive(
                    lat=lat,
                    lon=lon,
                    start=start_d.isoformat(),
                    end=archive_end.isoformat(),
                    requested_vars=requested_vars,
                )
                if archive_rows:
                    timeseries.extend(archive_rows)
                    provider_parts.append("Open-Meteo archive")

            if end_d >= today:
                forecast_start = max(start_d, today)
                forecast_rows = self._fetch_openmeteo_forecast(
                    lat=lat,
                    lon=lon,
                    start=forecast_start.isoformat(),
                    end=end_d.isoformat(),
                    requested_vars=requested_vars,
                )
                if forecast_rows:
                    timeseries.extend(forecast_rows)
                    provider_parts.append("Open-Meteo forecast")

            timeseries = self._dedup_timeseries_by_date(timeseries)
            available_vars = self._extract_available_vars_from_timeseries(timeseries)

            if not timeseries:
                raise RuntimeError("No meteorological rows returned by Open-Meteo")

            result = self._build_success_result(
                provider=" + ".join(provider_parts) if provider_parts else "Open-Meteo",
                fallback_used=True,
                requested_vars=requested_vars,
                available_vars=available_vars,
                time_range={"start": start, "end": end},
                timeseries=timeseries,
                raw_source_metadata={
                    "fallback_reason": fallback_reason,
                    "lat": lat,
                    "lon": lon,
                },
            )
            return result

        except Exception as e:
            logger.warning("Open-Meteo fallback failed: %s", e)
            return self._build_error_result(
                provider="open-meteo",
                requested_vars=requested_vars,
                time_range={"start": start, "end": end},
                error=f"Open-Meteo fallback failed: {e}",
                fallback_used=True,
                fallback_reason=fallback_reason,
            )

    # ─────────────────────────────────────────
    # Open-Meteo fetchers
    # ─────────────────────────────────────────

    def _fetch_openmeteo_archive(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
        requested_vars: List[str],
    ) -> List[Dict[str, Any]]:
        daily_fields = self._openmeteo_daily_fields(requested_vars)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(daily_fields),
            "timezone": "UTC",
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)

        with urlopen(url, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        return self._parse_openmeteo_payload(
            payload=payload,
            requested_vars=requested_vars,
            provider="Open-Meteo archive",
        )

    def _fetch_openmeteo_forecast(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
        requested_vars: List[str],
    ) -> List[Dict[str, Any]]:
        daily_fields = self._openmeteo_daily_fields(requested_vars)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": ",".join(daily_fields),
            "timezone": "UTC",
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)

        with urlopen(url, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        return self._parse_openmeteo_payload(
            payload=payload,
            requested_vars=requested_vars,
            provider="Open-Meteo forecast",
        )

    # ─────────────────────────────────────────
    # Payload parsers
    # ─────────────────────────────────────────

    def _parse_openmeteo_payload(
        self,
        payload: Dict[str, Any],
        requested_vars: List[str],
        provider: str,
    ) -> List[Dict[str, Any]]:
        daily = payload.get("daily") or {}
        dates = daily.get("time") or []

        rows: List[Dict[str, Any]] = []

        # Open-Meteo -> logical vars
        om_temp = daily.get("temperature_2m_mean") or []
        om_hum = daily.get("relative_humidity_2m_mean") or []
        om_prec = daily.get("precipitation_sum") or []
        om_wind = daily.get("wind_speed_10m_max") or []

        for idx, day in enumerate(dates):
            row = {
                "date": day,
                "provider": provider,
                "temperature": self._safe_get_float(om_temp, idx),
                "humidity": self._percent_to_ratio(self._safe_get_float(om_hum, idx)),
                "wind_speed": self._safe_get_float(om_wind, idx),
                "precipitation": self._safe_get_float(om_prec, idx),
                "u_wind": None,
                "v_wind": None,
            }
            rows.append(row)

        return rows

    def _parse_cds_json_payload(
        self,
        payload: Dict[str, Any],
        requested_vars: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Parser best-effort cho JSON trả về từ CDS.

        CDS JSON có thể khác nhau theo môi trường, nên parser này cố gắng đọc
        một số schema phổ biến.
        """
        rows: List[Dict[str, Any]] = []

        # Schema kiểu [{"date": ..., "2m_temperature": ...}, ...]
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rows.append(self._normalize_cds_row(item))
            return rows

        # Schema kiểu {"data":[...]}
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    rows.append(self._normalize_cds_row(item))
            return rows

        # Schema kiểu dims/coords/data_vars xarray-json-like
        data_vars = payload.get("data_vars")
        coords = payload.get("coords")
        if isinstance(data_vars, dict) and isinstance(coords, dict):
            dates = []
            time_coord = coords.get("time") or coords.get("valid_time") or {}
            if isinstance(time_coord, dict):
                dates = time_coord.get("data") or []
            elif isinstance(time_coord, list):
                dates = time_coord

            for idx, dt_value in enumerate(dates):
                row = {
                    "date": str(dt_value)[:10],
                    "provider": "CDS ERA5",
                    "temperature": None,
                    "humidity": None,
                    "wind_speed": None,
                    "precipitation": None,
                    "u_wind": None,
                    "v_wind": None,
                }

                for logical_var, cds_var in self.CDS_VAR_MAP.items():
                    series = data_vars.get(cds_var, {})
                    values = []
                    if isinstance(series, dict):
                        values = series.get("data") or []
                    elif isinstance(series, list):
                        values = series

                    value = self._safe_get_float(values, idx)

                    if logical_var == "2m_temperature":
                        row["temperature"] = value - 273.15 if value is not None else None
                    elif logical_var == "relative_humidity_2m":
                        row["humidity"] = self._percent_to_ratio(value)
                    elif logical_var == "total_precipitation":
                        row["precipitation"] = value
                    elif logical_var == "10m_u_component_of_wind":
                        row["u_wind"] = value
                    elif logical_var == "10m_v_component_of_wind":
                        row["v_wind"] = value

                row["wind_speed"] = self._wind_speed_from_uv(row.get("u_wind"), row.get("v_wind"))
                rows.append(row)

            return rows

        return rows

    def _normalize_cds_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize row CDS về schema chung.
        """
        date_value = (
            item.get("date")
            or item.get("time")
            or item.get("valid_time")
            or item.get("datetime")
            or ""
        )
        row = {
            "date": str(date_value)[:10],
            "provider": "CDS ERA5",
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "precipitation": None,
            "u_wind": None,
            "v_wind": None,
        }

        # possible keys
        temp = (
            item.get("2m_temperature")
            or item.get("temperature")
            or item.get("t2m")
        )
        humidity = (
            item.get("relative_humidity")
            or item.get("relative_humidity_2m")
            or item.get("humidity")
        )
        precip = (
            item.get("total_precipitation")
            or item.get("precipitation")
            or item.get("tp")
        )
        u_wind = item.get("10m_u_component_of_wind") or item.get("u10")
        v_wind = item.get("10m_v_component_of_wind") or item.get("v10")
        wind_speed = item.get("wind_speed") or item.get("10m_wind_speed")

        temp_v = self._to_float(temp)
        row["temperature"] = temp_v - 273.15 if temp_v is not None and temp_v > 150 else temp_v
        row["humidity"] = self._percent_to_ratio(self._to_float(humidity))
        row["precipitation"] = self._to_float(precip)
        row["u_wind"] = self._to_float(u_wind)
        row["v_wind"] = self._to_float(v_wind)
        row["wind_speed"] = self._to_float(wind_speed) or self._wind_speed_from_uv(
            row["u_wind"], row["v_wind"]
        )

        return row

    # ─────────────────────────────────────────
    # Result builders
    # ─────────────────────────────────────────

    def _build_success_result(
        self,
        provider: str,
        fallback_used: bool,
        requested_vars: List[str],
        available_vars: List[str],
        time_range: Dict[str, str],
        timeseries: List[Dict[str, Any]],
        raw_source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = self._build_summary(timeseries)
        start = time_range["start"]
        end = time_range["end"]
        horizon_days = (self._parse_date(end) - self._parse_date(start)).days + 1
        is_forecast_window = self._parse_date(start) >= date.today()

        return {
            "summary": summary,
            "timeseries": timeseries,
            "provider": provider,
            "fallback_used": fallback_used,
            "requested_vars": requested_vars,
            "available_vars": available_vars,
            "time_range": {"start": start, "end": end},
            "granularity": "day",
            "horizon_days": horizon_days,
            "is_forecast_window": is_forecast_window,
            "source_metadata": raw_source_metadata or {},
            "error": None,
        }

    def _build_error_result(
        self,
        provider: str,
        requested_vars: List[str],
        time_range: Dict[str, str],
        error: str,
        fallback_used: bool,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time_range.get("start", "")
        end = time_range.get("end", "")
        horizon_days = 0
        is_forecast_window = False

        try:
            if start and end:
                horizon_days = (self._parse_date(end) - self._parse_date(start)).days + 1
                is_forecast_window = self._parse_date(start) >= date.today()
        except Exception:
            pass

        return {
            "summary": {},
            "timeseries": [],
            "provider": provider,
            "fallback_used": fallback_used,
            "requested_vars": requested_vars,
            "available_vars": [],
            "time_range": {"start": start, "end": end},
            "granularity": "day",
            "horizon_days": horizon_days,
            "is_forecast_window": is_forecast_window,
            "source_metadata": {
                "fallback_reason": fallback_reason,
            },
            "error": error,
        }

    def _build_summary(self, timeseries: List[Dict[str, Any]]) -> Dict[str, Any]:
        temps = [self._to_float(row.get("temperature")) for row in timeseries]
        hums = [self._to_float(row.get("humidity")) for row in timeseries]
        winds = [self._to_float(row.get("wind_speed")) for row in timeseries]
        precs = [self._to_float(row.get("precipitation")) for row in timeseries]

        return {
            "temperature": self._avg(temps),
            "humidity": self._avg(hums),
            "wind_speed": self._avg(winds),
            "precipitation": self._avg(precs),
        }

    # ─────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────

    def _normalize_time_range(
        self,
        time_range: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, int, bool]:
        start = str((time_range or {}).get("start", "") or (time_range or {}).get("start_date", "")).strip()
        end = str((time_range or {}).get("end", "") or (time_range or {}).get("end_date", "")).strip()

        today = date.today().isoformat()
        if not start and not end:
            start = today
            end = today
        elif not start:
            start = end
        elif not end:
            end = start

        start_d = self._parse_date(start)
        end_d = self._parse_date(end)
        if end_d < start_d:
            start_d, end_d = end_d, start_d
            start, end = start_d.isoformat(), end_d.isoformat()

        horizon_days = (end_d - start_d).days + 1
        is_forecast_window = end_d >= date.today()

        return start, end, horizon_days, is_forecast_window

    def _normalize_requested_vars(self, variables: Optional[List[str]]) -> List[str]:
        requested = variables or list(self.DEFAULT_VARS)
        out: List[str] = []

        for var in requested:
            if not var:
                continue
            text = str(var).strip()

            # normalize aliases
            if text in {"temperature", "2m_temperature", "t2m"}:
                norm = "2m_temperature"
            elif text in {"humidity", "relative_humidity", "relative_humidity_2m"}:
                norm = "relative_humidity_2m"
            elif text in {"precipitation", "total_precipitation", "tp"}:
                norm = "total_precipitation"
            elif text in {"u_wind", "u10", "10m_u_component_of_wind"}:
                norm = "10m_u_component_of_wind"
            elif text in {"v_wind", "v10", "10m_v_component_of_wind"}:
                norm = "10m_v_component_of_wind"
            elif text in {"wind_speed", "10m_wind_speed"}:
                norm = "10m_wind_speed"
            else:
                norm = text

            if norm not in out:
                out.append(norm)

        return out

    def _openmeteo_daily_fields(self, requested_vars: List[str]) -> List[str]:
        fields: List[str] = []

        # Always request the minimal set needed to build normalized rows
        for logical_var in requested_vars:
            mapped = self.OPENMETEO_DAILY_MAP.get(logical_var)
            if mapped and mapped not in fields:
                fields.append(mapped)

        # Wind speed often useful downstream
        if "wind_speed_10m_max" not in fields:
            fields.append("wind_speed_10m_max")

        return fields

    def _extract_available_vars_from_timeseries(self, timeseries: List[Dict[str, Any]]) -> List[str]:
        available = set()
        for row in timeseries:
            if row.get("temperature") is not None:
                available.add("2m_temperature")
            if row.get("humidity") is not None:
                available.add("relative_humidity_2m")
            if row.get("precipitation") is not None:
                available.add("total_precipitation")
            if row.get("wind_speed") is not None:
                available.add("10m_wind_speed")
            if row.get("u_wind") is not None:
                available.add("10m_u_component_of_wind")
            if row.get("v_wind") is not None:
                available.add("10m_v_component_of_wind")
        return sorted(available)

    def _dedup_timeseries_by_date(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {}
        for row in rows:
            d = row.get("date")
            if d:
                seen[d] = row
        return [seen[k] for k in sorted(seen.keys())]

    def _expand_dates_for_cds(self, start: str, end: str) -> Tuple[List[str], List[str], List[str]]:
        start_d = self._parse_date(start)
        end_d = self._parse_date(end)

        years = sorted({f"{d.year:04d}" for d in self._daterange(start_d, end_d)})
        months = sorted({f"{d.month:02d}" for d in self._daterange(start_d, end_d)})
        days = sorted({f"{d.day:02d}" for d in self._daterange(start_d, end_d)})
        return years, months, days

    def _daterange(self, start_d: date, end_d: date):
        d = start_d
        while d <= end_d:
            yield d
            d += timedelta(days=1)

    def _cache_key(
        self,
        lat: float,
        lon: float,
        start: str,
        end: str,
        requested_vars: List[str],
    ) -> str:
        return json.dumps(
            {
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
                "start": start,
                "end": end,
                "requested_vars": requested_vars,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def _safe_get_float(self, seq: List[Any], idx: int) -> Optional[float]:
        try:
            value = seq[idx]
            return self._to_float(value)
        except Exception:
            return None

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _avg(self, values: List[Optional[float]]) -> Optional[float]:
        numeric = [float(v) for v in values if v is not None]
        if not numeric:
            return None
        return sum(numeric) / len(numeric)

    def _percent_to_ratio(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value > 1.0:
            return value / 100.0
        return value

    def _wind_speed_from_uv(self, u: Optional[float], v: Optional[float]) -> Optional[float]:
        if u is None or v is None:
            return None
        return (u ** 2 + v ** 2) ** 0.5

    def _parse_date(self, value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()