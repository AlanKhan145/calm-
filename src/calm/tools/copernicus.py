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
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import Any

logger = logging.getLogger(__name__)


class CopernicusTool:
    """Copernicus Climate Data Store (ERA5) tool."""

    def __init__(
        self,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        """Initialize with safety checker and config."""
        self.safety_checker = safety_checker
        self.config = config or {}

    def fetch_era5(
        self,
        lat: float,
        lon: float,
        time_range: dict,
        variables: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch ERA5 met data, with Open-Meteo fallback when CDS is unavailable."""
        action = f"CDS fetch_era5 lat={lat} lon={lon} time_range={time_range}"
        self.safety_checker.check_or_raise(action)
        variables = variables or [
            "2m_temperature",
            "total_precipitation",
            "10m_u_component_of_wind",
        ]
        start = str((time_range or {}).get("start", "")).strip()
        end = str((time_range or {}).get("end", "")).strip()
        if not start and not end:
            today = date.today().isoformat()
            start = today
            end = today
        if not start:
            start = end
        if not end:
            end = start
        try:
            import cdsapi

            cdsapi.Client()
            # Keep this interface side-effect free for lightweight usage.
            # Actual CDS batch retrieval can be wired here if credentials are configured.
            raise RuntimeError("CDS direct retrieval not configured in lightweight mode")
        except Exception:
            # Fallback to Open-Meteo archive/forecast APIs for practical online met context.
            try:
                q = urlencode(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start,
                        "end_date": end,
                        "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_max",
                        "timezone": "UTC",
                    }
                )
                url = f"https://archive-api.open-meteo.com/v1/archive?{q}"
                with urlopen(url, timeout=20) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                daily = payload.get("daily") or {}
                temps = daily.get("temperature_2m_mean") or []
                hums = daily.get("relative_humidity_2m_mean") or []
                precs = daily.get("precipitation_sum") or []
                winds = daily.get("wind_speed_10m_max") or []

                def _avg(vals: list[Any]) -> float | None:
                    numeric = [float(v) for v in vals if v is not None]
                    return sum(numeric) / len(numeric) if numeric else None

                return {
                    "summary": {
                        "temperature": _avg(temps),
                        "humidity": (_avg(hums) / 100.0) if _avg(hums) is not None else None,
                        "wind_speed": _avg(winds),
                        "precipitation": _avg(precs),
                    },
                    "timeseries": {"daily": daily},
                    "source": "Open-Meteo (CDS fallback)",
                    "time_range": {"start": start, "end": end},
                    "variables": variables,
                }
            except Exception as e:
                logger.warning("CDS/Open-Meteo met retrieval failed: %s", e)
                return {"error": "CDS unavailable", "summary": {}}
        return {
            "error": "CDS client ready but retrieval not configured",
            "summary": {},
            "variables": variables,
            "time_range": {"start": start, "end": end},
        }
