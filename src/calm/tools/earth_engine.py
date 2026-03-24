"""
File: earth_engine.py
Description: GEE wrapper (FR-D01). All GEE calls through this tool.
             Safety check + feature extraction, no raw rasters to LLM.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EarthEngineTool:
    """Google Earth Engine tool. Safety check + caching."""

    def __init__(
        self,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        """Initialize with safety checker and config."""
        self.safety_checker = safety_checker
        self.config = config or {}

    def fetch_satellite_stats(
        self,
        location: str | dict,
        time_range: dict,
        product: str = "LANDSAT/LC08/C02/T1_L2",
    ) -> dict[str, Any]:
        """Fetch satellite stats from GEE. Returns summary only, no raw rasters."""
        action = (
            f"GEE fetch_satellite_stats location={location} "
            f"time_range={time_range} product={product}"
        )
        self.safety_checker.check_or_raise(action)
        point = self._to_ee_point(location)
        if point is None:
            return {"error": "Invalid location for GEE", "stats": {}}
        start, end = self._normalize_time_range(time_range)
        if not start or not end:
            return {"error": "Invalid time_range for GEE", "stats": {}}
        try:
            import ee

            ee.Initialize(project=self.config.get("gee_project") or None)
            landsat = (
                ee.ImageCollection(product)
                .filterDate(start, end)
                .filterBounds(point)
                .sort("CLOUD_COVER")
            )
            first_img = landsat.first()
            if first_img is None:
                return {"error": "No GEE image found for region/time", "stats": {}}

            # Landsat L2 surface reflectance scale factor for NDVI proxy.
            red = first_img.select("SR_B4").multiply(0.0000275).add(-0.2)
            nir = first_img.select("SR_B5").multiply(0.0000275).add(-0.2)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
            reducer = ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True).combine(
                ee.Reducer.stdDev(), sharedInputs=True
            )
            region_stats = ndvi.reduceRegion(
                reducer=reducer,
                geometry=point.buffer(5000),
                scale=30,
                maxPixels=1_000_000,
            )
            stats = region_stats.getInfo() or {}
            image_id = first_img.get("system:index").getInfo()
        except Exception as e:
            logger.warning("GEE not available: %s", e)
            return {
                "error": "GEE unavailable",
                "stats": {},
            }
        return {
            "stats": {
                "ndvi_mean": stats.get("ndvi_mean"),
                "ndvi_min": stats.get("ndvi_min"),
                "ndvi_max": stats.get("ndvi_max"),
                "ndvi_stdDev": stats.get("ndvi_stdDev"),
            },
            "image_id": image_id,
            "time_range": {"start": start, "end": end},
            "source": product,
        }

    @staticmethod
    def _to_ee_point(location: str | dict) -> Any:
        try:
            import ee
        except Exception:
            return None
        if isinstance(location, dict):
            lat = location.get("lat")
            lon = location.get("lon")
            if lat is None or lon is None:
                return None
            try:
                return ee.Geometry.Point([float(lon), float(lat)])
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_time_range(time_range: dict) -> tuple[str, str]:
        if not isinstance(time_range, dict):
            return "", ""
        start = str(time_range.get("start", "")).strip()
        end = str(time_range.get("end", "")).strip()
        if not start and not end:
            today = dt.date.today().isoformat()
            return today, today
        if not start:
            start = end
        if not end:
            end = start
        return start, end
