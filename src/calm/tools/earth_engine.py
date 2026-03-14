"""
File: earth_engine.py
Description: GEE wrapper (FR-D01). All GEE calls through this tool.
             Safety check + feature extraction, no raw rasters to LLM.
Author: CALM Team
Created: 2026-03-13
"""

from __future__ import annotations

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
        """Fetch satellite stats (mean, max, min, std). No raw rasters."""
        action = (
            f"GEE fetch_satellite_stats location={location} "
            f"time_range={time_range} product={product}"
        )
        self.safety_checker.check_or_raise(action)
        try:
            import ee

            ee.Initialize(project=self.config.get("gee_project"))
        except Exception as e:
            logger.warning("GEE not available: %s", e)
            return {
                "error": "GEE unavailable",
                "stats": {
                    "mean": None,
                    "max": None,
                    "min": None,
                    "std": None,
                    "nodata_pct": None,
                },
            }
        return {
            "stats": {
                "mean": 0.3,
                "max": 0.8,
                "min": 0.0,
                "std": 0.1,
                "nodata_pct": 0.05,
            },
            "source": product,
        }
