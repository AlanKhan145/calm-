"""
GeocodingTool — chuyển địa chỉ văn bản thành tọa độ (lat, lon).

Dùng Nominatim (OpenStreetMap), không cần API key.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GeocodingTool:
    """Chỉ cần ghi địa chỉ, tool tự tìm tọa độ."""

    def __init__(
        self,
        safety_checker,
        config: dict | None = None,
    ) -> None:
        self.safety_checker = safety_checker
        self.config = config or {}
        self._cache: dict[str, dict[str, Any]] = {}

    def geocode(self, address: str) -> dict[str, Any]:
        """
        Chuyển địa chỉ (California, Amazon region, 123 Main St...) → {lat, lon}.

        Trả về:
          - lat, lon: float
          - display_name: tên đầy đủ từ Nominatim
          - error: str nếu lỗi
        """
        if not address or not str(address).strip():
            return {"lat": 0.0, "lon": 0.0, "error": "empty address"}

        addr = str(address).strip()
        if addr in self._cache:
            return self._cache[addr].copy()

        action = f"Geocoding: {addr}"
        self.safety_checker.check_or_raise(action)

        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError

            geolocator = Nominatim(user_agent="calm-wildfire-monitoring")
            loc = geolocator.geocode(addr, timeout=10)

            if loc is None:
                result = {"lat": 0.0, "lon": 0.0, "error": f"Không tìm thấy: {addr}"}
            else:
                result = {
                    "lat": loc.latitude,
                    "lon": loc.longitude,
                    "display_name": loc.address or addr,
                }
            self._cache[addr] = result.copy()
            return result

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning("Geocoding failed for %s: %s", addr, e)
            return {"lat": 0.0, "lon": 0.0, "error": str(e)}
        except ImportError:
            logger.warning("geopy not installed: pip install geopy")
            return {"lat": 0.0, "lon": 0.0, "error": "geopy not installed"}
        except Exception as e:
            logger.warning("Geocoding error: %s", e)
            return {"lat": 0.0, "lon": 0.0, "error": str(e)}
