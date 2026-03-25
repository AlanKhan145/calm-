"""
Offline SeasFire feature builder.

IMPORTANT:
- Đây là OFFLINE FALLBACK dùng local SeasFire dataset để tìm sample gần nhất
  rồi build tensor cho mô hình.
- Đây KHÔNG PHẢI online production path.
- Online feature assembly (từ GEE / Copernicus / retrieval pipeline) phải được xử lý
  ở module khác, không đi qua file này như đường chính.

Vai trò file này:
1) Khi có local SeasFire dataset:
   - đọc dataset
   - tìm sample gần nhất theo không gian (và nếu có thể thì có cân nhắc thời gian)
   - trả tensor để model runner dùng ở chế độ offline / fallback
2) Khi không có dataset local:
   - trả None
   - caller phải quyết định fallback heuristic hoặc re-retrieval

File này không tự đi lấy dữ liệu từ GEE/CDS, và không nên được dùng để đại diện
cho online prediction pipeline.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_SEASFIRE_AVAILABLE = False
GRUDataset = None
GRUTransform = None

try:
    seasfire_path = os.environ.get("SEASFIRE_ML_PATH")
    if seasfire_path and Path(seasfire_path).exists():
        if seasfire_path not in sys.path:
            sys.path.insert(0, seasfire_path)
        from utils import GRUDataset, GRUTransform  # type: ignore

        _SEASFIRE_AVAILABLE = True
except Exception:
    GRUDataset = None
    GRUTransform = None
    _SEASFIRE_AVAILABLE = False


class SeasFireFeatureBuilder:
    """
    Backward-compatible class name for OFFLINE local-dataset feature building.

    Rõ vai trò:
    - Dùng local SeasFire dataset để build tensor gần đúng bằng nearest sample.
    - Không phải online production feature builder.
    - Không thay thế pipeline online feature assembly từ GEE/Copernicus.

    Khuyến nghị:
    - Online prediction path nên dùng model_inputs / online feature assembly ở layer khác.
    - Class này chỉ nên được dùng như offline fallback khi có dataset local.
    """

    def __init__(
        self,
        dataset_path: str = "",
        timesteps: int = 6,
        target_week: int = 4,
        include_oci: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.timesteps = timesteps
        self.target_week = target_week
        self.include_oci = include_oci
        self.config = config or {}

        self._dataset = None
        self._transform = None

        # Cân nặng cho temporal distance khi sample có time
        self.temporal_weight = float(self.config.get("temporal_weight", 0.05))
        self.max_temporal_penalty_days = int(self.config.get("max_temporal_penalty_days", 365))
        self.strict_require_local_dataset = bool(
            self.config.get("strict_require_local_dataset", False)
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def build(self, params: Dict[str, Any]) -> Any:
        """
        Backward-compatible wrapper.

        OFFLINE ONLY:
        - dùng local dataset nếu có
        - không tự assemble online features
        - trả tensor hoặc None
        """
        return self.build_from_local_dataset(params)

    def build_from_local_dataset(self, params: Dict[str, Any]) -> Any:
        """
        Xây tensor từ local SeasFire dataset bằng nearest-sample retrieval.

        Đây là OFFLINE FALLBACK, không phải online production path.

        Input có thể chứa:
        - location / coordinates / geometry / bbox
        - time_range
        - model_inputs (nhưng file này không dùng model_inputs để tự assemble tensor online)

        Output:
        - torch.Tensor shape (1, timesteps, features), hoặc None nếu không build được
        """
        if not self._ensure_dataset():
            logger.info(
                "SeasFireFeatureBuilder skipped: local dataset unavailable. "
                "This offline builder is not the online production path."
            )
            return None

        lat, lon, query_time, context_info = self._extract_query_context(params)
        if lat is None or lon is None:
            logger.warning(
                "SeasFireFeatureBuilder: missing spatial context for offline local lookup: %s",
                context_info,
            )
            return None

        try:
            best_idx, debug_info = self._find_nearest_sample_index(
                query_lat=lat,
                query_lon=lon,
                query_time=query_time,
            )
            if best_idx is None:
                logger.warning(
                    "SeasFireFeatureBuilder: no nearest local sample found. debug=%s",
                    debug_info,
                )
                return None

            x, y = self._dataset.get(best_idx)
            x_t, y_t = self._transform((x, y))
            logger.info(
                "SeasFireFeatureBuilder: built offline tensor from local dataset. "
                "sample_idx=%s debug=%s",
                best_idx,
                debug_info,
            )
            return x_t.unsqueeze(0)
        except Exception as e:
            logger.warning("SeasFireFeatureBuilder build_from_local_dataset failed: %s", e)
            return None

    def describe_role(self) -> Dict[str, Any]:
        """
        Trả metadata mô tả rõ vai trò để caller/debugger không dùng nhầm.
        """
        return {
            "name": self.__class__.__name__,
            "role": "offline_local_dataset_feature_builder",
            "is_online_production_path": False,
            "requires_local_dataset": True,
            "uses_nearest_sample": True,
            "dataset_available": self._ensure_dataset(),
        }

    # ─────────────────────────────────────────
    # Dataset loading
    # ─────────────────────────────────────────

    def _ensure_dataset(self) -> bool:
        """
        Load GRUDataset nếu có. Trả về True khi thành công.
        """
        if not _SEASFIRE_AVAILABLE:
            return False
        if not self.dataset_path or not self.dataset_path.exists():
            return False
        if self._dataset is not None and self._transform is not None:
            return True

        try:
            self._dataset = GRUDataset(
                root_dir=str(self.dataset_path),
                target_week=self.target_week,
                include_oci_variables=self.include_oci,
                transform=None,
            )
            self._transform = GRUTransform(str(self.dataset_path), self.timesteps)
            return True
        except Exception as e:
            logger.warning("SeasFireFeatureBuilder: dataset load failed: %s", e)
            self._dataset = None
            self._transform = None
            return False

    # ─────────────────────────────────────────
    # Query context extraction
    # ─────────────────────────────────────────

    def _extract_query_context(
        self,
        params: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float], Optional[datetime], Dict[str, Any]]:
        """
        Tách context truy vấn dùng cho local nearest-sample search.

        Ưu tiên:
        1) location dict / coordinates
        2) geometry point
        3) bbox centroid
        4) polygon centroid (best-effort)

        Time:
        - ưu tiên time_range.start hoặc start_date
        """
        lat = None
        lon = None

        location = params.get("location") or {}
        coordinates = params.get("coordinates") or {}
        geometry = params.get("geometry") or {}

        # 1) location dict
        if isinstance(location, dict):
            lat = self._to_float(location.get("lat") or location.get("latitude"))
            lon = self._to_float(location.get("lon") or location.get("lng") or location.get("longitude"))

        # 2) coordinates dict
        if lat is None or lon is None:
            if isinstance(coordinates, dict):
                lat = self._to_float(coordinates.get("lat") or coordinates.get("latitude"))
                lon = self._to_float(coordinates.get("lon") or coordinates.get("lng") or coordinates.get("longitude"))

        # 3) geometry point
        if (lat is None or lon is None) and isinstance(geometry, dict):
            gtype = geometry.get("type")
            gvalue = geometry.get("value", {})
            if gtype == "point" and isinstance(gvalue, dict):
                lat = self._to_float(gvalue.get("lat") or gvalue.get("latitude"))
                lon = self._to_float(gvalue.get("lon") or gvalue.get("lng") or gvalue.get("longitude"))

        # 4) bbox centroid
        if (lat is None or lon is None) and isinstance(params.get("bbox"), dict):
            lat, lon = self._bbox_centroid(params.get("bbox"))

        if (lat is None or lon is None) and isinstance(geometry, dict) and geometry.get("type") == "bbox":
            lat, lon = self._bbox_centroid(geometry.get("value"))

        # 5) polygon centroid
        if (lat is None or lon is None) and params.get("polygon") is not None:
            lat, lon = self._polygon_centroid(params.get("polygon"))

        if (
            (lat is None or lon is None)
            and isinstance(geometry, dict)
            and geometry.get("type") == "polygon"
        ):
            lat, lon = self._polygon_centroid(geometry.get("value"))

        time_range = params.get("time_range") or {}
        query_time = self._extract_query_datetime(time_range)

        context_info = {
            "lat": lat,
            "lon": lon,
            "query_time": query_time.isoformat() if query_time else None,
        }
        return lat, lon, query_time, context_info

    def _extract_query_datetime(
        self,
        time_range: Dict[str, Any],
    ) -> Optional[datetime]:
        if not isinstance(time_range, dict):
            return None

        raw = (
            time_range.get("start")
            or time_range.get("start_date")
            or time_range.get("date")
            or time_range.get("day")
        )
        if not raw:
            return None

        return self._parse_datetime(raw)

    # ─────────────────────────────────────────
    # Nearest-sample search
    # ─────────────────────────────────────────

    def _find_nearest_sample_index(
        self,
        query_lat: float,
        query_lon: float,
        query_time: Optional[datetime],
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        OFFLINE local nearest-sample lookup.

        Khác bản cũ:
        - tách riêng logic nearest-sample
        - có cân nhắc cả khoảng cách thời gian nếu dataset sample có time
        - không giả vờ là online feature assembly
        """
        samples = getattr(self._dataset, "_samples", None)
        if not samples:
            return None, {"error": "dataset has no _samples"}

        best_idx = None
        best_score = float("inf")
        best_debug: Dict[str, Any] = {}

        for idx, sample in enumerate(samples):
            sample_lat, sample_lon, sample_time = self._parse_sample_tuple(sample)

            if sample_lat is None or sample_lon is None:
                continue

            spatial_dist_sq = self._spatial_distance_sq(
                query_lat=query_lat,
                query_lon=query_lon,
                sample_lat=sample_lat,
                sample_lon=sample_lon,
            )

            temporal_penalty = self._temporal_penalty(
                query_time=query_time,
                sample_time=sample_time,
            )

            score = spatial_dist_sq + temporal_penalty

            if score < best_score:
                best_score = score
                best_idx = idx
                best_debug = {
                    "sample_lat": sample_lat,
                    "sample_lon": sample_lon,
                    "sample_time": sample_time.isoformat() if sample_time else None,
                    "spatial_dist_sq": spatial_dist_sq,
                    "temporal_penalty": temporal_penalty,
                    "score": score,
                }

        return best_idx, best_debug

    def _parse_sample_tuple(
        self,
        sample: Any,
    ) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
        """
        Hỗ trợ sample tuple kiểu:
        - (lat, lon, time)
        - (lat, lon)
        - object lạ -> best effort
        """
        try:
            if isinstance(sample, (tuple, list)):
                if len(sample) >= 3:
                    return (
                        self._to_float(sample[0]),
                        self._to_float(sample[1]),
                        self._parse_datetime(sample[2]),
                    )
                if len(sample) >= 2:
                    return (
                        self._to_float(sample[0]),
                        self._to_float(sample[1]),
                        None,
                    )
        except Exception:
            pass

        return None, None, None

    def _spatial_distance_sq(
        self,
        query_lat: float,
        query_lon: float,
        sample_lat: float,
        sample_lon: float,
    ) -> float:
        return (sample_lat - query_lat) ** 2 + (sample_lon - query_lon) ** 2

    def _temporal_penalty(
        self,
        query_time: Optional[datetime],
        sample_time: Optional[datetime],
    ) -> float:
        """
        Temporal penalty nhẹ để ưu tiên sample gần thời điểm truy vấn.
        Nếu sample/query time không có -> penalty 0.
        """
        if query_time is None or sample_time is None:
            return 0.0

        day_diff = abs((sample_time.date() - query_time.date()).days)
        clipped = min(day_diff, self.max_temporal_penalty_days)
        return clipped * self.temporal_weight

    # ─────────────────────────────────────────
    # Geometry helpers
    # ─────────────────────────────────────────

    def _bbox_centroid(
        self,
        bbox: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(bbox, dict):
            return None, None

        min_lat = self._to_float(bbox.get("min_lat"))
        min_lon = self._to_float(bbox.get("min_lon"))
        max_lat = self._to_float(bbox.get("max_lat"))
        max_lon = self._to_float(bbox.get("max_lon"))

        if None in (min_lat, min_lon, max_lat, max_lon):
            return None, None

        return ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)

    def _polygon_centroid(
        self,
        polygon: Any,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Best-effort centroid cho polygon GeoJSON-like hoặc list points.
        """
        coords = None

        if isinstance(polygon, dict):
            if polygon.get("type") == "Polygon":
                coords = polygon.get("coordinates", [[]])
                if coords and isinstance(coords, list):
                    coords = coords[0]
            elif polygon.get("coordinates"):
                coords = polygon.get("coordinates")
                if coords and isinstance(coords, list) and coords and isinstance(coords[0], list):
                    coords = coords[0]
        elif isinstance(polygon, list):
            coords = polygon
            if coords and isinstance(coords[0], list) and len(coords[0]) > 0 and isinstance(coords[0][0], (list, tuple)):
                coords = coords[0]

        if not coords:
            return None, None

        lats = []
        lons = []
        for pt in coords:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                lon = self._to_float(pt[0])
                lat = self._to_float(pt[1])
                if lat is not None and lon is not None:
                    lats.append(lat)
                    lons.append(lon)

        if not lats or not lons:
            return None, None

        return (sum(lats) / len(lats), sum(lons) / len(lons))

    # ─────────────────────────────────────────
    # Parsing helpers
    # ─────────────────────────────────────────

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        text = str(value).strip()
        if not text:
            return None

        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue

        try:
            # Python >= 3.7 isoformat parser
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None


# Alias rõ nghĩa hơn để caller mới dùng đúng role.
OfflineDatasetFeatureBuilder = SeasFireFeatureBuilder