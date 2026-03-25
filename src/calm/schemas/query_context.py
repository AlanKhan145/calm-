"""
Shared query context schema dùng chung toàn hệ thống.

Mục tiêu:
- Chuẩn hóa location / time / task thành 1 schema thống nhất.
- Dùng chung giữa:
  - QueryNormalizerAgent
  - Planner
  - Data agents
  - Orchestrator
  - Prediction / refinement agents
- Tránh truyền dict rời rạc.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TaskType(str, Enum):
    UNKNOWN = "unknown"
    QA = "qa"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    MONITORING = "monitoring"


class TimeRange(BaseModel):
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    timezone: Optional[str] = None
    source: Optional[str] = None

    @validator("end")
    def validate_end_after_start(cls, v: Optional[datetime], values: Dict[str, Any]) -> Optional[datetime]:
        start = values.get("start")
        if start is not None and v is not None and v < start:
            raise ValueError("time_range.end must be >= time_range.start")
        return v


class BoundingBox(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float

    @validator("max_lat")
    def validate_lat_order(cls, v: float, values: Dict[str, Any]) -> float:
        min_lat = values.get("min_lat")
        if min_lat is not None and v < min_lat:
            raise ValueError("bbox.max_lat must be >= bbox.min_lat")
        return v

    @validator("max_lon")
    def validate_lon_order(cls, v: float, values: Dict[str, Any]) -> float:
        min_lon = values.get("min_lon")
        if min_lon is not None and v < min_lon:
            raise ValueError("bbox.max_lon must be >= bbox.min_lon")
        return v


class SourceTraceItem(BaseModel):
    field: str
    source: str
    note: Optional[str] = None
    confidence: Optional[float] = None
    raw_value: Optional[Any] = None


class QueryContext(BaseModel):
    """
    Schema chuẩn truyền xuyên suốt pipeline.

    Các agent về sau nên nhận QueryContext thay vì dict rời rạc.
    """

    task_type: TaskType = TaskType.UNKNOWN

    location_text: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[BoundingBox] = None

    time_range: Optional[TimeRange] = None

    requested_output: List[str] = Field(default_factory=list)

    confidence: float = 0.0

    source_trace: List[SourceTraceItem] = Field(default_factory=list)

    @validator("lat")
    def validate_lat(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (-90.0 <= v <= 90.0):
            raise ValueError("lat must be between -90 and 90")
        return v

    @validator("lon")
    def validate_lon(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (-180.0 <= v <= 180.0):
            raise ValueError("lon must be between -180 and 180")
        return v

    @validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return float(v)

    @validator("requested_output", pre=True, always=True)
    def normalize_requested_output(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            out: List[str] = []
            seen = set()
            for item in v:
                if item is None:
                    continue
                s = str(item).strip()
                if s and s not in seen:
                    out.append(s)
                    seen.add(s)
            return out
        return [str(v)]

    def has_coordinates(self) -> bool:
        return self.lat is not None and self.lon is not None

    def has_time_range(self) -> bool:
        return self.time_range is not None and self.time_range.start is not None and self.time_range.end is not None

    def add_trace(
        self,
        field: str,
        source: str,
        note: Optional[str] = None,
        confidence: Optional[float] = None,
        raw_value: Optional[Any] = None,
    ) -> None:
        self.source_trace.append(
            SourceTraceItem(
                field=field,
                source=source,
                note=note,
                confidence=confidence,
                raw_value=raw_value,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        # compatible với pydantic v1 / v2
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    @classmethod
    def from_normalizer_output(cls, payload: Dict[str, Any]) -> "QueryContext":
        """
        Helper để convert output của QueryNormalizerAgent sang schema chung.
        """
        payload = payload or {}

        location_resolved = payload.get("location_resolved") or {}
        time_range_raw = payload.get("time_range") or {}
        requested_output = payload.get("requested_output") or []

        source_trace: List[SourceTraceItem] = []

        if payload.get("task_type"):
            source_trace.append(
                SourceTraceItem(
                    field="task_type",
                    source="query_normalizer",
                    note="normalized from raw query",
                    raw_value=payload.get("task_type"),
                )
            )

        if payload.get("location_text") is not None:
            source_trace.append(
                SourceTraceItem(
                    field="location_text",
                    source="query_normalizer",
                    note="location text extracted from raw query",
                    raw_value=payload.get("location_text"),
                )
            )

        if location_resolved:
            source_trace.append(
                SourceTraceItem(
                    field="location_resolved",
                    source=str(location_resolved.get("source") or "query_normalizer"),
                    note="resolved location candidate",
                    raw_value=location_resolved,
                )
            )

        if time_range_raw:
            source_trace.append(
                SourceTraceItem(
                    field="time_range",
                    source=str(time_range_raw.get("source") or "query_normalizer"),
                    note="resolved time range",
                    raw_value=time_range_raw,
                )
            )

        if requested_output:
            source_trace.append(
                SourceTraceItem(
                    field="requested_output",
                    source="query_normalizer",
                    note="requested outputs inferred from query",
                    raw_value=requested_output,
                )
            )

        bbox = location_resolved.get("bbox")
        bbox_obj = BoundingBox(**bbox) if isinstance(bbox, dict) else None

        time_range_obj = TimeRange(**time_range_raw) if time_range_raw else None

        return cls(
            task_type=payload.get("task_type", TaskType.UNKNOWN),
            location_text=payload.get("location_text"),
            lat=location_resolved.get("lat"),
            lon=location_resolved.get("lon"),
            bbox=bbox_obj,
            time_range=time_range_obj,
            requested_output=requested_output,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            source_trace=source_trace,
        )