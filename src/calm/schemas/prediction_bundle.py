"""
Shared prediction bundle schema cho toàn bộ prediction pipeline.

Mục tiêu:
- Là contract chung giữa:
  - data retrieval
  - feature builder
  - model runner
  - RSEN / feedback refinement
- Tránh truyền dict rời rạc, thiếu chuẩn.
- Gom mọi input/output trung gian vào cùng một bundle thống nhất.

Schema chính gồm:
- met_data
- satellite_data
- static_geo
- text_context
- feature_tensor
- data_quality
- source_metadata
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


JSONDict = Dict[str, Any]
JSONList = List[JSONDict]
FlexibleData = Union[JSONDict, JSONList, None]


class TextContextItem(BaseModel):
    text: str
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataQuality(BaseModel):
    """
    Tóm tắt chất lượng dữ liệu cho toàn pipeline.
    """

    score: float = 0.0
    completeness: Optional[float] = None
    observed_ratio: Optional[float] = None
    missing_ratio: Optional[float] = None
    missing_features: List[str] = Field(default_factory=list)
    stale: bool = False
    issues: List[str] = Field(default_factory=list)
    feature_status: Optional[str] = None
    feature_manifest: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)

    @validator("score", "completeness", "observed_ratio", "missing_ratio", pre=True, always=True)
    def _normalize_ratio_fields(cls, v: Any, field):
        if v is None:
            return None if field.name != "score" else 0.0
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None if field.name != "score" else 0.0
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return x

    @validator("missing_features", "issues", "notes", pre=True, always=True)
    def _normalize_str_list(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
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
        s = str(v).strip()
        return [s] if s else []


class SourceMetadataItem(BaseModel):
    """
    Metadata cho từng nguồn dữ liệu trong pipeline.
    """

    source_name: str
    source_type: Optional[str] = None
    status: Optional[str] = None
    fetched_at: Optional[datetime] = None
    coverage_start: Optional[datetime] = None
    coverage_end: Optional[datetime] = None
    location_hint: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PredictionBundle(BaseModel):
    """
    Contract chung của prediction pipeline.

    Field chính:
    - met_data: dữ liệu khí tượng / time series / met inputs
    - satellite_data: dữ liệu viễn thám / satellite features
    - static_geo: dữ liệu tĩnh địa lý như DEM, slope, landcover, fuel...
    - text_context: textual context / supporting text / summaries
    - feature_tensor: tensor hoặc array-like đã build cho model
    - data_quality: chất lượng dữ liệu / coverage / missingness
    - source_metadata: metadata của từng nguồn dữ liệu
    """

    met_data: FlexibleData = None
    satellite_data: FlexibleData = None
    static_geo: Optional[JSONDict] = None
    text_context: List[TextContextItem] = Field(default_factory=list)
    feature_tensor: Any = None
    data_quality: DataQuality = Field(default_factory=DataQuality)
    source_metadata: List[SourceMetadataItem] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @validator("static_geo", pre=True, always=True)
    def _normalize_static_geo(cls, v: Any) -> Optional[JSONDict]:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        return {"value": v}

    @validator("text_context", pre=True, always=True)
    def _normalize_text_context(cls, v: Any) -> List[TextContextItem]:
        if v is None:
            return []

        if isinstance(v, str):
            s = v.strip()
            return [TextContextItem(text=s)] if s else []

        if isinstance(v, dict):
            if "text" in v:
                return [TextContextItem(**v)]
            return [TextContextItem(text=str(v), metadata={"raw": v})]

        if isinstance(v, (list, tuple)):
            out: List[TextContextItem] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, TextContextItem):
                    out.append(item)
                elif isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(TextContextItem(text=s))
                elif isinstance(item, dict):
                    if "text" in item:
                        out.append(TextContextItem(**item))
                    else:
                        out.append(TextContextItem(text=str(item), metadata={"raw": item}))
                else:
                    out.append(TextContextItem(text=str(item)))
            return out

        return [TextContextItem(text=str(v))]

    @validator("source_metadata", pre=True, always=True)
    def _normalize_source_metadata(cls, v: Any) -> List[SourceMetadataItem]:
        if v is None:
            return []

        if isinstance(v, dict):
            if "source_name" in v:
                return [SourceMetadataItem(**v)]
            items = []
            for k, val in v.items():
                if isinstance(val, dict):
                    items.append(SourceMetadataItem(source_name=str(k), details=val))
                else:
                    items.append(SourceMetadataItem(source_name=str(k), details={"value": val}))
            return items

        if isinstance(v, (list, tuple)):
            out: List[SourceMetadataItem] = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, SourceMetadataItem):
                    out.append(item)
                elif isinstance(item, dict):
                    out.append(SourceMetadataItem(**item))
                else:
                    out.append(SourceMetadataItem(source_name=str(item)))
            return out

        return [SourceMetadataItem(source_name=str(v))]

    def has_met_data(self) -> bool:
        return self.met_data not in (None, {}, [])

    def has_satellite_data(self) -> bool:
        return self.satellite_data not in (None, {}, [])

    def has_static_geo(self) -> bool:
        return bool(self.static_geo)

    def has_feature_tensor(self) -> bool:
        return self.feature_tensor is not None

    def add_text_context(
        self,
        text: str,
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text_context.append(
            TextContextItem(
                text=text,
                source=source,
                timestamp=timestamp,
                metadata=metadata or {},
            )
        )

    def add_source_metadata(
        self,
        source_name: str,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
        fetched_at: Optional[datetime] = None,
        coverage_start: Optional[datetime] = None,
        coverage_end: Optional[datetime] = None,
        location_hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source_metadata.append(
            SourceMetadataItem(
                source_name=source_name,
                source_type=source_type,
                status=status,
                fetched_at=fetched_at,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                location_hint=location_hint,
                details=details or {},
            )
        )

    def set_feature_tensor(
        self,
        feature_tensor: Any,
        feature_status: Optional[str] = None,
        feature_manifest: Optional[Dict[str, Any]] = None,
        missing_features: Optional[List[str]] = None,
        score: Optional[float] = None,
    ) -> None:
        """
        Helper để feature builder cập nhật bundle theo contract chung.
        """
        self.feature_tensor = feature_tensor

        if feature_status is not None:
            self.data_quality.feature_status = feature_status

        if feature_manifest is not None:
            self.data_quality.feature_manifest = feature_manifest

        if missing_features is not None:
            self.data_quality.missing_features = list(dict.fromkeys(str(x) for x in missing_features if x is not None))

        if score is not None:
            self.data_quality.score = max(0.0, min(1.0, float(score)))

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    @classmethod
    def from_parts(
        cls,
        met_data: FlexibleData = None,
        satellite_data: FlexibleData = None,
        static_geo: Optional[JSONDict] = None,
        text_context: Optional[List[Any]] = None,
        feature_tensor: Any = None,
        data_quality: Optional[Dict[str, Any]] = None,
        source_metadata: Optional[List[Any]] = None,
    ) -> "PredictionBundle":
        return cls(
            met_data=met_data,
            satellite_data=satellite_data,
            static_geo=static_geo,
            text_context=text_context or [],
            feature_tensor=feature_tensor,
            data_quality=DataQuality(**(data_quality or {})),
            source_metadata=source_metadata or [],
        )