"""
OnlineFeatureBuilder — build online inference tensor cho SeasFire từ dữ liệu thật.

Mục tiêu:
- Đường chính cho online inference.
- Không phụ thuộc nearest sample từ local dataset.
- Nhận dữ liệu online:
    - met_timeseries
    - satellite_features
    - static_geo
    - time_range
    - location
- Trả:
    - tensor (1, timesteps, input_size)
    - feature_manifest
    - missing_features
- Hỗ trợ normalization bằng stats từ lúc training.
- Log rõ feature nào là:
    - real
    - fallback
    - imputed

Tương thích với model_runner:
build(parameters) -> {
    "features": tensor,
    "feature_status": "online_ready" | "missing",
    "feature_manifest": ...,
    "missing_features": [...],
    "metadata": {...},
}
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OnlineFeatureBuilder:
    """
    Online builder để dựng tensor đầu vào cho SeasFire model.

    parameters đầu vào thường có:
    - met_timeseries: list[dict] | dict[columnar]
    - satellite_features: list[dict] | dict[columnar]
    - static_geo: dict
    - time_range: {"start": ..., "end": ...}
    - location: {"lat": ..., "lon": ...} hoặc dict geocoding

    Cấu hình quan trọng:
    - feature_schema: list[dict]
        Mỗi feature spec nên có:
        {
            "name": "t2m",
            "aliases": ["t2m", "temperature", "temp"],
            "source": "met_timeseries",   # met_timeseries | satellite_features | static_geo | location | derived | auto
            "kind": "dynamic",            # dynamic | static | derived
            "default": None,
            "normalize": True
        }

    - stats_path:
        file stats từ training, hỗ trợ json/pkl/pt/pth
        Có thể chứa:
        {
            "feature_names": [...],
            "mean": [...],
            "std": [...]
        }
        hoặc
        {
            "means": {"t2m": ...},
            "stds": {"t2m": ...}
        }
        hoặc
        {
            "stats": {"t2m": {"mean": ..., "std": ...}}
        }
    """

    def __init__(
        self,
        feature_schema: list[dict[str, Any]] | None = None,
        stats_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.stats_path = Path(stats_path) if stats_path else None

        self.default_timesteps = int(self.config.get("default_timesteps", 8))
        self.input_size = int(self.config.get("input_size", 59))
        self.normalize_enabled = bool(self.config.get("normalize", True))
        self.strict_input_size = bool(self.config.get("strict_input_size", True))
        self.log_each_feature = bool(self.config.get("log_each_feature", True))
        self.min_observed_ratio = float(self.config.get("min_observed_ratio", 0.10))
        self.min_observed_features = int(self.config.get("min_observed_features", 1))

        self._stats = self._load_stats(self.stats_path)
        self.feature_schema = self._resolve_feature_schema(feature_schema, self._stats)

        if self.strict_input_size and len(self.feature_schema) != self.input_size:
            raise ValueError(
                f"OnlineFeatureBuilder feature_schema size={len(self.feature_schema)} "
                f"không khớp input_size={self.input_size}"
            )

        if not self.strict_input_size:
            self.input_size = len(self.feature_schema)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Build online tensor.

        Trả dict theo format builder chuẩn:
        {
            "features": tensor | None,
            "feature_status": "online_ready" | "missing",
            "feature_manifest": {...},
            "missing_features": [...],
            "metadata": {...},
        }
        """
        parameters = parameters or {}

        met_timeseries = parameters.get("met_timeseries")
        satellite_features = parameters.get("satellite_features")
        static_geo = parameters.get("static_geo") or {}
        time_range = parameters.get("time_range") or {}
        location = parameters.get("location") or {}

        met_records = self._coerce_timeseries_records(met_timeseries)
        sat_records = self._coerce_timeseries_records(satellite_features)

        static_geo_flat = self._flatten_mapping(static_geo)
        location_flat = self._flatten_mapping(location)
        derived_context = self._build_derived_context(time_range=time_range, location=location)

        timesteps = self._resolve_timesteps(
            met_records=met_records,
            sat_records=sat_records,
            time_range=time_range,
        )

        if timesteps <= 0:
            manifest = {
                "timesteps": 0,
                "input_size": self.input_size,
                "status": "missing",
                "summary": {
                    "real_count": 0,
                    "fallback_count": 0,
                    "imputed_count": len(self.feature_schema),
                    "observed_count": 0,
                },
                "features": [],
            }
            return {
                "features": None,
                "feature_status": "missing",
                "feature_manifest": manifest,
                "missing_features": [spec["name"] for spec in self.feature_schema],
                "metadata": {
                    "feature_source": "online",
                    "error": "No valid timesteps could be resolved from online inputs.",
                    "feature_manifest": manifest,
                    "missing_features": [spec["name"] for spec in self.feature_schema],
                },
            }

        feature_matrix: list[list[float]] = [[0.0 for _ in range(self.input_size)] for _ in range(timesteps)]
        feature_entries: list[dict[str, Any]] = []
        missing_features: list[str] = []
        fallback_features: list[str] = []
        real_count = 0
        fallback_count = 0
        imputed_count = 0
        observed_count = 0

        for idx, spec in enumerate(self.feature_schema):
            series, entry = self._build_one_feature(
                spec=spec,
                timesteps=timesteps,
                met_records=met_records,
                sat_records=sat_records,
                static_geo_flat=static_geo_flat,
                location_flat=location_flat,
                derived_context=derived_context,
            )

            for t in range(timesteps):
                feature_matrix[t][idx] = float(series[t])

            feature_entries.append(entry)

            status = entry["status"]
            if status == "real":
                real_count += 1
                observed_count += 1
            elif status == "fallback":
                fallback_count += 1
                observed_count += 1
                fallback_features.append(spec["name"])
            else:
                imputed_count += 1
                missing_features.append(spec["name"])

            if self.log_each_feature:
                logger.info(
                    "OnlineFeatureBuilder feature=%s status=%s source=%s observed_steps=%s/%s",
                    entry["name"],
                    entry["status"],
                    entry.get("source_used"),
                    entry.get("observed_steps", 0),
                    timesteps,
                )

        observed_ratio = observed_count / max(1, len(self.feature_schema))
        feature_status = (
            "online_ready"
            if observed_count >= self.min_observed_features and observed_ratio >= self.min_observed_ratio
            else "missing"
        )

        tensor = self._to_tensor(feature_matrix) if feature_status == "online_ready" else None

        manifest = {
            "timesteps": timesteps,
            "input_size": len(self.feature_schema),
            "status": feature_status,
            "summary": {
                "real_count": real_count,
                "fallback_count": fallback_count,
                "imputed_count": imputed_count,
                "observed_count": observed_count,
                "observed_ratio": observed_ratio,
                "fallback_features": fallback_features,
                "missing_features": missing_features,
            },
            "sources": {
                "has_met_timeseries": bool(met_records),
                "has_satellite_features": bool(sat_records),
                "has_static_geo": bool(static_geo_flat),
                "has_location": bool(location_flat),
                "has_time_range": bool(time_range),
            },
            "features": feature_entries,
        }

        return {
            "features": tensor,
            "feature_status": feature_status,
            "feature_manifest": manifest,
            "missing_features": missing_features,
            "metadata": {
                "feature_source": "online",
                "feature_manifest": manifest,
                "missing_features": missing_features,
                "timesteps": timesteps,
                "input_size": len(self.feature_schema),
                "normalization_enabled": self.normalize_enabled,
                "stats_path": str(self.stats_path) if self.stats_path else None,
            },
        }

    # ------------------------------------------------------------------
    # Feature schema / stats
    # ------------------------------------------------------------------

    def _resolve_feature_schema(
        self,
        feature_schema: list[dict[str, Any]] | None,
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Ưu tiên:
        1) feature_schema truyền thẳng
        2) config["feature_schema"]
        3) feature_names từ stats
        """
        raw_schema = feature_schema or self.config.get("feature_schema")

        if raw_schema:
            return [self._normalize_feature_spec(x) for x in raw_schema]

        feature_names = stats.get("feature_names") or []
        if feature_names:
            return [
                self._normalize_feature_spec(
                    {
                        "name": name,
                        "aliases": [name],
                        "source": "auto",
                        "kind": "dynamic",
                        "default": None,
                        "normalize": True,
                    }
                )
                for name in feature_names
            ]

        raise ValueError(
            "OnlineFeatureBuilder cần feature_schema hoặc stats có feature_names để dựng đúng input order."
        )

    def _normalize_feature_spec(self, spec: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(spec, str):
            spec = {"name": spec}

        name = str(spec["name"])
        aliases = spec.get("aliases") or [name]

        if name not in aliases:
            aliases = [name, *aliases]

        return {
            "name": name,
            "aliases": self._unique_keep_order([str(x) for x in aliases]),
            "source": spec.get("source", "auto"),
            "kind": spec.get("kind", "dynamic"),
            "default": spec.get("default"),
            "normalize": bool(spec.get("normalize", True)),
        }

    def _load_stats(self, stats_path: Path | None) -> dict[str, Any]:
        if not stats_path:
            return {}

        if not stats_path.exists():
            logger.warning("OnlineFeatureBuilder stats not found: %s", stats_path)
            return {}

        try:
            suffix = stats_path.suffix.lower()

            if suffix in {".json"}:
                raw = json.loads(stats_path.read_text(encoding="utf-8"))
            elif suffix in {".pkl", ".pickle"}:
                with open(stats_path, "rb") as f:
                    raw = pickle.load(f)
            elif suffix in {".pt", ".pth"}:
                import torch

                raw = torch.load(str(stats_path), map_location="cpu")
            else:
                logger.warning("OnlineFeatureBuilder unsupported stats format: %s", stats_path)
                return {}

            return self._normalize_stats_bundle(raw)
        except Exception as e:
            logger.warning("OnlineFeatureBuilder failed to load stats %s: %s", stats_path, e)
            return {}

    def _normalize_stats_bundle(self, raw: Any) -> dict[str, Any]:
        """
        Chuẩn hóa nhiều format stats về:
        {
            "feature_names": [...],
            "mean_by_name": {...},
            "std_by_name": {...},
        }
        """
        out = {
            "feature_names": [],
            "mean_by_name": {},
            "std_by_name": {},
        }

        if not isinstance(raw, dict):
            return out

        feature_names = (
            raw.get("feature_names")
            or raw.get("features")
            or raw.get("columns")
            or raw.get("input_features")
            or []
        )
        if isinstance(feature_names, tuple):
            feature_names = list(feature_names)

        means = raw.get("mean") or raw.get("means") or raw.get("feature_mean")
        stds = raw.get("std") or raw.get("stds") or raw.get("feature_std")

        stats_nested = raw.get("stats") or raw.get("feature_stats") or raw.get("normalization")

        if feature_names and isinstance(means, (list, tuple)) and isinstance(stds, (list, tuple)):
            out["feature_names"] = list(feature_names)
            for name, mean_val, std_val in zip(feature_names, means, stds):
                out["mean_by_name"][str(name)] = self._safe_float(mean_val)
                out["std_by_name"][str(name)] = self._safe_float(std_val)

        if isinstance(means, dict):
            for k, v in means.items():
                out["mean_by_name"][str(k)] = self._safe_float(v)

        if isinstance(stds, dict):
            for k, v in stds.items():
                out["std_by_name"][str(k)] = self._safe_float(v)

        if isinstance(stats_nested, dict):
            # format: {"t2m": {"mean": ..., "std": ...}}
            for k, v in stats_nested.items():
                if isinstance(v, dict):
                    if "mean" in v:
                        out["mean_by_name"][str(k)] = self._safe_float(v.get("mean"))
                    if "std" in v:
                        out["std_by_name"][str(k)] = self._safe_float(v.get("std"))

            if not out["feature_names"]:
                out["feature_names"] = list(stats_nested.keys())

        if not out["feature_names"]:
            all_names = set(out["mean_by_name"].keys()) | set(out["std_by_name"].keys())
            out["feature_names"] = list(sorted(all_names))

        return out

    # ------------------------------------------------------------------
    # Build one feature
    # ------------------------------------------------------------------

    def _build_one_feature(
        self,
        spec: dict[str, Any],
        timesteps: int,
        met_records: list[dict[str, Any]],
        sat_records: list[dict[str, Any]],
        static_geo_flat: dict[str, Any],
        location_flat: dict[str, Any],
        derived_context: dict[str, Any],
    ) -> tuple[list[float], dict[str, Any]]:
        name = spec["name"]
        preferred_source = spec["source"]
        kind = spec["kind"]
        aliases = spec["aliases"]

        raw_series: list[float] | None = None
        source_used: str | None = None
        source_key: str | None = None
        status = "imputed"
        notes: list[str] = []

        # 1) Thử source chính
        raw_series, source_used, source_key, status, note = self._extract_feature_series(
            spec=spec,
            preferred_source=preferred_source,
            aliases=aliases,
            kind=kind,
            timesteps=timesteps,
            met_records=met_records,
            sat_records=sat_records,
            static_geo_flat=static_geo_flat,
            location_flat=location_flat,
            derived_context=derived_context,
        )
        if note:
            notes.append(note)

        # 2) Impute nếu vẫn chưa có
        if raw_series is None:
            raw_series, note = self._impute_series(spec, timesteps)
            status = "imputed"
            source_used = "imputation"
            source_key = None
            notes.append(note)

        # 3) Normalize
        normalized_series, norm_info = self._normalize_series(name, raw_series, spec.get("normalize", True))

        entry = {
            "name": name,
            "preferred_source": preferred_source,
            "source_used": source_used,
            "source_key": source_key,
            "status": status,  # real | fallback | imputed
            "observed_steps": self._count_non_none(raw_series),
            "timesteps": timesteps,
            "notes": notes,
            "normalization": norm_info,
        }

        return normalized_series, entry

    def _extract_feature_series(
        self,
        spec: dict[str, Any],
        preferred_source: str,
        aliases: list[str],
        kind: str,
        timesteps: int,
        met_records: list[dict[str, Any]],
        sat_records: list[dict[str, Any]],
        static_geo_flat: dict[str, Any],
        location_flat: dict[str, Any],
        derived_context: dict[str, Any],
    ) -> tuple[list[float] | None, str | None, str | None, str, str | None]:
        """
        real:
          lấy đúng từ source mong muốn
        fallback:
          lấy được từ source khác / derived / broadcast / pad
        imputed:
          không lấy được, phải fill bằng stats/default
        """
        source_order = self._resolve_source_order(preferred_source)

        for source_name in source_order:
            if source_name == "met_timeseries":
                values, key = self._extract_from_records(met_records, aliases)
                if values is not None:
                    values, resized_note, resized_status = self._resize_series(values, timesteps)
                    status = "real" if source_name == preferred_source and resized_status == "real" else "fallback"
                    note = resized_note
                    return values, source_name, key, status, note

            elif source_name == "satellite_features":
                values, key = self._extract_from_records(sat_records, aliases)
                if values is not None:
                    values, resized_note, resized_status = self._resize_series(values, timesteps)
                    status = "real" if source_name == preferred_source and resized_status == "real" else "fallback"
                    note = resized_note
                    return values, source_name, key, status, note

            elif source_name == "static_geo":
                value, key = self._extract_from_mapping(static_geo_flat, aliases)
                if value is not None:
                    series = [value for _ in range(timesteps)]
                    status = "real" if source_name == preferred_source else "fallback"
                    return series, source_name, key, status, "broadcast static feature across timesteps"

            elif source_name == "location":
                value, key = self._extract_from_mapping(location_flat, aliases)
                if value is not None:
                    series = [value for _ in range(timesteps)]
                    status = "real" if source_name == preferred_source else "fallback"
                    return series, source_name, key, status, "broadcast location feature across timesteps"

            elif source_name == "derived":
                value, key = self._extract_from_mapping(derived_context, aliases)
                if value is not None:
                    series = [value for _ in range(timesteps)]
                    return series, source_name, key, "fallback", "derived feature from time/location context"

        return None, None, None, "imputed", "feature not found in online sources"

    def _resolve_source_order(self, preferred_source: str) -> list[str]:
        if preferred_source == "met_timeseries":
            return ["met_timeseries", "satellite_features", "static_geo", "location", "derived"]
        if preferred_source == "satellite_features":
            return ["satellite_features", "met_timeseries", "static_geo", "location", "derived"]
        if preferred_source == "static_geo":
            return ["static_geo", "location", "derived", "met_timeseries", "satellite_features"]
        if preferred_source == "location":
            return ["location", "static_geo", "derived", "met_timeseries", "satellite_features"]
        if preferred_source == "derived":
            return ["derived", "location", "static_geo", "met_timeseries", "satellite_features"]
        return ["met_timeseries", "satellite_features", "static_geo", "location", "derived"]

    # ------------------------------------------------------------------
    # Timeseries / mapping extraction
    # ------------------------------------------------------------------

    def _coerce_timeseries_records(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []

        if isinstance(value, list):
            records = [self._flatten_mapping(x) for x in value if isinstance(x, dict)]
            return self._sort_records_by_time(records)

        if isinstance(value, dict):
            # nested wrapper
            for key in ("records", "series", "data", "items"):
                if isinstance(value.get(key), list):
                    records = [self._flatten_mapping(x) for x in value[key] if isinstance(x, dict)]
                    return self._sort_records_by_time(records)

            # dict columnar -> records
            if any(isinstance(v, (list, tuple)) for v in value.values()):
                records = self._columnar_to_records(value)
                return self._sort_records_by_time(records)

            # single record
            return [self._flatten_mapping(value)]

        return []

    def _columnar_to_records(self, mapping: dict[str, Any]) -> list[dict[str, Any]]:
        flat = self._flatten_mapping(mapping)

        list_keys = [k for k, v in flat.items() if isinstance(v, (list, tuple))]
        if not list_keys:
            return [flat]

        max_len = max(len(flat[k]) for k in list_keys)
        records: list[dict[str, Any]] = []

        for i in range(max_len):
            rec: dict[str, Any] = {}
            for k, v in flat.items():
                if isinstance(v, (list, tuple)):
                    rec[k] = v[i] if i < len(v) else None
                else:
                    rec[k] = v
            records.append(rec)

        return records

    def _sort_records_by_time(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def extract_ts(rec: dict[str, Any]):
            for key in ("timestamp", "time", "datetime", "date", "ts"):
                raw = rec.get(key)
                parsed = self._parse_datetime(raw)
                if parsed is not None:
                    return parsed
            return None

        decorated = [(extract_ts(r), r) for r in records]
        if all(ts is None for ts, _ in decorated):
            return records

        return [r for _, r in sorted(decorated, key=lambda x: x[0] or datetime.min)]

    def _extract_from_records(
        self,
        records: list[dict[str, Any]],
        aliases: list[str],
    ) -> tuple[list[float] | None, str | None]:
        if not records:
            return None, None

        matched_key = self._find_matching_key(records[0], aliases)
        if matched_key is None:
            return None, None

        values: list[float] = []
        for rec in records:
            values.append(self._safe_float(rec.get(matched_key)))

        if all(v is None for v in values):
            return None, matched_key

        return values, matched_key

    def _extract_from_mapping(
        self,
        mapping: dict[str, Any],
        aliases: list[str],
    ) -> tuple[float | None, str | None]:
        if not mapping:
            return None, None

        matched_key = self._find_matching_key(mapping, aliases)
        if matched_key is None:
            return None, None

        value = self._safe_float(mapping.get(matched_key))
        return value, matched_key

    def _find_matching_key(self, mapping: dict[str, Any], aliases: list[str]) -> str | None:
        if not mapping:
            return None

        keys = list(mapping.keys())
        lower_to_real = {k.lower(): k for k in keys}

        # 1) exact
        for alias in aliases:
            real = lower_to_real.get(alias.lower())
            if real is not None:
                return real

        # 2) last segment exact
        for alias in aliases:
            alias_lower = alias.lower()
            for k in keys:
                if k.lower().split(".")[-1] == alias_lower:
                    return k

        # 3) contains
        for alias in aliases:
            alias_lower = alias.lower()
            for k in keys:
                leaf = k.lower().split(".")[-1]
                if alias_lower in k.lower() or alias_lower == leaf:
                    return k

        return None

    # ------------------------------------------------------------------
    # Derived context
    # ------------------------------------------------------------------

    def _build_derived_context(
        self,
        time_range: dict[str, Any],
        location: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Derived features có thể dùng cho schema nếu training có:
        - lat, lon
        - month
        - day_of_year
        - duration_hours
        """
        out: dict[str, Any] = {}

        location_flat = self._flatten_mapping(location)
        lat = self._safe_float(
            location_flat.get("lat")
            or location_flat.get("latitude")
            or location_flat.get("coords.lat")
        )
        lon = self._safe_float(
            location_flat.get("lon")
            or location_flat.get("lng")
            or location_flat.get("longitude")
            or location_flat.get("coords.lon")
            or location_flat.get("coords.lng")
        )

        if lat is not None:
            out["lat"] = lat
            out["latitude"] = lat
        if lon is not None:
            out["lon"] = lon
            out["longitude"] = lon

        start_raw = time_range.get("start")
        end_raw = time_range.get("end")
        start_dt = self._parse_datetime(start_raw)
        end_dt = self._parse_datetime(end_raw)

        if start_dt is not None:
            out["month"] = float(start_dt.month)
            out["day_of_year"] = float(start_dt.timetuple().tm_yday)
            out["dayofyear"] = float(start_dt.timetuple().tm_yday)
            out["year"] = float(start_dt.year)

        if start_dt is not None and end_dt is not None:
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
            out["duration_hours"] = duration_hours
            out["window_hours"] = duration_hours

        return out

    # ------------------------------------------------------------------
    # Imputation / normalization / tensor
    # ------------------------------------------------------------------

    def _resize_series(
        self,
        values: list[float | None],
        timesteps: int,
    ) -> tuple[list[float | None], str | None, str]:
        """
        Nếu series không khớp timesteps:
        - đúng độ dài -> real
        - dài hơn -> truncate recent window
        - ngắn hơn -> pad bằng giá trị cuối hợp lệ
        """
        if not values:
            return [], "empty series", "fallback"

        if len(values) == timesteps:
            return values, None, "real"

        if len(values) > timesteps:
            return values[-timesteps:], "truncated series to target timesteps", "fallback"

        # len(values) < timesteps
        last_valid = None
        for v in reversed(values):
            if v is not None:
                last_valid = v
                break

        padded = list(values)
        while len(padded) < timesteps:
            padded.append(last_valid)

        return padded, "padded shorter series using last valid observation", "fallback"

    def _impute_series(
        self,
        spec: dict[str, Any],
        timesteps: int,
    ) -> tuple[list[float], str]:
        name = spec["name"]
        default_value = spec.get("default")
        mean_val = self._stats.get("mean_by_name", {}).get(name)

        if default_value is not None:
            fill = self._safe_float(default_value)
            note = "imputed using feature default"
        elif mean_val is not None:
            fill = self._safe_float(mean_val)
            note = "imputed using training mean"
        else:
            fill = 0.0
            note = "imputed using hard zero fallback"

        if fill is None:
            fill = 0.0

        return [float(fill) for _ in range(timesteps)], note

    def _normalize_series(
        self,
        feature_name: str,
        values: list[float | None],
        do_normalize: bool,
    ) -> tuple[list[float], dict[str, Any]]:
        clean_values = [0.0 if v is None else float(v) for v in values]

        if not self.normalize_enabled or not do_normalize:
            return clean_values, {
                "applied": False,
                "mean": None,
                "std": None,
            }

        mean_val = self._stats.get("mean_by_name", {}).get(feature_name)
        std_val = self._stats.get("std_by_name", {}).get(feature_name)

        if mean_val is None or std_val is None or std_val == 0:
            return clean_values, {
                "applied": False,
                "mean": mean_val,
                "std": std_val,
            }

        normalized = [(float(v) - float(mean_val)) / float(std_val) for v in clean_values]
        return normalized, {
            "applied": True,
            "mean": float(mean_val),
            "std": float(std_val),
        }

    def _to_tensor(self, matrix: list[list[float]]):
        try:
            import torch

            x = torch.tensor(matrix, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return x
        except Exception as e:
            logger.warning("OnlineFeatureBuilder failed to create torch tensor: %s", e)
            return None

    # ------------------------------------------------------------------
    # Timesteps
    # ------------------------------------------------------------------

    def _resolve_timesteps(
        self,
        met_records: list[dict[str, Any]],
        sat_records: list[dict[str, Any]],
        time_range: dict[str, Any],
    ) -> int:
        candidates = [
            len(met_records),
            len(sat_records),
        ]

        explicit_timesteps = time_range.get("timesteps")
        if explicit_timesteps is not None:
            try:
                candidates.append(int(explicit_timesteps))
            except (TypeError, ValueError):
                pass

        max_steps = max(candidates) if any(candidates) else 0
        if max_steps > 0:
            return max_steps

        return self.default_timesteps

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _flatten_mapping(
        self,
        value: Any,
        prefix: str = "",
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        if isinstance(value, dict):
            for k, v in value.items():
                full_key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    out.update(self._flatten_mapping(v, full_key))
                else:
                    out[full_key] = v
                    # thêm leaf key nếu chưa có để dễ match alias
                    leaf = str(k)
                    if leaf not in out:
                        out[leaf] = v

        return out

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
            if f != f:  # NaN
                return None
            return f
        except (TypeError, ValueError):
            return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            raw = raw.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None
        return None

    def _count_non_none(self, values: list[float | None]) -> int:
        return sum(1 for v in values if v is not None)

    def _unique_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for item in items:
            key = item.strip().lower()
            if not key:
                continue
            if key not in seen:
                out.append(item.strip())
                seen.add(key)
        return out