"""
SeasFireModelRunner — adapter predict(params) -> SeasFireRunner.predict_batch().

Dùng cho PredictionReasoningAgent.

Điểm chính:
- Hỗ trợ 2 nguồn feature:
  1) online_feature_builder
  2) offline_feature_builder
- Trả rõ trạng thái feature_status:
  - online_ready
  - offline_ready
  - missing
- Trả prediction metadata để tầng trên biết:
  - feature lấy từ đâu
  - đã thử builder nào
  - có đang fallback heuristic hay không
"""

from __future__ import annotations

import logging
from typing import Any

from calm.artifact.seasfire_runner import SeasFireRunner

logger = logging.getLogger(__name__)


class SeasFireModelRunner:
    """
    Adapter: predict(parameters) -> gọi SeasFireRunner.predict_batch().

    parameters có thể chứa:
    - location: {lat, lon}
    - time_range: {start, end}
    - met_data
    - spatial_data

    Luồng build feature:
    1) thử online_feature_builder trước
    2) nếu không có feature thì thử offline_feature_builder
    3) nếu vẫn không có thì runner tự fallback heuristic từ params
    """

    def __init__(
        self,
        seasfire_runner: SeasFireRunner,
        feature_builder=None,
        online_feature_builder=None,
        offline_feature_builder=None,
    ) -> None:
        self.runner = seasfire_runner

        # Backward compatibility:
        # code cũ chỉ truyền feature_builder thì mặc định xem như online builder.
        if feature_builder is not None and online_feature_builder is None and offline_feature_builder is None:
            online_feature_builder = feature_builder

        self.online_feature_builder = online_feature_builder
        self.offline_feature_builder = offline_feature_builder

        # Giữ lại alias cũ nếu nơi khác còn truy cập self.feature_builder
        self.feature_builder = feature_builder or online_feature_builder or offline_feature_builder

    def _builder_name(self, builder: Any) -> str:
        return getattr(builder, "__class__", type(builder)).__name__

    def _normalize_build_output(
        self,
        built: Any,
        source: str,
        builder_name: str,
    ) -> tuple[Any, str, dict[str, Any]]:
        """
        Cho phép builder trả về 2 kiểu:
        - raw features
        - dict có dạng:
          {
              "features": ...,
              "feature_status": "...",
              "metadata": {...}
          }
        """
        metadata: dict[str, Any] = {
            "builder_name": builder_name,
            "feature_source": source,
        }

        if built is None:
            return None, "missing", metadata

        if isinstance(built, dict) and "features" in built:
            features = built.get("features")
            metadata.update(built.get("metadata") or {})

            # builder có thể tự báo status, nhưng nếu thiếu thì chuẩn hoá theo source
            feature_status = built.get("feature_status")
            if feature_status not in {"online_ready", "offline_ready", "missing"}:
                feature_status = "online_ready" if source == "online" else "offline_ready"

            metadata.setdefault("builder_reported_status", feature_status)
            return features, feature_status if features is not None else "missing", metadata

        # builder trả raw features
        feature_status = "online_ready" if source == "online" else "offline_ready"
        return built, feature_status, metadata

    def _try_build_with_source(
        self,
        builder: Any,
        parameters: dict[str, Any],
        source: str,
    ) -> tuple[Any, str, dict[str, Any]]:
        builder_name = self._builder_name(builder)

        try:
            built = builder.build(parameters)
            features, feature_status, metadata = self._normalize_build_output(
                built=built,
                source=source,
                builder_name=builder_name,
            )
            metadata["success"] = features is not None
            return features, feature_status, metadata
        except Exception as e:
            logger.warning(
                "SeasFireModelRunner: %s feature build failed: %s",
                source,
                e,
            )
            return None, "missing", {
                "builder_name": builder_name,
                "feature_source": source,
                "success": False,
                "error": str(e),
            }

    def _resolve_features(
        self,
        parameters: dict[str, Any],
    ) -> tuple[Any, str, str | None, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []

        # 1) ưu tiên online builder
        if self.online_feature_builder is not None:
            features, feature_status, metadata = self._try_build_with_source(
                self.online_feature_builder,
                parameters,
                source="online",
            )
            attempts.append(metadata)
            if features is not None:
                prediction_metadata = {
                    "feature_status": feature_status,
                    "feature_source": "online",
                    "feature_attempts": attempts,
                }
                return features, feature_status, "online", prediction_metadata

        # 2) fallback sang offline builder
        if self.offline_feature_builder is not None:
            features, feature_status, metadata = self._try_build_with_source(
                self.offline_feature_builder,
                parameters,
                source="offline",
            )
            attempts.append(metadata)
            if features is not None:
                prediction_metadata = {
                    "feature_status": feature_status,
                    "feature_source": "offline",
                    "feature_attempts": attempts,
                }
                return features, feature_status, "offline", prediction_metadata

        # 3) không có feature
        prediction_metadata = {
            "feature_status": "missing",
            "feature_source": None,
            "feature_attempts": attempts,
        }
        return None, "missing", None, prediction_metadata

    def predict(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Chạy dự đoán và trả rõ metadata về feature.

        Trả thêm các field:
        - feature_status: online_ready | offline_ready | missing
        - feature_source: online | offline | None
        - prediction_metadata: metadata chi tiết cho tầng trên
        """
        features, feature_status, feature_source, prediction_metadata = self._resolve_features(parameters)

        # Theo comment hệ thống hiện tại:
        # nếu features=None thì SeasFireRunner sẽ fallback heuristic từ met_data/params
        prediction_metadata.update(
            {
                "runner_name": self.runner.__class__.__name__,
                "prediction_mode": "model" if features is not None else "heuristic_fallback",
                "used_runner_fallback": features is None,
                "has_met_data": bool(parameters.get("met_data")),
                "has_spatial_data": bool(parameters.get("spatial_data")),
            }
        )

        raw_result = self.runner.predict_batch(features, parameters)

        # Chuẩn hoá output để chắc chắn luôn là dict
        if not isinstance(raw_result, dict):
            raw_result = {"result": raw_result}

        # Merge metadata cũ nếu runner đã trả prediction_metadata
        existing_prediction_metadata = raw_result.get("prediction_metadata") or {}
        if not isinstance(existing_prediction_metadata, dict):
            existing_prediction_metadata = {"runner_prediction_metadata": existing_prediction_metadata}

        raw_result["feature_status"] = feature_status
        raw_result["feature_source"] = feature_source
        raw_result["prediction_metadata"] = {
            **existing_prediction_metadata,
            **prediction_metadata,
        }

        return raw_result