"""
SeasFireModelRunner — adapter predict(params) → SeasFireRunner.predict_batch().

Dùng cho PredictionReasoningAgent. Gọi feature_builder.build() rồi runner.predict_batch().
Khi không có features: runner dùng heuristic từ met_data trong params.
"""

from __future__ import annotations

import logging
from typing import Any

from calm.artifact.seasfire_runner import SeasFireRunner

logger = logging.getLogger(__name__)


class SeasFireModelRunner:
    """
    Adapter: predict(parameters) → gọi SeasFireRunner.predict_batch().
    parameters có location, time_range; có thể có met_data, spatial_data từ orchestrator.
    """

    def __init__(
        self,
        seasfire_runner: SeasFireRunner,
        feature_builder=None,
    ) -> None:
        self.runner = seasfire_runner
        self.feature_builder = feature_builder

    def predict(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Chạy dự đoán. parameters từ orchestrator có thể có:
        - location: {lat, lon}
        - time_range: {start, end}
        - met_data: từ GEE/CDS (temperature, humidity...)
        - spatial_data: từ GEE
        """
        features = None
        if self.feature_builder:
            try:
                features = self.feature_builder.build(parameters)
            except Exception as e:
                logger.warning("SeasFireModelRunner: feature build failed %s", e)

        return self.runner.predict_batch(features, parameters)
