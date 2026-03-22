"""
SeasFireFeatureBuilder — chuyển (location, time_range, met_data) → tensor cho seasfire-ml.

Khi có seasfire dataset (metadata.pt, local.h5...): đọc và tạo tensor.
Khi không: trả về None → SeasFireRunner dùng heuristic fallback.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SEASFIRE_AVAILABLE = False
try:
    seasfire_path = os.environ.get("SEASFIRE_ML_PATH")
    if seasfire_path and Path(seasfire_path).exists():
        if seasfire_path not in sys.path:
            sys.path.insert(0, seasfire_path)
        from utils import GRUDataset, GRUTransform
        _SEASFIRE_AVAILABLE = True
except ImportError:
    GRUDataset = None
    GRUTransform = None


class SeasFireFeatureBuilder:
    """
    Xây dựng features từ location + time_range (+ met_data nếu có).
    Cần seasfire dataset (data/train hoặc data/test) với metadata.pt, local.h5.
    """

    def __init__(
        self,
        dataset_path: str | Path = "",
        timesteps: int = 6,
        target_week: int = 4,
        include_oci: bool = False,
        config: dict | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.timesteps = timesteps
        self.target_week = target_week
        self.include_oci = include_oci
        self.config = config or {}
        self._dataset = None
        self._transform = None

    def _ensure_dataset(self) -> bool:
        """Load GRUDataset nếu có. Trả về True khi thành công."""
        if not _SEASFIRE_AVAILABLE or not self.dataset_path or not self.dataset_path.exists():
            return False
        if self._dataset is not None:
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
            logger.warning("SeasFireFeatureBuilder: dataset load failed %s", e)
            return False

    def build(
        self,
        params: dict[str, Any],
    ) -> Any:
        """
        Tạo tensor từ params (location, time_range).
        params có thể có met_data, spatial_data từ orchestrator.
        Trả về torch.Tensor (1, timesteps, features) hoặc None nếu không có dataset.
        """
        if not self._ensure_dataset():
            return None

        # Cần lat, lon, time để lấy sample từ dataset
        location = params.get("location") or {}
        if isinstance(location, str):
            return None
        lat = location.get("lat") if isinstance(location, dict) else None
        lon = location.get("lon") if isinstance(location, dict) else None
        time_range = params.get("time_range") or {}
        start = time_range.get("start", "")

        if lat is None or lon is None:
            return None

        # Tìm sample gần nhất trong dataset (lat, lon, time)
        try:
            import numpy as np
            samples = self._dataset._samples
            best_idx = 0
            best_dist = float("inf")
            for idx, (slat, slon, stime) in enumerate(samples):
                d = (float(slat) - lat) ** 2 + (float(slon) - lon) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            x, y = self._dataset.get(best_idx)
            x_t, y_t = self._transform((x, y))
            return x_t.unsqueeze(0)  # (1, timesteps, features)
        except Exception as e:
            logger.warning("SeasFireFeatureBuilder build failed: %s", e)
            return None
