"""
PredictionArtifactStore — cache predictions theo location + time + model_version.

Agent chỉ đọc lát cắt cần thiết, không gọi model lặp.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PredictionArtifactStore:
    """
    Lưu/đọc predictions từ batch inference.
    Key: task_type + location + start_time + end_time + model_version.
    """

    def __init__(self, base_path: str | Path = ".artifact/predictions", config: dict | None = None) -> None:
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

    def _key(self, params: dict[str, Any]) -> str:
        """Tạo key dedup từ params."""
        loc = params.get("location", {})
        tr = params.get("time_range", {})
        model = params.get("model", params.get("model_version", ""))
        parts = [
            str(loc.get("lat", "")),
            str(loc.get("lon", "")),
            str(tr.get("start", "")),
            str(tr.get("end", "")),
            str(model),
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Đọc prediction từ cache nếu có."""
        k = self._key(params)
        path = self.base / f"{k}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("ArtifactStore get failed: %s", e)
            return None

    def put(self, params: dict[str, Any], prediction: dict[str, Any]) -> None:
        """Lưu prediction vào cache."""
        k = self._key(params)
        path = self.base / f"{k}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prediction, default=str), encoding="utf-8")
