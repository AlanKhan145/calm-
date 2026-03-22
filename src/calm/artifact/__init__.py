"""
Artifact layer — lưu predictions/features từ SeasFire, agent đọc lại thay vì gọi model lặp.

- PredictionArtifactStore: cache predictions.parquet theo location + week + model_version
- SeasFireRunner: batch inference, lưu output
"""

from calm.artifact.prediction_store import PredictionArtifactStore
from calm.artifact.seasfire_runner import SeasFireRunner

__all__ = ["PredictionArtifactStore", "SeasFireRunner"]
