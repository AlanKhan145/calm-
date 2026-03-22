"""
Artifact layer — lưu predictions/features từ SeasFire, agent đọc lại thay vì gọi model lặp.

- PredictionArtifactStore: cache predictions theo location + week
- SeasFireRunner: load checkpoint, batch inference, heuristic fallback
- SeasFireFeatureBuilder: (lat, lon, time) → tensor từ seasfire dataset
- SeasFireModelRunner: adapter predict(params) cho PredictionReasoningAgent
"""

from calm.artifact.prediction_store import PredictionArtifactStore
from calm.artifact.seasfire_runner import SeasFireRunner
from calm.artifact.feature_builder import SeasFireFeatureBuilder
from calm.artifact.model_runner import SeasFireModelRunner

__all__ = [
    "PredictionArtifactStore",
    "SeasFireRunner",
    "SeasFireFeatureBuilder",
    "SeasFireModelRunner",
]
