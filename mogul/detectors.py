"""Replaceable detector interfaces; no fabricated AI predictions."""
from dataclasses import dataclass, field
from typing import Protocol, Sequence
import numpy as np


@dataclass
class DetectionResult:
    success: bool
    message: str
    polygons: list = field(default_factory=list)
    confidence: float | None = None
    model: str | None = None


class RegionBackend(Protocol):
    def detect(self, frames: Sequence[np.ndarray], **context) -> DetectionResult: ...


class MiddlePartDetector:
    def __init__(self, backend: RegionBackend | None = None):
        self.backend = backend

    def detect(self, frames):
        if self.backend is None:
            return DetectionResult(False, '自動検出不可：ミドルパート認識モデル未導入。映像上をドラッグして手動設定してください。')
        return self.backend.detect(frames)


class LaneDetector:
    def __init__(self, backend: RegionBackend | None = None):
        self.backend = backend

    def detect(self, frames, middle, count):
        if self.backend is None:
            return DetectionResult(False, '自動検出不可：レーン認識モデル未導入。「手動均等分割」で初期配置し補正してください。')
        return self.backend.detect(frames, middle=middle, count=count)
