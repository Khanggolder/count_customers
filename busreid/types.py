from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import numpy as np


class CameraRole(StrEnum):
    FRONT = "front"
    REAR = "rear"


class Direction(StrEnum):
    IN = "IN"
    OUT = "OUT"


class CrossingDirection(StrEnum):
    A_TO_B = "A_TO_B"
    B_TO_A = "B_TO_A"


@dataclass(slots=True)
class Detection:
    track_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))

    def anchor(self, mode: str = "bottom_center") -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        if mode == "center":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if mode != "bottom_center":
            raise ValueError(f"Unsupported anchor mode: {mode}")
        return self.bottom_center


@dataclass(slots=True)
class CrossingEvent:
    camera: str
    track_id: str
    crossing_direction: CrossingDirection
    direction: Direction
    occupancy_delta: int
    frame_index: int
    timestamp_s: float
    confidence: float | None = None
    crop_path: str | None = None
    synthetic: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capture_time_utc: datetime | None = None

    @property
    def delta(self) -> int:
        """Compatibility alias: delta means occupancy change, never polygon direction."""
        return self.occupancy_delta


@dataclass(slots=True)
class TrackCrop:
    track_id: str
    image: np.ndarray
    quality: float
    frame_index: int


JSONDict = dict[str, Any]
