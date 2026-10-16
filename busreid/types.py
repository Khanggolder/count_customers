from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class CameraRole(StrEnum):
    FRONT = "front"
    REAR = "rear"


class Direction(StrEnum):
    IN = "IN"
    OUT = "OUT"


class MatchStatus(StrEnum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    UNVERIFIED = "UNVERIFIED"
    REJECTED = "REJECTED"


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


@dataclass(slots=True)
class CrossingEvent:
    camera: str
    track_id: str
    direction: Direction
    frame_index: int
    timestamp_s: float
    confidence: float | None = None
    crop_path: str | None = None
    synthetic: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class TrackCrop:
    track_id: str
    image: np.ndarray
    quality: float
    frame_index: int


@dataclass(slots=True)
class EmbeddingRecord:
    event_id: int | None
    track_id: str
    timestamp_s: float
    embedding: np.ndarray
    passenger_id: str | None = None
    crop_path: Path | None = None


@dataclass(slots=True)
class MatchDecision:
    out_event_id: int | None
    in_event_id: int | None
    status: MatchStatus
    similarity: float | None
    reason: str
    passenger_id: str | None = None


JSONDict = dict[str, Any]

