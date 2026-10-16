from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from .types import CrossingEvent, Detection, Direction


class Zone(StrEnum):
    A = "A"
    B = "B"
    NONE = "NONE"


@dataclass(slots=True)
class TrackState:
    stable_zone: Zone = Zone.NONE
    candidate_zone: Zone = Zone.NONE
    candidate_frames: int = 0
    origin_zone: Zone = Zone.NONE
    last_event_frame: int = -1_000_000
    seen_zones: list[Zone] = field(default_factory=list)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class DirectionCounter:
    """Debounced A<->B crossing state machine, scoped to exactly one camera."""

    def __init__(
        self,
        camera: str,
        zone_a: list[tuple[float, float]],
        zone_b: list[tuple[float, float]],
        forward_direction: Direction,
        min_stable_frames: int = 2,
        debounce_frames: int = 12,
    ) -> None:
        if min_stable_frames < 1:
            raise ValueError("min_stable_frames phải >= 1")
        self.camera = camera
        self.zone_a_normalized = zone_a
        self.zone_b_normalized = zone_b
        self.forward_direction = forward_direction
        self.min_stable_frames = min_stable_frames
        self.debounce_frames = debounce_frames
        self.states: dict[str, TrackState] = {}

    def _zone(self, point: tuple[float, float], frame_shape: tuple[int, ...]) -> Zone:
        height, width = frame_shape[:2]
        normalized = (point[0] / width, point[1] / height)
        if point_in_polygon(normalized, self.zone_a_normalized):
            return Zone.A
        if point_in_polygon(normalized, self.zone_b_normalized):
            return Zone.B
        return Zone.NONE

    def update(
        self,
        detection: Detection,
        frame_shape: tuple[int, ...],
        frame_index: int,
        timestamp_s: float,
    ) -> CrossingEvent | None:
        zone = self._zone(detection.bottom_center, frame_shape)
        state = self.states.setdefault(detection.track_id, TrackState())
        if zone != state.candidate_zone:
            state.candidate_zone = zone
            state.candidate_frames = 1
        else:
            state.candidate_frames += 1
        if state.candidate_frames < self.min_stable_frames or zone == state.stable_zone:
            return None

        previous = state.stable_zone
        state.stable_zone = zone
        if zone != Zone.NONE:
            state.seen_zones.append(zone)
            state.seen_zones = state.seen_zones[-4:]
        if previous in (Zone.A, Zone.B) and zone == Zone.NONE:
            state.origin_zone = previous
            return None

        origin = state.origin_zone if previous == Zone.NONE else previous
        if origin not in (Zone.A, Zone.B) or zone not in (Zone.A, Zone.B) or origin == zone:
            if zone in (Zone.A, Zone.B):
                state.origin_zone = zone
            return None
        state.origin_zone = zone
        if frame_index - state.last_event_frame < self.debounce_frames:
            return None

        forward = origin == Zone.A and zone == Zone.B
        direction = self.forward_direction if forward else (
            Direction.OUT if self.forward_direction == Direction.IN else Direction.IN
        )
        state.last_event_frame = frame_index
        return CrossingEvent(
            camera=self.camera,
            track_id=detection.track_id,
            direction=direction,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            confidence=detection.confidence,
        )

    def prune(self, active_track_ids: set[str]) -> None:
        self.states = {key: value for key, value in self.states.items() if key in active_track_ids}

