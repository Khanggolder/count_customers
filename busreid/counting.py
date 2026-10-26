from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from .business import expected_direction, occupancy_change
from .types import CrossingDirection, CrossingEvent, Detection, Direction


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
    last_crossing_sign: int = 0
    seen_zones: list[Zone] = field(default_factory=list)
    last_seen_frame: int = -1
    last_point: tuple[float, float] | None = None


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
        state_ttl_frames: int = 90,
        anchor_mode: str = "bottom_center",
        id_switch_grace_frames: int = 8,
        id_switch_max_distance: float = 0.35,
    ) -> None:
        if min_stable_frames < 1:
            raise ValueError("min_stable_frames phải >= 1")
        if forward_direction != expected_direction(camera, CrossingDirection.A_TO_B):
            raise ValueError(f"Camera {camera} A_TO_B direction conflicts with calibration")
        self.camera = camera
        self.zone_a_normalized = zone_a
        self.zone_b_normalized = zone_b
        self.forward_direction = forward_direction
        self.min_stable_frames = min_stable_frames
        self.debounce_frames = debounce_frames
        self.state_ttl_frames = state_ttl_frames
        self.anchor_mode = anchor_mode
        self.id_switch_grace_frames = id_switch_grace_frames
        self.id_switch_max_distance = id_switch_max_distance
        self.states: dict[str, TrackState] = {}

    def _recover_id_switch(
        self, track_id: str, point: tuple[float, float], frame_shape: tuple[int, ...], frame_index: int
    ) -> TrackState:
        height, width = frame_shape[:2]
        normalized = (point[0] / width, point[1] / height)
        candidates: list[tuple[float, str, TrackState]] = []
        for old_id, state in self.states.items():
            if state.last_point is None or state.last_seen_frame >= frame_index:
                continue
            age = frame_index - state.last_seen_frame
            distance = (
                (state.last_point[0] - normalized[0]) ** 2 + (state.last_point[1] - normalized[1]) ** 2
            ) ** 0.5
            if age <= self.id_switch_grace_frames and distance <= self.id_switch_max_distance:
                candidates.append((distance, old_id, state))
        if len(candidates) == 1:
            _, old_id, state = candidates[0]
            del self.states[old_id]
            self.states[track_id] = state
            return state
        state = TrackState()
        self.states[track_id] = state
        return state

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
        point = detection.anchor(self.anchor_mode)
        zone = self._zone(point, frame_shape)
        state = self.states.get(detection.track_id)
        if state is None:
            state = self._recover_id_switch(detection.track_id, point, frame_shape, frame_index)
        height, width = frame_shape[:2]
        state.last_seen_frame = frame_index
        state.last_point = (point[0] / width, point[1] / height)
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
        crossing = CrossingDirection.A_TO_B if origin == Zone.A and zone == Zone.B else CrossingDirection.B_TO_A
        crossing_sign = 1 if crossing == CrossingDirection.A_TO_B else -1
        if crossing_sign == state.last_crossing_sign and frame_index - state.last_event_frame < self.debounce_frames:
            return None

        state.last_event_frame = frame_index
        state.last_crossing_sign = crossing_sign
        direction = expected_direction(self.camera, crossing)
        return CrossingEvent(
            camera=self.camera,
            track_id=detection.track_id,
            crossing_direction=crossing,
            direction=direction,
            occupancy_delta=occupancy_change(direction),
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            confidence=detection.confidence,
        )

    def prune(self, active_track_ids: set[str]) -> None:
        self.states = {key: value for key, value in self.states.items() if key in active_track_ids}

    def prune_stale(self, frame_index: int) -> None:
        self.states = {
            key: value
            for key, value in self.states.items()
            if frame_index - value.last_seen_frame <= self.state_ttl_frames
        }
