from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class StopState(StrEnum):
    MOVING = "MOVING"
    APPROACHING = "APPROACHING"
    DWELL_CANDIDATE = "DWELL_CANDIDATE"
    STOP_CONFIRMED = "STOP_CONFIRMED"
    DEPARTING = "DEPARTING"
    CLOSED = "CLOSED"


@dataclass(slots=True)
class StopPoint:
    stop_id: str
    sequence: int
    latitude: float
    longitude: float
    radius_m: float = 50.0


@dataclass(slots=True)
class TelemetryPoint:
    event_time_utc: datetime
    latitude: float | None
    longitude: float | None
    speed_kmh: float | None
    door_front: bool | None
    door_rear: bool | None
    quality: str = "GOOD"

    @property
    def any_door_open(self) -> bool | None:
        values = (self.door_front, self.door_rear)
        if any(value is True for value in values):
            return True
        if all(value is False for value in values):
            return False
        return None


@dataclass(slots=True)
class DetectedVisit:
    stop_id: str
    arrived_at: datetime
    departed_at: datetime | None
    status: str
    confidence: float
    evidence: dict = field(default_factory=dict)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class StopVisitDetector:
    """Sequential geofence/speed/door detector with dwell and departure hysteresis."""

    def __init__(
        self,
        stops: list[StopPoint],
        speed_threshold_kmh: float = 3.0,
        dwell_seconds: float = 3.0,
        exit_radius_factor: float = 1.35,
        stale_after_seconds: float = 5.0,
        allow_doorless_uncertain: bool = True,
    ) -> None:
        self.stops = sorted(stops, key=lambda item: item.sequence)
        self.speed_threshold_kmh = speed_threshold_kmh
        self.dwell_seconds = dwell_seconds
        self.exit_radius_factor = exit_radius_factor
        self.stale_after_seconds = stale_after_seconds
        self.allow_doorless_uncertain = allow_doorless_uncertain

    def detect(self, samples: list[TelemetryPoint]) -> list[DetectedVisit]:
        visits: list[DetectedVisit] = []
        next_stop_index = 0
        candidate_started: datetime | None = None
        candidate_min_distance = float("inf")
        candidate_door: bool | None = None
        active: DetectedVisit | None = None
        previous_time: datetime | None = None

        for sample in sorted(samples, key=lambda item: item.event_time_utc):
            if (
                previous_time
                and (sample.event_time_utc - previous_time).total_seconds() > self.stale_after_seconds
            ):
                candidate_started = None
            previous_time = sample.event_time_utc
            if next_stop_index >= len(self.stops):
                break
            stop = self.stops[next_stop_index]
            valid_position = (
                sample.latitude is not None
                and sample.longitude is not None
                and sample.quality not in {"STALE", "MISSING"}
            )
            distance = (
                haversine_m(sample.latitude, sample.longitude, stop.latitude, stop.longitude)
                if valid_position
                else float("inf")
            )
            near = distance <= stop.radius_m
            slow = sample.speed_kmh is not None and sample.speed_kmh <= self.speed_threshold_kmh
            door = sample.any_door_open

            if active is not None:
                departed = (
                    valid_position
                    and distance > stop.radius_m * self.exit_radius_factor
                    and (sample.speed_kmh or 0) > self.speed_threshold_kmh
                    and door is not True
                )
                if departed:
                    active.departed_at = sample.event_time_utc
                    active.evidence["departure_distance_m"] = round(distance, 2)
                    visits.append(active)
                    active = None
                    next_stop_index += 1
                    candidate_started = None
                continue

            if near and slow:
                if candidate_started is None:
                    candidate_started = sample.event_time_utc
                    candidate_min_distance = distance
                    candidate_door = door
                else:
                    candidate_min_distance = min(candidate_min_distance, distance)
                    if door is True:
                        candidate_door = True
                dwell = (sample.event_time_utc - candidate_started).total_seconds()
                door_evidence = candidate_door is True
                door_missing = candidate_door is None
                if dwell >= self.dwell_seconds and (
                    door_evidence or (door_missing and self.allow_doorless_uncertain)
                ):
                    status = "CONFIRMED" if door_evidence else "UNCERTAIN"
                    confidence = 0.98 if door_evidence else 0.55
                    active = DetectedVisit(
                        stop_id=stop.stop_id,
                        arrived_at=candidate_started,
                        departed_at=None,
                        status=status,
                        confidence=confidence,
                        evidence={
                            "gps_near": True,
                            "min_distance_m": round(candidate_min_distance, 2),
                            "speed_below_kmh": self.speed_threshold_kmh,
                            "dwell_seconds": dwell,
                            "door_open": candidate_door,
                            "quality": sample.quality,
                        },
                    )
            else:
                candidate_started = None
                candidate_min_distance = float("inf")
                candidate_door = None

        if active is not None:
            active.evidence["warning"] = "Trip ended before departure was observed"
            visits.append(active)
        return visits


def assign_visit(
    event_time: datetime,
    visits: list[DetectedVisit],
    grace_before_s: float = 1.0,
    grace_after_s: float = 2.0,
) -> str | None:
    candidates: list[tuple[float, str]] = []
    for visit in visits:
        if visit.status != "CONFIRMED" or visit.departed_at is None:
            continue
        start = visit.arrived_at.timestamp() - grace_before_s
        end = visit.departed_at.timestamp() + grace_after_s
        stamp = event_time.timestamp()
        if start <= stamp <= end:
            midpoint = (start + end) / 2
            candidates.append((abs(stamp - midpoint), visit.stop_id))
    return min(candidates)[1] if candidates else None
