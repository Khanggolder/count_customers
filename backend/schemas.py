from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TripCreate(BaseModel):
    trip_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    route_id: str = Field(min_length=1, max_length=64)
    vehicle_id: str = Field(min_length=1, max_length=80)
    start_time_utc: datetime
    mode: Literal["PAIRED_VERIFIED", "UNPAIRED", "ILLUSTRATIVE_COMPOSITE", "SYNTHETIC_PAIRED"] = "UNPAIRED"
    initial_occupancy: int = Field(default=0, ge=0, le=500)

    @field_validator("start_time_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Timestamp phải có timezone")
        return value


class TelemetryCreate(BaseModel):
    event_time_utc: datetime
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    speed_kmh: float | None = Field(default=None, ge=0, le=250)
    heading: float | None = Field(default=None, ge=0, lt=360)
    door_front: bool | None = None
    door_rear: bool | None = None
    source: Literal["SIMULATED", "CSV", "JSON", "SENSOR", "OCR_OVERLAY", "MANUAL"] = "JSON"
    quality: Literal["GOOD", "NOISY", "STALE", "MISSING"] = "GOOD"

    @field_validator("event_time_utc")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Timestamp phải có timezone")
        return value


class AssignmentPatch(BaseModel):
    stop_visit_id: str | None = Field(default=None, max_length=100)
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=3, max_length=1000)


class PassengerEventCreate(BaseModel):
    event_uuid: str = Field(min_length=8, max_length=100)
    source_id: str = Field(min_length=1, max_length=100)
    camera: Literal["FRONT", "REAR"]
    direction: Literal["IN", "OUT"]
    track_id: str = Field(min_length=1, max_length=100)
    video_frame: int | None = Field(default=None, ge=0)
    source_pts: float | None = Field(default=None, ge=0)
    event_time_utc: datetime
    source_type: Literal["REAL_VIDEO_AI", "REAL_VIDEO_AI_PROCESSING_TIME", "SIMULATED_EVENT_GENERATOR"]
    source_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_path: str | None = Field(default=None, max_length=1000)
    provenance: Literal["REAL_VIDEO_AI", "SIMULATED_EVENTS"]

    @field_validator("event_time_utc")
    @classmethod
    def event_timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Timestamp phải có timezone")
        return value


class RunCreate(BaseModel):
    trip_id: str | None = None
    detail: dict = Field(default_factory=dict)
