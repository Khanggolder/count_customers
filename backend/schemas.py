from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceCreate(BaseModel):
    camera: Literal["front", "rear"]
    source_type: Literal["mp4", "rtsp", "webcam"] = "mp4"
    uri: str = Field(min_length=1, max_length=2048)
    provenance: str | None = Field(default=None, max_length=1000)
    verified: bool = False

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("URI không hợp lệ")
        return value


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    mode: Literal["UNPAIRED", "PAIRED"] = "UNPAIRED"
    initial_occupancy: int = Field(default=0, ge=0, le=500)
    sources: list[SourceCreate] = Field(default_factory=list, max_length=2)


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    status: str
    mode: str
    initial_occupancy: int
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None


class EventCreate(BaseModel):
    camera: Literal["front", "rear"]
    direction: Literal["IN", "OUT"]
    track_id: str = Field(min_length=1, max_length=80)
    timestamp_s: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    crop_path: str | None = None
    synthetic: bool = False


class EventRead(EventCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    passenger_id: str | None
    created_at: datetime


class MatchPatch(BaseModel):
    status: Literal["CONFIRMED", "REJECTED", "UNKNOWN"]
    reviewer_note: str | None = Field(default=None, max_length=1000)


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    in_event_id: int | None
    out_event_id: int
    similarity: float | None
    status: str
    reason: str
    reviewer_note: str | None

