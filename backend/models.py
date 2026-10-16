from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionModel(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(24), default="CREATED")
    mode: Mapped[str] = mapped_column(String(24), default="UNPAIRED")
    initial_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sources: Mapped[list["CameraSourceModel"]] = relationship(cascade="all, delete-orphan")
    events: Mapped[list["CountEventModel"]] = relationship(cascade="all, delete-orphan")


class CameraSourceModel(Base):
    __tablename__ = "camera_sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    camera: Mapped[str] = mapped_column(String(16))
    source_type: Mapped[str] = mapped_column(String(16))
    uri: Mapped[str] = mapped_column(Text)
    provenance: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)


class CountEventModel(Base):
    __tablename__ = "count_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    camera: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    track_id: Mapped[str] = mapped_column(String(80))
    timestamp_s: Mapped[float] = mapped_column(Float)
    frame_index: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)
    crop_path: Mapped[str | None] = mapped_column(Text)
    passenger_id: Mapped[str | None] = mapped_column(String(40))
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PassengerTrackModel(Base):
    __tablename__ = "passenger_tracks"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    passenger_id: Mapped[str] = mapped_column(String(40))
    state: Mapped[str] = mapped_column(String(24), default="ONBOARD")
    in_event_id: Mapped[int | None] = mapped_column(ForeignKey("count_events.id"))
    out_event_id: Mapped[int | None] = mapped_column(ForeignKey("count_events.id"))


class ReIDMatchModel(Base):
    __tablename__ = "reid_matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    in_event_id: Mapped[int | None] = mapped_column(ForeignKey("count_events.id"))
    out_event_id: Mapped[int] = mapped_column(ForeignKey("count_events.id"))
    similarity: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(Text)
    reviewer_note: Mapped[str | None] = mapped_column(Text)


class CameraCalibrationModel(Base):
    __tablename__ = "camera_calibrations"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera: Mapped[str] = mapped_column(String(16), index=True)
    zone_a: Mapped[list] = mapped_column(JSON)
    zone_b: Mapped[list] = mapped_column(JSON)
    event_direction: Mapped[str] = mapped_column(String(8))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationReportModel(Base):
    __tablename__ = "evaluation_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("sessions.id"))
    dataset: Mapped[str] = mapped_column(String(80))
    metrics: Mapped[dict] = mapped_column(JSON)
    report_kind: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

