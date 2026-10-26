from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RouteModel(Base):
    __tablename__ = "routes"
    route_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    direction: Mapped[str] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(Text)
    retrieved_date: Mapped[str] = mapped_column(String(10))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    stops: Mapped[list["StopModel"]] = relationship(
        cascade="all, delete-orphan", order_by="StopModel.sequence"
    )


class StopModel(Base):
    __tablename__ = "stops"
    stop_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.route_id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    radius_m: Mapped[float] = mapped_column(Float, default=50.0)
    coordinate_source: Mapped[str] = mapped_column(Text)
    name_source_url: Mapped[str] = mapped_column(Text)
    retrieved_date: Mapped[str] = mapped_column(String(10))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    __table_args__ = (UniqueConstraint("route_id", "sequence", name="uq_stop_route_sequence"),)


class TripModel(Base):
    __tablename__ = "trips"
    trip_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    route_id: Mapped[str] = mapped_column(ForeignKey("routes.route_id"), index=True)
    vehicle_id: Mapped[str] = mapped_column(String(80))
    start_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    mode: Mapped[str] = mapped_column(String(40), default="UNPAIRED")
    initial_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="CREATED")
    coverage_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    provenance: Mapped[str] = mapped_column(String(50), default="SIMULATED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CameraSourceModel(Base):
    __tablename__ = "camera_sources"
    source_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    door_role: Mapped[str] = mapped_column(String(16))
    input_type: Mapped[str] = mapped_column(String(16))
    safe_uri: Mapped[str] = mapped_column(Text)
    calibration_id: Mapped[str | None] = mapped_column(String(80))
    paired_flag: Mapped[str] = mapped_column(String(30), default="UNPAIRED")
    time_offset_ms: Mapped[int] = mapped_column(Integer, default=0)


class TelemetrySampleModel(Base):
    __tablename__ = "telemetry_samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    event_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    heading: Mapped[float | None] = mapped_column(Float)
    door_front: Mapped[bool | None] = mapped_column(Boolean)
    door_rear: Mapped[bool | None] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(40))
    quality: Mapped[str] = mapped_column(String(30), default="GOOD")
    __table_args__ = (UniqueConstraint("trip_id", "event_time_utc", name="uq_telemetry_trip_time"),)


class StopVisitModel(Base):
    __tablename__ = "stop_visits"
    stop_visit_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    stop_id: Mapped[str] = mapped_column(ForeignKey("stops.stop_id"), index=True)
    arrived_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    departed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30))
    evidence: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    override_audit: Mapped[list] = mapped_column(JSON, default=list)
    __table_args__ = (UniqueConstraint("trip_id", "stop_id", "arrived_at", name="uq_visit_trip_stop_time"),)


class PassengerEventModel(Base):
    __tablename__ = "passenger_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    event_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    source_id: Mapped[str] = mapped_column(String(100), index=True)
    camera: Mapped[str] = mapped_column(String(16))
    crossing_direction: Mapped[str | None] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    occupancy_delta: Mapped[int | None] = mapped_column(Integer)
    direction_status: Mapped[str] = mapped_column(String(32), default="SOURCE_ASSERTED")
    track_id: Mapped[str] = mapped_column(String(100))
    video_frame: Mapped[int | None] = mapped_column(Integer)
    source_pts: Mapped[float | None] = mapped_column(Float)
    event_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source_type: Mapped[str] = mapped_column(String(24))
    source_confidence: Mapped[float | None] = mapped_column(Float)
    stop_visit_id: Mapped[str | None] = mapped_column(ForeignKey("stop_visits.stop_visit_id"), index=True)
    assignment_status: Mapped[str] = mapped_column(String(30), default="UNASSIGNED")
    evidence_path: Mapped[str | None] = mapped_column(Text)
    provenance: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewAuditModel(Base):
    __tablename__ = "review_audits"
    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str] = mapped_column(String(100))
    record_id: Mapped[str] = mapped_column(String(100), index=True)
    before: Mapped[dict] = mapped_column(JSON)
    after: Mapped[dict] = mapped_column(JSON)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reason: Mapped[str] = mapped_column(Text)


class RunModel(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    trip_id: Mapped[str | None] = mapped_column(ForeignKey("trips.trip_id"), index=True)
    status: Mapped[str] = mapped_column(String(30), default="QUEUED")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


Index(
    "ix_passenger_trip_stop_time",
    PassengerEventModel.trip_id,
    PassengerEventModel.stop_visit_id,
    PassengerEventModel.event_time_utc,
)
Index("ix_visit_trip_time", StopVisitModel.trip_id, StopVisitModel.arrived_at)
