from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import (
    CameraSourceModel,
    PassengerEventModel,
    RouteModel,
    StopModel,
    TelemetrySampleModel,
    TripModel,
)
from busreid.service import detect_and_assign, validate_event_source


def test_real_event_needs_matching_source_capture_clock_and_door_evidence(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'assignment.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with factory() as db:
        db.add(
            RouteModel(
                route_id="R",
                name="R",
                direction="R",
                source="TEST",
                retrieved_date="2026-01-01",
                verified=False,
            )
        )
        db.add(
            StopModel(
                stop_id="S",
                route_id="R",
                sequence=1,
                name="S",
                latitude=10,
                longitude=106,
                radius_m=50,
                coordinate_source="TEST",
                name_source_url="TEST",
                retrieved_date="2026-01-01",
                verified=False,
            )
        )
        for trip_id in ("REAL", "PUBLIC"):
            db.add(
                TripModel(
                    trip_id=trip_id,
                    route_id="R",
                    vehicle_id="V",
                    start_time_utc=start,
                    end_time_utc=None,
                    mode="UNPAIRED",
                    initial_occupancy=0,
                    status="COMPLETED",
                    coverage_complete=False,
                    provenance="REAL_VIDEO_AI",
                )
            )
            db.add(
                CameraSourceModel(
                    source_id=f"{trip_id}:front",
                    trip_id=trip_id,
                    door_role="FRONT",
                    input_type="MP4",
                    safe_uri="fixture.mp4",
                    calibration_id="test",
                    paired_flag="SYNCHRONIZED",
                    time_offset_ms=0,
                )
            )
        db.flush()
        for second in range(7):
            db.add(
                TelemetrySampleModel(
                    trip_id="REAL",
                    event_time_utc=start + timedelta(seconds=second),
                    latitude=10 if second < 6 else 10.001,
                    longitude=106,
                    speed_kmh=0 if second < 6 else 15,
                    heading=None,
                    door_front=None,
                    door_rear=second < 6,
                    source="SENSOR",
                    quality="GOOD",
                )
            )
        for trip_id in ("REAL", "PUBLIC"):
            db.add(
                PassengerEventModel(
                    event_uuid=f"AI-{trip_id}",
                    trip_id=trip_id,
                    source_id=f"{trip_id}:front",
                    camera="FRONT",
                    direction="IN",
                    track_id="front:1",
                    video_frame=30,
                    source_pts=3.2,
                    event_time_utc=start + timedelta(seconds=3, milliseconds=200),
                    source_type="REAL_VIDEO_AI",
                    source_confidence=0.9,
                    stop_visit_id=None,
                    assignment_status="UNASSIGNED",
                    evidence_path=None,
                    provenance="REAL_VIDEO_AI",
                )
            )
        db.commit()
        with pytest.raises(ValueError, match="source_id"):
            validate_event_source(db, "PUBLIC", "REAL:front", "FRONT", "REAL_VIDEO_AI", "REAL_VIDEO_AI")
        result = detect_and_assign(db, "REAL")
        assert result["detected"] == 1
        assert result["events_review_required"] == 1
        assert (
            db.query(PassengerEventModel).filter_by(trip_id="REAL").one().assignment_status
            == "REVIEW_REQUIRED"
        )
        public = detect_and_assign(db, "PUBLIC")
        assert public["detected"] == 0
        assert (
            db.query(PassengerEventModel).filter_by(trip_id="PUBLIC").one().assignment_status == "UNASSIGNED"
        )
