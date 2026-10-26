"""Software integration fixture: FakeDetector is NOT a measurement of AI accuracy."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.database import Base
from backend.models import PassengerEventModel, RouteModel, StopModel, TripModel
from busreid.config import AppConfig, CameraConfig, DetectorConfig
from busreid.pipeline import CameraPipeline
from busreid.service import flow_rows
from busreid.types import Detection, Direction
from scripts import run_count


class FakeDetector:
    def __init__(self, camera: str) -> None:
        self.camera = camera
        self.index = 0

    def track(self, _frame: np.ndarray) -> list[Detection]:
        y = 25 if self.index < 30 or self.index >= 45 else 80
        self.index += 1
        return [Detection(f"{self.camera}:fixture:1", (40, y - 20, 60, y), 0.99)]


def fixture_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(60):
        writer.write(np.full((100, 100, 3), 100, dtype=np.uint8))
    writer.release()


def test_two_camera_video_to_stop_flow_and_provenance_guard(tmp_path: Path, monkeypatch) -> None:
    """Two MP4 pipelines, deterministic tracks, sensor samples, DB assignment and flow."""
    engine = create_engine(f"sqlite:///{tmp_path / 'integration.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with factory() as db:
        db.add(
            RouteModel(
                route_id="R1",
                name="Fixture route",
                direction="test",
                source="TEST",
                retrieved_date="2026-01-01",
                verified=False,
            )
        )
        db.add(
            StopModel(
                stop_id="S1",
                route_id="R1",
                sequence=1,
                name="Fixture stop",
                latitude=10,
                longitude=106,
                radius_m=50,
                coordinate_source="TEST",
                name_source_url="TEST",
                retrieved_date="2026-01-01",
                verified=False,
            )
        )
        db.add(
            TripModel(
                trip_id="TRIP-1",
                route_id="R1",
                vehicle_id="BUS-1",
                start_time_utc=start,
                end_time_utc=None,
                mode="UNPAIRED",
                initial_occupancy=0,
                status="CREATED",
                coverage_complete=False,
                provenance="REAL_OR_USER_INPUT",
            )
        )
        db.commit()
    front_video = tmp_path / "front_software_fixture.mp4"
    rear_video = tmp_path / "rear_software_fixture.mp4"
    fixture_video(front_video)
    fixture_video(rear_video)
    zone_a = [(0, 0), (1, 0), (1, 0.45), (0, 0.45)]
    zone_b = [(0, 0.55), (1, 0.55), (1, 1), (0, 1)]
    cameras = {
        name: CameraConfig(name, name, direction, str(front_video if name == "front" else rear_video), zone_a, zone_b, 2, 4)
        for name, direction in (("front", Direction.IN), ("rear", Direction.OUT))
    }
    config = AppConfig("sqlite://", DetectorConfig(), cameras)
    monkeypatch.setattr(run_count, "engine", engine)
    monkeypatch.setattr(run_count, "SessionLocal", factory)
    monkeypatch.setattr(run_count, "load_config", lambda _: config)
    monkeypatch.setattr(
        "busreid.workers.CameraPipeline",
        lambda camera, app_config: CameraPipeline(camera, app_config, FakeDetector(camera.name)),
    )
    telemetry = [
        {
            "event_time_utc": (start + timedelta(seconds=i)).isoformat(),
            "latitude": 10 if i < 6 else 10.001,
            "longitude": 106,
            "speed_kmh": 0 if i < 6 else 15,
            "door_front": i < 6,
            "door_rear": i < 6,
            "source": "SENSOR",
            "quality": "GOOD",
        }
        for i in range(7)
    ]
    telemetry_file = tmp_path / "telemetry.json"
    telemetry_file.write_text(json.dumps(telemetry), encoding="utf-8")
    code = run_count.main(
        [
            "--front",
            str(front_video),
            "--rear",
            str(rear_video),
            "--trip-id",
            "TRIP-1",
            "--recording-start-time",
            start.isoformat(),
            "--telemetry-json",
            str(telemetry_file),
            "--telemetry-synchronized",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert code == 0
    with factory() as db:
        events = list(db.scalars(select(PassengerEventModel).order_by(PassengerEventModel.camera)))
        assert {(item.camera, item.crossing_direction, item.direction, item.occupancy_delta, item.assignment_status) for item in events} == {
            ("FRONT", "A_TO_B", "IN", 1, "AUTO_ASSIGNED"),
            ("FRONT", "B_TO_A", "OUT", -1, "AUTO_ASSIGNED"),
            ("REAR", "A_TO_B", "OUT", -1, "AUTO_ASSIGNED"),
            ("REAR", "B_TO_A", "IN", 1, "AUTO_ASSIGNED"),
        }
        assert all(item.source_type == "REAL_VIDEO_AI" for item in events)
        assert all(
            item.event_time_utc.replace(tzinfo=timezone.utc) < item.created_at.replace(tzinfo=timezone.utc)
            for item in events
        )
        row = flow_rows(db, "TRIP-1")[0]
        assert (row["front_in"], row["front_out"], row["rear_in"], row["rear_out"]) == (1, 1, 1, 1)
        assert (row["total_in"], row["total_out"]) == (2, 2)
        assert row["onboard_after_stop"] is None  # coverage not independently verified
    summary = json.loads((tmp_path / "output" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregation"]["events_assigned"] == 4
    assert summary["video_model_validation"] == "NO_GROUND_TRUTH_NA"
    try:
        run_count.main(
            [
                "--front",
                str(tmp_path / "different.mp4"),
                "--trip-id",
                "TRIP-1",
                "--output",
                str(tmp_path / "should_not_run"),
            ]
        )
    except ValueError as exc:
        assert "Source mismatch" in str(exc)
    else:
        raise AssertionError("Mismatched source was accepted")
    assert not (tmp_path / "should_not_run").exists()


def test_real_inference_rejects_synthetic_trip_before_video_work(tmp_path: Path, monkeypatch) -> None:
    from busreid.scenario import build_scenario
    from busreid.service import replay_scenario

    engine = create_engine(f"sqlite:///{tmp_path / 'guard.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        replay_scenario(db, build_scenario())
    monkeypatch.setattr(run_count, "engine", engine)
    monkeypatch.setattr(run_count, "SessionLocal", factory)
    video = tmp_path / "fixture.mp4"
    fixture_video(video)
    try:
        run_count.main(
            [
                "--front",
                str(video),
                "--trip-id",
                    "SIM-R01-v3-20261019-0800",
                "--output",
                str(tmp_path / "output"),
            ]
        )
    except ValueError as exc:
        assert "synthetic trip" in str(exc)
    else:
        raise AssertionError("AI video was accepted into a synthetic trip")
    assert not (tmp_path / "output").exists()
