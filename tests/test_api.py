from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.main import app
from backend.models import PassengerEventModel, ReviewAuditModel


def make_client(tmp_path):
    test_engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(test_engine)
    factory = sessionmaker(bind=test_engine, expire_on_commit=False)

    def override_db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app), factory


def test_demo_api_lifecycle_flow_export_and_audit(tmp_path) -> None:
    client, factory = make_client(tmp_path)
    result = client.post("/demo/generate")
    assert result.status_code == 201
    trip_id = result.json()["trip_id"]
    assert len(client.get("/routes/SIM-R01/stops").json()) == 8
    assert len(client.get(f"/trips/{trip_id}/stop-visits").json()) == 8
    flow = client.get(f"/trips/{trip_id}/flow").json()
    assert sum(row["total_in"] for row in flow) == 18
    assert sum(row["total_out"] for row in flow) == 25
    assert all(row["warning"] is None and row["onboard_after_stop"] >= 0 for row in flow)
    assert client.get(f"/trips/{trip_id}/exports?format=csv").status_code == 200
    assert client.get(f"/trips/{trip_id}/exports?format=json").status_code == 200
    unassigned = client.get(f"/trips/{trip_id}/passenger-events?assignment_status=UNASSIGNED").json()
    visit = client.get(f"/trips/{trip_id}/stop-visits").json()[0]
    event_time = visit["arrived_at"]
    if "+" not in event_time and not event_time.endswith("Z"):
        event_time += "Z"
    synthetic_event = {
        "event_uuid": "SIM-IDEMPOTENT-001",
        "source_id": f"{trip_id}:front:sim",
        "camera": "FRONT",
        "direction": "IN",
        "track_id": "front:test:1",
        "video_frame": 10,
        "source_pts": 1.0,
        "event_time_utc": event_time,
        "source_type": "SIMULATED_EVENT_GENERATOR",
        "source_confidence": 0.9,
        "provenance": "SIMULATED_EVENTS",
    }
    assert (
        client.post(f"/trips/{trip_id}/passenger-events", json=synthetic_event).json()["duplicate"] is False
    )
    assert client.post(f"/trips/{trip_id}/passenger-events", json=synthetic_event).json()["duplicate"] is True
    ai_event = {
        **synthetic_event,
        "event_uuid": "AI-REJECTED-001",
        "source_type": "REAL_VIDEO_AI",
        "provenance": "REAL_VIDEO_AI",
    }
    assert client.post(f"/trips/{trip_id}/passenger-events", json=ai_event).status_code == 422
    for camera, direction, source_id in (
        ("FRONT", "OUT", f"{trip_id}:front:sim"),
        ("REAR", "IN", f"{trip_id}:rear:sim"),
    ):
        invalid = {
            **synthetic_event,
            "event_uuid": f"BAD-{camera}-{direction}-001",
            "camera": camera,
            "direction": direction,
            "source_id": source_id,
        }
        response = client.post(f"/trips/{trip_id}/passenger-events", json=invalid)
        assert response.status_code == 422
        assert "accepts only" in response.json()["detail"]
    detected = client.post(f"/trips/{trip_id}/detect-stop-visits").json()
    assert detected["detected"] == 8
    assert detected["events_assigned"] == 1
    response = client.patch(
        f"/passenger-events/{unassigned[0]['id']}/assignment",
        json={"stop_visit_id": visit["stop_visit_id"], "actor": "tester", "reason": "manual test correction"},
    )
    assert response.status_code == 200
    assert response.json()["assignment_status"] == "MANUAL_ASSIGNED"
    assert len(client.get(f"/trips/{trip_id}/audits").json()) == 1
    with factory() as db:
        assert len(list(db.scalars(select(ReviewAuditModel)))) == 1
    app.dependency_overrides.clear()


def test_legacy_reverse_events_are_preserved_but_excluded_from_flow_and_exports(tmp_path) -> None:
    client, factory = make_client(tmp_path)
    trip_id = client.post("/demo/generate").json()["trip_id"]
    baseline = client.get(f"/trips/{trip_id}/flow").json()
    visits = client.get(f"/trips/{trip_id}/stop-visits").json()
    for camera, direction in (("FRONT", "OUT"), ("REAR", "IN")):
        with factory() as db:
            db.add(
                PassengerEventModel(
                    event_uuid=f"LEGACY-{camera}-{direction}",
                    trip_id=trip_id,
                    source_id=f"{trip_id}:{camera.lower()}:sim",
                    camera=camera,
                    direction=direction,
                    track_id="legacy:1",
                    video_frame=None,
                    source_pts=None,
                    event_time_utc=datetime.fromisoformat(baseline[0]["arrived_at"]),
                    source_type="SIMULATED_EVENT_GENERATOR",
                    source_confidence=1.0,
                    stop_visit_id=visits[0]["stop_visit_id"],
                    assignment_status="AUTO_ASSIGNED",
                    evidence_path=None,
                    provenance="SIMULATED_EVENTS",
                )
            )
            db.commit()
    assert client.get(f"/trips/{trip_id}/flow").json() == baseline
    assert client.get(f"/trips/{trip_id}/exports?format=json").json() == baseline
    csv_text = client.get(f"/trips/{trip_id}/exports?format=csv").text
    assert "boarding_count" in csv_text and "alighting_count" in csv_text
    assert "front_out" in csv_text  # deprecated compatibility column is zero
    assert all(row["front_out"] == row["rear_in"] == 0 for row in baseline)
    assert client.post(f"/trips/{trip_id}/detect-stop-visits").status_code == 200
    with factory() as db:
        legacy = list(
            db.scalars(select(PassengerEventModel).where(PassengerEventModel.event_uuid.like("LEGACY-%")))
        )
        assert len(legacy) == 2  # history was not deleted
    response = client.patch(
        f"/passenger-events/{legacy[0].id}/assignment",
        json={"stop_visit_id": visits[0]["stop_visit_id"], "actor": "tester", "reason": "legacy check"},
    )
    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_validation_rejects_naive_time_and_invalid_coordinates(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    client.post("/demo/generate")
    bad_trip = {
        "trip_id": "BAD",
        "route_id": "SIM-R01",
        "vehicle_id": "V",
        "start_time_utc": "2026-01-01T00:00:00",
    }
    assert client.post("/trips", json=bad_trip).status_code == 422
    payload = [{"event_time_utc": "2026-01-01T00:00:00Z", "latitude": 100, "longitude": 106, "speed_kmh": 0}]
    assert client.post("/trips/SIM-R01-v2-20261019-0800/telemetry", json=payload).status_code == 422
    real_telemetry = [
        {
            "event_time_utc": "2026-10-19T01:00:03Z",
            "latitude": 10.0,
            "longitude": 106.0,
            "speed_kmh": 0,
            "source": "SENSOR",
        }
    ]
    assert client.post("/trips/SIM-R01-v2-20261019-0800/telemetry", json=real_telemetry).status_code == 422
    assert client.get("/trips/missing/flow").status_code == 404
    app.dependency_overrides.clear()


def test_run_status_and_cancel(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    created = client.post("/runs", json={"detail": {"kind": "smoke"}}).json()
    assert created["status"] == "QUEUED"
    cancelled = client.post(f"/runs/{created['run_id']}/cancel").json()
    assert cancelled == {"run_id": created["run_id"], "status": "CANCEL_REQUESTED", "cancel_requested": True}
    app.dependency_overrides.clear()
