from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.main import app
from backend.models import ReviewAuditModel


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
    assert sum(row["total_in"] for row in flow) == 24
    assert sum(row["total_out"] for row in flow) == 32
    assert flow[-1]["warning"] == "NEGATIVE_OCCUPANCY_REVIEW_REQUIRED"
    assert client.get(f"/trips/{trip_id}/exports?format=csv").status_code == 200
    assert client.get(f"/trips/{trip_id}/exports?format=json").status_code == 200
    unassigned = client.get(f"/trips/{trip_id}/passenger-events?assignment_status=UNASSIGNED").json()
    visit = client.get(f"/trips/{trip_id}/stop-visits").json()[0]
    event_time = visit["arrived_at"]
    if "+" not in event_time and not event_time.endswith("Z"):
        event_time += "Z"
    ai_event = {
        "event_uuid": "AI-IDEMPOTENT-001",
        "source_id": "test-front",
        "camera": "FRONT",
        "direction": "IN",
        "track_id": "front:test:1",
        "video_frame": 10,
        "source_pts": 1.0,
        "event_time_utc": event_time,
        "source_type": "REAL_VIDEO_AI",
        "source_confidence": 0.9,
        "provenance": "REAL_VIDEO_AI",
    }
    assert client.post(f"/trips/{trip_id}/passenger-events", json=ai_event).json()["duplicate"] is False
    assert client.post(f"/trips/{trip_id}/passenger-events", json=ai_event).json()["duplicate"] is True
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
    assert client.post("/trips/SIM-R01-20261019-0800/telemetry", json=payload).status_code == 422
    assert client.get("/trips/missing/flow").status_code == 404
    app.dependency_overrides.clear()


def test_run_status_and_cancel(tmp_path) -> None:
    client, _ = make_client(tmp_path)
    created = client.post("/runs", json={"detail": {"kind": "smoke"}}).json()
    assert created["status"] == "QUEUED"
    cancelled = client.post(f"/runs/{created['run_id']}/cancel").json()
    assert cancelled == {"run_id": created["run_id"], "status": "CANCEL_REQUESTED", "cancel_requested": True}
    app.dependency_overrides.clear()
