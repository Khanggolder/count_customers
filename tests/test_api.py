from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, get_db
from backend.main import app


def make_client(tmp_path: Path) -> TestClient:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def test_session_event_stats_and_exports(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post("/sessions", json={"name": "test", "mode": "PAIRED", "initial_occupancy": 1, "sources": []})
    assert response.status_code == 201
    session_id = response.json()["id"]
    assert client.post(f"/sessions/{session_id}/start").status_code == 200
    event = {"camera": "front", "direction": "IN", "track_id": "front:t1", "timestamp_s": 1.2, "frame_index": 12, "confidence": 0.9, "synthetic": True}
    assert client.post(f"/sessions/{session_id}/events", json=event).status_code == 201
    stats = client.get(f"/sessions/{session_id}/stats").json()
    assert stats == {"in": 1, "out": 0, "occupancy": 2, "warning": None, "hardcoded": False}
    assert client.get(f"/sessions/{session_id}/export?format=csv").status_code == 200
    assert client.get(f"/sessions/{session_id}/export?format=json").json()[0]["synthetic"] is True
    assert client.post(f"/sessions/{session_id}/stop").status_code == 200


def test_duplicate_camera_source_is_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    payload = {"name": "bad", "sources": [{"camera": "front", "uri": "a.mp4"}, {"camera": "front", "uri": "b.mp4"}]}
    assert client.post("/sessions", json=payload).status_code == 422

