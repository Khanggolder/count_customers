from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import CameraSourceModel, CountEventModel, ReIDMatchModel, SessionModel
from .schemas import EventCreate, EventRead, MatchPatch, MatchRead, SessionCreate, SessionRead

Base.metadata.create_all(bind=engine)
app = FastAPI(title="BusReID AI API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "busreid-ai", "result_kind": "SYSTEM_STATUS"}


@app.get("/device")
def device() -> dict:
    try:
        import torch

        cuda = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "CPU",
        }
    except ImportError:
        return {"torch": None, "cuda_available": False, "cuda_version": None, "device": "CPU", "warning": "PyTorch chưa cài"}


@app.post("/sessions", response_model=SessionRead, status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)) -> SessionModel:
    if len({source.camera for source in payload.sources}) != len(payload.sources):
        raise HTTPException(422, "Mỗi camera chỉ được khai báo một nguồn")
    row = SessionModel(name=payload.name, mode=payload.mode, initial_occupancy=payload.initial_occupancy)
    db.add(row)
    db.flush()
    for source in payload.sources:
        db.add(CameraSourceModel(session_id=row.id, **source.model_dump()))
    db.commit()
    db.refresh(row)
    return row


@app.get("/sessions", response_model=list[SessionRead])
def list_sessions(db: Session = Depends(get_db)) -> list[SessionModel]:
    return list(db.scalars(select(SessionModel).order_by(SessionModel.id.desc())))


def _session_or_404(session_id: int, db: Session) -> SessionModel:
    row = db.get(SessionModel, session_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy session")
    return row


@app.get("/sessions/{session_id}", response_model=SessionRead)
def get_session(session_id: int, db: Session = Depends(get_db)) -> SessionModel:
    return _session_or_404(session_id, db)


@app.post("/sessions/{session_id}/start", response_model=SessionRead)
def start_session(session_id: int, db: Session = Depends(get_db)) -> SessionModel:
    row = _session_or_404(session_id, db)
    if row.status == "RUNNING":
        raise HTTPException(409, "Session đang chạy")
    row.status = "RUNNING"
    row.started_at = datetime.now(timezone.utc)
    row.ended_at = None
    db.commit()
    return row


@app.post("/sessions/{session_id}/stop", response_model=SessionRead)
def stop_session(session_id: int, db: Session = Depends(get_db)) -> SessionModel:
    row = _session_or_404(session_id, db)
    if row.status != "RUNNING":
        raise HTTPException(409, "Session không ở trạng thái RUNNING")
    row.status = "STOPPED"
    row.ended_at = datetime.now(timezone.utc)
    db.commit()
    return row


@app.post("/sessions/{session_id}/events", response_model=EventRead, status_code=201)
def add_event(session_id: int, payload: EventCreate, db: Session = Depends(get_db)) -> CountEventModel:
    _session_or_404(session_id, db)
    row = CountEventModel(session_id=session_id, **payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@app.get("/sessions/{session_id}/events", response_model=list[EventRead])
def events(session_id: int, db: Session = Depends(get_db)) -> list[CountEventModel]:
    _session_or_404(session_id, db)
    return list(db.scalars(select(CountEventModel).where(CountEventModel.session_id == session_id).order_by(CountEventModel.timestamp_s)))


@app.get("/sessions/{session_id}/stats")
def stats(session_id: int, db: Session = Depends(get_db)) -> dict:
    row = _session_or_404(session_id, db)
    grouped = dict(db.execute(select(CountEventModel.direction, func.count()).where(CountEventModel.session_id == session_id).group_by(CountEventModel.direction)).all())
    count_in, count_out = int(grouped.get("IN", 0)), int(grouped.get("OUT", 0))
    occupancy = row.initial_occupancy + count_in - count_out if row.mode == "PAIRED" else None
    warning = "Occupancy âm; kiểm tra đồng bộ/sự kiện" if occupancy is not None and occupancy < 0 else None
    if row.mode != "PAIRED":
        warning = "UNPAIRED: không suy ra occupancy"
    return {"in": count_in, "out": count_out, "occupancy": occupancy, "warning": warning, "hardcoded": False}


@app.get("/sessions/{session_id}/matches", response_model=list[MatchRead])
def matches(session_id: int, db: Session = Depends(get_db)) -> list[ReIDMatchModel]:
    _session_or_404(session_id, db)
    return list(db.scalars(select(ReIDMatchModel).where(ReIDMatchModel.session_id == session_id)))


@app.patch("/matches/{match_id}", response_model=MatchRead)
def review_match(match_id: int, payload: MatchPatch, db: Session = Depends(get_db)) -> ReIDMatchModel:
    row = db.get(ReIDMatchModel, match_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy match")
    session = db.get(SessionModel, row.session_id)
    if payload.status == "CONFIRMED" and session is not None and session.mode != "PAIRED":
        raise HTTPException(409, "Không thể CONFIRMED identity trong session UNPAIRED")
    row.status = payload.status
    row.reviewer_note = payload.reviewer_note
    db.commit()
    return row


@app.get("/sessions/{session_id}/export")
def export_session(session_id: int, format: str = Query("json", pattern="^(csv|json)$"), db: Session = Depends(get_db)) -> Response:
    _session_or_404(session_id, db)
    rows = list(db.scalars(select(CountEventModel).where(CountEventModel.session_id == session_id).order_by(CountEventModel.timestamp_s)))
    data = [{"id": item.id, "session_id": item.session_id, "camera": item.camera, "direction": item.direction, "track_id": item.track_id, "timestamp_s": item.timestamp_s, "frame_index": item.frame_index, "confidence": item.confidence, "passenger_id": item.passenger_id, "synthetic": item.synthetic} for item in rows]
    if format == "json":
        return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="session_{session_id}.json"'})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(data[0].keys()) if data else ["id", "session_id", "camera", "direction", "track_id", "timestamp_s", "frame_index", "confidence", "passenger_id", "synthetic"])
    writer.writeheader()
    writer.writerows(data)
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="session_{session_id}.csv"'})

