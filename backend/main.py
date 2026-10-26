from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from busreid.business import validate_passenger_direction
from busreid.scenario import build_scenario
from busreid.service import (
    detect_and_assign,
    flow_rows,
    replay_scenario,
    update_assignment,
    update_event_direction,
    validate_event_source,
)

from .database import Base, engine, get_db
from .models import (
    PassengerEventModel,
    ReviewAuditModel,
    RouteModel,
    RunModel,
    StopModel,
    StopVisitModel,
    TelemetrySampleModel,
    TripModel,
)
from .schemas import (
    AssignmentPatch,
    EventDirectionPatch,
    PassengerEventCreate,
    RunCreate,
    TelemetryCreate,
    TripCreate,
)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="BusStop Flow AI API", version="0.2.0")


def _trip_or_404(trip_id: str, db: Session) -> TripModel:
    row = db.get(TripModel, trip_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy chuyến")
    return row


def _trip_dict(row: TripModel) -> dict:
    return {
        "trip_id": row.trip_id,
        "route_id": row.route_id,
        "vehicle_id": row.vehicle_id,
        "start_time_utc": row.start_time_utc,
        "end_time_utc": row.end_time_utc,
        "mode": row.mode,
        "initial_occupancy": row.initial_occupancy,
        "status": row.status,
        "coverage_complete": row.coverage_complete,
        "provenance": row.provenance,
    }


def _event_dict(row: PassengerEventModel) -> dict:
    return {
        "id": row.id,
        "event_uuid": row.event_uuid,
        "trip_id": row.trip_id,
        "source_id": row.source_id,
        "camera": row.camera,
        "crossing_direction": row.crossing_direction,
        "direction": row.direction,
        "occupancy_delta": row.occupancy_delta,
        "direction_status": row.direction_status,
        "policy_valid": _event_policy_valid(row),
        "track_id": row.track_id,
        "video_frame": row.video_frame,
        "source_pts": row.source_pts,
        "event_time_utc": row.event_time_utc.isoformat(),
        "source_type": row.source_type,
        "source_confidence": row.source_confidence,
        "stop_visit_id": row.stop_visit_id,
        "assignment_status": row.assignment_status,
        "evidence_path": row.evidence_path,
        "provenance": row.provenance,
    }


def _event_policy_valid(row: PassengerEventModel) -> bool:
    if row.direction_status == "REVIEW_REQUIRED" or row.crossing_direction is None:
        return False
    try:
        validate_passenger_direction(
            row.camera, row.direction, row.crossing_direction, row.occupancy_delta
        )
        return row.occupancy_delta is not None
    except ValueError:
        return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "busstop-flow-ai", "version": "0.2.0"}


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
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "CPU",
            "warning": "PyTorch chưa cài",
        }


@app.get("/routes")
def routes(db: Session = Depends(get_db)) -> list[dict]:
    return [
        {
            "route_id": row.route_id,
            "name": row.name,
            "direction": row.direction,
            "source": row.source,
            "retrieved_date": row.retrieved_date,
            "verified": row.verified,
        }
        for row in db.scalars(select(RouteModel).order_by(RouteModel.route_id))
    ]


@app.get("/routes/{route_id}/stops")
def route_stops(route_id: str, db: Session = Depends(get_db)) -> list[dict]:
    if db.get(RouteModel, route_id) is None:
        raise HTTPException(404, "Không tìm thấy tuyến")
    rows = db.scalars(select(StopModel).where(StopModel.route_id == route_id).order_by(StopModel.sequence))
    return [
        {
            "stop_id": row.stop_id,
            "route_id": row.route_id,
            "sequence": row.sequence,
            "name": row.name,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "radius_m": row.radius_m,
            "coordinate_source": row.coordinate_source,
            "name_source_url": row.name_source_url,
            "retrieved_date": row.retrieved_date,
            "verified": row.verified,
        }
        for row in rows
    ]


@app.post("/trips", status_code=201)
def create_trip(payload: TripCreate, db: Session = Depends(get_db)) -> dict:
    if payload.mode == "SYNTHETIC_PAIRED":
        raise HTTPException(422, "Create synthetic trips through the scenario generator")
    if db.get(RouteModel, payload.route_id) is None:
        raise HTTPException(422, "route_id chưa tồn tại")
    row = TripModel(
        **payload.model_dump(), status="CREATED", coverage_complete=False, provenance="REAL_OR_USER_INPUT"
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "trip_id đã tồn tại") from exc
    return _trip_dict(row)


@app.get("/trips")
def trips(
    limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> list[dict]:
    rows = db.scalars(select(TripModel).order_by(TripModel.start_time_utc.desc()).limit(limit).offset(offset))
    return [_trip_dict(row) for row in rows]


@app.get("/trips/{trip_id}")
def get_trip(trip_id: str, db: Session = Depends(get_db)) -> dict:
    return _trip_dict(_trip_or_404(trip_id, db))


@app.post("/trips/{trip_id}/start")
def start_trip(trip_id: str, db: Session = Depends(get_db)) -> dict:
    row = _trip_or_404(trip_id, db)
    if row.status == "RUNNING":
        raise HTTPException(409, "Chuyến đã chạy")
    row.status = "RUNNING"
    db.commit()
    return _trip_dict(row)


@app.post("/trips/{trip_id}/stop")
def stop_trip(trip_id: str, db: Session = Depends(get_db)) -> dict:
    row = _trip_or_404(trip_id, db)
    row.status = "COMPLETED"
    row.end_time_utc = datetime.now(timezone.utc)
    db.commit()
    return _trip_dict(row)


@app.post("/trips/{trip_id}/telemetry", status_code=201)
def ingest_telemetry(trip_id: str, payload: list[TelemetryCreate], db: Session = Depends(get_db)) -> dict:
    trip = _trip_or_404(trip_id, db)
    if any((sample.source == "SIMULATED") != (trip.provenance == "SIMULATED") for sample in payload):
        raise HTTPException(422, "Telemetry provenance conflicts with trip provenance")
    created = 0
    existing = set(
        db.scalars(select(TelemetrySampleModel.event_time_utc).where(TelemetrySampleModel.trip_id == trip_id))
    )
    for sample in payload:
        when = sample.event_time_utc.astimezone(timezone.utc)
        if when in existing:
            continue
        db.add(TelemetrySampleModel(trip_id=trip_id, **{**sample.model_dump(), "event_time_utc": when}))
        existing.add(when)
        created += 1
    db.commit()
    return {"created": created, "duplicates_skipped": len(payload) - created}


@app.get("/trips/{trip_id}/telemetry")
def trip_telemetry(
    trip_id: str,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[dict]:
    _trip_or_404(trip_id, db)
    rows = db.scalars(
        select(TelemetrySampleModel)
        .where(TelemetrySampleModel.trip_id == trip_id)
        .order_by(TelemetrySampleModel.event_time_utc)
        .limit(limit)
        .offset(offset)
    )
    return [
        {
            "id": row.id,
            "event_time_utc": row.event_time_utc.isoformat(),
            "latitude": row.latitude,
            "longitude": row.longitude,
            "speed_kmh": row.speed_kmh,
            "heading": row.heading,
            "door_front": row.door_front,
            "door_rear": row.door_rear,
            "source": row.source,
            "quality": row.quality,
        }
        for row in rows
    ]


@app.get("/trips/{trip_id}/stop-visits")
def stop_visits(trip_id: str, db: Session = Depends(get_db)) -> list[dict]:
    _trip_or_404(trip_id, db)
    rows = db.scalars(
        select(StopVisitModel).where(StopVisitModel.trip_id == trip_id).order_by(StopVisitModel.arrived_at)
    )
    return [
        {
            "stop_visit_id": row.stop_visit_id,
            "trip_id": row.trip_id,
            "stop_id": row.stop_id,
            "arrived_at": row.arrived_at.isoformat(),
            "departed_at": row.departed_at.isoformat() if row.departed_at else None,
            "status": row.status,
            "evidence": row.evidence,
            "confidence": row.confidence,
            "override_audit": row.override_audit,
        }
        for row in rows
    ]


@app.post("/trips/{trip_id}/detect-stop-visits")
def detect_stop_visits(trip_id: str, db: Session = Depends(get_db)) -> dict:
    _trip_or_404(trip_id, db)
    try:
        return detect_and_assign(db, trip_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/trips/{trip_id}/passenger-events")
def passenger_events(
    trip_id: str,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    assignment_status: str | None = None,
    db: Session = Depends(get_db),
) -> list[dict]:
    _trip_or_404(trip_id, db)
    query = select(PassengerEventModel).where(PassengerEventModel.trip_id == trip_id)
    if assignment_status:
        query = query.where(PassengerEventModel.assignment_status == assignment_status)
    rows = db.scalars(query.order_by(PassengerEventModel.event_time_utc).limit(limit).offset(offset))
    return [_event_dict(row) for row in rows]


@app.post("/trips/{trip_id}/passenger-events", status_code=201)
def create_passenger_event(
    trip_id: str, payload: PassengerEventCreate, db: Session = Depends(get_db)
) -> dict:
    _trip_or_404(trip_id, db)
    try:
        validate_passenger_direction(
            payload.camera, payload.direction, payload.crossing_direction, payload.occupancy_delta
        )
        validate_event_source(
            db, trip_id, payload.source_id, payload.camera, payload.provenance, payload.source_type
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    existing = db.scalar(
        select(PassengerEventModel).where(PassengerEventModel.event_uuid == payload.event_uuid)
    )
    if existing is not None:
        if existing.trip_id != trip_id or existing.source_id != payload.source_id:
            raise HTTPException(409, "event_uuid already belongs to a different trip/source")
        if (
            existing.camera != payload.camera
            or existing.crossing_direction != payload.crossing_direction
            or existing.direction != payload.direction
            or existing.occupancy_delta != payload.occupancy_delta
        ):
            raise HTTPException(409, "event_uuid conflicts with an existing crossing")
        return {**_event_dict(existing), "duplicate": True}
    row = PassengerEventModel(
        trip_id=trip_id,
        stop_visit_id=None,
        assignment_status="UNASSIGNED",
        direction_status="SOURCE_ASSERTED",
        **{**payload.model_dump(), "event_time_utc": payload.event_time_utc.astimezone(timezone.utc)},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {**_event_dict(row), "duplicate": False}


@app.patch("/passenger-events/{event_id}/assignment")
def patch_assignment(event_id: int, payload: AssignmentPatch, db: Session = Depends(get_db)) -> dict:
    try:
        return _event_dict(
            update_assignment(db, event_id, payload.stop_visit_id, payload.actor, payload.reason)
        )
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy event") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.patch("/passenger-events/{event_id}/direction")
def patch_event_direction(event_id: int, payload: EventDirectionPatch, db: Session = Depends(get_db)) -> dict:
    try:
        return _event_dict(update_event_direction(
            db, event_id, payload.crossing_direction, payload.direction,
            payload.occupancy_delta, payload.actor, payload.reason,
        ))
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy event") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/trips/{trip_id}/audits")
def audits(trip_id: str, db: Session = Depends(get_db)) -> list[dict]:
    event_ids = [
        str(value)
        for value in db.scalars(select(PassengerEventModel.id).where(PassengerEventModel.trip_id == trip_id))
    ]
    if not event_ids:
        return []
    rows = db.scalars(select(ReviewAuditModel).where(ReviewAuditModel.record_id.in_(event_ids)))
    return [
        {
            "id": row.id,
            "actor": row.actor,
            "record_id": row.record_id,
            "before": row.before,
            "after": row.after,
            "time": row.time.isoformat(),
            "reason": row.reason,
        }
        for row in rows
    ]


@app.get("/trips/{trip_id}/flow")
def trip_flow(trip_id: str, db: Session = Depends(get_db)) -> list[dict]:
    try:
        return flow_rows(db, trip_id)
    except KeyError as exc:
        raise HTTPException(404, "Không tìm thấy chuyến") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/trips/{trip_id}/exports")
def export_trip(
    trip_id: str, format: str = Query("csv", pattern="^(csv|json)$"), db: Session = Depends(get_db)
) -> Response:
    rows = trip_flow(trip_id, db)
    if format == "json":
        return Response(
            json.dumps(rows, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{trip_id}_flow.json"'},
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[key for key in rows[0] if key != "evidence"] if rows else [])
    writer.writeheader()
    for row in rows:
        writer.writerow({key: value for key, value in row.items() if key != "evidence"})
    return Response(
        "\ufeff" + output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{trip_id}_flow.csv"'},
    )


@app.post("/demo/generate", status_code=201)
def generate_demo(
    kind: str = Query("normal", pattern="^(normal|negative_occupancy|incomplete_camera)$"), db: Session = Depends(get_db)
) -> dict:
    return replay_scenario(db, build_scenario(kind=kind))


@app.post("/runs", status_code=201)
def create_run(payload: RunCreate, db: Session = Depends(get_db)) -> dict:
    if payload.trip_id:
        _trip_or_404(payload.trip_id, db)
    row = RunModel(
        run_id=str(uuid.uuid4()),
        trip_id=payload.trip_id,
        status="QUEUED",
        progress=0.0,
        detail=payload.detail,
    )
    db.add(row)
    db.commit()
    return {
        "run_id": row.run_id,
        "trip_id": row.trip_id,
        "status": row.status,
        "progress": row.progress,
        "cancel_requested": row.cancel_requested,
        "detail": row.detail,
    }


@app.get("/runs")
def list_runs(db: Session = Depends(get_db)) -> list[dict]:
    return [
        {
            "run_id": row.run_id,
            "trip_id": row.trip_id,
            "status": row.status,
            "progress": row.progress,
            "cancel_requested": row.cancel_requested,
            "detail": row.detail,
        }
        for row in db.scalars(select(RunModel).order_by(RunModel.created_at.desc()))
    ]


@app.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    row = db.get(RunModel, run_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy run")
    return {
        "run_id": row.run_id,
        "trip_id": row.trip_id,
        "status": row.status,
        "progress": row.progress,
        "cancel_requested": row.cancel_requested,
        "detail": row.detail,
    }


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    row = db.get(RunModel, run_id)
    if row is None:
        raise HTTPException(404, "Không tìm thấy run")
    row.cancel_requested = True
    if row.status in {"QUEUED", "RUNNING"}:
        row.status = "CANCEL_REQUESTED"
    db.commit()
    return {"run_id": row.run_id, "status": row.status, "cancel_requested": row.cancel_requested}
