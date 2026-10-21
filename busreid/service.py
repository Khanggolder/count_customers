from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import (
    CameraSourceModel,
    PassengerEventModel,
    ReviewAuditModel,
    RouteModel,
    StopModel,
    StopVisitModel,
    TelemetrySampleModel,
    TripModel,
)
from busreid.business import validate_passenger_direction
from busreid.stop_detection import StopPoint, StopVisitDetector, TelemetryPoint, assign_visit


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return result.astimezone(timezone.utc)


def validate_event_source(
    db: Session, trip_id: str, source_id: str, camera: str, provenance: str, source_type: str
) -> CameraSourceModel:
    trip = db.get(TripModel, trip_id)
    source = db.get(CameraSourceModel, source_id)
    if trip is None or source is None or source.trip_id != trip_id or source.door_role != camera:
        raise ValueError("source_id must belong to trip_id and camera")
    synthetic = trip.provenance == "SIMULATED"
    if synthetic != (provenance == "SIMULATED_EVENTS"):
        raise ValueError("Synthetic and real AI provenance cannot be mixed")
    if synthetic != (source.input_type == "SIMULATION"):
        raise ValueError("Camera source input_type conflicts with trip provenance")
    if synthetic != (source_type == "SIMULATED_EVENT_GENERATOR"):
        raise ValueError("Event source_type conflicts with trip provenance")
    return source


def detect_and_assign(db: Session, trip_id: str) -> dict:
    trip = db.get(TripModel, trip_id)
    if trip is None:
        raise KeyError(trip_id)
    stops = list(
        db.scalars(select(StopModel).where(StopModel.route_id == trip.route_id).order_by(StopModel.sequence))
    )
    telemetry = list(
        db.scalars(
            select(TelemetrySampleModel)
            .where(TelemetrySampleModel.trip_id == trip_id)
            .order_by(TelemetrySampleModel.event_time_utc)
        )
    )
    if trip.provenance == "SIMULATED" and any(row.source != "SIMULATED" for row in telemetry):
        raise ValueError("Synthetic trip contains non-synthetic telemetry")
    if trip.provenance != "SIMULATED" and any(row.source == "SIMULATED" for row in telemetry):
        raise ValueError("Real trip cannot use synthetic telemetry")
    points = [
        TelemetryPoint(
            parse_time(
                row.event_time_utc.replace(tzinfo=timezone.utc)
                if row.event_time_utc.tzinfo is None
                else row.event_time_utc
            ),
            row.latitude,
            row.longitude,
            row.speed_kmh,
            row.door_front,
            row.door_rear,
            row.quality,
        )
        for row in telemetry
    ]
    detected = StopVisitDetector(
        [StopPoint(row.stop_id, row.sequence, row.latitude, row.longitude, row.radius_m) for row in stops]
    ).detect(points)
    visit_ids: dict[str, str] = {}
    created = updated = 0
    for visit in detected:
        visit_id = f"{trip_id}:{visit.stop_id}:{int(visit.arrived_at.timestamp())}"
        visit_ids[visit.stop_id] = visit_id
        existing_visit = db.get(StopVisitModel, visit_id)
        if existing_visit is None:
            db.add(
                StopVisitModel(
                    stop_visit_id=visit_id,
                    trip_id=trip_id,
                    stop_id=visit.stop_id,
                    arrived_at=visit.arrived_at,
                    departed_at=visit.departed_at,
                    status=visit.status,
                    evidence=visit.evidence,
                    confidence=visit.confidence,
                    override_audit=[],
                )
            )
            created += 1
        elif not existing_visit.override_audit:
            existing_visit.departed_at = visit.departed_at
            existing_visit.status = visit.status
            existing_visit.evidence = visit.evidence
            existing_visit.confidence = visit.confidence
            updated += 1
    assigned = review = 0
    events = list(
        db.scalars(
            select(PassengerEventModel).where(
                PassengerEventModel.trip_id == trip_id,
                PassengerEventModel.assignment_status.in_(["UNASSIGNED", "REVIEW_REQUIRED"]),
            )
        )
    )
    for event in events:
        try:
            validate_passenger_direction(event.camera, event.direction)
        except ValueError:
            # Retain legacy rows for audit without assigning or counting them.
            continue
        source = validate_event_source(
            db, trip_id, event.source_id, event.camera, event.provenance, event.source_type
        )
        if trip.provenance != "SIMULATED" and (
            source.paired_flag not in {"SYNCHRONIZED", "PAIRED_VERIFIED"}
            or event.source_type != "REAL_VIDEO_AI"
        ):
            continue
        event_time = parse_time(
            event.event_time_utc.replace(tzinfo=timezone.utc)
            if event.event_time_utc.tzinfo is None
            else event.event_time_utc
        )
        stop_id = assign_visit(event_time, detected)
        if not stop_id:
            continue
        # Use a recent preceding sensor sample; future readings cannot prove an open door.
        evidence = [
            point for point in points if 0 <= (event_time - point.event_time_utc).total_seconds() <= 2
        ]
        door = (
            (evidence[-1].door_front if event.camera == "FRONT" else evidence[-1].door_rear)
            if evidence
            else None
        )
        if door is True:
            event.stop_visit_id = visit_ids[stop_id]
            event.assignment_status = "AUTO_ASSIGNED"
            assigned += 1
        else:
            event.stop_visit_id = None
            event.assignment_status = "REVIEW_REQUIRED"
            review += 1
    db.commit()
    return {
        "detected": len(detected),
        "created": created,
        "updated": updated,
        "events_assigned": assigned,
        "events_review_required": review,
    }


def replay_scenario(db: Session, scenario: dict) -> dict:
    if scenario.get("schema_version") != "2.0":
        raise ValueError("Only FRONT-IN/REAR-OUT scenario schema 2.0 may be replayed")
    route_data, trip_data = scenario["route"], scenario["trip"]
    if trip_data["provenance"] != "SIMULATED":
        raise ValueError("Scenario trip must be SIMULATED")
    for item in scenario["passenger_events"]:
        validate_passenger_direction(item["camera"], item["direction"])
        if item["source_type"] != "SIMULATED_EVENT_GENERATOR" or item["provenance"] != "SIMULATED_EVENTS":
            raise ValueError("Scenario event must retain synthetic provenance")
    route = db.get(RouteModel, route_data["route_id"])
    if route is None:
        route = RouteModel(**route_data)
        db.add(route)
    for item in scenario["stops"]:
        if db.get(StopModel, item["stop_id"]) is None:
            db.add(StopModel(route_id=route_data["route_id"], **item))
    trip = db.get(TripModel, trip_data["trip_id"])
    if trip is None:
        trip = TripModel(
            **{
                **trip_data,
                "route_id": route_data["route_id"],
                "start_time_utc": parse_time(trip_data["start_time_utc"]),
                "end_time_utc": parse_time(trip_data["end_time_utc"]),
            },
        )
        db.add(trip)
    elif trip.provenance != "SIMULATED":
        raise ValueError("Scenario cannot be replayed into a real trip")
    elif trip.mode != "SYNTHETIC_PAIRED" or trip.initial_occupancy != trip_data["initial_occupancy"]:
        raise ValueError("Existing scenario trip metadata conflicts with schema 2.0")
    for camera in ("front", "rear"):
        source_id = f"{trip_data['trip_id']}:{camera}:sim"
        if db.get(CameraSourceModel, source_id) is None:
            db.add(
                CameraSourceModel(
                    source_id=source_id,
                    trip_id=trip_data["trip_id"],
                    door_role=camera.upper(),
                    input_type="SIMULATION",
                    safe_uri="simulation://generated",
                    calibration_id=f"{camera}-default",
                    paired_flag="SYNTHETIC_PAIRED",
                    time_offset_ms=0,
                )
            )
    db.flush()

    def time_key(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    existing_times = {
        time_key(value)
        for value in db.scalars(
            select(TelemetrySampleModel.event_time_utc).where(TelemetrySampleModel.trip_id == trip.trip_id)
        )
    }
    samples: list[TelemetryPoint] = []
    for item in scenario["telemetry"]:
        when = parse_time(item["event_time_utc"])
        sample = TelemetryPoint(
            when,
            item["latitude"],
            item["longitude"],
            item["speed_kmh"],
            item["door_front"],
            item["door_rear"],
            item["quality"],
        )
        samples.append(sample)
        if time_key(when) not in existing_times:
            db.add(
                TelemetrySampleModel(
                    trip_id=trip.trip_id,
                    event_time_utc=when,
                    latitude=item["latitude"],
                    longitude=item["longitude"],
                    speed_kmh=item["speed_kmh"],
                    heading=None,
                    door_front=item["door_front"],
                    door_rear=item["door_rear"],
                    source="SIMULATED",
                    quality=item["quality"],
                )
            )
            existing_times.add(time_key(when))

    stop_points = [
        StopPoint(item["stop_id"], item["sequence"], item["latitude"], item["longitude"], item["radius_m"])
        for item in scenario["stops"]
    ]
    detected = StopVisitDetector(stop_points).detect(samples)
    visit_ids: dict[str, str] = {}
    for visit in detected:
        visit_id = f"{trip.trip_id}:{visit.stop_id}:{int(visit.arrived_at.timestamp())}"
        visit_ids[visit.stop_id] = visit_id
        if db.get(StopVisitModel, visit_id) is None:
            db.add(
                StopVisitModel(
                    stop_visit_id=visit_id,
                    trip_id=trip.trip_id,
                    stop_id=visit.stop_id,
                    arrived_at=visit.arrived_at,
                    departed_at=visit.departed_at,
                    status=visit.status,
                    evidence=visit.evidence,
                    confidence=visit.confidence,
                    override_audit=[],
                )
            )
    db.flush()

    existing_events = set(
        db.scalars(select(PassengerEventModel.event_uuid).where(PassengerEventModel.trip_id == trip.trip_id))
    )
    for item in scenario["passenger_events"]:
        if item["event_uuid"] in existing_events:
            continue
        event_time = parse_time(item["event_time_utc"])
        assigned_stop = assign_visit(event_time, detected)
        visit_id = visit_ids.get(assigned_stop) if assigned_stop else None
        db.add(
            PassengerEventModel(
                event_uuid=item["event_uuid"],
                trip_id=trip.trip_id,
                source_id=f"{trip.trip_id}:{item['camera']}:sim",
                camera=item["camera"].upper(),
                direction=item["direction"],
                track_id=item["track_id"],
                video_frame=item["video_frame"],
                source_pts=item["source_pts"],
                event_time_utc=event_time,
                source_type=item["source_type"],
                source_confidence=item["source_confidence"],
                stop_visit_id=visit_id,
                assignment_status="AUTO_ASSIGNED" if visit_id else "UNASSIGNED",
                evidence_path=None,
                provenance=item["provenance"],
            )
        )
    db.commit()
    return {
        "trip_id": trip.trip_id,
        "stop_visits": len(detected),
        "events": len(scenario["passenger_events"]),
        "provenance": "SIMULATED_EVENTS",
        "idempotent": True,
    }


def flow_rows(db: Session, trip_id: str) -> list[dict]:
    trip = db.get(TripModel, trip_id)
    if trip is None:
        raise KeyError(trip_id)
    stops = list(
        db.scalars(select(StopModel).where(StopModel.route_id == trip.route_id).order_by(StopModel.sequence))
    )
    visits = list(db.scalars(select(StopVisitModel).where(StopVisitModel.trip_id == trip_id)))
    visit_by_stop = {visit.stop_id: visit for visit in visits}
    events = list(db.scalars(select(PassengerEventModel).where(PassengerEventModel.trip_id == trip_id)))
    for event in events:
        validate_event_source(db, trip_id, event.source_id, event.camera, event.provenance, event.source_type)
    sources = list(db.scalars(select(CameraSourceModel).where(CameraSourceModel.trip_id == trip_id)))
    required_flag = "SYNTHETIC_PAIRED" if trip.provenance == "SIMULATED" else "PAIRED_VERIFIED"
    paired_roles = {source.door_role for source in sources if source.paired_flag == required_flag}
    occupancy_valid = (
        trip.mode in {"PAIRED_VERIFIED", "SYNTHETIC_PAIRED"}
        and trip.coverage_complete
        and paired_roles == {"FRONT", "REAR"}
    )
    occupancy = trip.initial_occupancy
    rows = []
    for stop in stops:
        visit = visit_by_stop.get(stop.stop_id)
        counts = {"front_in": 0, "front_out": 0, "rear_in": 0, "rear_out": 0}
        if visit:
            for event in events:
                if (
                    event.stop_visit_id == visit.stop_visit_id
                    and event.assignment_status in {"AUTO_ASSIGNED", "MANUAL_ASSIGNED"}
                    and (event.camera, event.direction) in {("FRONT", "IN"), ("REAR", "OUT")}
                ):
                    counts[f"{event.camera.lower()}_{event.direction.lower()}"] += 1
        total_in = counts["front_in"]
        total_out = counts["rear_out"]
        warning = None
        onboard = None
        if occupancy_valid:
            occupancy += total_in - total_out
            onboard = occupancy
            if occupancy < 0:
                warning = "NEGATIVE_OCCUPANCY_REVIEW_REQUIRED"
        rows.append(
            {
                "trip_id": trip_id,
                "stop_id": stop.stop_id,
                "sequence": stop.sequence,
                "stop_name": stop.name,
                **counts,
                "total_in": total_in,
                "total_out": total_out,
                "boarding_count": total_in,
                "alighting_count": total_out,
                "onboard_after_stop": onboard,
                "arrived_at": visit.arrived_at.isoformat() if visit else None,
                "departed_at": visit.departed_at.isoformat() if visit and visit.departed_at else None,
                "visit_status": visit.status if visit else "NOT_OBSERVED",
                "evidence": visit.evidence if visit else None,
                "warning": warning,
            }
        )
    return rows


def update_assignment(
    db: Session, event_id: int, stop_visit_id: str | None, actor: str, reason: str
) -> PassengerEventModel:
    event = db.get(PassengerEventModel, event_id)
    if event is None:
        raise KeyError(event_id)
    validate_event_source(
        db, event.trip_id, event.source_id, event.camera, event.provenance, event.source_type
    )
    validate_passenger_direction(event.camera, event.direction)
    if stop_visit_id is not None:
        visit = db.get(StopVisitModel, stop_visit_id)
        if visit is None or visit.trip_id != event.trip_id:
            raise ValueError("StopVisit does not belong to this trip")
    before = {"stop_visit_id": event.stop_visit_id, "assignment_status": event.assignment_status}
    event.stop_visit_id = stop_visit_id
    event.assignment_status = "MANUAL_ASSIGNED" if stop_visit_id else "UNASSIGNED"
    after = {"stop_visit_id": event.stop_visit_id, "assignment_status": event.assignment_status}
    db.add(ReviewAuditModel(actor=actor, record_id=str(event.id), before=before, after=after, reason=reason))
    db.commit()
    db.refresh(event)
    return event


def load_scenario(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
