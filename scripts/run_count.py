from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.models import (  # noqa: E402
    CameraSourceModel,
    PassengerEventModel,
    RouteModel,
    TelemetrySampleModel,
    TripModel,
)
from backend.schemas import TelemetryCreate  # noqa: E402
from busreid.business import validate_passenger_direction  # noqa: E402
from busreid.config import load_config  # noqa: E402
from busreid.pipeline import OverlayOptions  # noqa: E402
from busreid.service import detect_and_assign, flow_rows, validate_event_source  # noqa: E402
from busreid.video import redact_source  # noqa: E402
from busreid.workers import DualCameraRunner  # noqa: E402


def source_type(value: str) -> str:
    return "RTSP" if value.lower().startswith(("rtsp://", "rtsps://")) else "MP4"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("--recording-start-time phải có timezone, ví dụ 2026-10-19T08:00:00+07:00")
    return result.astimezone(timezone.utc)


def event_uuid(
    trip_id: str, camera: str, track_id: str, frame: int, direction: str, recording_start: datetime | None
) -> str:
    anchor = recording_start.isoformat() if recording_start else "PROCESSING_TIME"
    raw = f"{trip_id}|{camera}|{anchor}|{track_id}|{frame}|{direction}".encode()
    return "AI-" + hashlib.sha256(raw).hexdigest()[:32]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run independent YOLO+ByteTrack doorway counters concurrently"
    )
    parser.add_argument("--front")
    parser.add_argument("--rear")
    parser.add_argument("--config", default="configs/cameras.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trip-id", default=None)
    parser.add_argument("--telemetry-json", type=Path, help="JSON array of timezone-aware telemetry samples")
    parser.add_argument(
        "--telemetry-synchronized",
        action="store_true",
        help="Operator attests telemetry and video clocks refer to the same physical trip",
    )
    parser.add_argument(
        "--recording-start-time",
        help="ISO-8601 capture time with timezone; required for trustworthy file clock",
    )
    parser.add_argument(
        "--paired-verified", action="store_true", help="Only if same trip and time synchronization are proven"
    )
    parser.add_argument("--cancel-file", type=Path, help="Local cancellation marker checked between frames")
    for layer in ("boxes", "track-ids", "zones", "trajectory", "events"):
        parser.add_argument(f"--hide-{layer}", action="store_true")
    args = parser.parse_args(argv)
    if not args.front and not args.rear:
        parser.error("Cần ít nhất --front hoặc --rear")
    recording_start = parse_time(args.recording_start_time)
    if args.telemetry_synchronized and not recording_start:
        parser.error("--telemetry-synchronized requires --recording-start-time")
    config = load_config(args.config)
    sources = {key: value for key, value in (("front", args.front), ("rear", args.rear)) if value}
    if args.paired_verified and (not args.telemetry_synchronized or len(sources) != 2):
        parser.error("--paired-verified requires both cameras and synchronized telemetry/capture time")
    Base.metadata.create_all(bind=engine)
    created = datetime.now(timezone.utc)
    trip_id = args.trip_id or f"VIDEO-{created.strftime('%Y%m%dT%H%M%S%fZ')}"
    with SessionLocal() as db:
        existing_trip = db.get(TripModel, trip_id)
        if existing_trip is not None:
            if existing_trip.provenance not in {"REAL_VIDEO_AI", "REAL_OR_USER_INPUT"}:
                raise ValueError("Real AI inference cannot be added to a synthetic trip")
            for camera, source in sources.items():
                existing_source = db.get(CameraSourceModel, f"{trip_id}:{camera}")
                if existing_source and (
                    existing_source.trip_id != trip_id
                    or existing_source.door_role != camera.upper()
                    or existing_source.safe_uri != redact_source(source)
                ):
                    raise ValueError(f"Source mismatch for {camera}")
    runner = DualCameraRunner(config)
    overlay = OverlayOptions(
        boxes=not args.hide_boxes,
        track_ids=not args.hide_track_ids,
        zones=not args.hide_zones,
        trajectory=not args.hide_trajectory,
        events=not args.hide_events,
    )
    outcome = runner.run(
        sources,
        args.output,
        args.max_frames,
        recording_start_time=recording_start,
        overlay=overlay,
        cancel_file=args.cancel_file,
    )
    if outcome.cancelled:
        print("Inference cancelled by operator", file=sys.stderr)
        return 2

    with SessionLocal() as db:
        if db.get(RouteModel, "VIDEO-ONLY") is None:
            db.add(
                RouteModel(
                    route_id="VIDEO-ONLY",
                    name="Video counting without synchronized route telemetry",
                    direction="N/A",
                    source="USER_OR_PUBLIC_VIDEO",
                    retrieved_date=created.date().isoformat(),
                    verified=False,
                )
            )
        trip = db.get(TripModel, trip_id)
        if trip is None:
            trip = TripModel(
                trip_id=trip_id,
                route_id="VIDEO-ONLY",
                vehicle_id="UNKNOWN",
                start_time_utc=recording_start or created,
                end_time_utc=None,
                mode="PAIRED_VERIFIED" if args.paired_verified and len(sources) == 2 else "UNPAIRED",
                initial_occupancy=0,
                status="COMPLETED" if not outcome.errors else "PARTIAL",
                coverage_complete=False,
                provenance="REAL_VIDEO_AI",
            )
            db.add(trip)
        else:
            trip.status = "COMPLETED" if not outcome.errors else "PARTIAL"
            trip.provenance = "REAL_VIDEO_AI"
            if args.paired_verified:
                trip.mode = "PAIRED_VERIFIED"
        for camera, source in sources.items():
            source_id = f"{trip_id}:{camera}"
            existing_source = db.get(CameraSourceModel, source_id)
            if existing_source is None:
                db.add(
                    CameraSourceModel(
                        source_id=source_id,
                        trip_id=trip_id,
                        door_role=camera.upper(),
                        input_type=source_type(source),
                        safe_uri=redact_source(source),
                        calibration_id=f"{camera}-yaml",
                        paired_flag=(
                            "PAIRED_VERIFIED"
                            if args.paired_verified
                            else "SYNCHRONIZED"
                            if args.telemetry_synchronized
                            else "UNPAIRED"
                        ),
                        time_offset_ms=config.cameras[camera].time_offset_ms,
                    )
                )
            else:
                existing_source.paired_flag = (
                    "PAIRED_VERIFIED"
                    if args.paired_verified
                    else "SYNCHRONIZED"
                    if args.telemetry_synchronized
                    else "UNPAIRED"
                )
        db.flush()
        if args.telemetry_json:
            samples = json.loads(args.telemetry_json.read_text(encoding="utf-8"))
            if not isinstance(samples, list):
                raise ValueError("Telemetry JSON must be an array")
            existing_times = {
                row.event_time_utc.replace(tzinfo=timezone.utc)
                if row.event_time_utc.tzinfo is None
                else row.event_time_utc
                for row in db.query(TelemetrySampleModel).filter_by(trip_id=trip_id)
            }
            for raw in samples:
                sample = TelemetryCreate.model_validate(raw)
                if sample.source == "SIMULATED":
                    raise ValueError("Real trip cannot ingest synthetic telemetry")
                when = sample.event_time_utc.astimezone(timezone.utc)
                if when not in existing_times:
                    db.add(
                        TelemetrySampleModel(
                            trip_id=trip_id, **{**sample.model_dump(), "event_time_utc": when}
                        )
                    )
                    existing_times.add(when)
        for camera, result in outcome.results.items():
            for event in result.events:
                validate_passenger_direction(
                    camera, event.direction, event.crossing_direction, event.occupancy_delta
                )
                capture_time = event.capture_time_utc or event.created_at
                validate_event_source(
                    db,
                    trip_id,
                    f"{trip_id}:{camera}",
                    camera.upper(),
                    "REAL_VIDEO_AI",
                    "REAL_VIDEO_AI" if recording_start else "REAL_VIDEO_AI_PROCESSING_TIME",
                )
                uuid_value = event_uuid(
                    trip_id, camera, event.track_id, event.frame_index, event.direction.value, recording_start
                )
                if db.query(PassengerEventModel).filter_by(event_uuid=uuid_value).first():
                    continue
                db.add(
                    PassengerEventModel(
                        event_uuid=uuid_value,
                        trip_id=trip_id,
                        source_id=f"{trip_id}:{camera}",
                        camera=camera.upper(),
                        crossing_direction=event.crossing_direction.value,
                        direction=event.direction.value,
                        occupancy_delta=event.occupancy_delta,
                        direction_status="SOURCE_ASSERTED",
                        track_id=event.track_id,
                        video_frame=event.frame_index,
                        source_pts=event.timestamp_s,
                        event_time_utc=capture_time,
                        source_type="REAL_VIDEO_AI" if recording_start else "REAL_VIDEO_AI_PROCESSING_TIME",
                        source_confidence=event.confidence,
                        stop_visit_id=None,
                        assignment_status="UNASSIGNED",
                        evidence_path=event.crop_path,
                        provenance="REAL_VIDEO_AI",
                    )
                )
        db.commit()
        aggregation = detect_and_assign(db, trip_id)
        flow = flow_rows(db, trip_id)
    summary = {
        "trip_id": trip_id,
        "mode": trip.mode,
        "coverage_complete": False,
        "clock": "CAPTURE_ANCHORED" if recording_start else "PROCESSING_TIME_NOT_CAPTURE_TIME",
        "sources": {key: redact_source(value) for key, value in sources.items()},
        "workers": {
            key: {
                "frames": value.frames,
                "events": len(value.events),
                "processing_fps": value.processing_fps,
                "device": value.device,
                "peak_vram_mb": value.peak_vram_mb,
                "annotated_path": value.annotated_path,
            }
            for key, value in outcome.results.items()
        },
        "errors": outcome.errors,
        "result_kind": "REAL_VIDEO_AI",
        "video_model_validation": "NO_GROUND_TRUTH_NA",
        "aggregation": aggregation,
        "flow": flow,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if outcome.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
