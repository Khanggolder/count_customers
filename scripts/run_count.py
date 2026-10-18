from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.models import CameraSourceModel, PassengerEventModel, RouteModel, TripModel  # noqa: E402
from busreid.config import load_config  # noqa: E402
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


def event_uuid(trip_id: str, camera: str, track_id: str, frame: int, direction: str) -> str:
    raw = f"{trip_id}|{camera}|{track_id}|{frame}|{direction}".encode()
    return "AI-" + hashlib.sha256(raw).hexdigest()[:32]


def main() -> int:
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
    parser.add_argument(
        "--recording-start-time",
        help="ISO-8601 capture time with timezone; required for trustworthy file clock",
    )
    parser.add_argument(
        "--paired-verified", action="store_true", help="Only if same trip and time synchronization are proven"
    )
    args = parser.parse_args()
    if not args.front and not args.rear:
        parser.error("Cần ít nhất --front hoặc --rear")
    recording_start = parse_time(args.recording_start_time)
    config = load_config(args.config)
    sources = {key: value for key, value in (("front", args.front), ("rear", args.rear)) if value}
    runner = DualCameraRunner(config)
    outcome = runner.run(sources, args.output, args.max_frames)

    Base.metadata.create_all(bind=engine)
    created = datetime.now(timezone.utc)
    trip_id = args.trip_id or f"VIDEO-{created.strftime('%Y%m%dT%H%M%SZ')}"
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
        if db.get(TripModel, trip_id) is not None:
            raise RuntimeError(f"trip-id already exists: {trip_id}")
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
        for camera, source in sources.items():
            db.add(
                CameraSourceModel(
                    source_id=f"{trip_id}:{camera}",
                    trip_id=trip_id,
                    door_role=camera.upper(),
                    input_type=source_type(source),
                    safe_uri=redact_source(source),
                    calibration_id=f"{camera}-yaml",
                    paired_flag=trip.mode,
                    time_offset_ms=config.cameras[camera].time_offset_ms,
                )
            )
        db.flush()
        for camera, result in outcome.results.items():
            offset = timedelta(milliseconds=config.cameras[camera].time_offset_ms)
            for event in result.events:
                capture_time = (
                    recording_start + timedelta(seconds=event.timestamp_s) + offset
                    if recording_start
                    else event.created_at
                )
                db.add(
                    PassengerEventModel(
                        event_uuid=event_uuid(
                            trip_id, camera, event.track_id, event.frame_index, event.direction.value
                        ),
                        trip_id=trip_id,
                        source_id=f"{trip_id}:{camera}",
                        camera=camera.upper(),
                        direction=event.direction.value,
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
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if outcome.errors and not outcome.results else 0


if __name__ == "__main__":
    raise SystemExit(main())
