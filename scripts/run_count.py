from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.database import Base, SessionLocal, engine  # noqa: E402
from backend.models import CameraSourceModel, CountEventModel, SessionModel  # noqa: E402
from busreid.config import load_config  # noqa: E402
from busreid.pipeline import CameraPipeline  # noqa: E402
from busreid.video import redact_source  # noqa: E402


def source_type(value: str) -> str:
    return "rtsp" if value.lower().startswith(("rtsp://", "rtsps://")) else "mp4"


def provenance_for(value: str) -> str | None:
    path = Path(value)
    manifest = path.with_suffix(".source.json")
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return str(data.get("source_page"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=str)
    parser.add_argument("--rear", type=str)
    parser.add_argument("--config", default="configs/cameras.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--paired", action="store_true", help="Chỉ bật khi hai nguồn thật sự cùng chuyến")
    args = parser.parse_args()
    if not args.front and not args.rear:
        parser.error("Cần ít nhất --front hoặc --rear")
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    pipeline_results = []
    for camera_name, source in (("front", args.front), ("rear", args.rear)):
        if not source:
            continue
        pipeline = CameraPipeline(config.cameras[camera_name], config)
        result = pipeline.run(source, args.output, max_frames=args.max_frames)
        pipeline_results.append(result)
        results.append({"camera": camera_name, "frames": result.frames, "events": len(result.events), "processing_fps": result.processing_fps, "device": result.device, "peak_vram_mb": result.peak_vram_mb, "annotated_path": result.annotated_path})
    Base.metadata.create_all(bind=engine)
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        session = SessionModel(
            name=args.session_name or args.output.name,
            status="COMPLETED",
            mode="PAIRED" if args.paired and args.front and args.rear else "UNPAIRED",
            started_at=now,
            ended_at=datetime.now(timezone.utc),
        )
        db.add(session)
        db.flush()
        for camera_name, source in (("front", args.front), ("rear", args.rear)):
            if source:
                db.add(CameraSourceModel(session_id=session.id, camera=camera_name, source_type=source_type(source), uri=source, provenance=provenance_for(source), verified=Path(source).is_file()))
        passenger_counter = 1
        for result in pipeline_results:
            for event in result.events:
                passenger_id = None
                if event.direction.value == "IN":
                    passenger_id = f"P{passenger_counter:04d}"
                    passenger_counter += 1
                db.add(CountEventModel(session_id=session.id, camera=event.camera, direction=event.direction.value, track_id=event.track_id, timestamp_s=event.timestamp_s, frame_index=event.frame_index, confidence=event.confidence, crop_path=event.crop_path, passenger_id=passenger_id, synthetic=False))
        db.commit()
        session_id = session.id
    summary = {"session_id": session_id, "mode": "PAIRED" if args.paired and args.front and args.rear else "UNPAIRED", "input_layout": "TWO_CAMERA" if args.front and args.rear else "SINGLE_CAMERA", "sources": {key: redact_source(value) if value else None for key, value in (("front", args.front), ("rear", args.rear))}, "results": results, "real_inference": True}
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

