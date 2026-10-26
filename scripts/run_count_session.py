"""Standalone AI counting session. Never creates a Trip or telemetry records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from busreid.config import load_config  # noqa: E402
from busreid.pipeline import OverlayOptions  # noqa: E402
from busreid.video import redact_source  # noqa: E402
from busreid.workers import DualCameraRunner  # noqa: E402


def parse_source(value: str) -> str | int:
    if value.startswith("webcam:"):
        index = value.removeprefix("webcam:")
        if not index.isdecimal():
            raise ValueError("Webcam index phải là số nguyên không âm")
        return int(index)
    return value


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Independent bidirectional FRONT/REAR AI counting")
    parser.add_argument("--front")
    parser.add_argument("--rear")
    parser.add_argument("--config", default="configs/cameras.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-file", type=Path, required=True)
    for layer in ("boxes", "track-ids", "zones", "trajectory", "events"):
        parser.add_argument(f"--hide-{layer}", action="store_true")
    args = parser.parse_args(argv)
    sources = {
        camera: parse_source(source)
        for camera, source in (("front", args.front), ("rear", args.rear))
        if source is not None
    }
    if not sources:
        parser.error("Chọn ít nhất một camera")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    overlay = OverlayOptions(
        boxes=not args.hide_boxes,
        track_ids=not args.hide_track_ids,
        zones=not args.hide_zones,
        trajectory=not args.hide_trajectory,
        events=not args.hide_events,
    )
    outcome = DualCameraRunner(load_config(args.config)).run(
        sources,
        output,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        overlay=overlay,
        stop_file=args.stop_file,
    )
    status = (
        "FAILED"
        if not outcome.results
        else "STOPPED"
        if any(result.status == "STOPPED" for result in outcome.results.values())
        else "COMPLETED"
    )
    summary = {
        "session_id": output.name,
        "status": status,
        "provenance": "REAL_VIDEO_AI",
        "association": "INDEPENDENT_SESSION_NO_TRIP",
        "sources": {camera: redact_source(source) for camera, source in sources.items()},
        "cameras": {
            camera: {
                "status": result.status,
                "frames": result.frames,
                "count": len(result.events),
                "counts": {
                    "IN": sum(event.direction.value == "IN" for event in result.events),
                    "OUT": sum(event.direction.value == "OUT" for event in result.events),
                },
                "occupancy_delta": sum(event.occupancy_delta for event in result.events),
                "crossings": len(result.events),
                "processing_fps": result.processing_fps,
            }
            for camera, result in outcome.results.items()
        },
        "errors": outcome.errors,
    }
    temporary = output / "session_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "session_summary.json")
    if outcome.errors:
        print(f"Camera errors: {outcome.errors}", file=sys.stderr)
    return 0 if outcome.results else 1


if __name__ == "__main__":
    raise SystemExit(main())
