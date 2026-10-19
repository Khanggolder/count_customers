from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import AppConfig, CameraConfig
from .counting import DirectionCounter
from .detector import PersonDetector
from .quality import QualitySelector
from .timebase import SourceClock
from .types import CrossingEvent, Detection
from .video import OpenCVVideoSource, VideoMetadata, redact_source

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    camera: str
    input_path: str
    annotated_path: str
    events_path: str
    events: list[CrossingEvent]
    frames: int
    elapsed_s: float
    processing_fps: float
    metadata: VideoMetadata
    device: str
    peak_vram_mb: float | None


def _scale_polygon(points: list[tuple[float, float]], shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    return np.asarray([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    camera: CameraConfig,
    counts: dict[str, int],
) -> np.ndarray:
    output = frame.copy()
    cv2.polylines(output, [_scale_polygon(camera.zone_a, frame.shape)], True, (0, 200, 255), 2)
    cv2.polylines(output, [_scale_polygon(camera.zone_b, frame.shape)], True, (255, 180, 0), 2)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (30, 220, 90), 2)
        cv2.putText(
            output,
            f"{detection.track_id} {detection.confidence:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 220, 90),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (310, 42), (10, 17, 30), -1)
    cv2.putText(
        output,
        f"{camera.name.upper()}  IN {counts['IN']}  OUT {counts['OUT']}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _write_events(path: Path, events: list[CrossingEvent]) -> None:
    fields = [
        "camera",
        "track_id",
        "direction",
        "frame_index",
        "timestamp_s",
        "confidence",
        "crop_path",
        "synthetic",
        "created_at",
        "capture_time_utc",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            row["direction"] = event.direction.value
            row["created_at"] = event.created_at.isoformat()
            row["capture_time_utc"] = event.capture_time_utc.isoformat() if event.capture_time_utc else ""
            writer.writerow(row)


class CameraPipeline:
    def __init__(
        self, camera: CameraConfig, config: AppConfig, detector: PersonDetector | None = None
    ) -> None:
        self.camera = camera
        self.detector = detector or PersonDetector(camera.name, **asdict(config.detector))
        self.counter = DirectionCounter(
            camera=camera.name,
            zone_a=camera.zone_a,
            zone_b=camera.zone_b,
            forward_direction=camera.event_direction,
            min_stable_frames=camera.min_stable_frames,
            debounce_frames=camera.debounce_frames,
            state_ttl_frames=camera.state_ttl_frames,
            anchor_mode=camera.anchor_mode,
        )
        self.quality = QualitySelector()

    def run(
        self,
        source_path: str,
        output_dir: str | Path,
        max_frames: int | None = None,
        on_event: Callable[[CrossingEvent, list[np.ndarray]], None] | None = None,
        recording_start_time: datetime | None = None,
    ) -> PipelineResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source = OpenCVVideoSource(source_path)
        metadata = source.open()
        clock = SourceClock(recording_start_time or datetime.now(timezone.utc), self.camera.time_offset_ms)
        generation = source.generation
        annotated = destination / f"{self.camera.name}_annotated.mp4"
        events_file = destination / f"{self.camera.name}_events.csv"
        writer = cv2.VideoWriter(
            str(annotated),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            source.close()
            raise RuntimeError(f"Không tạo được video output: {annotated}")
        events: list[CrossingEvent] = []
        counts = {"IN": 0, "OUT": 0}
        progress_path = destination / f"{self.camera.name}_progress.json"
        frame_index = 0
        device_name = "CPU"
        peak_vram_mb: float | None = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                device_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        started = time.perf_counter()
        try:
            while max_frames is None or frame_index < max_frames:
                ok, frame, timestamp_s = source.read()
                if not ok or frame is None:
                    break
                if source.generation != generation:
                    clock.reconnect()
                    generation = source.generation
                capture_time = clock.map_pts(timestamp_s)
                detections = self.detector.track(frame)
                for detection in detections:
                    self.quality.consider(frame, detection, frame_index)
                    event = self.counter.update(detection, frame.shape, frame_index, timestamp_s)
                    if event is not None:
                        event.capture_time_utc = capture_time if recording_start_time else None
                        crops = [item.image for item in self.quality.pop(detection.track_id)]
                        if crops:
                            crop_dir = destination / "crops"
                            crop_dir.mkdir(exist_ok=True)
                            crop_path = (
                                crop_dir
                                / f"{self.camera.name}_{detection.track_id.replace(':', '_')}_{frame_index}.jpg"
                            )
                            cv2.imwrite(str(crop_path), crops[0])
                            event.crop_path = str(crop_path)
                        counts[event.direction.value] += 1
                        events.append(event)
                        if on_event:
                            on_event(event, crops)
                self.counter.prune_stale(frame_index)
                writer.write(annotate_frame(frame, detections, self.camera, counts))
                frame_index += 1
                if frame_index % 10 == 0:
                    progress_path.write_text(
                        json.dumps(
                            {
                                "frames": frame_index,
                                "total_frames": metadata.frame_count,
                                "events": len(events),
                                "status": "RUNNING",
                            }
                        ),
                        encoding="utf-8",
                    )
        finally:
            writer.release()
            source.close()
        elapsed = time.perf_counter() - started
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        _write_events(events_file, events)
        progress_path.write_text(
            json.dumps(
                {
                    "frames": frame_index,
                    "total_frames": metadata.frame_count,
                    "events": len(events),
                    "status": "COMPLETED",
                }
            ),
            encoding="utf-8",
        )
        result = PipelineResult(
            camera=self.camera.name,
            input_path=redact_source(source_path),
            annotated_path=str(annotated),
            events_path=str(events_file),
            events=events,
            frames=frame_index,
            elapsed_s=elapsed,
            processing_fps=frame_index / elapsed if elapsed else 0.0,
            metadata=metadata,
            device=device_name,
            peak_vram_mb=peak_vram_mb,
        )
        summary_path = destination / f"{self.camera.name}_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "camera": result.camera,
                    "input_path": result.input_path,
                    "annotated_path": result.annotated_path,
                    "events_path": result.events_path,
                    "frames": result.frames,
                    "elapsed_s": result.elapsed_s,
                    "processing_fps": result.processing_fps,
                    "device": result.device,
                    "peak_vram_mb": result.peak_vram_mb,
                    "counts": counts,
                    "metadata": asdict(metadata),
                    "result_kind": "REAL_INFERENCE",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result
