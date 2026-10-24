from __future__ import annotations

import csv
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .business import validate_passenger_direction
from .config import AppConfig, CameraConfig
from .counting import DirectionCounter
from .detector import PersonDetector
from .quality import QualitySelector
from .timebase import SourceClock
from .types import CrossingEvent, Detection
from .video import OpenCVVideoSource, VideoMetadata, redact_source
from .video_output import make_browser_video, validate_video

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
    status: str = "COMPLETED"


@dataclass(slots=True)
class OverlayOptions:
    boxes: bool = True
    track_ids: bool = True
    zones: bool = True
    trajectory: bool = True
    events: bool = True


class PipelineCancelled(RuntimeError):
    pass


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Windows readers/scanners can briefly lock an existing preview file."""
    for attempt in range(5):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_progress(path: Path, **values: object) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _scale_polygon(points: list[tuple[float, float]], shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    return np.asarray([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    camera: CameraConfig,
    counts: dict[str, int],
    *,
    options: OverlayOptions | None = None,
    trails: dict[str, deque[tuple[int, int]]] | None = None,
    frame_events: list[CrossingEvent] | None = None,
    banner_events: list[CrossingEvent] | None = None,
) -> np.ndarray:
    options = options or OverlayOptions()
    output = frame.copy()
    scale = max(1.0, min(frame.shape[1] / 960, frame.shape[0] / 540))
    stroke = max(2, round(2 * scale))
    origin = round(10 * scale)
    highlighted = {item.track_id for item in frame_events or []} if options.events else set()
    if options.zones:
        for label, polygon, color in (
            ("A", camera.zone_a, (0, 200, 255)),
            ("B", camera.zone_b, (255, 180, 0)),
        ):
            if len(polygon) < 3:
                continue
            contour = _scale_polygon(polygon, frame.shape)
            cv2.polylines(output, [contour], True, color, stroke)
            center = np.mean(contour, axis=0).astype(int)
            cv2.putText(
                output, f"ZONE {label}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, color, stroke
            )
        cv2.putText(
            output,
            "VALID: A -> B",
            (origin, frame.shape[0] - round(12 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            (255, 255, 255),
            stroke,
        )
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (0, 255, 255) if detection.track_id in highlighted else (30, 220, 90)
        if options.boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, stroke)
        if options.track_ids:
            track_label = f"{detection.track_id}  {detection.confidence:.2f}"
            font_scale = 0.45 * scale
            label_width = cv2.getTextSize(
                track_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, round(scale))
            )[0][0]
            cv2.putText(
                output,
                track_label,
                (
                    max(0, min(x1, frame.shape[1] - label_width - 2)),
                    max(y1 - round(8 * scale), round(18 * scale)),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                max(1, round(scale)),
                cv2.LINE_AA,
            )
        anchor = tuple(int(value) for value in detection.anchor(camera.anchor_mode))
        cv2.circle(output, anchor, round(5 * scale), color, -1)
        if options.trajectory and trails and len(trails.get(detection.track_id, ())) > 1:
            points = np.asarray(trails[detection.track_id], dtype=np.int32)
            cv2.polylines(output, [points], False, color, stroke)
    cv2.rectangle(
        output, (0, 0), (min(frame.shape[1], round(530 * scale)), round(42 * scale)), (10, 17, 30), -1
    )
    label = "FRONT - BOARDING / IN" if camera.name.lower() == "front" else "REAR - ALIGHTING / OUT"
    cv2.putText(
        output,
        f"{label}: {counts[camera.event_direction.value]}",
        (origin, round(28 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7 * scale,
        (255, 255, 255),
        stroke,
        cv2.LINE_AA,
    )
    shown_events = banner_events if banner_events is not None else frame_events
    if options.events and shown_events:
        cv2.putText(
            output,
            f"+{len(shown_events)} {camera.event_direction.value}  "
            + ", ".join(item.track_id for item in shown_events[:2]),
            (origin, min(frame.shape[0] - round(30 * scale), round(78 * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            (0, 255, 255),
            stroke,
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
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            row["direction"] = event.direction.value
            row["created_at"] = event.created_at.isoformat()
            row["capture_time_utc"] = event.capture_time_utc.isoformat() if event.capture_time_utc else ""
            writer.writerow(row)
    _replace_with_retry(temporary, path)


class CameraPipeline:
    def __init__(
        self, camera: CameraConfig, config: AppConfig, detector: PersonDetector | None = None
    ) -> None:
        validate_passenger_direction(camera.name, camera.event_direction)
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
        overlay: OverlayOptions | None = None,
        cancel_file: Path | None = None,
        stop_file: Path | None = None,
        start_frame: int = 0,
    ) -> PipelineResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source = OpenCVVideoSource(source_path)
        metadata = source.open()
        try:
            if start_frame < 0:
                raise ValueError("Frame bắt đầu phải không âm")
            if start_frame:
                source.seek(start_frame)
        except Exception:
            source.close()
            raise
        available_frames = metadata.frame_count - start_frame if metadata.frame_count is not None else None
        total_frames = min(max_frames, available_frames) if max_frames and available_frames is not None else available_frames or max_frames
        clock = SourceClock(recording_start_time or datetime.now(timezone.utc), self.camera.time_offset_ms)
        generation = source.generation
        annotated = destination / f"{self.camera.name}_annotated.mp4"
        intermediate = destination / f"{self.camera.name}_encoding.mp4"
        events_file = destination / f"{self.camera.name}_events.csv"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            source.close()
            raise RuntimeError(f"Không tạo được video output: {intermediate}")
        events: list[CrossingEvent] = []
        counts = {"IN": 0, "OUT": 0}
        progress_path = destination / f"{self.camera.name}_progress.json"
        frame_index = start_frame
        processed_frames = 0
        trails: dict[str, deque[tuple[int, int]]] = {}
        recent_events: list[tuple[CrossingEvent, int]] = []
        stopped = False
        device_name = "CPU"
        peak_vram_mb: float | None = None
        if isinstance(self.detector, PersonDetector):
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                device_name = torch.cuda.get_device_name(0)
        started = time.perf_counter()
        _write_progress(
            progress_path,
            frames=0,
            total_frames=total_frames,
            start_frame=start_frame,
            next_frame_index=start_frame,
            events=0,
            counts=counts,
            elapsed_s=0.0,
            processing_fps=0.0,
            status="RUNNING",
        )
        try:
            while max_frames is None or processed_frames < max_frames:
                if cancel_file and cancel_file.is_file():
                    raise PipelineCancelled("Inference cancelled by operator")
                if stop_file and stop_file.is_file():
                    stopped = True
                    break
                ok, frame, timestamp_s = source.read()
                if not ok or frame is None:
                    break
                if source.generation != generation:
                    clock.reconnect()
                    generation = source.generation
                capture_time = clock.map_pts(timestamp_s)
                detections = self.detector.track(frame)
                frame_events: list[CrossingEvent] = []
                for detection in detections:
                    trail = trails.setdefault(detection.track_id, deque(maxlen=25))
                    trail.append(tuple(int(value) for value in detection.anchor(self.camera.anchor_mode)))
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
                        frame_events.append(event)
                        recent_events.append((event, frame_index))
                        _write_events(events_file, events)
                        if on_event:
                            on_event(event, crops)
                self.counter.prune_stale(frame_index)
                active = set(self.counter.states)
                trails = {key: value for key, value in trails.items() if key in active}
                recent_events = [
                    (event, first_frame)
                    for event, first_frame in recent_events
                    if frame_index - first_frame < max(4, round(metadata.fps * 0.7))
                ]
                annotated_frame = annotate_frame(
                    frame,
                    detections,
                    self.camera,
                    counts,
                    options=overlay,
                    trails=trails,
                    frame_events=frame_events,
                    banner_events=[event for event, _ in recent_events],
                )
                writer.write(annotated_frame)
                frame_index += 1
                processed_frames += 1
                if processed_frames == 1 or processed_frames % 15 == 0 or frame_events:
                    preview = destination / f"{self.camera.name}_preview.jpg"
                    temporary = preview.with_suffix(".tmp.jpg")
                    display_frame = annotated_frame
                    if annotated_frame.shape[1] > 1280:
                        width = annotated_frame.shape[1]
                        display_frame = cv2.resize(
                            annotated_frame, (1280, round(annotated_frame.shape[0] * 1280 / width))
                        )
                    if cv2.imwrite(str(temporary), display_frame):
                        try:
                            _replace_with_retry(temporary, preview)
                        except PermissionError:
                            logger.warning("Preview file temporarily locked: %s", preview)
                            temporary.unlink(missing_ok=True)
                if processed_frames % 10 == 0 or frame_events:
                    elapsed_now = time.perf_counter() - started
                    _write_progress(
                        progress_path,
                        frames=processed_frames,
                        total_frames=total_frames,
                        start_frame=start_frame,
                        next_frame_index=frame_index,
                        events=len(events),
                        counts=counts,
                        elapsed_s=elapsed_now,
                        processing_fps=processed_frames / elapsed_now if elapsed_now else 0.0,
                        status="RUNNING",
                    )
        except PipelineCancelled:
            _write_progress(
                progress_path,
                frames=processed_frames,
                total_frames=total_frames,
                events=len(events),
                counts=counts,
                elapsed_s=time.perf_counter() - started,
                processing_fps=0.0,
                status="CANCELLED",
            )
            raise
        except Exception:
            _write_progress(
                progress_path,
                frames=processed_frames,
                total_frames=total_frames,
                events=len(events),
                counts=counts,
                elapsed_s=time.perf_counter() - started,
                processing_fps=0.0,
                status="FAILED",
            )
            raise
        finally:
            writer.release()
            source.close()
        elapsed = time.perf_counter() - started
        try:
            if processed_frames:
                validate_video(intermediate, processed_frames)
                make_browser_video(intermediate, annotated, processed_frames)
            elif not stopped:
                raise RuntimeError("Nguồn video không cung cấp frame nào")
            else:
                intermediate.unlink(missing_ok=True)
        except Exception:
            _write_progress(
                progress_path,
                frames=processed_frames,
                total_frames=total_frames,
                events=len(events),
                counts=counts,
                elapsed_s=elapsed,
                processing_fps=processed_frames / elapsed if elapsed else 0.0,
                status="FAILED",
            )
            raise
        if isinstance(self.detector, PersonDetector):
            import torch

            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        _write_events(events_file, events)
        _write_progress(
            progress_path,
            frames=processed_frames,
            total_frames=total_frames,
            start_frame=start_frame,
            next_frame_index=frame_index,
            events=len(events),
            counts=counts,
            elapsed_s=elapsed,
            processing_fps=processed_frames / elapsed if elapsed else 0.0,
            status="STOPPED" if stopped else "COMPLETED",
        )
        result = PipelineResult(
            camera=self.camera.name,
            input_path=redact_source(source_path),
            annotated_path=str(annotated),
            events_path=str(events_file),
            events=events,
            frames=processed_frames,
            elapsed_s=elapsed,
            processing_fps=processed_frames / elapsed if elapsed else 0.0,
            metadata=metadata,
            device=device_name,
            peak_vram_mb=peak_vram_mb,
            status="STOPPED" if stopped else "COMPLETED",
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
                    "start_frame": start_frame,
                    "next_frame_index": frame_index,
                    "elapsed_s": result.elapsed_s,
                    "processing_fps": result.processing_fps,
                    "device": result.device,
                    "peak_vram_mb": result.peak_vram_mb,
                    "counts": counts,
                    "metadata": asdict(metadata),
                    "result_kind": "REAL_INFERENCE",
                    "status": result.status,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return result
