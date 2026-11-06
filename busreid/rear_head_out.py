"""Experimental REAR head tracking and A -> B OUT counting, isolated from Trip data."""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import RearHeadOutConfig
from .counting import Zone, point_in_polygon
from .pipeline import OverlayOptions, _replace_with_retry, _write_progress
from .rear_occupancy import HeadDetector
from .timebase import SourceClock
from .types import Detection
from .video import OpenCVVideoSource, VideoMetadata, redact_source
from .video_output import make_browser_video, validate_video


class HeadTracker(HeadDetector):
    """One single-class YOLO model and one ByteTrack/BoT-SORT state per run."""

    def __init__(self, config: RearHeadOutConfig) -> None:
        super().__init__(config)
        self.tracker = config.tracker
        self.last_detection_count = 0

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            source=frame, persist=True, classes=[0], conf=self.confidence, iou=self.iou,
            imgsz=self.imgsz, device=self.device, tracker=self.tracker, verbose=False,
        )
        self.last_detection_count = len(results[0].boxes.xyxy) if results and results[0].boxes is not None else 0
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        ids = boxes.id.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        return [
            Detection(f"rear:head{track_id:05d}", tuple(int(v) for v in box), float(confidence))
            for box, track_id, confidence in zip(xyxy, ids, confs, strict=True)
        ]


@dataclass(slots=True)
class HeadOutEvent:
    camera: str
    track_id: str
    direction: str
    frame_index: int
    timestamp_s: float
    confidence: float
    crossing_zone: str
    anchor_x: float
    anchor_y: float
    origin_zone: str = "A"
    capture_time_utc: str = ""


@dataclass(slots=True)
class HeadZoneState:
    stable: Zone = Zone.NONE
    candidate: Zone = Zone.NONE
    candidate_frames: int = 0
    armed: bool = False
    transitioned: bool = False
    counted: bool = False
    blocked: bool = False
    last_seen: int = -1
    last_point: tuple[float, float] | None = None
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=25))


class HeadZoneCounter:
    """Confirm A, transition outside both zones, then stable B for one OUT."""

    def __init__(self, config: RearHeadOutConfig) -> None:
        self.config = config
        self.states: dict[str, HeadZoneState] = {}
        self.total_out = 0
        self.stable_a_confirmations = 0
        self.transition_confirmations = 0
        self.stable_b_confirmations = 0

    def _zone(self, point: tuple[float, float]) -> Zone:
        if not point_in_polygon(point, self.config.roi):
            return Zone.NONE
        if point_in_polygon(point, self.config.zone_a):
            return Zone.A
        if point_in_polygon(point, self.config.zone_b):
            return Zone.B
        return Zone.NONE

    def _state(
        self, track_id: str, point: tuple[float, float], frame_index: int,
        active_track_ids: set[str],
    ) -> HeadZoneState:
        current = self.states.get(track_id)
        if current is not None:
            return current
        candidates = [
            (old_id, state)
            for old_id, state in self.states.items()
            if old_id not in active_track_ids and state.last_point is not None
            and 0 < frame_index - state.last_seen <= self.config.id_switch_grace_frames
            and np.linalg.norm(np.subtract(state.last_point, point)) <= self.config.id_switch_max_distance
        ]
        if len(candidates) == 1:
            old_id, state = candidates[0]
            del self.states[old_id]
            self.states[track_id] = state
            return state
        state = HeadZoneState(blocked=len(candidates) > 1)
        self.states[track_id] = state
        return state

    def update(
        self, detection: Detection, shape: tuple[int, ...], frame_index: int, timestamp_s: float,
        active_track_ids: set[str] | None = None,
    ) -> HeadOutEvent | None:
        height, width = shape[:2]
        x, y = detection.anchor("center")
        point = (x / width, y / height)
        state = self._state(detection.track_id, point, frame_index, active_track_ids or {detection.track_id})
        state.last_seen = frame_index
        state.last_point = point
        state.trail.append((round(x), round(y)))
        if not point_in_polygon(point, self.config.roi):
            # Leaving the observation ROI is not a valid transition across the door.
            state.stable = state.candidate = Zone.NONE
            state.candidate_frames = 0
            state.armed = state.transitioned = state.counted = False
            return None
        zone = self._zone(point)
        if zone != state.candidate:
            state.candidate, state.candidate_frames = zone, 1
        else:
            state.candidate_frames += 1
        if state.candidate_frames < self.config.min_stable_frames or zone == state.stable:
            return None
        previous = state.stable
        state.stable = zone
        if zone == Zone.A:
            self.stable_a_confirmations += 1
            # B -> A cannot count. A new crossing requires another stable transition.
            state.armed, state.transitioned, state.counted = True, False, False
        elif zone == Zone.NONE and previous == Zone.A and state.armed:
            self.transition_confirmations += 1
            state.transitioned = True
        elif zone == Zone.B:
            self.stable_b_confirmations += 1
            if not (state.armed and state.transitioned) or state.counted or state.blocked:
                state.armed = False
                return None
            state.counted, state.armed, state.transitioned = True, False, False
            self.total_out += 1
            return HeadOutEvent(
                "rear", detection.track_id, "OUT", frame_index, timestamp_s,
                detection.confidence, "A->transition->B", x, y,
            )
        return None

    def prune(self, frame_index: int) -> None:
        self.states = {
            key: state for key, state in self.states.items()
            if frame_index - state.last_seen <= self.config.state_ttl_frames
        }


@dataclass(slots=True)
class RearHeadOutResult:
    camera: str
    input_path: str
    annotated_path: str
    events_path: str
    events: list[HeadOutEvent]
    frames: int
    elapsed_s: float
    processing_fps: float
    metadata: VideoMetadata
    model: str
    tracker: str
    device: str
    status: str


def _annotate(
    frame: np.ndarray, detections: list[Detection], counter: HeadZoneCounter,
    config: RearHeadOutConfig, options: OverlayOptions, frame_events: list[HeadOutEvent],
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    if options.zones:
        for label, polygon, color in (
            ("ROI", config.roi, (200, 200, 200)),
            ("A INSIDE", config.zone_a, (0, 200, 255)),
            ("B OUTSIDE", config.zone_b, (255, 170, 0)),
        ):
            points = np.asarray([(round(x * width), round(y * height)) for x, y in polygon], dtype=np.int32)
            cv2.polylines(output, [points], True, color, 2)
            center = np.mean(points, axis=0).astype(int)
            cv2.putText(output, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    highlighted = {event.track_id for event in frame_events}
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        center = tuple(round(v) for v in detection.anchor("center"))
        color = (0, 255, 255) if detection.track_id in highlighted else (0, 230, 70)
        if options.boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if options.track_ids:
            cv2.putText(output, f"{detection.track_id} {detection.confidence:.2f}",
                        (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        if options.trajectory:
            trail = counter.states[detection.track_id].trail
            if len(trail) > 1:
                cv2.polylines(output, [np.asarray(trail, dtype=np.int32)], False, color, 2)
        cv2.circle(output, center, 3, color, -1)
    cv2.rectangle(output, (0, 0), (min(width, 360), 35), (10, 17, 30), -1)
    cv2.putText(output, f"REAR HEAD-OUT: {counter.total_out}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if options.events and frame_events:
        cv2.putText(output, f"OUT +{len(frame_events)}", (8, min(height - 8, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return output


class RearHeadOutPipeline:
    def __init__(self, config: RearHeadOutConfig, tracker: Any | None = None) -> None:
        self.config = config
        self._injected_tracker = tracker

    def run(
        self, source_path: str | int, output_dir: str | Path, max_frames: int | None = None,
        *, stop_file: Path | None = None, start_frame: int = 0,
        overlay: OverlayOptions | None = None,
    ) -> RearHeadOutResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        tracker = self._injected_tracker or HeadTracker(self.config)
        counter = HeadZoneCounter(self.config)
        source = OpenCVVideoSource(source_path)
        metadata = source.open()
        progress = destination / "rear_head_out_progress.json"
        preview = destination / "rear_head_out_preview.jpg"
        csv_path = destination / "rear_head_out_events.csv"
        intermediate = destination / "rear_head_out_encoding.mp4"
        annotated = destination / "rear_head_out_annotated.mp4"
        writer: cv2.VideoWriter | None = None
        started = time.perf_counter()
        events: list[HeadOutEvent] = []
        processed = 0
        detected_heads = 0
        tracked_heads = 0
        stopped = False
        total: int | None = None
        clock = SourceClock(datetime.now(timezone.utc))
        generation = source.generation

        def write_progress(status: str, error: str | None = None) -> None:
            elapsed = time.perf_counter() - started
            _write_progress(
                progress, mode="HEAD_TRACKING_ZONE_OUT", status=status,
                frames=processed, total_frames=total, count=counter.total_out,
                detected_heads=detected_heads, tracked_heads=tracked_heads,
                stable_a_confirmations=counter.stable_a_confirmations,
                transition_confirmations=counter.transition_confirmations,
                stable_b_confirmations=counter.stable_b_confirmations,
                processing_fps=processed / elapsed if elapsed else 0.0,
                model=self.config.model, tracker=self.config.tracker,
                confidence=self.config.confidence, error=error,
            )

        try:
            if start_frame:
                source.seek(start_frame)
            available = metadata.frame_count - start_frame if metadata.frame_count is not None else None
            total = min(max_frames, available) if max_frames and available is not None else available or max_frames
            writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
                                     metadata.fps, (metadata.width, metadata.height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create REAR head OUT video")
            write_progress("RUNNING")
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(HeadOutEvent.__dataclass_fields__))
                csv_writer.writeheader()
                while max_frames is None or processed < max_frames:
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
                    stable_timestamp_s = (capture_time - clock.recording_start_time).total_seconds()
                    index = start_frame + processed
                    detections = tracker.track(frame)
                    active_track_ids = {detection.track_id for detection in detections}
                    detected_heads += getattr(tracker, "last_detection_count", len(detections))
                    tracked_heads += len(detections)
                    frame_events = []
                    for detection in detections:
                        event = counter.update(
                            detection, frame.shape, index, stable_timestamp_s, active_track_ids
                        )
                        if event:
                            # MP4 has no verified recording start; do not invent capture UTC.
                            event.capture_time_utc = (
                                capture_time.isoformat() if metadata.source_type == "rtsp" else ""
                            )
                            events.append(event)
                            frame_events.append(event)
                            csv_writer.writerow(asdict(event))
                            handle.flush()
                    counter.prune(index)
                    output_frame = _annotate(frame, detections, counter, self.config,
                                             overlay or OverlayOptions(), frame_events)
                    writer.write(output_frame)
                    processed += 1
                    if processed == 1 or processed % 15 == 0 or frame_events:
                        temporary = preview.with_suffix(".tmp.jpg")
                        if cv2.imwrite(str(temporary), output_frame):
                            try:
                                _replace_with_retry(temporary, preview)
                            except PermissionError:
                                temporary.unlink(missing_ok=True)
                    if processed % 10 == 0 or frame_events:
                        write_progress("RUNNING")
            if not processed:
                raise RuntimeError("REAR source provided no frames; no head OUT result is available")
        except Exception as exc:
            write_progress("FAILED", str(exc))
            raise
        finally:
            if writer is not None:
                writer.release()
            source.close()
        try:
            validate_video(intermediate, processed)
            make_browser_video(intermediate, annotated, processed)
        except Exception as exc:
            write_progress("FAILED", str(exc))
            raise
        elapsed = time.perf_counter() - started
        result = RearHeadOutResult(
            "rear", redact_source(source_path), str(annotated), str(csv_path), events,
            processed, elapsed, processed / elapsed if elapsed else 0.0, metadata,
            self.config.model, self.config.tracker,
            str(getattr(getattr(getattr(tracker, "model", None), "predictor", None), "device", "N/A")),
            "STOPPED" if stopped else "COMPLETED",
        )
        summary = {**asdict(result), "mode": "HEAD_TRACKING_ZONE_OUT", "count": len(events),
                   "config": asdict(self.config), "source": result.input_path,
                   "evaluation": "N/A: no labeled ground truth",
                   "diagnostics": {
                       "detected_heads": detected_heads, "tracked_heads": tracked_heads,
                       "stable_a_confirmations": counter.stable_a_confirmations,
                       "transition_confirmations": counter.transition_confirmations,
                       "stable_b_confirmations": counter.stable_b_confirmations,
                   }}
        summary["events"] = [asdict(event) for event in events]
        (destination / "rear_head_out_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_progress(result.status)
        return result
