"""Experimental REAR head tracking and A -> B OUT counting, isolated from Trip data."""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
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
    """Detect on a masked ROI crop, then track only accepted head centers."""

    def __init__(self, config: RearHeadOutConfig) -> None:
        super().__init__(config)
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, yaml_load
        from ultralytics.utils.checks import check_yaml

        tracker_args = IterableSimpleNamespace(**yaml_load(check_yaml(config.tracker)))
        self.tracker = TRACKER_MAP[tracker_args.tracker_type](args=tracker_args, frame_rate=30)
        self.roi_frame: HeadRoiFrame | None = None
        self.roi = config.roi
        self.last_detection_count = 0
        self.last_inference_s = 0.0

    def track(self, frame: np.ndarray) -> list[Detection]:
        if self.roi_frame is None:
            self.roi_frame = HeadRoiFrame(frame.shape, self.roi)
        if frame.shape[:2] != (self.roi_frame.height, self.roi_frame.width):
            raise ValueError("REAR frame size changed during tracking; start a new session")
        crop = self.roi_frame.crop_masked(frame)
        started = time.perf_counter()
        results = self.model.predict(
            source=crop, classes=[0], conf=self.confidence, iou=self.iou,
            imgsz=self.imgsz, device=self.device, verbose=False,
        )
        self.last_inference_s = time.perf_counter() - started
        boxes = results[0].boxes if results else None
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None else np.empty((0, 4))
        scores = boxes.conf.detach().cpu().numpy() if boxes is not None else np.empty(0)
        accepted = [i for i, box in enumerate(xyxy) if self.roi_frame.contains_crop_box(box)]
        self.last_detection_count = len(accepted)
        valid_boxes = xyxy[accepted]
        centers = (valid_boxes[:, :2] + valid_boxes[:, 2:]) / 2
        sizes = valid_boxes[:, 2:] - valid_boxes[:, :2]
        tracker_input = SimpleNamespace(
            xywh=np.concatenate((centers, sizes), axis=1),
            conf=scores[accepted], cls=np.zeros(len(accepted)),
        )
        tracks = self.tracker.update(tracker_input, crop)
        # ByteTrack/BoT-SORT may retain predicted or lost states after a head
        # leaves the polygon. Do not keep those states for a later re-entry.
        for name in ("tracked_stracks", "lost_stracks"):
            states = getattr(self.tracker, name, None)
            if states is not None:
                setattr(self.tracker, name, [
                    state for state in states if self.roi_frame.contains_crop_box(state.xyxy)
                ])
        detections = []
        for track in tracks:
            box = self.roi_frame.to_frame_box(track[:4])
            if box is not None and self.roi_frame.contains_frame_box(box):
                detections.append(Detection(f"rear:head{int(track[4]):05d}", box, float(track[5])))
        return detections


class HeadRoiFrame:
    """Fixed pixel geometry for the lifetime of a REAR tracking session."""

    def __init__(self, shape: tuple[int, ...], polygon: list[tuple[float, float]]) -> None:
        self.height, self.width = shape[:2]
        self.contour = np.asarray([
            (min(round(x * self.width), self.width - 1),
             min(round(y * self.height), self.height - 1)) for x, y in polygon
        ], dtype=np.int32)
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.contour)
        local = self.contour - (self.x, self.y)
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [local], 255)

    def crop_masked(self, frame: np.ndarray) -> np.ndarray:
        crop = frame[self.y:self.y + self.h, self.x:self.x + self.w]
        return cv2.bitwise_and(crop, crop, mask=self.mask)

    def contains_crop_box(self, box: np.ndarray) -> bool:
        center = ((float(box[0] + box[2]) / 2) + self.x,
                  (float(box[1] + box[3]) / 2) + self.y)
        return cv2.pointPolygonTest(self.contour, center, False) >= 0

    def to_frame_box(self, box: np.ndarray) -> tuple[int, int, int, int] | None:
        x1 = max(0, min(self.width - 1, round(float(box[0]) + self.x)))
        y1 = max(0, min(self.height - 1, round(float(box[1]) + self.y)))
        x2 = max(0, min(self.width - 1, round(float(box[2]) + self.x)))
        y2 = max(0, min(self.height - 1, round(float(box[3]) + self.y)))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def contains_frame_box(self, box: tuple[int, int, int, int]) -> bool:
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return cv2.pointPolygonTest(self.contour, center, False) >= 0


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
        if not point_in_polygon(point, self.config.roi):
            # Leaving the observation ROI is not a valid transition across the door.
            state.stable = state.candidate = Zone.NONE
            state.candidate_frames = 0
            state.armed = state.transitioned = state.counted = False
            state.trail.clear()
            return None
        state.trail.append((round(x), round(y)))
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

    def invalidate_missing(self, active_track_ids: set[str]) -> None:
        """A gap cannot stand in for a verified transition inside the ROI."""
        for track_id, state in self.states.items():
            if track_id not in active_track_ids:
                state.stable = state.candidate = Zone.NONE
                state.candidate_frames = 0
                state.armed = state.transitioned = state.counted = False
                state.trail.clear()


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
        inference_s = 0.0
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
                inference_s=round(inference_s, 4),
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
                    inference_s += getattr(tracker, "last_inference_s", 0.0)
                    counter.invalidate_missing(active_track_ids)
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
                       "inference_s": round(inference_s, 4),
                       "roi_crop_size": (
                           [tracker.roi_frame.w, tracker.roi_frame.h]
                           if getattr(tracker, "roi_frame", None) is not None else None
                       ),
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
