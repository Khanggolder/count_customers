"""Experimental REAR optical flow line crossing; motion events are not people."""

from __future__ import annotations

import csv
import json
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import RearFlowOutConfig
from .pipeline import _replace_with_retry, _write_progress
from .roi import to_pixels, validate_flow_geometry
from .timebase import SourceClock
from .video import OpenCVVideoSource, VideoMetadata, redact_source
from .video_output import make_browser_video, validate_video


@dataclass(slots=True)
class FlowMotionEvent:
    event_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    direction: str
    motion_intensity: float
    duration_s: float
    estimated_passenger_count: int | None = None
    review_status: str = "REVIEW_REQUIRED"


@dataclass(slots=True)
class FlowResult:
    camera: str
    input_path: str
    annotated_path: str
    events_path: str
    events: list[FlowMotionEvent]
    frames: int
    elapsed_s: float
    processing_fps: float
    metadata: VideoMetadata
    status: str


class FlowLineAnalyzer:
    """Sample signed dense flow across a fixed line, group active time intervals."""

    def __init__(self, config: RearFlowOutConfig, shape: tuple[int, ...]):
        validate_flow_geometry(config.roi, config.counting_line, config.out_direction)
        self.config = config
        self.height, self.width = shape[:2]
        self.roi = np.zeros((self.height, self.width), np.uint8)
        contour = np.asarray([to_pixels(point, self.width, self.height) for point in config.roi], np.int32)
        cv2.fillPoly(self.roi, [contour], 255)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        p0, p1 = (np.asarray(to_pixels(point, self.width, self.height), np.float32)
                  for point in config.counting_line)
        samples = max(2, round(float(np.linalg.norm(p1 - p0))))
        coords = p0[None, :] + np.linspace(0, 1, samples, dtype=np.float32)[:, None] * (p1 - p0)
        self.sample_x = np.clip(np.rint(coords[:, 0]).astype(int), 0, self.width - 1)
        self.sample_y = np.clip(np.rint(coords[:, 1]).astype(int), 0, self.height - 1)
        self.valid = self.roi[self.sample_y, self.sample_x] > 0
        vector = np.asarray([config.out_direction[0] * self.width,
                             config.out_direction[1] * self.height], np.float32)
        self.out_unit = vector / np.linalg.norm(vector)
        self.previous: np.ndarray | None = None
        self.previous_full: np.ndarray | None = None
        self.scores: deque[tuple[float, float]] = deque(maxlen=config.smoothing_window)
        self.slice_rows: list[np.ndarray] = []
        self.signed_rows: list[np.ndarray] = []
        self.events: list[FlowMotionEvent] = []
        self.active: dict | None = None
        self.gap = 0
        self.last_end = -config.debounce_frames
        self.global_shifts: list[float] = []
        self.motion_trace: list[dict] = []
        self.last_out_mask = np.zeros(len(self.sample_x), bool)
        self.last_in_mask = np.zeros(len(self.sample_x), bool)

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if self.config.gaussian_blur:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _global_shift(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        if not self.config.stabilization:
            return np.zeros(2, np.float32)
        background = (self.roi == 0).astype(np.float32)
        if np.count_nonzero(background) < .2 * background.size:
            return np.zeros(2, np.float32)
        shift, response = cv2.phaseCorrelate(previous.astype(np.float32) * background,
                                            current.astype(np.float32) * background)
        if response < .1 or not np.isfinite(shift).all():
            return np.zeros(2, np.float32)
        result = np.asarray(shift, np.float32)
        self.global_shifts.append(float(np.linalg.norm(result)))
        return result

    def update(self, frame: np.ndarray, index: int, timestamp_s: float) -> tuple[list[FlowMotionEvent], np.ndarray | None]:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("REAR frame size changed during Optical Flow session")
        gray = self._gray(frame)
        current = cv2.bitwise_and(gray[self.y:self.y + self.h, self.x:self.x + self.w],
                                  self.roi[self.y:self.y + self.h, self.x:self.x + self.w])
        self.slice_rows.append(gray[self.sample_y, self.sample_x].copy())
        if self.previous is None:
            self.previous, self.previous_full = current, gray
            self.signed_rows.append(np.zeros(len(self.sample_x), np.uint8))
            return [], None
        shift = self._global_shift(self.previous_full, gray)
        flow = cv2.calcOpticalFlowFarneback(self.previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
        self.previous, self.previous_full = current, gray
        flow -= shift
        local_x, local_y = self.sample_x - self.x, self.sample_y - self.y
        vectors = flow[local_y, local_x]
        projected = vectors @ self.out_unit
        magnitude = np.linalg.norm(vectors, axis=1)
        accepted = self.valid & (magnitude >= self.config.min_motion_magnitude)
        self.last_out_mask = accepted & (projected >= self.config.min_motion_magnitude * .6)
        self.last_in_mask = accepted & (projected <= -self.config.min_motion_magnitude * .6)
        out_fraction = float(np.count_nonzero(self.last_out_mask) / len(accepted))
        in_fraction = float(np.count_nonzero(self.last_in_mask) / len(accepted))
        self.scores.append((out_fraction, in_fraction))
        smoothed = np.mean(self.scores, axis=0)
        direction = ("OUT" if smoothed[0] >= self.config.min_active_fraction and smoothed[0] > smoothed[1] * 1.5
                     else "IN" if smoothed[1] >= self.config.min_active_fraction and smoothed[1] > smoothed[0] * 1.5
                     else None)
        intensity = float(np.mean(np.abs(projected[accepted]))) if np.any(accepted) else 0.0
        signed = np.clip(127 + projected * 20, 0, 255).astype(np.uint8)
        signed[~accepted] = 127
        self.signed_rows.append(signed)
        self.motion_trace.append({"frame": index, "time_s": timestamp_s, "out_fraction": out_fraction,
                                  "in_fraction": in_fraction, "intensity": intensity})
        finished: list[FlowMotionEvent] = []
        if self.active is not None and direction != self.active["direction"]:
            self.gap += 1
            if self.gap > self.config.max_gap_frames:
                event = self._finish()
                if event is not None:
                    finished.append(event)
        if direction is not None and self.active is None and index - self.last_end >= self.config.debounce_frames:
            self.active = {"direction": direction, "start_frame": index, "start_time_s": timestamp_s,
                           "end_frame": index, "end_time_s": timestamp_s, "frames": 0, "intensity": 0.0}
        if self.active is not None and direction == self.active["direction"]:
            self.active["end_frame"] = index
            self.active["end_time_s"] = timestamp_s
            self.active["frames"] += 1
            self.active["intensity"] += intensity
            self.gap = 0
        return finished, flow

    def _finish(self) -> FlowMotionEvent | None:
        active, self.active = self.active, None
        self.gap = 0
        if active is None or active["frames"] < self.config.min_event_frames:
            return None
        self.last_end = active["end_frame"]
        event = FlowMotionEvent(
            uuid.uuid4().hex[:12], active["start_frame"], active["end_frame"],
            active["start_time_s"], active["end_time_s"], active["direction"],
            round(active["intensity"] / active["frames"], 3),
            round(active["end_time_s"] - active["start_time_s"], 3),
        )
        self.events.append(event)
        return event

    def finish(self) -> FlowMotionEvent | None:
        return self._finish()

    def temporal_image(self) -> np.ndarray:
        gray = np.asarray(self.slice_rows, np.uint8)
        signed = np.asarray(self.signed_rows, np.uint8)
        return cv2.vconcat([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                            cv2.applyColorMap(signed, cv2.COLORMAP_TURBO)])


def annotate_flow(frame: np.ndarray, analyzer: FlowLineAnalyzer, flow: np.ndarray | None,
                  *, show_vectors: bool = False) -> np.ndarray:
    output = frame.copy()
    contour = np.asarray([to_pixels(point, analyzer.width, analyzer.height)
                          for point in analyzer.config.roi], np.int32)
    cv2.polylines(output, [contour], True, (0, 220, 180), 2)
    start, end = [to_pixels(point, analyzer.width, analyzer.height)
                  for point in analyzer.config.counting_line]
    cv2.line(output, start, end, (0, 200, 255), 2)
    middle = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    arrow_end = (round(middle[0] + analyzer.out_unit[0] * 35),
                 round(middle[1] + analyzer.out_unit[1] * 35))
    cv2.arrowedLine(output, middle, arrow_end, (30, 30, 255), 2)
    cv2.putText(output, "OUT", arrow_end, cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 255), 2)
    for mask, color in ((analyzer.last_out_mask, (30, 30, 255)),
                        (analyzer.last_in_mask, (255, 120, 30))):
        for x, y in zip(analyzer.sample_x[mask][::3], analyzer.sample_y[mask][::3], strict=False):
            cv2.circle(output, (int(x), int(y)), 3, color, -1)
    if show_vectors and flow is not None:
        for x, y in zip(analyzer.sample_x[::12], analyzer.sample_y[::12], strict=False):
            vector = flow[y - analyzer.y, x - analyzer.x]
            cv2.arrowedLine(output, (int(x), int(y)),
                            (round(x + vector[0] * 3), round(y + vector[1] * 3)), (255, 230, 0), 1)
    cv2.putText(output, f"OUT motion events: {sum(e.direction == 'OUT' for e in analyzer.events)} | passengers: N/A",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2)
    return output


class RearFlowPipeline:
    def __init__(self, config: RearFlowOutConfig):
        self.config = config

    def run(self, source_path: str | int, output_dir: str | Path, max_frames: int | None = None,
            *, stop_file: Path | None = None, start_frame: int = 0,
            show_vectors: bool = False) -> FlowResult:
        if not self.config.enabled:
            raise ValueError("Configure and save REAR Optical Flow ROI/line before inference")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source = OpenCVVideoSource(source_path)
        progress = destination / "rear_flow_progress.json"
        intermediate = destination / "rear_flow_encoding.mp4"
        annotated = destination / "rear_flow_annotated.mp4"
        events_path = destination / "rear_flow_events.csv"
        started = time.perf_counter()
        processed = 0
        stopped = False
        analyzer = None
        writer = None
        clock = SourceClock(datetime.now(timezone.utc))
        try:
            metadata = source.open()
            generation = source.generation
            if start_frame:
                source.seek(start_frame)
            total = min(max_frames, metadata.frame_count - start_frame) if max_frames and metadata.frame_count else metadata.frame_count
            writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps,
                                     (metadata.width, metadata.height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create REAR Optical Flow video")
            analyzer = FlowLineAnalyzer(self.config, (metadata.height, metadata.width))
            fields = list(FlowMotionEvent.__dataclass_fields__)
            with events_path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=fields)
                csv_writer.writeheader()
                _write_progress(progress, status="RUNNING", frames=0, total_frames=total, motion_events=0,
                                estimated_out=None, processing_fps=0.0)
                while max_frames is None or processed < max_frames:
                    if stop_file and stop_file.exists():
                        stopped = True
                        break
                    ok, frame, timestamp_s = source.read()
                    if not ok or frame is None:
                        break
                    if source.generation != generation:
                        clock.reconnect()
                        generation = source.generation
                    stable_timestamp_s = (clock.map_pts(timestamp_s) - clock.recording_start_time).total_seconds()
                    index = start_frame + processed
                    finished, flow = analyzer.update(frame, index, stable_timestamp_s)
                    for event in finished:
                        csv_writer.writerow(asdict(event))
                    if finished:
                        handle.flush()
                    output = annotate_flow(frame, analyzer, flow, show_vectors=show_vectors)
                    writer.write(output)
                    processed += 1
                    if processed == 1 or processed % 15 == 0:
                        temporary = destination / "rear_flow_preview.tmp.jpg"
                        if cv2.imwrite(str(temporary), output):
                            try:
                                _replace_with_retry(temporary, destination / "rear_flow_preview.jpg")
                            except PermissionError:
                                temporary.unlink(missing_ok=True)
                    if processed % 10 == 0:
                        elapsed = time.perf_counter() - started
                        _write_progress(progress, status="RUNNING", frames=processed, total_frames=total,
                                        motion_events=len(analyzer.events), estimated_out=None,
                                        processing_fps=processed / elapsed if elapsed else 0.0)
                last = analyzer.finish()
                if last is not None:
                    csv_writer.writerow(asdict(last))
            if processed == 0:
                raise RuntimeError("REAR source provided no frames")
        except Exception as exc:
            _write_progress(progress, status="FAILED", frames=processed, error=str(exc))
            raise
        finally:
            if writer is not None:
                writer.release()
            source.close()
        try:
            validate_video(intermediate, processed)
            make_browser_video(intermediate, annotated, processed)
            if not cv2.imwrite(str(destination / "rear_flow_temporal_slice.png"), analyzer.temporal_image()):
                raise RuntimeError("Cannot save REAR temporal slice")
        except Exception as exc:
            _write_progress(progress, status="FAILED", frames=processed, error=str(exc))
            raise
        elapsed = time.perf_counter() - started
        result = FlowResult("rear", redact_source(source_path), str(annotated), str(events_path),
                            analyzer.events, processed, elapsed, processed / elapsed if elapsed else 0.0,
                            metadata, "STOPPED" if stopped else "COMPLETED")
        diagnostics = {"motion_trace": analyzer.motion_trace,
                       "global_shifts_px": analyzer.global_shifts,
                       "roi_crop_size": [analyzer.w, analyzer.h]}
        (destination / "rear_flow_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        summary = {"mode": "FLOW_LINE_OUT", "status": result.status, "frames": processed,
                   "elapsed_s": elapsed, "processing_fps": result.processing_fps,
                   "motion_events": len(analyzer.events),
                   "out_motion_events": sum(event.direction == "OUT" for event in analyzer.events),
                   "estimated_out": None, "uncertain_events": len(analyzer.events),
                   "ground_truth_evaluation": "N/A", "source": result.input_path,
                   "timestamp_basis": "MP4_PTS_OR_RTSP_MONOTONIC_SOURCE_CLOCK",
                   "config": asdict(self.config)}
        (destination / "rear_flow_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_progress(progress, status=result.status, frames=processed, total_frames=total,
                        motion_events=len(analyzer.events), out_motion_events=summary["out_motion_events"],
                        estimated_out=None, processing_fps=result.processing_fps)
        return result
