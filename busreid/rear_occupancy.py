"""Experimental REAR cabin headcount. No door crossing or PassengerEvent is produced."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np

from .config import RearOccupancyConfig
from .pipeline import _replace_with_retry, _write_progress
from .video import OpenCVVideoSource, VideoMetadata, redact_source
from .video_output import make_browser_video, validate_video

MODEL_REPO = "wesjos/YOLO-head-detection"
MODEL_NAME = "11n-head.pt"


@dataclass(slots=True, frozen=True)
class HeadDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class RearOccupancyResult:
    camera: str
    input_path: str
    annotated_path: str
    occupancy_path: str
    frames: int
    elapsed_s: float
    processing_fps: float
    metadata: VideoMetadata
    model: str
    device: str
    confidence: float
    last_count: int
    mean_count: float
    min_count: int
    max_count: int
    status: str
    events: tuple = ()  # Explicitly no PassengerEvent / REAR-OUT.


def _weights_path(model: str) -> Path:
    local = Path(model).expanduser()
    if local.is_file():
        if local.suffix.lower() != ".pt":
            raise ValueError("Head Detection checkpoint must be a .pt file")
        return local.resolve()
    if local.name != model or not model.endswith("-head.pt"):
        raise FileNotFoundError(f"Head Detection checkpoint not found: {model}")
    cache = Path(__file__).resolve().parents[1] / "runs" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / model
    if target.is_file() and target.stat().st_size > 1024:
        return target
    temporary = target.with_suffix(".download")
    try:
        urlretrieve(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{model}", temporary)
        if temporary.stat().st_size < 1024:
            raise RuntimeError("downloaded checkpoint is empty")
        temporary.replace(target)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cannot download {model} from {MODEL_REPO}; place a trusted .pt file locally "
            "and set rear_occupancy.model"
        ) from exc
    return target


class HeadDetector:
    """Load the dedicated single-class head model once, then predict per frame."""

    def __init__(self, config: RearOccupancyConfig) -> None:
        settings = Path(__file__).resolve().parents[1] / "runs" / "ultralytics"
        settings.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for head detection") from exc
        weights = _weights_path(config.model)
        try:
            self.model: Any = YOLO(str(weights))
        except Exception as exc:
            raise RuntimeError(f"Cannot load Head Detection checkpoint: {weights.name}") from exc
        if self.model.task != "detect" or len(self.model.names) != 1:
            raise RuntimeError("Head Detection requires a single-class detection checkpoint, not COCO person")
        self.confidence = config.confidence
        self.iou = config.iou
        self.imgsz = config.imgsz
        self.device = None if config.device == "auto" else config.device

    def detect(self, frame: np.ndarray) -> list[HeadDetection]:
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        return [
            HeadDetection(tuple(int(value) for value in box), float(confidence))
            for box, confidence in zip(xyxy, confs, strict=True)
        ]


def _in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


class RearOccupancyCounter:
    def __init__(self, config: RearOccupancyConfig) -> None:
        self.config = config
        self.history: deque[int] = deque(maxlen=config.smoothing_window)

    def count(self, detections: list[HeadDetection], frame_shape: tuple[int, ...]) -> tuple[list[HeadDetection], int, int]:
        height, width = frame_shape[:2]
        accepted = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            if x2 <= x1 or y2 <= y1:
                continue
            point = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height))
            if _in_polygon(point, self.config.roi) and not any(
                _in_polygon(point, polygon) for polygon in self.config.ignore
            ):
                accepted.append(detection)
        raw = len(accepted)
        self.history.append(raw)
        return accepted, raw, int(statistics.median_low(self.history))


def _annotate(frame: np.ndarray, detections: list[HeadDetection], count: int, config: RearOccupancyConfig) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for polygon, color in [(config.roi, (0, 200, 255)), *[(p, (0, 0, 255)) for p in config.ignore]]:
        points = np.asarray([(round(x * width), round(y * height)) for x, y in polygon], dtype=np.int32)
        cv2.polylines(output, [points], True, color, 2)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 230, 70), 2)
        cv2.putText(output, f"head {detection.confidence:.2f}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 70), 2)
    cv2.rectangle(output, (0, 0), (min(width, 440), 42), (10, 17, 30), -1)
    cv2.putText(output, f"VISIBLE HEADS: {count}", (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    return output


class RearOccupancyPipeline:
    def __init__(self, config: RearOccupancyConfig, detector: HeadDetector | None = None) -> None:
        self.config = config
        self.detector = detector or HeadDetector(config)
        self.counter = RearOccupancyCounter(config)

    def run(
        self,
        source_path: str | int,
        output_dir: str | Path,
        max_frames: int | None = None,
        *,
        stop_file: Path | None = None,
        start_frame: int = 0,
    ) -> RearOccupancyResult:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.counter = RearOccupancyCounter(self.config)
        progress = destination / "rear_progress.json"
        source = OpenCVVideoSource(source_path)
        metadata = source.open()
        writer: cv2.VideoWriter | None = None
        started = time.perf_counter()
        processed = 0
        counts: list[int] = []
        stopped = False
        annotated = destination / "rear_annotated.mp4"
        intermediate = destination / "rear_encoding.mp4"
        csv_path = destination / "rear_occupancy.csv"
        try:
            if start_frame:
                source.seek(start_frame)
            available = metadata.frame_count - start_frame if metadata.frame_count is not None else None
            total = min(max_frames, available) if max_frames and available is not None else available or max_frames
            writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"),
                                     metadata.fps, (metadata.width, metadata.height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create REAR annotated video")
            _write_progress(progress, mode="OCCUPANCY_HEADCOUNT", status="RUNNING", frames=0,
                            total_frames=total, visible_head_count=None, processing_fps=0.0,
                            model=self.config.model, confidence=self.config.confidence)
            with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                csv_writer = csv.writer(handle)
                csv_writer.writerow(["frame_index", "timestamp_s", "visible_head_count", "smoothed_count"])
                while max_frames is None or processed < max_frames:
                    if stop_file and stop_file.is_file():
                        stopped = True
                        break
                    ok, frame, timestamp_s = source.read()
                    if not ok or frame is None:
                        break
                    accepted, raw, smoothed = self.counter.count(self.detector.detect(frame), frame.shape)
                    index = start_frame + processed
                    csv_writer.writerow([index, round(timestamp_s, 6), raw, smoothed])
                    handle.flush()
                    counts.append(raw)
                    output_frame = _annotate(frame, accepted, raw, self.config)
                    writer.write(output_frame)
                    processed += 1
                    if processed == 1 or processed % 15 == 0:
                        preview = destination / "rear_preview.jpg"
                        temporary = preview.with_suffix(".tmp.jpg")
                        if cv2.imwrite(str(temporary), output_frame):
                            try:
                                _replace_with_retry(temporary, preview)
                            except PermissionError:
                                # Preview is best effort on Windows; a reader can lock the old JPEG.
                                temporary.unlink(missing_ok=True)
                    if processed == 1 or processed % 10 == 0:
                        elapsed = time.perf_counter() - started
                        _write_progress(progress, mode="OCCUPANCY_HEADCOUNT", status="RUNNING",
                                        frames=processed, total_frames=total, visible_head_count=raw,
                                        smoothed_count=smoothed, processing_fps=processed / elapsed if elapsed else 0.0,
                                        model=self.config.model, confidence=self.config.confidence)
            if processed == 0:
                raise RuntimeError("REAR source provided no frames; no occupancy result is available")
        except Exception as exc:
            _write_progress(progress, mode="OCCUPANCY_HEADCOUNT", status="FAILED", frames=processed,
                            visible_head_count=None, error=str(exc))
            raise
        finally:
            if writer is not None:
                writer.release()
            source.close()
        elapsed = time.perf_counter() - started
        try:
            validate_video(intermediate, processed)
            make_browser_video(intermediate, annotated, processed)
        except Exception as exc:
            _write_progress(progress, mode="OCCUPANCY_HEADCOUNT", status="FAILED", frames=processed,
                            visible_head_count=None, error=str(exc))
            raise
        result = RearOccupancyResult(
            camera="rear", input_path=redact_source(source_path), annotated_path=str(annotated),
            occupancy_path=str(csv_path), frames=processed, elapsed_s=elapsed,
            processing_fps=processed / elapsed if elapsed else 0.0, metadata=metadata,
            model=self.config.model,
            device=str(getattr(getattr(getattr(self.detector, "model", None), "predictor", None), "device", "N/A")),
            confidence=self.config.confidence, last_count=counts[-1],
            mean_count=statistics.mean(counts), min_count=min(counts), max_count=max(counts),
            status="STOPPED" if stopped else "COMPLETED",
        )
        (destination / "rear_occupancy_summary.json").write_text(
            json.dumps({**asdict(result), "mode": "OCCUPANCY_HEADCOUNT", "events": [],
                        "source": result.input_path}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_progress(progress, mode="OCCUPANCY_HEADCOUNT", status=result.status, frames=processed,
                        total_frames=total, visible_head_count=counts[-1],
                        smoothed_count=int(statistics.median_low(self.counter.history)),
                        processing_fps=result.processing_fps, model=self.config.model,
                        confidence=self.config.confidence)
        return result
