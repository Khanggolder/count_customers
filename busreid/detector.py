from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .types import Detection

_MODEL_INIT_LOCK = Lock()


class PersonDetector:
    """Ultralytics person detector + per-instance tracker."""

    def __init__(
        self,
        camera_namespace: str,
        model: str = "yolo11n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str = "auto",
        tracker: str = "bytetrack.yaml",
    ) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "runs" / "ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        with _MODEL_INIT_LOCK:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Thiếu ultralytics; chạy pip install -e .") from exc
            model_instance = YOLO(model)
        self.namespace = camera_namespace
        self.confidence = confidence
        self.iou = iou
        self.tracker = tracker
        self.device = None if device == "auto" else device
        self.model: Any = model_instance

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        ids = boxes.id.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()
        return [
            Detection(
                track_id=f"{self.namespace}:track{track_id:05d}",
                bbox=tuple(int(v) for v in box),
                confidence=float(confidence),
            )
            for box, track_id, confidence in zip(xyxy, ids, confs, strict=True)
        ]
