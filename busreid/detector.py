from __future__ import annotations

from typing import Any

import numpy as np

from .types import Detection


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
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Thiếu ultralytics; chạy pip install -e .") from exc
        self.namespace = camera_namespace
        self.confidence = confidence
        self.iou = iou
        self.tracker = tracker
        self.device = None if device == "auto" else device
        self.model: Any = YOLO(model)

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

