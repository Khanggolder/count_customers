from __future__ import annotations

import cv2
import numpy as np

from .types import Detection, TrackCrop


class QualitySelector:
    def __init__(self, max_per_track: int = 4, min_height: int = 64, min_width: int = 24) -> None:
        self.max_per_track = max_per_track
        self.min_height = min_height
        self.min_width = min_width
        self._crops: dict[str, list[TrackCrop]] = {}

    @staticmethod
    def score(image: np.ndarray, frame_shape: tuple[int, ...]) -> float:
        if image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
        area_ratio = image.shape[0] * image.shape[1] / (frame_shape[0] * frame_shape[1])
        size_score = min(area_ratio / 0.08, 1.0)
        return 0.6 * sharpness + 0.4 * size_score

    def consider(self, frame: np.ndarray, detection: Detection, frame_index: int) -> None:
        x1, y1, x2, y2 = detection.bbox
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < self.min_width or y2 - y1 < self.min_height:
            return
        crop = frame[y1:y2, x1:x2].copy()
        candidate = TrackCrop(detection.track_id, crop, self.score(crop, frame.shape), frame_index)
        items = self._crops.setdefault(detection.track_id, [])
        items.append(candidate)
        items.sort(key=lambda item: item.quality, reverse=True)
        del items[self.max_per_track :]

    def best(self, track_id: str) -> list[TrackCrop]:
        return list(self._crops.get(track_id, []))

    def pop(self, track_id: str) -> list[TrackCrop]:
        return self._crops.pop(track_id, [])

