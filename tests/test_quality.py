import cv2
import numpy as np

from busreid.quality import QualitySelector
from busreid.types import Detection


def test_quality_keeps_only_best_crops() -> None:
    selector = QualitySelector(max_per_track=2, min_height=10, min_width=10)
    for index in range(4):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        if index % 2:
            cv2.line(frame, (20, 20), (80, 80), (255, 255, 255), 3)
        selector.consider(frame, Detection("front:t1", (10, 10, 90, 90), 0.9), index)
    crops = selector.best("front:t1")
    assert len(crops) == 2
    assert crops[0].quality >= crops[1].quality
