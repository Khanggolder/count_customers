"""Local manual IN/OUT label tool; labels are human ground truth, not AI output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def label_video(video: Path, output: Path, clip: str | None = None) -> int:
    source = cv2.VideoCapture(str(video))
    if not source.isOpened():
        raise FileNotFoundError(video)
    fps = source.get(cv2.CAP_PROP_FPS) or 25.0
    labels: list[dict] = []
    frame_index = 0
    paused = False
    frame = None
    try:
        while True:
            if not paused:
                ok, frame = source.read()
                if not ok:
                    break
                frame_index += 1
            assert frame is not None
            display = frame.copy()
            cv2.putText(
                display,
                "I=IN O=OUT SPACE=pause Q=save/quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Manual passenger flow labels", display)
            key = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key in (ord("i"), ord("o")):
                labels.append(
                    {
                        "clip": clip or "single",
                        "direction": "IN" if key == ord("i") else "OUT",
                        "frame_index": frame_index - 1,
                        "timestamp_s": round((frame_index - 1) / fps, 4),
                    }
                )
    finally:
        source.release()
        cv2.destroyAllWindows()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "direction", "frame_index", "timestamp_s"])
        writer.writeheader()
        writer.writerows(labels)
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip")
    args = parser.parse_args()
    print(f"Saved {label_video(args.video, args.output, args.clip)} manual labels to {args.output}")


if __name__ == "__main__":
    main()
